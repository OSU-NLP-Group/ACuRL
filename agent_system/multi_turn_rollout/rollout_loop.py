import torch
import numpy as np
import time
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import (
    process_image,
    to_list_of_dict,
    torch_to_numpy,
    filter_group_data,
    get_placeholder_image_array,
    load_task_data,
    get_size_mb,
    _random_sample_tasks,
)
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        # Thread pool for async image saving (reduce postprocess time).
        self._image_save_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="img_save")
        self._pending_image_saves = []
        # Dedicated thread pool for env pre-creation (avoid contention with image saving tasks).
        self._env_preload_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="env_preload")
        # Env pre-creation: after closing current env, create next-batch envs in parallel during policy update.
        self._next_env_future = None
        # Track async env-close task.
        self._env_close_future = None
    
    def _timing_enabled(self) -> bool:
        """Whether timing logs should be printed."""
        return bool(getattr(self.config.env, "timing_enabled", True))

    def _timing_verbose(self) -> bool:
        """Whether verbose per-step timing logs should be printed."""
        return bool(getattr(self.config.env, "timing_verbose", True))

    @staticmethod
    def _safe_mean(values) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _print_timing_summary(
        self,
        *,
        env_init_time: float,
        total_async_eval_time: float,
        per_step_preprocess_times,
        per_step_action_times,
        per_step_decode_times,
        per_step_env_total_times,
        per_step_postprocess_times,
        post_step_timing: dict,
    ) -> None:
        """Print a clean timing summary (release-friendly by default)."""
        if not self._timing_enabled():
            return

        steps = max(
            len(per_step_preprocess_times),
            len(per_step_action_times),
            len(per_step_decode_times),
            len(per_step_env_total_times),
            len(per_step_postprocess_times),
        )

        avg_pre = self._safe_mean(per_step_preprocess_times)
        avg_act = self._safe_mean(per_step_action_times)
        avg_dec = self._safe_mean(per_step_decode_times)
        avg_env = self._safe_mean(per_step_env_total_times)
        avg_post = self._safe_mean(per_step_postprocess_times)
        avg_total = avg_pre + avg_act + avg_dec + avg_env + avg_post

        async_img = float(post_step_timing.get("async_image_save", 0.0) or 0.0)
        async_eval = float(post_step_timing.get("async_eval", 0.0) or 0.0)
        succ_eval = float(post_step_timing.get("success_evaluator", 0.0) or 0.0)
        data_filter = float(post_step_timing.get("data_filtering", 0.0) or 0.0)
        save_result = float(post_step_timing.get("save_result", 0.0) or 0.0)

        print(
            f"[Timing] env_init={env_init_time:.3f}s steps={steps} "
            f"step_avg(pre={avg_pre:.3f}s act={avg_act:.3f}s dec={avg_dec:.3f}s env={avg_env:.3f}s post={avg_post:.3f}s total≈{avg_total:.3f}s)"
        )
        if total_async_eval_time:
            print(f"[Timing] async_eval_total={total_async_eval_time:.3f}s (non-blocking)")
        print(
            "[Timing] post_step("
            f"img_save={async_img:.3f}s async_eval={async_eval:.3f}s success_eval={succ_eval:.3f}s "
            f"filter={data_filter:.3f}s save_result={save_result:.3f}s"
            ")"
        )

    def __del__(self):
        """Clean up thread pool resources."""
        if hasattr(self, '_image_save_executor'):
            try:
                self._image_save_executor.shutdown(wait=False)
            except Exception:
                pass
        if hasattr(self, '_env_preload_executor'):
            try:
                self._env_preload_executor.shutdown(wait=False)
            except Exception:
                pass

    def _get_task_category(self, task_name):
        """Get the category of a task based on task name"""
        parts = task_name.split('.')
        return parts[0]
    
    def _process_chat_with_image_tokens(self, chat: List[Dict], images_his: List) -> List[Dict]:
        """
        Process chat messages to replace generic <|image_pad|> tokens with indexed <|image_pad|> tokens.
        
        Args:
            chat: List of chat messages, each containing 'content' field
            images_his: List of historical images for indexing
            
        Returns:
            Modified chat messages with indexed image tokens
        """
        # Count image tokens in the chat structure
        n_img_tokens = 0
        for msg in chat:
            if isinstance(msg.get("content"), list):
                for content_item in msg["content"]:
                    if content_item.get("type") == "image" and content_item.get("image") == "<|image_pad|>":
                        n_img_tokens += 1
        
        if n_img_tokens:
            start_idx = max(len(images_his) - n_img_tokens + 1, 0)
            idx_iter = iter(range(start_idx, start_idx + n_img_tokens))

            chat_mod = []
            for msg in chat:
                new_msg = copy.deepcopy(msg)
                if isinstance(new_msg.get("content"), list):
                    for content_item in new_msg["content"]:
                        if content_item.get("type") == "image" and content_item.get("image") == "<|image_pad|>":
                            content_item["image"] = f"<|image_{next(idx_iter)}_pad|>"
                chat_mod.append(new_msg)
            return chat_mod
        else:
            return chat
    
    def _get_task_goal(self, task_name, obs, idx):
        """Get the task goal"""
        if 'tasks' in obs and idx < len(obs['tasks']):
            return obs['tasks'][idx]
        else:
            raise ValueError(f"No instruction found for task {task_name}")

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        envs: EnvironmentManagerBase,
        preprocessed_images: List,  # Preprocessed images for this sample.
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text' and 'image' keys
            preprocessed_images (List): Pre-processed images to avoid redundant processing (may be empty)
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]        
        # Get observation components
        obs_chats = obs['chats']
        obs_chat = obs_chats[item]
        # Check if placeholder was used (from environment manager)
        used_placeholder = obs['placeholder_used'][item]
        
        # Use preprocessed images for this sample.
        if preprocessed_images is None:
            raise ValueError("preprocessed_images must be a list")
        keep_history_images = preprocessed_images

        is_multi_modal = len(keep_history_images) > 0

        
        chat = obs_chat
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # print("Using multi modal mode!")

            raw_prompt = prompt_with_chat_template.replace('<|image_pad|>', '<|vision_start|><|image_pad|><|vision_end|>')
            
            # Images were already preprocessed at the batch stage; use directly.
            processed_images = keep_history_images
            placeholder_from_processing = any(getattr(img, '_from_process_image_error', False) for img in processed_images)
            used_placeholder = used_placeholder or placeholder_from_processing
            
            row_dict['multi_modal_data'] = {'image': processed_images}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')


            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                # Replace <|image_pad|> with vision tokens and actual number of placeholders
                while '<|image_pad|>' in prompt_with_chat_template and index < len(image_grid_thw):
                    num_placeholders = image_grid_thw[index].prod() // merge_length
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<|image_pad|>',
                        '<|vision_start|>' + '<|placeholder|>' * num_placeholders + '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            print("Using single text modal mode!")
            raw_prompt = prompt_with_chat_template
            # Ensure consistent structure - set empty multi_modal fields
            row_dict['multi_modal_data'] = None
            row_dict['multi_modal_inputs'] = None
            
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                    tokenizer=self.tokenizer,
                                                                    max_length=self.config.data.max_prompt_length,
                                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                                    left_pad=True,
                                                                    truncation='error')
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index
            
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build final output dict
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")
        
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'index': item,
            'used_placeholder': used_placeholder
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
        envs: EnvironmentManagerBase,
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        
        # Optimization: preprocess all images in a batch to avoid per-sample processing in preprocess_single_sample.
        obs_images = obs['images']
        images_his = obs['images_his']
        all_batch_images = []
        for item in range(batch_size):
            obs_image = obs_images[item]
            image_his = images_his[item]
            all_current_images = image_his + [obs_image]
            keep_history_images = all_current_images[-envs.image_history_length:]
            
            # Batch-preprocess images (only if present, to avoid unnecessary work).
            processed_images = []
            if len(keep_history_images) > 0:
                for img in keep_history_images:
                    processed_img = process_image(img)
                    processed_images.append(processed_img)
            # Append either an empty list or the processed list to the batch.
            all_batch_images.append(processed_images)
        
        processed_samples = []
        
        # Process each sample
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                envs=envs,
                preprocessed_images=all_batch_images[item],  # Pass in preprocessed images.
            )
            processed_samples.append(processed)
        
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def preprocess_batch_threadpool(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
        envs: EnvironmentManagerBase,
        n_workers: int = 8,
    ) -> DataProto:
        """
        Process a batch of observation samples using thread pool for acceleration.
        
        This method uses ThreadPoolExecutor to process multiple samples in parallel.
        Thread pool is more suitable than process pool here because:
        1. Avoids pickle issues with tokenizer and processor
        2. Lower overhead for data sharing
        3. Good for I/O-bound and mixed CPU/I-O workloads
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
            envs (EnvironmentManagerBase): Environment manager
            n_workers (int): Number of worker threads to use (default: 4)
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        
        # Step 1: preprocess all images upfront (thread-pool parallelism over history + current frame to avoid
        # main-thread serial bottlenecks).
        image_stage_start = time.time()
        obs_images = obs['images']
        images_his = obs['images_his']
        all_batch_images = [None] * batch_size

        def prepare_images(item_idx):
            obs_image = obs_images[item_idx]
            image_his = images_his[item_idx]
            all_current_images = image_his + [obs_image]
            keep_history_images = all_current_images[-envs.image_history_length:]

            processed_images = []
            if len(keep_history_images) > 0:
                for img in keep_history_images:
                    processed_img = process_image(img)
                    processed_images.append(processed_img)
            return processed_images

        image_workers = min(n_workers, batch_size) if batch_size > 0 else 1
        with ThreadPoolExecutor(max_workers=image_workers, thread_name_prefix="preprocess_image") as image_executor:
            futures = {image_executor.submit(prepare_images, i): i for i in range(batch_size)}
            for future in as_completed(futures):
                idx = futures[future]
                all_batch_images[idx] = future.result()
        image_stage_end = time.time()
        sample_stage_start = image_stage_end
        
        # Step 2: process each sample in parallel with a thread pool.
        processed_samples = [None] * batch_size
        
        def process_single_item(item_idx):
            """Internal helper to process a single sample."""
            return self.preprocess_single_sample(
                item=item_idx,
                gen_batch=gen_batch,
                obs=obs,
                envs=envs,
                preprocessed_images=all_batch_images[item_idx],
            )
        
        # Process in parallel using a thread pool.
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="preprocess") as executor:
            # Submit all tasks and preserve ordering.
            futures = {executor.submit(process_single_item, i): i for i in range(batch_size)}
            
            # Collect results and place them back into the correct positions.
            errors = []
            for future in as_completed(futures):
                item_idx = futures[future]
                try:
                    result = future.result()
                    processed_samples[item_idx] = result
                except Exception as e:
                    errors.append((item_idx, e))
                    print(f"Error processing sample {item_idx} in threadpool mode: {e}")
                    import traceback
                    traceback.print_exc()
            
            # If there are errors, raise after all threads complete.
            if errors:
                error_msg = f"Failed to process {len(errors)}/{batch_size} samples. First error: {errors[0][1]}"
                raise RuntimeError(error_msg)
        
        # Step 3: collate all samples.
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        sample_stage_end = time.time()
        total_preprocess_time = sample_stage_end - image_stage_start
        if getattr(self.config.env, "log_preprocess_breakdown", True):
            image_duration = image_stage_end - image_stage_start
            sample_duration = sample_stage_end - sample_stage_start
            print(
                f"[PREPROCESS_BREAKDOWN] images={image_duration:.3f}s "
                f"samples={sample_duration:.3f}s "
                f"total={total_preprocess_time:.3f}s"
            )

        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            task_names: List[str] = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            task_names (List[str]): Task names for each environment (optional, for per-category metrics)
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        # Calculate per-category success rates if task_names are provided
        if task_names is not None:
            from collections import defaultdict
            category_success = defaultdict(lambda: defaultdict(list))
            
            for idx, task_name in enumerate(task_names):
                category = self._get_task_category(task_name)
                for key, value in success.items():
                    category_success[category][key].append(value[idx])
            
            # Compute average success rate per category
            for category, metrics in category_success.items():
                for metric_key, values in metrics.items():
                    # Create a new key with category suffix
                    category_metric_key = f"{metric_key}_cat_{category}"
                    success_rate[category_metric_key] = np.mean(values)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train,
            training_step: int = 0,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        # Clear pending saves that may have been left from a previous call (e.g., if an exception occurred).
        self._pending_image_saves.clear()
        
        # Initial observations from the environment
        _t_env_reset_s = time.time()
        if hasattr(envs, '_preloaded_obs'):
            obs = envs._preloaded_obs
            infos = envs._preloaded_infos
            del envs._preloaded_obs, envs._preloaded_infos
        else:
            obs, infos = envs.reset()
        _t_env_reset_e = time.time()
        env_init_time = (_t_env_reset_e - _t_env_reset_s)

        if is_train:
            save_type = "train"
        else:
            save_type = "test"

        # Initialize trajectory collection
        lenght_obs = len(obs['images'])
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)

        record_results = []
        task_names_list = []  # Store task names for per-category metrics
        for idx in range(batch_size):
            task_name = obs["task_names"][idx]
            task_names_list.append(task_name)
            task_goal = self._get_task_goal(task_name, obs, idx)
            record_results.append(
                {
                    "task_name": task_name,
                    "task_goal": task_goal,
                    "traj_uid": traj_uid[idx],
                    "chat_history": [],
                    "raw_actions": [],
                    "parsed_actions": [],
                    'is_action_valid': [],
                    "infos": [],
                    "used_placeholder": [],
                }
            )
        # Asynchronously save initial images (reduce postprocess time).
        initial_placeholder_flags = obs.get("placeholder_used", [False] * batch_size)
        for idx, (task_name, uid, image) in enumerate(zip(obs["task_names"], traj_uid, obs["images"])):
            category = self._get_task_category(task_name)
            image_to_save = image
            if idx < len(initial_placeholder_flags) and initial_placeholder_flags[idx]:
                image_to_save = get_placeholder_image_array()
            future = self._image_save_executor.submit(
                envs.save_image,
                image_to_save,
                f"{save_type}/{task_name}/training_step_{training_step}/{uid}/0",
            )
            self._pending_image_saves.append(future)
        # Trajectory collection loop
        per_step_action_times = []
        per_step_env_times = []
        per_step_eval_times = []
        per_step_total_times = []
        per_step_preprocess_times = []
        per_step_decode_times = []
        per_step_postprocess_times = []
        per_step_env_total_times = []  # Actual wall-clock time for envs.step()
        per_step_image_submit_times = []  # Time to submit async image saves
        for _step in range(self.config.env.max_steps):
            _t_total_s = time.time()
            active_masks = np.logical_not(is_done)

            for j in range(batch_size):
                chat_mod = self._process_chat_with_image_tokens(obs["chats"][j], obs["images_his"][j])
                if not is_done[j]:
                    record_results[j]["chat_history"].append(chat_mod)

            _t_preprocess_s = time.time()
            # Use threadpool-based parallel preprocessing for better performance
            # n_workers can be configured via config.env.preprocess_workers or defaults to 8
            n_workers = getattr(self.config.env, 'preprocess_workers', 128)
            batch = self.preprocess_batch_threadpool(gen_batch=gen_batch, obs=obs, envs=envs, n_workers=n_workers)
            
            # Extract placeholder usage information
            placeholder_usage = batch.non_tensor_batch.get('used_placeholder')
            if placeholder_usage is None:
                placeholder_usage = [False] * batch_size
            else:
                placeholder_usage = list(map(bool, placeholder_usage))

            # Pop keys that are only needed for generation, not for training
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "used_placeholder"]
            # Remove large multi-modal data (PIL images) - processed tensors in multi_modal_inputs are kept
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            # Remove fields not used in PPO training to reduce memory
            if "index" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("index")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # Pad batch_input to be divisible by world_size to avoid chunking errors
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            _t_preprocess_e = time.time()
            per_step_preprocess_times.append(_t_preprocess_e - _t_preprocess_s)

            _t_gen_s = time.time()
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            
            # Unpad the output to original size
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            _t_gen_e = time.time()
            per_step_action_times.append(_t_gen_e - _t_gen_s)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            # Remove timing from batch_output.meta_info to avoid union conflicts
            batch_output.meta_info.pop("timing", None)

            batch = batch.union(batch_output)

            _t_decode_s = time.time()
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            _t_decode_e = time.time()
            per_step_decode_times.append(_t_decode_e - _t_decode_s)

            for j in range(batch_size):
                if not is_done[j]:
                    record_results[j]["raw_actions"].append(text_actions[j])

            _t_step_s = time.time()
            step_result = envs.step(text_actions)
            _t_step_e = time.time()

            if len(step_result) == 4:
                next_obs, rewards, dones, infos = step_result
                truncateds = np.zeros(batch_size, dtype=bool)  # Default truncateds
            else:
                raise ValueError(f"Unexpected step result length: {len(step_result)}")

            # Record actual wall-clock time for envs.step()
            env_total_time = (_t_step_e - _t_step_s)
            per_step_env_total_times.append(env_total_time)
            
            # Split env step vs eval using infos if available (for internal breakdown)
            if isinstance(infos, list) and len(infos) > 0 and isinstance(infos[0], dict):
                import numpy as _np
                env_time = _np.mean([info.get('t_env_step', env_total_time) for info in infos]).item()
                eval_time = _np.mean([info.get('t_eval', 0.0) for info in infos]).item()
            else:
                env_time = env_total_time
                eval_time = 0.0
            per_step_env_times.append(env_time)
            per_step_eval_times.append(eval_time)
            
            _t_postprocess_s = time.time()
            parsed_actions = next_obs["parsed_actions"]
            for j in range(batch_size):
                if not is_done[j]:
                    record_results[j]["parsed_actions"].append(parsed_actions[j])
                    record_results[j]["infos"].append(infos[j])
                    record_results[j]["used_placeholder"].append(placeholder_usage[j])
                    
                    # Log placeholder usage at rollout level
                    if placeholder_usage[j]:
                        print(f"[ROLLOUT_PLACEHOLDER] Step {_step+1}, Env {j}, Traj_UID: {traj_uid[j]} - Placeholder image used")

            # Separate timing: submit async image-save tasks (submission is fast and does not block the main loop).
            _t_img_submit_s = time.time()
            for idx, (task_name, uid, image, image_his, parsed_action) in enumerate(zip(next_obs["task_names"], traj_uid, next_obs["images"], next_obs["images_his"], parsed_actions)):
                if not is_done[idx]:
                    current_step = _step + 1
                    category = self._get_task_category(task_name)
                    image_source = image_his[-1] if len(image_his) > 0 else image
                    image_to_save = image_source
                    if idx < len(placeholder_usage) and placeholder_usage[idx]:
                        image_to_save = get_placeholder_image_array()
                    future = self._image_save_executor.submit(
                        envs.save_image,
                        image_to_save,
                        f"{save_type}/{task_name}/training_step_{training_step}/{uid}/step{current_step}",
                    )
                    self._pending_image_saves.append(future)
            _t_img_submit_e = time.time()
            per_step_image_submit_times.append(_t_img_submit_e - _t_img_submit_s)
            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
                for j in range(batch_size):
                    if not is_done[j]:
                        record_results[j]["is_action_valid"].append(infos[j]['is_action_valid'])
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            _t_postprocess_e = time.time()
            per_step_postprocess_times.append(_t_postprocess_e - _t_postprocess_s)
            _t_total_e = time.time()
            per_step_total_times.append(_t_total_e - _t_total_s)
            if is_done.all():
                break
        
        # Initialize timing dictionary for post-step operations
        post_step_timing = {}
        
        # Wait for all async image saves here (keeps the main loop non-blocking).
        async_image_save_time = 0.0
        if self._pending_image_saves:
            _t_wait_img_s = time.time()
            failed_count = 0
            try:
                for future in self._pending_image_saves:
                    try:
                        future.result(timeout=30)  # 30s timeout
                    except Exception as e:
                        failed_count += 1
                        print(f"[WARNING] Async image save failed: {e}")
            finally:
                # Ensure the list is cleared regardless of exceptions.
                _t_wait_img_e = time.time()
                async_image_save_time = _t_wait_img_e - _t_wait_img_s
                total_count = len(self._pending_image_saves)
                success_count = total_count - failed_count
                if self._timing_verbose() or failed_count:
                    print(
                        f"[Timing] async_image_save waited={async_image_save_time:.3f}s "
                        f"(total={total_count}, ok={success_count}, failed={failed_count})"
                    )
                self._pending_image_saves.clear()
        post_step_timing["async_image_save"] = async_image_save_time
        
        # Wait for all pending async evaluations to complete
        total_async_eval_time = 0.0
        if hasattr(envs, 'wait_all_pending_evals'):
            print("[Async Eval] Waiting for all pending evaluations before success evaluation...")
            _t_async_eval_s = time.time()
            episode_rewards, total_async_eval_time = envs.wait_all_pending_evals(total_batch_list, episode_rewards, total_infos)
            _t_async_eval_e = time.time()
            post_step_timing["async_eval"] = _t_async_eval_e - _t_async_eval_s
            
            # Sync updated total_infos (with CUAJudge results) back to record_results
            for idx, record_result in enumerate(record_results):
                if idx < len(total_infos):
                    record_result["infos"] = total_infos[idx]
                    print(f"[CUAJudge] Synced updated infos to record_results[{idx}]")
        else:
            post_step_timing["async_eval"] = 0.0
        
        # Success evaluator
        _t_success_eval_s = time.time()
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        _t_success_eval_e = time.time()
        post_step_timing["success_evaluator"] = _t_success_eval_e - _t_success_eval_s
        
        # Minimal filter: always drop only trajectories that used placeholder in vanilla loop
        _t_filter_s = time.time()
        if is_train:
            keep_indices = []
            for i, rec in enumerate(record_results):
                used_ph_list = rec.get("used_placeholder", [])
                if not any(used_ph_list):
                    keep_indices.append(i)
                else:
                    # Print filtered trajectory info
                    env_traj_uid = rec.get('traj_uid', 'unknown')
                    print(f"[FILTER_PLACEHOLDER] Filtering Trajectory {i}, Traj_UID: {env_traj_uid}")

            if len(keep_indices) != len(record_results):
                num_removed = len(record_results) - len(keep_indices)
                print(f"Filtering out {num_removed} trajectories due to placeholder usage")
                # Filter collections to keep only valid trajectories
                total_batch_list = [total_batch_list[i] for i in keep_indices]
                episode_rewards = episode_rewards[keep_indices]
                episode_lengths = episode_lengths[keep_indices]
                traj_uid = traj_uid[keep_indices]
                task_names_list = [task_names_list[i] for i in keep_indices]
                # Filter success dict if aligned by trajectory length
                orig_len = len(record_results)
                success = {k: v[keep_indices] for k, v in success.items() if len(v) == orig_len}
                # Also shrink record_results used for saving
                record_results = [record_results[i] for i in keep_indices]
        _t_filter_e = time.time()
        post_step_timing["data_filtering"] = _t_filter_e - _t_filter_s

        # Save results
        _t_save_result_s = time.time()
        for idx, record_result in enumerate(record_results):
            for k, v in success.items():
                record_result[k] = v[idx]
            envs.save_result(record_result, f"{save_type}/{record_result['task_name']}/training_step_{training_step}/{record_result['traj_uid']}/{record_result['traj_uid']}")
        _t_save_result_e = time.time()
        post_step_timing["save_result"] = _t_save_result_e - _t_save_result_s
        # Timing logs (clean by default; enable verbose mode for per-step breakdown).
        if self._timing_verbose() and self._timing_enabled():
            print(
                f"[Timing] verbose per-step breakdown (steps={len(per_step_action_times)}). "
                "Note: with async eval enabled, eval time inside steps may show 0 (non-blocking)."
            )
            for i in range(len(per_step_action_times)):
                prep = per_step_preprocess_times[i] if i < len(per_step_preprocess_times) else 0.0
                ag = per_step_action_times[i] if i < len(per_step_action_times) else 0.0
                dec = per_step_decode_times[i] if i < len(per_step_decode_times) else 0.0
                et = per_step_env_total_times[i] if i < len(per_step_env_total_times) else 0.0
                post = per_step_postprocess_times[i] if i < len(per_step_postprocess_times) else 0.0
                print(
                    f"[Timing] step={i} pre={prep:.3f}s act={ag:.3f}s dec={dec:.3f}s env={et:.3f}s post={post:.3f}s"
                )

        self._print_timing_summary(
            env_init_time=env_init_time,
            total_async_eval_time=total_async_eval_time,
            per_step_preprocess_times=per_step_preprocess_times,
            per_step_action_times=per_step_action_times,
            per_step_decode_times=per_step_decode_times,
            per_step_env_total_times=per_step_env_total_times,
            per_step_postprocess_times=per_step_postprocess_times,
            post_step_timing=post_step_timing,
        )
            
            
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, task_names_list
    

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase = None,
            is_train: bool = True,
            training_step: int = 0,
            next_batch_info: dict = None,  # Next batch info (contains next_batch_dict and next_training_step).
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager (optional if using preloaded).
            is_train (bool): Whether in training mode.
            training_step (int): Current training step number.
            next_batch_info (dict): Info for next batch (next_batch_dict, next_training_step).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        ## initialize the environment
        from agent_system.environments import make_envs

        # Check if we should use random task sampling
        if is_train and getattr(self.config.env, 'random_task_sampling', False):
            task_names = _random_sample_tasks(self.config, training_step)
        else:
            task_names = gen_batch.non_tensor_batch["task_name"]
            task_names = task_names.tolist()
        print("Task names:", task_names)
        
        tasks = []
        # Get subfolder from config based on train/val mode
        if is_train:
            subfolder = getattr(self.config.env, 'train_subfolder', None)
        else:
            subfolder = getattr(self.config.env, 'val_subfolder', None)
        for task_name in task_names:
            task_data = load_task_data(task_name, subfolder=subfolder)
            tasks.append(task_data)

        # Try to use a pre-created environment (if creation was started in the previous step).
        env_create_time = 0.0  # Initialize env_create_time.
        if self._next_env_future is not None:
            print(f"[EnvPreload] Waiting for pre-created environment...")
            try:
                wait_start = time.time()
                envs = self._next_env_future.result(timeout=3000)  # 3000s (50min) to accommodate startup of 128 containers.
                wait_time = time.time() - wait_start
                print(f"[EnvPreload] Got pre-created environment (wait={wait_time:.2f}s)")
                # When using a pre-created env, env_create_time should only count waiting time (usually ~0).
                # Do not include background creation time since it finished asynchronously.
                env_create_time = wait_time
                self._next_env_future = None
            except Exception as e:
                print(f"[EnvPreload] Failed to use pre-created env: {e}, creating new")
                self._next_env_future = None
                envs = None
        
        # If no pre-created env is available, create a new one.
        # In validation, task_names may already be repeated, so env count = len(task_names) = len(tasks).
        if envs is None:
            env_create_start = time.time()
            envs = make_envs(tasks=tasks, config=self.config, is_train=is_train, processor=self.processor)
            env_create_time = time.time() - env_create_start
            print(f"[EnvPreload] Created environment (time={env_create_time:.2f}s)")


        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_task_names = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                training_step=training_step,
            )
        else:
            # Vanilla Sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_task_names = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
                training_step=training_step,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        
        # Print memory size of total_batch_list in GB
        size_mb = get_size_mb(total_batch_list)
        size_gb = size_mb / 1024
        print(f"total_batch_list size: {size_gb:.4f} GB ({size_mb:.2f} MB)")

        # Create trajectory data
        _t_gather_s = time.time()
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            task_names=total_task_names,
        )
        _t_gather_e = time.time()
        gather_rollout_data_time = _t_gather_e - _t_gather_s
        if self._timing_verbose() and self._timing_enabled():
            print(f"[Timing] gather_rollout_data: {gather_rollout_data_time:.3f}s")
        
        # Add timing to meta_info
        if "timing" not in gen_batch_output.meta_info:
            gen_batch_output.meta_info["timing"] = {}
        # Record env_init time (when using pre-created envs, only count waiting time; exclude background creation).
        gen_batch_output.meta_info["timing"]["env_init"] = env_create_time
        if self._timing_verbose() and self._timing_enabled():
            print(f"[Timing] Setting env_init to meta_info: {env_create_time:.3f}s")
        gen_batch_output.meta_info["timing"]["gather_rollout_data"] = gather_rollout_data_time
        
        
        # Close environment (async version).
        _t_env_close_s = time.time()
        # Keep a reference to envs for use inside the async function.
        envs_to_close = envs
        
        def _close_env_async():
            """Internal helper to close the environment asynchronously."""
            try:
                # Release build: always close all envs (no selective-close / placeholder keep-alive).
                envs_to_close.close()
                _t_env_close_e = time.time()
                env_close_time = _t_env_close_e - _t_env_close_s
                if self._timing_verbose() and self._timing_enabled():
                    print(f"[Timing] env_close completed (async): {env_close_time:.3f}s")
                return env_close_time
            except Exception as e:
                print(f"[ERROR] Async env close failed: {e}")
                import traceback
                traceback.print_exc()
                return 0.0
        
        # Submit async env-close task.
        self._env_close_future = self._env_preload_executor.submit(_close_env_async)
        print("[EnvClose] Submitted async env close task")
        
        # After env close, if next batch info is provided, start background creation (parallel with policy update).
        if next_batch_info is not None:
            try:
                next_training_step = next_batch_info.get('next_training_step', training_step + 1)
                next_is_train = next_batch_info.get('next_is_train', is_train)  # Default to current is_train.
                
                # Determine next batch tasks.
                if next_is_train and getattr(self.config.env, 'random_task_sampling', False):
                    next_task_names = _random_sample_tasks(self.config, next_training_step)
                else:
                    next_batch_dict = next_batch_info.get('next_batch_dict')
                    if next_batch_dict:
                        next_batch = DataProto.from_single_dict(next_batch_dict)
                        next_task_names = next_batch.non_tensor_batch.get("task_name")
                        if next_task_names is not None:
                            next_task_names = next_task_names.tolist() if hasattr(next_task_names, 'tolist') else list(next_task_names)
                            # Validation: val_group_n=1 so env count = len(task_names),
                            # but test_batch is repeated val_kwargs.n times, so repeat the task list as well.
                            if not next_is_train:
                                val_kwargs_n = getattr(self.config.actor_rollout_ref.rollout.val_kwargs, 'n', 1)
                                original_count = len(next_task_names)
                                next_task_names = next_task_names * val_kwargs_n
                                print(f"[EnvPreload] Validation: repeating {original_count} tasks {val_kwargs_n}x -> {len(next_task_names)} tasks")
                    else:
                        next_task_names = None
                
                # Load task data and start background creation (after async close completes).
                if next_task_names:
                    subfolder = getattr(self.config.env, 'train_subfolder', None) if next_is_train else getattr(self.config.env, 'val_subfolder', None)
                    next_tasks = [load_task_data(name, subfolder=subfolder) for name in next_task_names]
                    
                    # Save a reference to _env_close_future before submitting the task. Otherwise, the next step
                    # may overwrite self._env_close_future and we might end up waiting on the wrong Future.
                    close_future_ref = self._env_close_future
                    
                    def _create_env_after_close():
                        """Internal helper: wait for env close completion, then create a new env."""
                        try:
                            # Wait for async close to complete (Future.result() is safe to call multiple times).
                            # Use the saved reference instead of self._env_close_future, which may be overwritten.
                            if close_future_ref is not None:
                                print("[EnvPreload] Waiting for previous env close to complete...")
                                close_time = close_future_ref.result(timeout=600)
                                print(f"[EnvPreload] Previous env close completed (time={close_time:.2f}s)")
                            else:
                                print("[EnvPreload] Warning: close_future_ref is None, skipping wait for env close")
                            
                            # Now it is safe to create a new environment.
                            print(f"[EnvPreload] Background creating env for next step (is_train={next_is_train}, {len(next_tasks)} tasks)...")
                            t0 = time.time()
                            env = make_envs(tasks=next_tasks, config=self.config, is_train=next_is_train, processor=self.processor)
                            t1 = time.time()
                            obs, infos = env.reset()
                            t2 = time.time()
                            env._preloaded_obs = obs
                            env._preloaded_infos = infos
                            print(f"[EnvPreload] Background env created and reset with {len(next_tasks)} tasks (create={t1-t0:.1f}s, reset={t2-t1:.1f}s, total={t2-t0:.1f}s)")
                            return env
                        except Exception as e:
                            print(f"[EnvPreload] Background creation failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return None
                    
                    self._next_env_future = self._env_preload_executor.submit(_create_env_after_close)
                    print(f"[EnvPreload] Submitted background task for {len(next_tasks)} tasks (next_is_train={next_is_train})")
            except Exception as e:
                print(f"[EnvPreload] Failed to start preload: {e}")
        
        import gc
        gc.collect()
        
        return gen_batch_output