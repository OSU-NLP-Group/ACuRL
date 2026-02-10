from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import time
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.env_package.OSWorld.ui_tars_action_parser import add_box_token
from agent_system.environments.prompts.ui_tars import (
    COMPUTER_USE_DOUBAO_2,
    COMPUTER_USE_DOUBAO_2_without_password,
)
from agent_system.reward_manager import CUAJudgeEvaluator


class OSWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(
        self,
        task_names,
        envs,
        projection_f,
        env_name,
        image_history_length=2,
        image_save_dir=None,
        use_password=True,
        model_type="qwen25vl",
        enable_cuajudge=False,
        **kwargs,
    ):
        self.buffers = None
        self.original_tasks = envs.sample_tasks
        self.image_history_length = image_history_length
        self.use_password = use_password
        self.model_type = model_type  # Store model type for message construction
        
        # Flags
        # - enable_cuajudge: use CUAJudge to override terminal rewards (required for training)
        # - enable_rule_based: additionally run rule-based evaluation for comparison/metrics
        self.enable_cuajudge = bool(kwargs.get("enable_cuajudge", enable_cuajudge))
        self.enable_rule_based = bool(kwargs.get("enable_rule_based", False))

        # Initialize CUAJudge if enabled
        if self.enable_cuajudge:
            cuajudge_config = kwargs.get("cuajudge_config", {})
            self.cuajudge_evaluator = CUAJudgeEvaluator(**cuajudge_config)
            self.cuajudge_evaluated_envs = set()  # Track which environments have been evaluated
            # Set CUAJudge temp directory
            self.cuajudge_temp_dir = kwargs.get("cuajudge_temp_dir", "./tmp")
            print(f"[OSWorldEnvironmentManager] CUAJudge enabled with config: {cuajudge_config}")
            print(f"[OSWorldEnvironmentManager] CUAJudge temp directory: {self.cuajudge_temp_dir}")
            print(f"[OSWorldEnvironmentManager] Rule-based enabled: {self.enable_rule_based}")
            
            # Initialize async evaluation infrastructure
            from concurrent.futures import ThreadPoolExecutor
            self.eval_executor = ThreadPoolExecutor(max_workers=128)
            self.pending_eval_futures = []  # List of (future, context_data) tuples
            print(f"[OSWorldEnvironmentManager] Async evaluation enabled with ThreadPoolExecutor")
        else:
            self.cuajudge_evaluator = None
            self.cuajudge_evaluated_envs = set()
            self.cuajudge_temp_dir = "./tmp"
            self.eval_executor = None
            self.pending_eval_futures = []
            # No CUAJudge: rule-based must be explicitly enabled, otherwise reward is undefined.
            if not self.enable_rule_based:
                raise ValueError(
                    "Invalid config: enable_cuajudge=False and enable_rule_based=False. "
                    "At least one reward path must be enabled."
                )
            print("[OSWorldEnvironmentManager] CUAJudge disabled; using rule-based reward calculation")

        self.task_names = []
        self.tasks = []
        
        for task in self.original_tasks:
            task_id = task['id']
            if task.get("snapshot") == "sci_bench":
                category = task.get("type")
            elif 'snapshot' in task:
                category = task['snapshot']  # OSWorld often uses 'snapshot' as category
            elif task_id.startswith('ow-'):
                # Extract category from ow-{category}-{number} format
                parts = task_id.split('-')
                if len(parts) >= 3:
                    category = f"{parts[0]}-{parts[1]}"  # Extract category as ow-{category}
                else:
                    raise ValueError(f"Invalid ow task ID format: {task_id}")
            else:
                raise ValueError(f"Category not found in task: {task}")
            
            task_name = f"{category}.{task_id}"
            self.task_names.append(task_name)
            
            if 'instruction' in task:
                self.tasks.append(task['instruction'])
            else:
                raise ValueError(f"Instruction not found in task: {task}")
            
        super().__init__(envs, projection_f, env_name, image_save_dir)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
                
        self.pre_obs = obs

        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        
        # Reset CUAJudge evaluation tracking for new episode
        if hasattr(self, "cuajudge_evaluated_envs"):
            self.cuajudge_evaluated_envs.clear()
            print("[OSWorldEnvironmentManager] Reset CUAJudge evaluation tracking")
        
        # Clear any pending evaluations from previous batch
        if hasattr(self, 'pending_eval_futures') and self.pending_eval_futures:
            print(f"[OSWorldEnvironmentManager] Warning: {len(self.pending_eval_futures)} pending evals from previous batch, clearing...")
            self.pending_eval_futures.clear()

        images, placeholder_used = self._extract_images(obs)
        observations = {
            "tasks": self.tasks,
            "task_names": self.task_names,
            "images": images,
            "images_his": [[] for _ in range(len(obs))],
            "placeholder_used": placeholder_used,
        }
        
        # Initialize with system prompt for each environment
        actions = []
        for env_idx in range(len(obs)):
            # Use different prompt based on model type
            if self.model_type == "qwen3vl":
                # Qwen3-VL: save raw instruction (prompt will be built in build_chat_obs with history)
                prompt = self.tasks[env_idx]
            else:
                # UI-TARS format (default)
                if self.use_password:
                    prompt = COMPUTER_USE_DOUBAO_2.format(
                        language="English",
                        instruction=self.tasks[env_idx],
                    )
                else:
                    prompt = COMPUTER_USE_DOUBAO_2_without_password.format(
                        language="English",
                        instruction=self.tasks[env_idx],
                    )
            actions.append(prompt)
        self.save_to_history_buffer(observations["images"], actions)

        full_chat_obs = self.build_chat_obs(obs)
        observations["chats"] = full_chat_obs

        return observations, infos

    def step(self, text_actions: List[str]):
        # Per-env real screen size for projection (REQUIRED; do NOT use global env vars).
        screen_sizes = [(int(o["screen_width"]), int(o["screen_height"])) for o in self.pre_obs]
        actions, valids = self.projection_f(text_actions, screen_sizes=screen_sizes)
        _t_env_s = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        _t_env_e = time.time()
        self.pre_obs = next_obs

        images, placeholder_used = self._extract_images(next_obs)
        next_observations = {
            "tasks": self.tasks,
            "task_names": self.task_names,
            "raw_actions": text_actions,
            "parsed_actions": actions,
            'images': images,
            "images_his": [[rec["image"] for rec in buf] for buf in self.buffers],
            "placeholder_used": placeholder_used,
        }
        self.save_to_history_buffer(next_observations["images"], text_actions)
        full_chat_obs = self.build_chat_obs(next_obs)
        next_observations["chats"] = full_chat_obs

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            # attach per-env env_step timing
            info['t_env_step'] = float(_t_env_e - _t_env_s)

        # Handle CUAJudge evaluation if enabled (ASYNC version)
        if self.enable_cuajudge and self.cuajudge_evaluator:
            # Only evaluate environments that are done and haven't been evaluated yet
            done_indices = np.where(dones)[0]
            new_done_indices = [i for i in done_indices if i not in self.cuajudge_evaluated_envs]
            
            if new_done_indices:
                print(f"[CUAJudge] Found {len(new_done_indices)} new done environments to evaluate (async)")
                _t_eval_s = time.time()
                
                # Submit individual async evaluation task for each environment (true parallelism)
                # This allows ThreadPoolExecutor to parallelize across environments
                for env_idx in new_done_indices:
                    future = self.eval_executor.submit(
                        self._evaluate_with_cuajudge,
                        text_actions, next_observations, infos, dones, rewards, [env_idx]
                    )
                    
                    # Store future with context for later retrieval
                    # Note: We only need the original reward for this specific environment, not the whole array
                    context = {
                        'eval_start_time': _t_eval_s,
                        'done_indices': [env_idx],
                        'original_reward': float(rewards[env_idx]) if isinstance(rewards, np.ndarray) else rewards[env_idx],
                        'original_infos': infos.copy()  # Keep full copy for info updates
                    }
                    self.pending_eval_futures.append((future, context))
                
                # Mark these environments as evaluated (to avoid re-evaluation)
                self.cuajudge_evaluated_envs.update(new_done_indices)
                print(
                    f"[CUAJudge] Submitted {len(new_done_indices)} individual async eval tasks. "
                    f"Total pending: {len(self.pending_eval_futures)}"
                )
            else:
                print(
                    f"[CUAJudge] No new done environments to evaluate "
                    f"(already evaluated: {len(self.cuajudge_evaluated_envs)})"
                )
            
            rewards = to_numpy(rewards)
        else:
            rewards = to_numpy(rewards)
            
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def wait_all_pending_evals(self, total_batch_list, episode_rewards, total_infos):
        """
        Wait for all pending async evaluations to complete and update the batch results.
        Should be called at the end of each rollout batch.
        
        Args:
            total_batch_list: List of trajectory data for each environment
            episode_rewards: Episode rewards array to update
            total_infos: List of info dictionaries for each environment (will be updated in place)
            
        Returns:
            Tuple of (updated_episode_rewards, total_eval_wall_time)
        """
        if not self.pending_eval_futures:
            print("[CUAJudge] No pending evaluations to wait for")
            return episode_rewards, 0.0
        
        print(f"[CUAJudge] Waiting for {len(self.pending_eval_futures)} pending evaluations...")
        total_eval_start = time.time()
        
        updated_episode_rewards = episode_rewards.copy()
        total_eval_time = 0.0
        
        # Wait for all futures to complete (order doesn't matter since we use done_indices to map results)
        for future, context in self.pending_eval_futures:
            eval_start_time = context['eval_start_time']
            done_indices = context['done_indices']
            
            # Wait for this evaluation to complete
            try:
                updated_rewards, updated_infos = future.result()
                eval_end_time = time.time()
                eval_duration = eval_end_time - eval_start_time
                total_eval_time += eval_duration
                
                print(f"[CUAJudge] Completed eval for envs {done_indices}, duration: {eval_duration:.2f}s")
                
                # Update episode rewards for the evaluated environments
                # Since we submit each environment separately, done_indices should always have exactly one element
                for idx in done_indices:
                    if idx < len(updated_episode_rewards):
                        # Get original reward from context (stored per environment)
                        original_reward = context['original_reward']
                        # Subtract the original reward and add the CUAJudge reward
                        updated_episode_rewards[idx] = updated_episode_rewards[idx] - original_reward + updated_rewards[idx]
                        print(
                            f"[CUAJudge] Updated reward for env {idx}: {original_reward:.2f} -> {updated_rewards[idx]:.2f}, "
                            f"episode_reward: {updated_episode_rewards[idx]:.2f}"
                        )
                        
                        # Update info in total_infos for the last step of this trajectory
                        if idx < len(total_infos) and len(total_infos[idx]) > 0:
                            # Find the last active step (where active_masks=True) in total_batch_list
                            last_step_idx = None
                            if idx < len(total_batch_list):
                                for step_idx in reversed(range(len(total_batch_list[idx]))):
                                    if total_batch_list[idx][step_idx].get('active_masks', False):
                                        last_step_idx = step_idx
                                        break
                            
                            # Fallback to absolute last step if no active step found
                            if last_step_idx is None:
                                last_step_idx = len(total_infos[idx]) - 1
                                print(f"[CUAJudge] Warning: No active step found for env {idx}, using last step {last_step_idx}")
                            
                            if idx < len(updated_infos):
                                # Update the last active step's info in total_infos
                                total_infos[idx][last_step_idx].update(
                                    {
                                        "won": updated_infos[idx].get(
                                            "won", total_infos[idx][last_step_idx].get("won", False)
                                        ),
                                        "task_score": updated_infos[idx].get(
                                            "task_score", total_infos[idx][last_step_idx].get("task_score", 0.0)
                                        ),
                                        "reward": updated_infos[idx].get(
                                            "reward", total_infos[idx][last_step_idx].get("reward", 0.0)
                                        ),
                                        "cuajudge_evaluated": updated_infos[idx].get("cuajudge_evaluated", True),
                                        "cuajudge_response": updated_infos[idx].get("cuajudge_response", ""),
                                        "cuajudge_evaluation": updated_infos[idx].get("cuajudge_evaluation", {}),
                                    }
                                )
                                # Preserve rule_based results if they exist
                                for key in ['rule_based_task_score', 'rule_based_won', 'rule_based_reward']:
                                    if key in updated_infos[idx]:
                                        total_infos[idx][last_step_idx][key] = updated_infos[idx][key]
                                print(
                                    f"[CUAJudge] Updated total_infos[{idx}][{last_step_idx}] info: "
                                    f"won={total_infos[idx][last_step_idx]['won']}, task_score={total_infos[idx][last_step_idx]['task_score']}"
                                )
                            
            except Exception as e:
                print(f"[CUAJudge] Error waiting for eval future: {e}")
                import traceback
                traceback.print_exc()
        
        total_wait_time = time.time() - total_eval_start
        print(
            f"[CUAJudge] All evaluations complete. Total eval time: {total_eval_time:.2f}s, "
            f"Wall time: {total_wait_time:.2f}s (parallel speedup: {total_eval_time/max(total_wait_time, 0.1):.2f}x)"
        )
        
        # Clear pending futures
        self.pending_eval_futures.clear()
        
        return updated_episode_rewards, total_wait_time

    def _evaluate_with_cuajudge(
        self,
        text_actions: List[str],
        next_observations: Dict,
        infos: List[Dict],
        dones: np.ndarray,
        original_rewards: np.ndarray,
        done_indices: List[int] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Evaluate rewards using CUAJudge for environments that are done.
        """
        
        # Convert to numpy if needed
        dones = to_numpy(dones)
        original_rewards = to_numpy(original_rewards)
        
        # Start with original rewards and deep copy infos to avoid modifying original data
        rewards = copy.deepcopy(original_rewards)  # Deep copy rewards to avoid modifying original
        updated_infos = copy.deepcopy(infos)
        
        # Use provided done_indices or find all done environments
        if done_indices is None:
            done_indices = np.where(dones)[0]
        
        if len(done_indices) == 0:
            print("[CUAJudge] No environments are done, using original rewards")
            return rewards, updated_infos
        
        print(f"[CUAJudge] Evaluating {len(done_indices)} done environments with CUAJudge")
        
        # Create temporary directory for saving images
        os.makedirs(self.cuajudge_temp_dir, exist_ok=True)
        print(f"[CUAJudge] Created/verified temp directory: {self.cuajudge_temp_dir}")
        
        # Create a unique subdirectory for this evaluation session
        # Use done_indices in the name to avoid conflicts when multiple futures run in parallel
        done_indices_str = "_".join(map(str, done_indices))
        temp_dir = os.path.join(self.cuajudge_temp_dir, f"cuajudge_session_{done_indices_str}_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create tasks for CUAJudge evaluation
        cuajudge_tasks = []
        for i in done_indices:
            # Get action history from buffer
            action_history = []
            if i < len(self.buffers) and self.buffers[i]:
                action_history = [rec["action"] for rec in self.buffers[i]]
            # Add current action
            if text_actions[i]:
                action_history.append(text_actions[i])
            
            # Save only historical images from buffer (in chronological order)
            all_image_paths = []
            
            if i < len(self.buffers) and self.buffers[i]:
                for j, record in enumerate(self.buffers[i]):
                    if 'image' in record and record['image'] is not None:
                        hist_image_path = os.path.join(temp_dir, f"task_{i}_hist_{j}.png")
                        hist_saved = self._save_image_for_cuajudge(record['image'], hist_image_path)
                        if hist_saved and os.path.exists(hist_image_path):
                            all_image_paths.append(hist_image_path)
                            print(f"[CUAJudge] Saved historical image {j} for task {i}: {hist_image_path}")
                
                if all_image_paths:  # Only add task if we have images
                    print(f"[CUAJudge] Task {i} prepared with {len(all_image_paths)} historical images: {all_image_paths}")
                    
                    task = {
                        'task': self.tasks[i],
                        'action_thoughts': None,
                        'last_actions': action_history,
                        'images_path': all_image_paths,
                        'input_image_paths': None
                    }
                    cuajudge_tasks.append((i, task))
                else:
                    print(f"[CUAJudge] Warning: No historical images available for task {i}")
            else:
                print(f"[CUAJudge] Warning: No buffer history available for task {i}")
        
        if not cuajudge_tasks:
            print("[CUAJudge] No valid tasks for CUAJudge evaluation")
            return rewards, updated_infos
        
        # Run CUAJudge evaluation asynchronously for all done tasks
        # Create event loop if not exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Create async tasks for parallel evaluation
        async def evaluate_single_task(task_idx, task_data):
            """Evaluate a single task asynchronously"""
            try:
                print(f"[CUAJudge] Starting evaluation for task {task_idx}...")
                
                # Verify all image files exist before calling CUAJudge
                image_paths = task_data['images_path'] if task_data['images_path'] else []
                missing_files = []
                for img_path in image_paths:
                    if not os.path.exists(img_path):
                        missing_files.append(img_path)
                
                if missing_files:
                    raise FileNotFoundError(f"Missing image files for task {task_idx}: {missing_files}")
                
                print(f"[CUAJudge] Verified {len(image_paths)} image files exist for task {task_idx}")
                
                # Run CUAJudge evaluation
                result = await self.cuajudge_evaluator.cuajudge_general_eval(
                    task=task_data['task'],
                    input_image_paths=task_data['input_image_paths'],
                    action_thoughts=task_data['action_thoughts'],
                    last_actions=task_data['last_actions'],
                    images_path=task_data['images_path']
                )
                
                # Unpack the result (must be a 3-tuple)
                if len(result) != 3:
                    raise ValueError(f"cuajudge_general_eval must return (response, reward, details); got: {result}")
                response, reward, cuajudge_details = result
                
                print(f"[CUAJudge] Task {task_idx} evaluation completed: reward={reward}")
                
                return {
                    'idx': task_idx,
                    'response': response,
                    'reward': reward,
                    'cuajudge_details': cuajudge_details,
                    'task_data': task_data,
                    'success': True
                }
                
            except Exception as e:
                print(f"[CUAJudge] Error evaluating task {task_idx}: {e}")
                return {
                    'idx': task_idx,
                    'error': str(e),
                    'success': False
                }
        
        # Create all async tasks
        async_tasks = [
            evaluate_single_task(idx, task_data) 
            for idx, task_data in cuajudge_tasks
        ]
        
        print(f"[CUAJudge] Starting parallel evaluation of {len(async_tasks)} tasks...")
        
        # Execute all tasks concurrently and wait for completion
        results = loop.run_until_complete(asyncio.gather(*async_tasks, return_exceptions=True))
        
        print(f"[CUAJudge] All {len(results)} tasks completed")
        
        # Process results in order
        for result in results:
            if isinstance(result, Exception):
                print(f"[CUAJudge] Task evaluation failed with exception: {result}")
                continue
                
            if not result['success']:
                # Handle failed evaluation
                idx = result['idx']
                if idx < len(updated_infos):
                    updated_infos[idx]["cuajudge_evaluated"] = False
                    updated_infos[idx]["cuajudge_error"] = result["error"]
                continue
            
            # Process successful evaluation
            idx = result['idx']
            response = result['response']
            reward = result['reward']
            cuajudge_details = result['cuajudge_details']
            task_data = result['task_data']
            
            # Update rewards and infos
            rewards[idx] = reward
            
            if idx < len(updated_infos):
                updated_infos[idx]['task_score'] = reward
                updated_infos[idx]['won'] = (reward == 1.0)
                updated_infos[idx]['reward'] = reward
                updated_infos[idx]["cuajudge_evaluated"] = True
                updated_infos[idx]["cuajudge_response"] = response
                
                # Add comprehensive CUAJudge evaluation details to info
                updated_infos[idx]["cuajudge_evaluation"] = {
                    'response': response,
                    'task_score': reward,
                    'reward': reward,
                    'predicted_label': int(reward == 1.0),
                    'thoughts': cuajudge_details.get('thoughts', []),
                    'image_judge_record': cuajudge_details.get('image_judge_record', []),
                    'key_points': cuajudge_details.get('key_points', ''),
                    'cuajudge_config': {
                        'key_model': getattr(self.cuajudge_evaluator, 'key_identification_screenshot_model', 'unknown'),
                        'outcome_model': getattr(self.cuajudge_evaluator, 'key_points_outcome_model', 'unknown'),
                        'max_image': getattr(self.cuajudge_evaluator, 'max_image', 50),
                        'score_threshold': getattr(self.cuajudge_evaluator, 'score_threshold', 3),
                    }
                }
                
                # Add rule-based comparison results only when explicitly enabled
                if self.enable_rule_based:
                    original_info = infos[idx] if idx < len(infos) else {}
                    if "rule_based_task_score" in original_info:
                        updated_infos[idx]["rule_based_comparison"] = {
                            "task_score": original_info.get("rule_based_task_score", 0.0),
                            "won": original_info.get("rule_based_won", False),
                            "reward": original_info.get("rule_based_reward", 0.0),
                            "evaluation_failed": original_info.get("rule_based_evaluation_failed", False),
                            "evaluation_error": original_info.get("rule_based_evaluation_error", None),
                            "evaluation_method": original_info.get("rule_based_evaluation_method", "rule_based"),
                        }
                    else:
                        updated_infos[idx]["rule_based_comparison"] = {
                            "task_score": 0.0,
                            "won": False,
                            "reward": 0.0,
                            "evaluation_failed": True,
                            "evaluation_error": "Rule-based evaluation was enabled but no rule-based results were found",
                            "evaluation_method": "rule_based_missing",
                        }
                
                # Add task execution details
                updated_infos[idx]['task_execution'] = {
                    'images_used': task_data['images_path'],
                    'total_images': len(task_data['images_path']),
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            
            print(f"[CUAJudge] Task {idx} processed: reward={reward}, response={response[:100]}...")
        
        # Safe cleanup: ignore errors if directory doesn't exist or was already deleted
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[CUAJudge] Cleaned up temporary directory: {temp_dir}")
            else:
                print(f"[CUAJudge] Temporary directory already cleaned or doesn't exist: {temp_dir}")
        except Exception as e:
            print(f"[CUAJudge] Warning: Failed to clean up temporary directory {temp_dir}: {e}")
            # Continue execution - cleanup failure should not stop training
        
        return rewards, updated_infos
    
    def save_cuajudge_results(self, task_name: str, task_id: str, cuajudge_results: Dict, save_dir: str = "cuajudge_results"):
        """
        Save CUAJudge evaluation results to JSON file
        
        Args:
            task_name: Name of the task
            task_id: Unique identifier for the task
            cuajudge_results: Dictionary containing CUAJudge evaluation results
            save_dir: Directory to save the results
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with timestamp and task info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CUAJudge_eval_{task_name}_{task_id}_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        # Add metadata to results
        cuajudge_results['metadata'] = {
            'task_name': task_name,
            'task_id': task_id,
            'timestamp': timestamp,
            'evaluation_method': 'cuajudge_with_rule_based_comparison',
            'model_config': {
                'key_model': getattr(self.cuajudge_evaluator, 'key_identification_screenshot_model', 'unknown'),
                'outcome_model': getattr(self.cuajudge_evaluator, 'key_points_outcome_model', 'unknown'),
                'max_image': getattr(self.cuajudge_evaluator, 'max_image', 50),
                'score_threshold': getattr(self.cuajudge_evaluator, 'score_threshold', 3),
            },
            'note': 'This file contains both CUAJudge and rule-based evaluation results for comparison'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cuajudge_results, f, indent=2, ensure_ascii=False)
        print(f"[CUAJudge] Results saved to {filepath}")
        return filepath
    
    def _save_image_for_cuajudge(self, image, file_path: str):
        """
        Save image to file for CUAJudge evaluation
        """
        from PIL import Image
        
        # Validate image data
        if image is None:
            print(f"[CUAJudge] Error: Image is None for {file_path}")
            return False
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, bytes):
            import io
            pil_image = Image.open(io.BytesIO(image))
            print(f"[CUAJudge] Successfully converted bytes to PIL Image for {file_path}")
        else:
            print(f"[CUAJudge] Warning: Unsupported image type {type(image)} for {file_path}")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save image
        pil_image.save(file_path)
        
        # Verify file was actually created
        if os.path.exists(file_path):
            print(f"[CUAJudge] Successfully saved image to {file_path}")
            return True
        else:
            print(f"[CUAJudge] Error: File was not created at {file_path}")
            return False
            
    def _create_placeholder_image(self):
        """Create a placeholder image when screenshot is None"""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        import logging
        logger = logging.getLogger("verl.env_manager")
        logger.warning("Screenshot is None, using placeholder image")
        
        # Create a placeholder image (512x512 gray image with error message)
        placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        # Add text to indicate this is a placeholder
        draw = ImageDraw.Draw(placeholder)
        font = ImageFont.load_default()
        text = "Screenshot\nUnavailable"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (placeholder.width - text_width) // 2
        y = (placeholder.height - text_height) // 2
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Convert PIL Image to numpy array for compatibility with save_image
        return np.array(placeholder)

    def _extract_images(self, obs):
        """Extract screenshots from OSWorld observations.

        OSWorld returns screenshots under the `screenshot` key.
        """
        images = []
        placeholder_used = []
        for ob in obs:
            image = ob.get("screenshot")
            if image is None:
                images.append(self._create_placeholder_image())
                placeholder_used.append(True)
            else:
                images.append(image)
                placeholder_used.append(False)
        return images, placeholder_used

    def save_to_history_buffer(self, images, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'image': images[i], 'action': actions[i]})

    def _build_qwen3vl_chat_thread(self, buf: List[dict]) -> List[dict]:
        """Build a Qwen3-VL style chat thread for a single environment buffer."""
        thread: List[dict] = []

        # 1) Add system message with tools definition (no password in system prompt)
        from agent_system.environments.prompts.qwen3vl import (
            QWEN3VL_SYSTEM_PROMPT,
            get_qwen3vl_instruction_prompt,
        )

        thread.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": QWEN3VL_SYSTEM_PROMPT}],
            }
        )

        # Determine which buffer indices to show
        # image_history_length = N means we show at most N images total (including current)
        # So we can show at most (N-1) historical response pairs
        # Each historical pair needs: 1 image + 1 response, plus 1 current image
        history_len = len(buf) - 1  # Number of completed turns
        num_responses_to_show = (
            min(self.image_history_length - 1, history_len) if self.image_history_length > 0 else 0
        )

        # Build previous actions string for instruction prompt
        # Only include actions that will NOT be shown in the chat history
        # These are actions before the start of visible history
        if num_responses_to_show > 0:
            start_idx = len(buf) - num_responses_to_show - 1
            # Include actions from buf[1] to buf[start_idx] (before visible history)
            previous_actions = []
            for i in range(1, start_idx + 1):
                previous_actions.append(f"Step {i}: {buf[i]['action']}")
            previous_actions_str = "\n".join(previous_actions)
        else:
            # No history shown, include all previous actions
            previous_actions = []
            for i in range(1, len(buf)):
                previous_actions.append(f"Step {i}: {buf[i]['action']}")
            previous_actions_str = "\n".join(previous_actions)

        # Generate instruction prompt with history and password (if enabled)
        if len(buf) > 0:
            instruction_prompt = get_qwen3vl_instruction_prompt(
                instruction=buf[0]["action"],  # Original instruction
                previous_actions_str=previous_actions_str,
                use_password=self.use_password,  # Add password note to instruction
            )
        else:
            instruction_prompt = ""

        if num_responses_to_show > 0:
            # Calculate start index for history
            # In buf: buf[i]['image'] pairs with buf[i+1]['action']
            # To show last N response pairs, we need images starting from buf[len(buf)-N-1]
            start_idx = len(buf) - num_responses_to_show - 1

            # First message: earliest history image + instruction (with ALL history)
            content_parts = []
            content_parts.append({"type": "image", "image": "<|image_pad|>"})
            content_parts.append({"type": "text", "text": instruction_prompt})
            thread.append({"role": "user", "content": content_parts})

            # Add first response (corresponding to start_idx image)
            thread.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": buf[start_idx + 1]["action"]}],
                }
            )

            # Add remaining history pairs: image then response
            for i in range(start_idx + 1, len(buf) - 1):
                thread.append(
                    {"role": "user", "content": [{"type": "image", "image": "<|image_pad|>"}]}
                )
                thread.append(
                    {"role": "assistant", "content": [{"type": "text", "text": buf[i + 1]["action"]}]}
                )

            # Add current image (last in buffer) for next action prediction
            thread.append({"role": "user", "content": [{"type": "image", "image": "<|image_pad|>"}]})
        else:
            # No history (first turn): user(image_0 + instruction)
            if len(buf) > 0:
                content_parts = []
                content_parts.append({"type": "image", "image": "<|image_pad|>"})
                content_parts.append({"type": "text", "text": instruction_prompt})
                thread.append({"role": "user", "content": content_parts})

        return thread

    def _build_ui_tars_chat_thread(self, buf: List[dict]) -> List[dict]:
        """Build a UI-TARS style chat thread for a single environment buffer."""
        thread: List[dict] = []

        user_indices = [idx for idx in range(len(buf))]
        keep_user_imgs = user_indices[-self.image_history_length:]

        for idx, rec in enumerate(buf):
            if idx == 0:
                thread.append({"role": "user", "content": rec["action"]})
                if idx in keep_user_imgs:
                    thread.append({"role": "user", "content": "<|image_pad|>"})
            else:
                if idx in keep_user_imgs:
                    thread.append({"role": "assistant", "content": add_box_token(rec["action"])})
                    thread.append({"role": "user", "content": "<|image_pad|>"})
                else:
                    thread.append({"role": "assistant", "content": add_box_token(rec["action"])})

        return thread

    def build_chat_obs(self, obs: List[dict]) -> List[str]:
        """
        Build chat observation for the agent.
        Constructs different message formats based on model_type to match qwen3vl_agent.py predict() logic.
        """
        postprocess_text_obs: List[List[dict]] = []

        for env_idx, _ in enumerate(obs):
            buf = self.buffers[env_idx]
            if self.model_type == "qwen3vl":
                thread = self._build_qwen3vl_chat_thread(buf)
            else:
                thread = self._build_ui_tars_chat_thread(buf)
            postprocess_text_obs.append(thread)

        return postprocess_text_obs


def make_envs(config = None, tasks = [], is_train = True, processor = None):
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    # Check if we're in sample mode - if not sample, val group_n should be 1, otherwise use the configured value
    is_sample_mode = getattr(config.env, 'is_sample_mode', False)
    val_group_n = group_n if is_sample_mode else 1

    
    if "osworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.OSWorld.envs import build_osworld_envs
        from agent_system.environments.env_package.OSWorld.projection import osworld_projection

        kwargs = {}

        if tasks != []:
            # Check if CUAJudge should be enabled BEFORE creating environments
            enable_cuajudge = getattr(config.env, "enable_cuajudge", False)
            # Rule-based can be enabled independently from CUAJudge (e.g., when CUAJudge is disabled).
            # Default: enable ONLY in validation (not is_train). Training: always disabled.
            enable_rule_based = (not is_train) and getattr(config.env, "enable_rule_based", True)
            if is_train and not enable_cuajudge:
                raise ValueError("Training mode requires enable_cuajudge=True")
            
            # Initialize CUAJudge configuration and prepare kwargs
            cuajudge_config = {}
            # Always pass these flags through to OSWorldEnvironmentManager.
            kwargs["enable_cuajudge"] = enable_cuajudge
            kwargs["enable_rule_based"] = enable_rule_based
            if enable_cuajudge:
                # Extract CUAJudge configuration from config.env
                cuajudge_config = {
                    "key_identification_screenshot_model": getattr(config.env, "cuajudge_key_model", "o4-mini"),
                    "key_points_outcome_model": getattr(config.env, "cuajudge_outcome_model", "o4-mini"),
                    "max_image": getattr(config.env, "cuajudge_max_image", 50),
                    "score_threshold": getattr(config.env, "cuajudge_score_threshold", 3),
                    "use_vllm_for_key_screenshot": getattr(config.env, "use_vllm_for_key_screenshot", False),
                    "vllm_base_url": getattr(config.env, "vllm_base_url", None),
                }
                print(f"CUAJudge config: {cuajudge_config}")
                # Get CUAJudge temp directory from config
                cuajudge_temp_dir = getattr(config.env, "cuajudge_temp_dir", "./tmp")

                # Modify kwargs to include temp directory (flags are set above)
                kwargs["cuajudge_temp_dir"] = cuajudge_temp_dir
                            
            if is_train:
                _envs = build_osworld_envs(
                    tasks=tasks, 
                    seed=config.env.seed, 
                    env_num=config.data.train_batch_size, 
                    group_n=group_n, 
                    is_train=True, 
                    kwargs=kwargs,
                    desktop_config_file=getattr(config.env, 'desktop_config_file', 
                                              "./data/config_examples/environment_config.json")
                )
            else:
                _envs = build_osworld_envs(
                    tasks=tasks, 
                    seed=config.env.seed, 
                    env_num=config.data.val_batch_size, 
                    group_n=val_group_n, 
                    is_train=False, 
                    kwargs=kwargs,
                    desktop_config_file=getattr(config.env, 'desktop_config_file', 
                                              "./data/config_examples/environment_config.json")
                )

            image_save_dir = getattr(config.env, 'image_save_dir', 'osworld')
            use_password = getattr(config.env, 'use_password', False)
            
            # Automatically detect model_type based on processor class name (similar to rollout_loop.py)
            if processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__:
                model_type = "qwen3vl"
            else:
                # Default to qwen25vl, can also be overridden by config
                model_type = 'qwen25vl'
            
            projection_f = partial(osworld_projection, model_type=model_type)
            
            envs = OSWorldEnvironmentManager(
                task_names=tasks,
                envs=_envs,
                projection_f=projection_f,
                env_name="osworld",
                image_save_dir=image_save_dir,
                cuajudge_config=cuajudge_config,
                use_password=use_password,
                model_type=model_type,
                **kwargs,
            )
            return envs
        else:
            return None, None
    
    else:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")
