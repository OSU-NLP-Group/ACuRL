import logging

import torch
import numpy as np
import os
import sys
import re
import json
from typing import List, Tuple, Dict
import math
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
from verl import DataProto
import pandas as pd

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


logger = logging.getLogger(__name__)


def _create_placeholder_image():
    placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
    draw = ImageDraw.Draw(placeholder)
    font = ImageFont.load_default()
    text = "Screenshot\nUnavailable"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (placeholder.width - text_width) // 2
    y = (placeholder.height - text_height) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    setattr(placeholder, '_from_process_image_error', True)
    return placeholder


def get_placeholder_image_array():
    """Return placeholder image as numpy array for saving to disk."""
    return np.array(_create_placeholder_image())


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    try:
        if isinstance(image, bytes):
            from io import BytesIO
            image = Image.open(BytesIO(image))
        # Handle torch tensor format
        elif isinstance(image, torch.Tensor):
            image = torch_to_numpy(image)
            if image.max() < 1:
                image = image * 255.0
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            image = Image.fromarray(image)
        # Handle numpy array format
        elif isinstance(image, np.ndarray):
            if image.max() < 1:
                image = image * 255.0
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            image = Image.fromarray(image)
        # Handle PIL Image (already in correct format)
        elif isinstance(image, Image.Image):
            pass  # Already PIL Image, no conversion needed
        else:
            raise ValueError(f"Unsupported image type: {type(image)}. Supported types: bytes, torch.Tensor, numpy.ndarray, PIL.Image")

        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Force lazy decoders (e.g., PIL PNG) to read all bytes now so that
        # corrupted screenshots are caught within this helper instead of
        # bubbling up later during batch preprocessing.
        image.load()

        return image
    except Exception as exc:
        logger.warning("Broken image detected, returning placeholder. %s", exc)
        return _create_placeholder_image()


def adjust_batch(config, data: DataProto, mode="copy") -> DataProto:
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    size_divisor_ref = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    size_divisor_actor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        return data
    
    if mode == "delete":
        # Generate indices to remove, rather than indices to keep
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    elif mode == "copy":
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(bs, to_add, replace=False)
        dup_proto = data.select_idxs(dup_indices)

        adjusted_batch = DataProto.concat([data, dup_proto])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return adjusted_batch


def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, and success based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid


def extract_xy(line):
    number_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    nums = number_pat.findall(line)
    if len(nums) < 2:
        return None
    return (float(nums[0]), float(nums[1]))

def get_size_mb(obj):
    """Calculate the size of an object in MB."""
    if obj is None:
        return 0
    elif isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement() / (1024 * 1024)
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 * 1024)
    elif isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) / (1024 * 1024) + sum(get_size_mb(item) for item in obj)
    elif isinstance(obj, dict):
        return sys.getsizeof(obj) / (1024 * 1024) + sum(get_size_mb(k) + get_size_mb(v) for k, v in obj.items())
    else:
        return sys.getsizeof(obj) / (1024 * 1024)

def load_task_data(task_name: str, data_dir: str = "./data/tasks/examples", subfolder: str = None):
    """
    Load task data based on task_name format.
    
    Args:
        task_name: Task name in format:
                  - "{category}.{task_id}" fortasks
        data_dir: Base directory containing data
        subfolder: Optional subfolder within category directory (e.g., "gpt-5-mini_difficulty")
        
    Returns:
        Dict with full JSON file content
        For WebArena: Task name string (backward compatibility)    """
    parts = task_name.split('.')

    # If data_dir is a relative path, interpret it relative to repository root (two levels above this file)
    if not os.path.isabs(data_dir):
        current_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        data_dir = os.path.normpath(os.path.join(repo_root, data_dir))
        
    print(f"Processing task: {task_name}")

    category = parts[0]
    task_id = ".".join(parts[1:])
    
    if subfolder:
        json_file_path = os.path.join(data_dir, category, subfolder, f"{task_id}.json")
    else:
        json_file_path = os.path.join(data_dir, category, f"{task_id}.json")
    
    # Load JSON content
    with open(json_file_path, 'r') as f:
        task_dict = json.load(f)
    # Convert them into DesktopEnv task_config.
    if isinstance(task_dict, dict) and task_dict.get("snapshot") == "sci_bench":
        try:
            from pathlib import Path
            from agent_system.environments.env_package.OSWorld.desktop_env.scienceboard.adapter import (
                to_desktopenv_task_config,
            )
            # Use filename stem as stable id (e.g., C-16) to keep task_name resolvable.
            task_id_override = Path(json_file_path).stem
            return to_desktopenv_task_config(json_file_path, port=8000, task_id_override=task_id_override)
        except Exception as e:
            raise RuntimeError(f"Failed to convert ScienceBoard task {json_file_path}: {e}") from e
    return task_dict
        

def _get_all_available_tasks(self, config):
    """Get all available tasks from the training parquet file for random sampling."""
    
    # Get the training data file path from config
    train_files = config.data.train_files
    if isinstance(train_files, list):
        train_file = train_files[0]  # Use the first training file
    else:
        train_file = train_files
        
    # Convert relative path to absolute path if needed
    if not os.path.isabs(train_file):
        # Get the current working directory (should be the project root)
        current_dir = os.getcwd()
        train_file = os.path.join(current_dir, train_file)
    
    print(f"Loading tasks from training file: {train_file}")
    
    # Read the parquet file
    df = pd.read_parquet(train_file)
    
    # Extract task names from the 'task_name' column
    if 'task_name' in df.columns:
        all_tasks = df['task_name'].unique().tolist()
        print(f"Found {len(all_tasks)} unique tasks in training data")
        return all_tasks
    else:
        raise ValueError("'task_name' column not found in training data")
            

def _random_sample_tasks(self, config, training_step=0):
    """Randomly sample tasks for training if random_task_sampling is enabled."""
    
    # Get all available tasks
    all_tasks = self._get_all_available_tasks(config)
    
    # Get batch size from config
    batch_size = config.data.train_batch_size
    
    # Randomly sample tasks
    import random
    random.seed(config.env.seed + training_step)
    sampled_tasks = random.sample(all_tasks, min(batch_size, len(all_tasks)))
    
    print(f"Randomly sampled {len(sampled_tasks)} tasks: {sampled_tasks}")
    return sampled_tasks