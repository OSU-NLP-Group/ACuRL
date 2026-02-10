from typing import Any, Dict, List, Tuple

import datetime
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)}")
    return data


class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, env_name=None, image_save_dir=None):
        """
        Initialize the environment manager.
        
        Parameters:
        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.
        - projection_f: A function that maps text actions to environment actions.
        - env_name (str): The name of the environment.
        - image_save_dir (str): Custom directory for saving images and results. If None, uses env_name.
        """
        self.envs = envs
        self.projection_f = projection_f
        self.env_name = env_name
        self.image_save_dir = image_save_dir or env_name

    def reset(self) -> Dict[str, Any]:
        """
        Reset all environments and return the initial observations.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs}, infos
    
    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
        
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
        - rewards (np.ndarray or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
        
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            "text": None,  # Implement if needed
            "image": next_obs,
        }
        # Add action_valid to infos
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        
        return next_observations, rewards, dones, infos

    def build_text_obs(self) -> List[str]:
        """
        This function builds the text observation for the agent.
        
        Returns:
        - postprocess_text_obs (List[str]): A list of processed text observations.
        """
        pass

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.envs.close()

    def set_worker_uids(self, uids: List[str]):
        """
        Set UIDs for all workers (default implementation does nothing).
        Override this method in subclasses that support UID tracking.
        """
        pass

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        # Initialize additional success rate fields for CUAJudge and rule-based comparison
        success["rule_based_success_rate"] = []
        success["cuajudge_success_rate"] = []
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success["success_rate"]) == batch_size
        assert len(success["rule_based_success_rate"]) == batch_size
        assert len(success["cuajudge_success_rate"]) == batch_size

        

        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                # Read info from total_infos (updated by CUAJudge in wait_all_pending_evals)
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                task_score = float(info["task_score"])
                
                # In validation mode with CUAJudge, also calculate rule-based success rate
                if (
                    hasattr(self, "enable_cuajudge")
                    and self.enable_cuajudge
                    and "rule_based_won" in info
                    and "rule_based_task_score" in info
                ):
                    # Validation: success_rate shows rule-based result, cuajudge_success_rate shows CUAJudge result
                    rule_based_won_value = float(info["rule_based_won"])
                    rule_based_task_score_value = float(info["rule_based_task_score"])
                    success["success_rate"].append(rule_based_won_value)  # Main success_rate uses rule-based in val
                    success["rule_based_success_rate"].append(rule_based_won_value)
                    success["rule_based_task_score"].append(rule_based_task_score_value)
                    success["cuajudge_success_rate"].append(won_value)
                    success["cuajudge_task_score"].append(task_score)
                elif (
                    hasattr(self, "enable_cuajudge")
                    and self.enable_cuajudge
                    and "rule_based_won" not in info
                    and "rule_based_task_score" not in info
                ):
                    # Training mode or no rule-based comparison
                    success["success_rate"].append(won_value)  # Main success_rate uses CUAJudge in train
                    success["rule_based_success_rate"].append(0.0)  # Default value
                    success["rule_based_task_score"].append(0.0)  # Default value
                    success["cuajudge_success_rate"].append(won_value)
                    success["cuajudge_task_score"].append(task_score)
                else:
                    # Original (rule-based) mode
                    success["success_rate"].append(won_value)
                    success["rule_based_success_rate"].append(won_value)
                    success["rule_based_task_score"].append(task_score)
                    success["cuajudge_success_rate"].append(0.0)
                    success["cuajudge_task_score"].append(0.0)
                
                return

    def _trajectory_root(self) -> Path:
        return Path(__file__).resolve().parent / "../../trajectory" / (self.image_save_dir or "")

    def save_image(self, image, name):
        """
        Save an image to a file.
        
        Parameters:
        - image (np.ndarray, torch.Tensor, or bytes): The image to save.
        - name (str): The name/path to save the image.
        """
        root = self._trajectory_root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle different image types
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, bytes):
            try:
                # Try to decode as image bytes (JPEG/PNG)
                import io
                from PIL import Image

                image_pil = Image.open(io.BytesIO(image))
                image_pil.save(path)
                return
            except Exception as e:
                print(f"Failed to decode bytes as image: {e}")
                try:
                    # Try base64 decode and decode again as image
                    import base64
                    import io
                    from PIL import Image

                    decoded = base64.b64decode(image)
                    image_pil = Image.open(io.BytesIO(decoded))
                    image_pil.save(path)
                    return
                except Exception as e2:
                    # Last resort: save as raw bytes file
                    raw_path = path.with_suffix(".bytes")
                    with open(raw_path, "wb") as f:
                        f.write(image)
                    print(f"Saved raw bytes to {raw_path}")
                    return
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)}")
        
        # Continue with numpy array processing
        if len(image.shape) == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)
        
        from PIL import Image

        image_pil = Image.fromarray(image)
        image_pil.save(path)

    def save_image_with_grounding(
        self,
        image,
        name,
        click_coords,
        dot_radius = 6,
        dot_color = (255, 0, 0),
    ):
        root = self._trajectory_root()
        root.mkdir(parents=True, exist_ok=True)
        raw_path        = root / f"{name}.png"
        grounding_path  = root / f"{name}_grounding.png"
        
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        grounding_path.parent.mkdir(parents=True, exist_ok=True)

        # ---------- Input format ----------
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if image.ndim == 4:  # (N, C, H, W)
            image = image[0]
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:  # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        from PIL import Image, ImageDraw
        img_pil = Image.fromarray(image)
        img_pil.save(raw_path)

        grounding_saved = None
        if click_coords:
            img_with_dots = img_pil.copy()
            draw = ImageDraw.Draw(img_with_dots)
            for x, y in click_coords:
                x, y = int(round(x)), int(round(y))
                bbox = [(x - dot_radius, y - dot_radius),
                        (x + dot_radius, y + dot_radius)]
                draw.ellipse(bbox, fill=dot_color, outline=None)
            img_with_dots.save(grounding_path)
            grounding_saved = grounding_path

        return raw_path, grounding_saved

    def _json_serial(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (set,)):
            return list(obj)
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return str(obj)
        
    def save_result(self, result, name):
        root = self._trajectory_root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                default=self._json_serial,
            )
        