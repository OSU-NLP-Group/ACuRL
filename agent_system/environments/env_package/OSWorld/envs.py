from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import ray
from urllib.parse import urlparse


_DESKTOPENV_KWARGS_TO_FILTER = {
    "enable_cuajudge",
    "enable_rule_based",
    "cuajudge_temp_dir",
}


def _is_scienceboard_task(task) -> bool:
    """Select ScienceBoard docker image in API mode.

    Assumption (per project convention): ScienceBoard tasks always use snapshot == "sci_bench".
    """
    return isinstance(task, dict) and task.get("snapshot") == "sci_bench"


@ray.remote(num_cpus=0.1)
class OSWorldWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *DesktopEnv* instance.
    """
    
    def __init__(self, seed: int, task: dict, kwargs: Optional[dict], desktop_config: dict):
        # Ensure kwargs is always a dict for .get()/.items() usage
        kwargs = kwargs or {}

        # Lazy import / path setup (kept for compatibility with existing runtime layout)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
        sys.path.append(project_root)
        
        # Configure logging for OSWorld worker to show INFO messages
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Set specific logger levels for OSWorld components
        logging.getLogger("desktopenv.env").setLevel(logging.INFO)
        logging.getLogger("desktopenv.setup").setLevel(logging.INFO)
        logging.getLogger("desktopenv.providers.aws.AWSProvider").setLevel(logging.INFO)
        logging.getLogger("desktopenv.providers.docker.DockerProvider").setLevel(logging.INFO)
        
        from desktop_env.desktop_env import DesktopEnv
        
        # Store task information (now as dict)
        self.task = task
        self.current_uid = None  # Will be set when traj_uid is available from rollout_loop
        
        # Decide which API image to request (only relevant when provider is Docker with api_base_url).
        # Rule: ScienceBoard tasks request image="scienceboard"; regular OSWorld tasks don't send image.
        api_image = "scienceboard" if _is_scienceboard_task(task) else None

        # Check if CUAJudge is enabled (passed through kwargs)
        self.enable_cuajudge = kwargs.get("enable_cuajudge", False)
        if self.enable_cuajudge:
            print(f"[OSWorldWorker] CUAJudge mode enabled for task {task.get('id', 'unknown')}")
        
        # Check if we should run rule-based evaluation for comparison (for validation mode)
        self.enable_rule_based = kwargs.get("enable_rule_based", False)
        if self.enable_rule_based:
            print(f"[OSWorldWorker] Rule-based comparison enabled for task {task.get('id', 'unknown')}")
        
        # Filter out agent-system-only parameters before passing to DesktopEnv
        desktop_env_kwargs = {k: v for k, v in kwargs.items() if k not in _DESKTOPENV_KWARGS_TO_FILTER}
        
        # Set environment variables for desktop configurations
        self._set_desktop_environment_variables(desktop_config)
        
        # Create DesktopEnv instance (uid will be set later via set_uid method)
        self.env = DesktopEnv(
            provider_name=desktop_config.get("provider_name", "vmware"),
            region=desktop_config.get("region"),
            path_to_vm=desktop_config.get("path_to_vm"),
            action_space=desktop_config.get("action_space", "pyautogui"),
            snapshot_name=desktop_config.get("snapshot_name", "init_state"),
            screen_size=(desktop_config.get("screen_width", 1920), desktop_config.get("screen_height", 1080)),
            headless=desktop_config.get("headless", False),
            os_type=desktop_config.get("os_type", "Ubuntu"),
            require_a11y_tree=desktop_config.get("observation_type") in ["a11y_tree", "screenshot_a11y_tree", "som"],
            client_password=desktop_config.get("client_password", "password"),
            enable_proxy=desktop_config.get("enable_proxy", False),
            uid=None,  # uid will be set later via set_uid method with traj_uid from rollout_loop
            # Docker API-backed provider configuration
            api_base_url=desktop_config.get("api_base_url"),
            api_token=desktop_config.get("api_token"),
            api_image=api_image,
            **desktop_env_kwargs,
        )
    
    def _set_desktop_environment_variables(self, desktop_config: dict) -> None:
        """Set environment variables for desktop environment with specific configurations"""
        # Set any desktop-specific environment variables here
        # This could include paths, credentials, or other configuration
        desktop_id = desktop_config.get("desktop_id", "default")
        description = desktop_config.get("description", f"Desktop instance {desktop_id}")
        
        # Optional: print for debugging
        print(f"[OSWorldWorker] Desktop {desktop_id} - {description}")

    def step(self, action: str):
        """Execute a step in the environment"""
        obs, reward, terminated, info = self.env.step(action)
        info = dict(info or {})
        
        # Check if screenshot is None (placeholder case) and print debug info
        if obs.get("screenshot") is None:
            vnc_port = getattr(self.env.provider, 'vnc_port', None)
            server_port = getattr(self.env.provider, 'server_port', None)
            chromium_port = getattr(self.env.provider, 'chromium_port', None)
            vlc_port = getattr(self.env.provider, 'vlc_port', None)
            env_name = getattr(self.env.provider, "container_name", None) or getattr(self.env.provider, "container_id", None)
            traj_uid = self.current_uid
            
            print(
                f"[ENV_PLACEHOLDER_DETECTED] Traj_UID: {traj_uid} | Env_Name: {env_name} | "
                f"Ports: VNC={vnc_port}, Server={server_port}, Chrome={chromium_port}, VLC={vlc_port}"
            )

        if terminated:
            # Handle terminated case based on CUAJudge mode
            if self.enable_cuajudge:
                # Check if we should also run rule-based evaluation for comparison
                enable_rule_based = getattr(self, "enable_rule_based", False)
                
                if enable_rule_based:
                    # Comparison mode: run both rule-based and CUAJudge for comparison
                    rule_based_reward = 0.0
                    rule_based_won = False
                    rule_based_evaluation_failed = False
                    rule_based_evaluation_error = None
                    
                    try:
                        # Run rule-based evaluation first
                        rule_based_reward = self.env.evaluate()
                        rule_based_won = (rule_based_reward == 1.0)
                    except Exception as e:
                        import logging
                        logger = logging.getLogger("desktopenv.worker")
                        logger.error(f"Rule-based evaluation failed: {e}")
                        rule_based_evaluation_failed = True
                        rule_based_evaluation_error = str(e)
                    
                    # Store rule-based results in info for later comparison
                    info["rule_based_task_score"] = rule_based_reward
                    info["rule_based_reward"] = rule_based_reward
                    info["rule_based_won"] = rule_based_won
                    info["rule_based_evaluation_failed"] = rule_based_evaluation_failed
                    info["rule_based_evaluation_error"] = rule_based_evaluation_error
                    info["rule_based_evaluation_method"] = "rule_based"
                
                # Set placeholder values for CUAJudge (will be updated by env_manager)
                info["won"] = False
                info["task_score"] = 0.0
                info["evaluation_method"] = "cuajudge_placeholder"
                reward = 0.0
                info["reward"] = reward
            else:
                # Rule-based mode: use original evaluation logic
                try:
                    reward = self.env.evaluate()
                    info["task_score"] = reward
                    if reward == 1.0:
                        info["won"] = True
                    else:
                        info["won"] = False
                        reward = 0
                    info["reward"] = reward
                except Exception as e:
                    import logging
                    logger = logging.getLogger("desktopenv.worker")
                    logger.error(f"Environment evaluation failed: {e}")
                    
                    # Set default values when evaluation fails
                    reward = 0
                    info["task_score"] = 0
                    info["won"] = False
                    info["evaluation_failed"] = True
                    info["evaluation_error"] = str(e)
                    info["reward"] = reward
                
        else:
            # Non-terminated case: always set default values
            info["won"] = False
            info["task_score"] = 0.0
            reward = 0.0
            info["reward"] = reward

        return obs, reward, terminated, info
    
    
    def reset(self):
        obs = self.env.reset(self.task)
        info: dict[str, Any] = {"won": False}
        return obs, info

    def set_uid(self, uid: str) -> None:
        """Set the uid for this worker and its environment"""
        self.current_uid = uid
        if hasattr(self, 'env') and self.env:
            self.env.set_uid(uid)

    def restart_worker(self):
        """Restart the worker environment"""
        try:
            self.env.restart_environment()
            return True
        except Exception as e:
            import logging
            logger = logging.getLogger("desktopenv.worker")
            logger.error(f"Failed to restart worker: {e}")
            return False

    def is_ready(self):
        """Check if worker is ready (used for batch initialization)"""
        return True
    
    def get_container_info(self):
        """Get container ID and name from the environment provider"""
        container_id = getattr(self.env.provider, 'container_id', None)
        container_name = getattr(self.env.provider, 'container_name', None)
        return {'container_id': container_id, 'container_name': container_name}

    def close(self):
        """Close the environment"""
        self.env.close()



class OSWorldMultiProcessEnv(gym.Env):
    """Vectorised OSWorld via **Ray** actors (similar to BrowserGymMultiProcessEnv)."""

    def __init__(
        self,
        tasks: list[dict],
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 5,
        is_train: bool = True,
        kwargs: Optional[dict] = None,
        desktop_config_file: str = "./data/config_examples/environment_config.json",
        
    ) -> None:
        super().__init__()
        if not ray.is_initialized():
            ray.init(address="auto")

        kwargs = kwargs or {}

        self.seed = seed
        self.group_n = group_n
        self.env_num = env_num
        self.is_train = is_train
        self.tasks = tasks
        self.kwargs = kwargs
        self.desktop_config_file = desktop_config_file
        
        self.sample_tasks = np.repeat(self.tasks, self.group_n).tolist()

        self.workers = []
        # NOTE: num_processes is determined by tasks * group_n (env_num is expected to match len(tasks))
        self.num_processes = len(self.tasks) * self.group_n
        
        # Generate desktop configurations for each worker
        self.desktop_configs = self._load_and_generate_desktop_configurations()
        
        # Batch startup configuration: start workers in batches to avoid overwhelming the system
        batch_size = 64  # Number of workers to start per batch
        batch_delay = 10  # Delay in seconds between batches
        
        print(
            f"[OSWorldMultiProcessEnv] Starting {self.num_processes} workers in batches "
            f"(batch_size={batch_size}, delay={batch_delay}s)"
        )
        
        for batch_start in range(0, self.num_processes, batch_size):
            batch_end = min(batch_start + batch_size, self.num_processes)
            batch_workers = []
            
            # Create a batch of workers
            print(f"[OSWorldMultiProcessEnv] Creating workers {batch_start}-{batch_end - 1}...")
            for i in range(batch_start, batch_end):
                desktop_config = self.desktop_configs[i]
                worker = OSWorldWorker.remote(
                    seed + (i // self.group_n), 
                    self.sample_tasks[i], 
                    kwargs,
                    desktop_config
                )
                self.workers.append(worker)
                batch_workers.append(worker)
            
            # Wait for this batch to be ready (this ensures __init__ completes, including pod startup)
            print(f"[OSWorldMultiProcessEnv] Waiting for workers {batch_start}-{batch_end - 1} to be ready...")
            ready_futures = [w.is_ready.remote() for w in batch_workers]
            ray.get(ready_futures)  # Block until all workers in this batch are initialized
            print(
                f"[OSWorldMultiProcessEnv] Workers {batch_start}-{batch_end - 1} ready "
                f"({batch_end}/{self.num_processes})"
            )
            
            # Add delay before starting next batch (but not after the last batch)
            if batch_end < self.num_processes:
                print(f"[OSWorldMultiProcessEnv] Waiting {batch_delay}s before next batch...")
                time.sleep(batch_delay)
        
        # Print all container IDs after all workers are initialized
        print(f"\n{'='*80}")
        print(f"[OSWorldMultiProcessEnv] All {self.num_processes} environments started")
        print(f"[OSWorldMultiProcessEnv] Container info:")
        print(f"{'='*80}")
        container_info_futures = [w.get_container_info.remote() for w in self.workers]
        container_infos = ray.get(container_info_futures)
        for idx, info in enumerate(container_infos):
            container_id = info.get('container_id', 'N/A')
            container_name = info.get('container_name', 'N/A')
            print(f"  Worker {idx:3d}: container_id={container_id}, container_name={container_name}")
        print(f"{'='*80}\n")
    
    def _load_and_generate_desktop_configurations(self):        
        if not os.path.isabs(self.desktop_config_file):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # From OSWorld/ to project root (ACuRL/) requires going up 4 levels:
            # OSWorld/ -> env_package/ -> environments/ -> agent_system/ -> ACuRL/
            config_file = os.path.join(current_dir, '..', '..', '..', '..', self.desktop_config_file)
            config_file = os.path.normpath(config_file)
        else:
            config_file = self.desktop_config_file
        
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"[OSWorldMultiProcessEnv] Loaded desktop config from: {config_file}")
        print(f"[OSWorldMultiProcessEnv] Found {len(config['desktop_configs'])} desktop configs")
        
        desktop_configs = config["desktop_configs"]
        
        # Generate a desktop config for each worker (round-robin assignment)
        desktop_configs_list = []
        for worker_idx in range(self.num_processes):
            # Round-robin over available desktop configs
            desktop_config = desktop_configs[worker_idx % len(desktop_configs)].copy()
            desktop_configs_list.append(desktop_config)
        
        print(f"[OSWorldMultiProcessEnv] Generated {len(desktop_configs_list)} desktop configs")
        print(
            f"[OSWorldMultiProcessEnv] {self.num_processes} workers assigned across "
            f"{len(desktop_configs)} desktop configs"
        )
        
        # Print per-host allocation
        ip_stats = {}
        for i, cfg in enumerate(desktop_configs_list):
            desktop_id = cfg.get("desktop_id", f"desktop_{i}")
            provider = cfg.get("provider_name", "vmware")
            # For API mode, group by API hostname; otherwise fall back to localhost.
            api_base_url = cfg.get("api_base_url")
            host = "localhost"
            if isinstance(api_base_url, str) and api_base_url:
                try:
                    host = urlparse(api_base_url).hostname or api_base_url
                except Exception:
                    host = api_base_url
            
            if host not in ip_stats:
                ip_stats[host] = 0
            ip_stats[host] += 1
            
            print(f"  Worker {i} (Desktop {desktop_id}) -> host={host}, provider={provider}")
        
        print(f"[OSWorldMultiProcessEnv] Host allocation:")
        for ip, count in ip_stats.items():
            print(f"  {ip}: {count} workers")
        
        return desktop_configs_list


    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        
        for i, (obs, reward, done, info) in enumerate(results):
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        results = ray.get(futures)
        obs_list, info_list = [], []
        
        for i, (obs, info) in enumerate(results):
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list
            

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)
        
        # Wait for all workers to close
        try:
            ray.get(futures, timeout=120)
        except ray.exceptions.GetTimeoutError:
            print("[OSWorldMultiProcessEnv] Timeout waiting for workers to close, force killing...")
        except Exception as e:
            print(f"[OSWorldMultiProcessEnv] Error during worker close: {e}")
        
        # Shutdown Ray actors
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception as e:
                print(f"[OSWorldMultiProcessEnv] Error killing worker: {e}")
        
        gc.collect()
        
        self.workers.clear()
        if hasattr(self, 'sample_tasks'):
            del self.sample_tasks
        if hasattr(self, 'desktop_configs'):
            del self.desktop_configs


def build_osworld_envs(
    tasks: list = [dict],
    seed: int = 42,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    kwargs: Optional[dict] = None,
    desktop_config_file: str = "./data/config_examples/environment_config.json",
):
    return OSWorldMultiProcessEnv(
        tasks = tasks,
        seed = seed,
        env_num = env_num,
        group_n = group_n,
        is_train = is_train,
        kwargs = kwargs,
        desktop_config_file = desktop_config_file,
    )