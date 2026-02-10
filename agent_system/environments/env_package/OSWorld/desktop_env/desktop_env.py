from __future__ import annotations

import logging
import os
import time
import re
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union

import gymnasium as gym

from desktop_env.controllers.python import PythonController, ScreenshotFailureException
from desktop_env.controllers.setup import SetupController
from desktop_env.evaluators import metrics, getters
from desktop_env.providers import create_vm_manager_and_provider

logger = logging.getLogger("desktopenv.env")
logger.setLevel(logging.INFO)

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]

MAX_RETRIES = 3 # Maximum retries for environment setup
            


def _fix_pyautogui_less_than_bug(command: str) -> str:
    """
    Fix PyAutoGUI '<' character bug by converting it to hotkey("shift", ',') calls.
    
    This fixes the known PyAutoGUI issue where typing '<' produces '>' instead.
    References:
    - https://github.com/asweigart/pyautogui/issues/198
    - https://github.com/xlang-ai/OSWorld/issues/257
    
    Args:
        command (str): The original pyautogui command
        
    Returns:
        str: The fixed command with '<' characters handled properly
    """
    # Pattern to match press('<') or press('\u003c') calls  
    press_pattern = r'pyautogui\.press\(["\'](?:<|\\u003c)["\']\)'

    # Handle press('<') calls
    def replace_press_less_than(match):
        return 'pyautogui.hotkey("shift", ",")'
    
    # First handle press('<') calls
    command = re.sub(press_pattern, replace_press_less_than, command)

    # Pattern to match typewrite calls with quoted strings
    typewrite_pattern = r'pyautogui\.typewrite\((["\'])(.*?)\1\)'
    
    # Then handle typewrite calls
    def process_typewrite_match(match):
        quote_char = match.group(1)
        content = match.group(2)
        
        # Preprocess: Try to decode Unicode escapes like \u003c to actual '<'
        # This handles cases where '<' is represented as escaped Unicode
        try:
            # Attempt to decode unicode escapes
            decoded_content = content.encode('utf-8').decode('unicode_escape')
            content = decoded_content
        except UnicodeDecodeError:
            # If decoding fails, proceed with original content to avoid breaking existing logic
            pass  # English comment: Graceful degradation - fall back to original content if decoding fails
        
        # Check if content contains '<'
        if '<' not in content:
            return match.group(0)
        
        # Split by '<' and rebuild
        parts = content.split('<')
        result_parts = []
        
        for i, part in enumerate(parts):
            if i == 0:
                # First part
                if part:
                    result_parts.append(f"pyautogui.typewrite({quote_char}{part}{quote_char})")
            else:
                # Add hotkey for '<' and then typewrite for the rest
                result_parts.append('pyautogui.hotkey("shift", ",")')
                if part:
                    result_parts.append(f"pyautogui.typewrite({quote_char}{part}{quote_char})")
        
        return '; '.join(result_parts)
    
    command = re.sub(typewrite_pattern, process_typewrite_match, command)
    
    return command


class DesktopEnv(gym.Env):
    """
    DesktopEnv with OpenAI Gym interface. It provides a desktop environment for setting and evaluating desktop automation tasks.
    """
    def __init__(
            self,
            provider_name: str = "vmware",
            region: str = None,
            path_to_vm: str = None,
            snapshot_name: str = "init_state",
            action_space: str = "pyautogui",
            cache_dir: str = "./cache",
            screen_size: Tuple[int] = (int(os.environ.get("SCREEN_WIDTH", 1920)), int(os.environ.get("SCREEN_HEIGHT", 1080))),
            headless: bool = False,
            require_a11y_tree: bool = True,
            require_terminal: bool = False,
            os_type: str = "Ubuntu",
            enable_proxy: bool = False,
            client_password: str = "",
            uid: str = None,
            api_base_url: str = None,
            api_token: str = None,
            # Optional: request a specific image when launching via API-backed Docker provider.
            # If None, do not send image field (backend default).
            api_image: str = None,
    ):
        """
        Args:
            provider_name (str): virtualization provider name, default to "vmware"
            region (str): the region for allocate machines, work for cloud services, default to  "us-east-1"
            path_to_vm (str): path to .vmx file
            snapshot_name (str): snapshot name to revert to, default to "init_state"
            action_space (str): "computer_13" | "pyautogui"
            cache_dir (str): cache directory to cache task-related stuffs like
              reference file for evaluation
            screen_size (Tuple[int]): screen size of the VM
            headless (bool): whether to run the VM in headless mode
            require_a11y_tree (bool): whether to require accessibility tree
            require_terminal (bool): whether to require terminal output
            os_type (str): operating system type, default to "Ubuntu"
            enable_proxy (bool): whether to enable proxy support, default to False
        """
        # Initialize VM manager and vitualization provider
        self.region = region
        self.provider_name = provider_name
        self.enable_proxy = enable_proxy  # Store proxy enablement setting
        if client_password == "":
            if self.provider_name == "aws":
                self.client_password = "osworld-public-evaluation"
            else:
                self.client_password = "password"
        else:
            self.client_password = client_password

        self.screen_width = screen_size[0]
        self.screen_height = screen_size[1]
        self.uid = uid  # Store uid for passing to controllers

        # Default 
        self.server_port = 5000
        self.chromium_port = 9222
        self.vnc_port = 8006
        self.vlc_port = 8080
        # ScienceBoard app server port (host-mapped). Default to 8000, may be overridden by provider.
        self.app_port = 8000
        
        # Initialize with default (no proxy) provider
        # Pass api_base_url and api_token when creating provider to avoid Docker socket permission issues
        self.current_use_proxy = False
        self.manager, self.provider = create_vm_manager_and_provider(
            provider_name, region, use_proxy=False, 
            api_base_url=api_base_url, api_token=api_token
        )
        
        # Configure Docker provider in API mode (release version)
        self._docker_api_mode = False
        if provider_name == "docker":
            # api_base_url and api_token are now set during provider creation
            # Only set api_image if provided
            if api_image is not None and hasattr(self.provider, "api_image"):
                self.provider.api_image = api_image
            # Treat provider.use_api (including env-var configured API base) as API mode.
            self._docker_api_mode = bool(getattr(self.provider, "use_api", False))

        self.os_type = os_type

        # Track whether environment has been used (step/setup) to optimize snapshot revert
        # docker, aws, gcp, azure are always unused as the emulator starts from a clean state
        # vmware, virtualbox are always used as the emulator starts from a dirty state
        if self.provider_name in {"docker", "aws", "gcp", "azure"}:
            self.is_environment_used = False
        elif self.provider_name in {"vmware", "virtualbox"}:
            self.is_environment_used = True
        else:
            raise ValueError(f"Invalid provider name: {self.provider_name}")

        # Initialize environment variables
        if path_to_vm:
            self.path_to_vm = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_vm))) \
                if provider_name in {"vmware", "virtualbox"} else path_to_vm
        else:
            # In docker+API mode, the VM image path is irrelevant; avoid heavy qcow2 downloads.
            if self.provider_name == "docker" and self._docker_api_mode:
                self.path_to_vm = "api://osworld"
            else:
                self.path_to_vm = self.manager.get_vm_path(
                    os_type=self.os_type,
                    region=region,
                    screen_size=(self.screen_width, self.screen_height),
                )
        
        try:
            self.snapshot_name = snapshot_name
            self.cache_dir_base: str = cache_dir
            # todo: add the logic to get the screen size from the VM
            self.headless = headless
            self.require_a11y_tree = require_a11y_tree
            self.require_terminal = require_terminal

            # Initialize emulator and controller
            logger.info("Initializing...")
            self._start_emulator()

            # mode: human or machine
            self.instruction = None
            assert action_space in ["computer_13", "pyautogui", "claude_computer_use"]
            self.action_space = action_space  # todo: refactor it to the ActType

            # episodic stuffs, like counters, will be updated or reset
            # when calling self.reset()
            self._traj_no: int = -1
            self._step_no: int = 0
            self.action_history: List[Dict[str, any]] = []
            
            # Flag to track if environment needs to be restarted due to screenshot failures
            self._needs_restart = False
        except Exception as e:
            logger.error(f"Failed to initialize DesktopEnv: {e}")
            # If initialization fails, we should clean up the VM
            try:
                self.close()
                self.manager.delete_vm(self.path_to_vm, self.region)
                logger.info(f"Cleaned up VM {self.path_to_vm}.")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up VM {self.path_to_vm}: {cleanup_error}")
            raise

    def _start_emulator(self):
        # Power on the virtual machine with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.provider.start_emulator(self.path_to_vm, self.headless, self.os_type)
                
                # Get the ip from the virtual machine, and setup the controller
                vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(':')
                self.vm_ip = vm_ip_ports[0]
                # Get the ports from the virtual machine (for Docker provider only)
                if len(vm_ip_ports) > 1:
                    self.server_port = int(vm_ip_ports[1])
                    self.chromium_port = int(vm_ip_ports[2])
                    self.vnc_port = int(vm_ip_ports[3])
                    self.vlc_port = int(vm_ip_ports[4])
                # Optional: ScienceBoard app port mapping (container port 8000 -> host app_port)
                if len(vm_ip_ports) > 5:
                    self.app_port = int(vm_ip_ports[5])
                self.controller = PythonController(vm_ip=self.vm_ip, server_port=self.server_port, uid=self.uid)

                # Query the *real* VM screen size and use it as per-env source of truth.
                # IMPORTANT: Do NOT propagate via process-level env vars (one job can host many envs).
                real_size = self.controller.get_vm_screen_size() or {}
                real_w = int(real_size.get("width", 0) or 0)
                real_h = int(real_size.get("height", 0) or 0)
                if real_w <= 0 or real_h <= 0:
                    raise ValueError(f"Failed to query VM screen size via /screen_size: {real_size}")
                logger.info("VM real screen size: %sx%s", real_w, real_h)
                self.screen_width, self.screen_height = real_w, real_h

                self.setup_controller = SetupController(
                    vm_ip=self.vm_ip,
                    server_port=self.server_port,
                    chromium_port=self.chromium_port,
                    vlc_port=self.vlc_port,
                    cache_dir=self.cache_dir_base,
                    client_password=self.client_password,
                    screen_width=self.screen_width,
                    screen_height=self.screen_height,
                )
                
                # If we get here, startup was successful
                logger.info("Emulator started successfully")
                break
                
            except Exception as e:
                logger.error(f"Failed to start emulator on attempt {attempt + 1}/{max_retries}: {e}")
                
                # Clean up failed instance before retry
                if attempt < max_retries - 1:
                    logger.info("Cleaning up failed instance before retry...")
                    try:
                        self.provider.stop_emulator(self.path_to_vm)
                        logger.info("Failed instance cleaned up, retrying...")
                        time.sleep(5)
                    except Exception as cleanup_error:
                        logger.warning(f"Error during cleanup: {cleanup_error}")
                else:
                    # Last attempt failed, re-raise the exception
                    logger.error(f"All {max_retries} attempts to start emulator failed")
                    raise

    def _revert_to_snapshot(self):
        # Revert to certain snapshot of the virtual machine, and refresh the path to vm and ip of vm
        # due to the fact it could be changed when implemented by cloud services
        path_to_vm = self.provider.revert_to_snapshot(self.path_to_vm, self.snapshot_name)
        if path_to_vm and not path_to_vm == self.path_to_vm:
            # path_to_vm has to be a new path 
            
            self.manager.delete_vm(self.path_to_vm, self.region)
            self.manager.add_vm(path_to_vm, self.region)
            self.manager.occupy_vm(path_to_vm, os.getpid(), self.region)
            self.path_to_vm = path_to_vm

    def _save_state(self, snapshot_name=None):
        # Save the current virtual machine state to a certain snapshot name
        self.provider.save_state(self.path_to_vm, snapshot_name)

    def _recreate_aws_instance(self):
        """
        Recreate AWS instance when setup fails.
        This closes the current instance and creates a new one.
        """
        if self.provider_name != "aws":
            logger.warning("Instance recreation is only supported for AWS provider")
            return
            
        logger.info("Closing current AWS instance...")
        try:
            # Stop the current emulator
            self.provider.stop_emulator(self.path_to_vm)
            
            # Delete the current VM instance
            self.manager.delete_vm(self.path_to_vm, self.region)
            logger.info(f"Deleted AWS instance: {self.path_to_vm}")
            
        except Exception as e:
            logger.warning(f"Error during instance cleanup: {e}")
            # Continue with recreation even if cleanup fails
        
        # Get a new VM path/instance
        logger.info("Allocating new AWS instance...")
        self.path_to_vm = self.manager.get_vm_path(os_type=self.os_type, region=self.region, screen_size=(self.screen_width, self.screen_height))
        
        # Start the new emulator
        logger.info("Starting new AWS instance...")
        self._start_emulator()
        
        # Reset environment state
        self.is_environment_used = False
        logger.info("AWS instance recreation completed")

    def _recreate_environment(self):
        """
        Recreate environment for any provider when setup fails.
        This closes the current environment and creates a new one.
        """
        logger.info(f"Recreating environment for provider: {self.provider_name}")
        
        try:
            # Stop the current emulator
            logger.info("Closing current environment...")
            self.provider.stop_emulator(self.path_to_vm)
            
            # Delete the current VM instance
            logger.info(f"Deleting current VM: {self.path_to_vm}")
            self.manager.delete_vm(self.path_to_vm, self.region)
            
        except Exception as e:
            logger.warning(f"Error during environment cleanup: {e}")
            # Continue with recreation even if cleanup fails
        
        # Get a new VM path/instance
        logger.info("Allocating new environment...")
        # self.path_to_vm = self.manager.get_vm_path(os_type=self.os_type, region=self.region, screen_size=(self.screen_width, self.screen_height))
        
        # Start the new emulator
        logger.info("Starting new environment...")
        self._start_emulator()
        
        # Reset environment state
        self.is_environment_used = False
        logger.info("Environment recreation completed")

    def set_uid(self, uid: str):
        """Set the uid for this environment instance and update controller"""
        self.uid = uid
        if hasattr(self, 'controller') and self.controller:
            self.controller.uid = uid

    def close(self):
        # Close (release) the virtual machine
        self.provider.stop_emulator(self.path_to_vm)

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        setup_success = False
        
        for attempt in range(MAX_RETRIES):
            # Only revert to snapshot if environment has been used (step/setup)
            # This optimization is especially important for cloud providers like AWS
            # where unnecessary snapshot operations are costly and time-consuming
            
            if task_config is not None:
                # Only consider task proxy requirement if proxy is enabled at system level
                task_use_proxy = task_config.get("proxy", False) and self.enable_proxy
                if not self.enable_proxy and task_config.get("proxy", False):
                    logger.info("Task requires proxy but proxy is disabled at system level, ignoring proxy requirement.")
                
                if task_use_proxy != self.current_use_proxy:
                    # keep because get_info_from_website depend on this
                    self.current_use_proxy = task_use_proxy
            
            if self.is_environment_used:
                logger.info("Environment has been used, reverting to snapshot {}...".format(self.snapshot_name))
                self._revert_to_snapshot()
                logger.info("Starting emulator...")
                self._start_emulator()
                logger.info("Emulator started.")
                # Reset the usage flag after reverting
                self.is_environment_used = False
            else:
                logger.info("Environment is clean, skipping snapshot revert (provider: {}).".format(self.provider_name))

            if task_config is not None:
                if task_config.get("proxy", False) and self.enable_proxy:
                    # If using proxy and proxy is enabled, set up the proxy configuration
                    self.setup_controller._proxy_setup(self.client_password)
                self._set_task_info(task_config)
                self.setup_controller.reset_cache_dir(self.cache_dir)
                logger.info("Setting up environment...")
                success = self.setup_controller.setup(self.config, task_config.get("proxy", False) and self.enable_proxy)
                if success:
                    # Mark environment as used when setup is successfully executed
                    if self.config:  # Only mark as used if there were actual setup operations
                        self.is_environment_used = True
                    setup_success = True
                    break
                else:
                    logger.error(
                        "Environment setup failed, retrying (%d/%d)...",
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    
                    # If this is not the last attempt, recreate the environment
                    if attempt < MAX_RETRIES - 1:
                        logger.info(f"Setup failed, recreating environment for provider: {self.provider_name}")
                        try:
                            if self.provider_name == "aws":
                                self._recreate_aws_instance()
                            else:
                                self._recreate_environment()
                            logger.info("Environment recreated successfully")
                        except Exception as e:
                            logger.error(f"Failed to recreate environment: {e}")
                            # Continue to next retry even if recreation fails
                    
                    time.sleep(5)
            else:
                setup_success = True
                break
            
        logger.info("Environment setup complete.")

        # Wait for applications to fully launch before taking first screenshot
        if setup_success:
            logger.info("Waiting for applications to fully launch...")
            time.sleep(60)  # Wait 5 seconds for UI to fully render
            logger.info("Wait complete, taking initial screenshot.")

        observation = self._get_obs()
        return observation

    def _get_obs(self):
        # We provide screenshot, accessibility_tree (optional), terminal (optional), and instruction.
        # can be customized and scaled
        try:
            screenshot = self.controller.get_screenshot(raise_on_failure=True)
        except ScreenshotFailureException as e:
            logger.error(f"Screenshot failure detected: {e}")
            # self._needs_restart = True  # COMMENTED OUT: No longer trigger restart on screenshot failure
            logger.warning("Screenshot failed but continuing task execution (restart disabled)")
            # Return observation with None screenshot but continue execution
            return {
                "screenshot": None,
                "screen_width": self.screen_width,
                "screen_height": self.screen_height,
                "accessibility_tree": self.controller.get_accessibility_tree() if self.require_a11y_tree else None,
                "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
                "instruction": self.instruction,
                "needs_restart": False  # CHANGED: Always return False to prevent restart
            }
        
        return {
            "screenshot": screenshot,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "accessibility_tree": self.controller.get_accessibility_tree() if self.require_a11y_tree else None,
            "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
            "instruction": self.instruction,
            "needs_restart": False
        }

    @property
    def vm_platform(self):
        return self.controller.get_vm_platform()

    @property
    def vm_screen_size(self):
        return self.controller.get_vm_screen_size()

    def _set_task_info(self, task_config: Dict[str, Any]):
        """Set task info (proxy logic is handled in reset method)"""
        self.task_id: str = task_config["id"]
        self.cache_dir: str = os.path.join(self.cache_dir_base, self.task_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = task_config["instruction"]
        self.config = task_config["config"] if "config" in task_config else []
        # ScienceBoard compatibility:
        # - "snapshot" may be used as a surface marker (e.g., "scienceboard")
        # - the actual VM snapshot name is stored in "vm_snapshot" when present.
        if "vm_snapshot" in task_config and isinstance(task_config["vm_snapshot"], str):
            self.snapshot_name = task_config["vm_snapshot"]
        elif "snapshot" in task_config and isinstance(task_config["snapshot"], str):
            self.snapshot_name = task_config["snapshot"]
        
        self._set_evaluator_info(task_config)

        # ScienceBoard compatibility: if docker/remote provider mapped app port, rewrite task_config ports.
        # - task_config["evaluator"]["port"] is the app server port used by ScienceBoard evaluators (default 8000).
        # - setup steps like grass_map/kalgebra_* also carry "port".
        #
        # IMPORTANT:
        # - Do NOT rewrite any in-VM commands (e.g. FLASK_PORT=8000) because those run inside the guest/container.
        # - Only rewrite host-side HTTP ports that the runner uses to reach app servers from outside.
        if getattr(self, "evaluator", {}).get("func") == "scienceboard":
            try:
                desired = int(self.evaluator.get("port", 8000))
                mapped = int(getattr(self, "app_port", desired))
            except Exception:
                desired, mapped = 8000, getattr(self, "app_port", 8000)

            if mapped and mapped != desired:
                task_type = str(self.evaluator.get("task_type", ""))
                # Only some task types use host-side HTTP to app servers on port=8000.
                if task_type in {"GrassGIS", "KAlgebra", "Celestia"}:
                    self.evaluator["port"] = mapped

                host_side_port_steps = {
                    "grass_map",
                    "grass_layer",
                    "grass_scale",
                    "grass_cmd",
                    "kalgebra_tab",
                    "kalgebra_func_2d",
                    "kalgebra_func_3d",
                }
                for cfg in self.config or []:
                    if not isinstance(cfg, dict):
                        continue
                    if cfg.get("type") not in host_side_port_steps:
                        continue
                    params = cfg.get("parameters")
                    if isinstance(params, dict) and params.get("port") == desired:
                        params["port"] = mapped

    def _set_evaluator_info(self, task_config: Dict[str, Any]):
        """Set evaluator information from task config"""
        # ScienceBoard compatibility: evaluator is handled as a single custom function
        if task_config.get("evaluator", {}).get("func") == "scienceboard":
            self.evaluator = task_config["evaluator"]
            self.metric = None
            self.metric_conj = "and"
            self.result_getter = None
            self.expected_getter = None
            self.metric_options = {}
            return

        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        self.evaluator = task_config["evaluator"]
        self.metric: Metric = [getattr(metrics, func) for func in self.evaluator["func"]] \
            if isinstance(self.evaluator["func"], list) \
            else getattr(metrics, self.evaluator["func"])
        self.metric_conj: str = self.evaluator.get("conj", "and")  # take conjunction of multiple metrics
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            self.result_getter: Getter = [getattr(getters, "get_{:}".format(res["type"])) for res in
                                          self.evaluator["result"]] \
                if isinstance(self.evaluator["result"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["result"]["type"]))
        else:
            self.result_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None

        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            self.expected_getter: Getter = [getattr(getters, "get_{:}".format(exp["type"])) if exp else None for exp in
                                            self.evaluator["expected"]] \
                if isinstance(self.evaluator["expected"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["expected"]["type"]))
        else:
            self.expected_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None
        self.metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = [opt if opt else {} for opt in
                                                                            self.evaluator["options"]] \
            if isinstance(self.evaluator.get("options", {}), list) \
            else self.evaluator["options"] \
            if "options" in self.evaluator \
            else [{}] * len(self.metric) \
            if isinstance(self.metric, list) \
            else {}

        assert (not isinstance(self.evaluator["func"], list)
                or (len(self.metric) == len(self.result_getter) == len(self.expected_getter) == len(
                    self.metric_options)))

    def step(self, action, pause=2):
        self._step_no += 1
        self.action_history.append(action)
        
        # Mark environment as used when step is called
        self.is_environment_used = True

        reward = 0  # todo: Define reward calculation for each example
        done = False  # todo: Define episode termination condition for each example
        info = {}
        logger.info(f"Step {self._step_no} in trajectory {self._traj_no} with action: {action}")
        # handle the special actions
        is_special_dict = isinstance(action, dict) and action.get("action_type") in ["WAIT", "FAIL", "DONE"]
        if action in ['WAIT', 'FAIL', 'DONE'] or is_special_dict:
            act_type = action if isinstance(action, str) else str(action.get("action_type"))
            if act_type == 'WAIT':
                time.sleep(pause)
            elif act_type == 'FAIL':
                done = True
                info = {"fail": True}
            elif act_type == 'DONE':
                done = True
                info = {"done": True}

        if self.action_space == "computer_13":
            # the set of all possible actions defined in the action representation
            # Do not execute special termination dict actions; they are control signals.
            if not is_special_dict:
                self.controller.execute_action(action)
        elif self.action_space == "pyautogui" or self.action_space == "claude_computer_use":
            if action in ['WAIT', 'FAIL', 'DONE'] or is_special_dict:
                self.controller.execute_action(action)
            else:
                # the set of all possible python commands insides `pyautogui`
                if type(action) == str:
                    # Fix PyAutoGUI '<' character bug before execution
                    fixed_command = _fix_pyautogui_less_than_bug(action)
                    execution_result = self.controller.execute_python_command(fixed_command)
                    info["execution_result"] = execution_result
                elif type(action) == dict:
                    # Fix PyAutoGUI '<' character bug before execution
                    fixed_command = _fix_pyautogui_less_than_bug(action['command'])
                    execution_result = self.controller.execute_python_command(fixed_command)
                    info["execution_result"] = execution_result

        time.sleep(pause)
        observation = self._get_obs()

        return observation, reward, done, info

    def evaluate(self):
        """
        Evaluate whether the task is successfully completed.
        """
        # ScienceBoard compatibility
        if getattr(self, "evaluator", {}).get("func") == "scienceboard":
            from desktop_env.scienceboard.evaluator import evaluate_scienceboard
            return 1 if evaluate_scienceboard(self, self.evaluator) else 0

        postconfig = self.evaluator.get("postconfig", [])
        self.setup_controller.setup(postconfig)
        # Mark environment as used if there were postconfig setup operations
        if postconfig:
            self.is_environment_used = True

        if self.evaluator['func'] == "infeasible":
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 1
            else:
                return 0
        else:
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 0

        if type(self.metric) == list:
            # Multiple metrics to evaluate whether the task is successfully completed
            results = []
            assert len(self.metric) == len(self.result_getter), "The number of metrics and result getters must be the same"
            if "expected" in self.evaluator:
                assert len(self.metric) == len(self.expected_getter), "The number of metrics and expected getters must be the same"
            for idx, metric in enumerate(self.metric):
                try:
                    config = self.evaluator["result"][idx]
                    result_state = self.result_getter[idx](self, config)
                except FileNotFoundError:
                    logger.error("File not found!")
                    if self.metric_conj == 'and':
                        return 0

                if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                    expected_state = self.expected_getter[idx](self, self.evaluator["expected"][idx])
                    metric: int = metric(result_state, expected_state, **self.metric_options[idx])
                else:
                    metric: int = metric(result_state, **self.metric_options[idx])

                if self.metric_conj == 'and' and float(metric) == 0.0:
                    return 0
                elif self.metric_conj == 'or' and float(metric) == 1.0:
                    return 1
                else:
                    results.append(metric)

            return sum(results) / len(results) if self.metric_conj == 'and' else max(results)
        else:
            # Single metric to evaluate whether the task is successfully completed
            try:
                result_state = self.result_getter(self, self.evaluator["result"])
            except FileNotFoundError:
                logger.error("File not found!")
                return 0

            if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                expected_state = self.expected_getter(self, self.evaluator["expected"])
                metric: float = self.metric(result_state, expected_state, **self.metric_options)
            else:
                metric: float = self.metric(result_state, **self.metric_options)

        return metric

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.controller.get_screenshot()
        else:
            raise ValueError('Unsupported render mode: {}'.format(mode))
    
    def restart_environment(self):
        """
        Restart the environment by closing and reinitializing the VM.
        This is called when screenshot acquisition fails permanently.
        """
        try:
            logger.info("Restarting environment due to screenshot failure...")
            
            # Close current environment
            self.close()
            
            # Restart the emulator
            self._start_emulator()
            
            # Reset the restart flag
            self._needs_restart = False
            
            logger.info("Environment restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart environment: {e}")
            self._needs_restart = True  # Keep the flag set if restart fails
            raise
    
    def needs_restart(self):
        """Check if the environment needs to be restarted."""
        return self._needs_restart
