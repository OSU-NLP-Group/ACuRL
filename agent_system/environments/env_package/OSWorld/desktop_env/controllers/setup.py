import json
import logging
import os
import os.path
import platform
import shutil
import sqlite3
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any, Union, Optional
from typing import Dict, List

import requests
from playwright.sync_api import sync_playwright, TimeoutError
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive, GoogleDriveFile, GoogleDriveFileList
from requests_toolbelt.multipart.encoder import MultipartEncoder

from desktop_env.controllers.python import PythonController
from desktop_env.evaluators.metrics.utils import compare_urls
from desktop_env.providers.aws.proxy_pool import get_global_proxy_pool, init_proxy_pool, ProxyInfo

import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()


PROXY_CONFIG_FILE = os.getenv("PROXY_CONFIG_FILE", "evaluation_examples/settings/proxy/dataimpulse.json")  # Default proxy config file

logger = logging.getLogger("desktopenv.setup")
logger.setLevel(logging.INFO)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

# init_proxy_pool(PROXY_CONFIG_FILE)  # initialize the global proxy pool

MAX_RETRIES = 3

class SetupController:
    def __init__(self, vm_ip: str, server_port: int = 5000, chromium_port: int = 9222, vlc_port: int = 8080, cache_dir: str = "cache", client_password: str = "", screen_width: int = 1920, screen_height: int = 1080):
        self.vm_ip: str = vm_ip
        self.server_port: int = server_port
        self.chromium_port: int = chromium_port
        self.vlc_port: int = vlc_port
        self.http_server: str = f"http://{vm_ip}:{server_port}"
        self.http_server_setup_root: str = f"http://{vm_ip}:{server_port}/setup"
        self.cache_dir: str = cache_dir
        self.use_proxy: bool = False
        self.client_password: str = client_password
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height

    def reset_cache_dir(self, cache_dir: str):
        self.cache_dir = cache_dir

    def setup(self, config: List[Dict[str, Any]], use_proxy: bool = False)-> bool:
        """
        Args:
            config (List[Dict[str, Any]]): list of dict like {str: Any}. each
              config dict has the structure like
                {
                    "type": str, corresponding to the `_{:}_setup` methods of
                      this class
                    "parameters": dict like {str, Any} providing the keyword
                      parameters
                }
        """  
        self.use_proxy = use_proxy
        # make sure connection can be established
        logger.info(f"try to connect {self.http_server}")
        retry = 0
        while retry < MAX_RETRIES:
            try:
                _ = requests.get(self.http_server + "/terminal", timeout=120)
                break
            except:
                time.sleep(5)
                retry += 1
                logger.info(f"retry: {retry}/{MAX_RETRIES}")
            
            if retry == MAX_RETRIES:
                return False
                

        for i, cfg in enumerate(config):
            config_type: str = cfg["type"]
            parameters: Dict[str, Any] = cfg["parameters"]

            # Assumes all the setup the functions should follow this name
            # protocol
            setup_function: str = "_{:}_setup".format(config_type)
            assert hasattr(self, setup_function), f'Setup controller cannot find init function {setup_function}'
            
            try:
                logger.info(f"Executing setup step {i+1}/{len(config)}: {setup_function}")
                logger.debug(f"Setup parameters: {parameters}")
                getattr(self, setup_function)(**parameters)
                logger.info(f"SETUP COMPLETED: {setup_function}({str(parameters)})")
            except Exception as e:
                logger.error(f"SETUP FAILED at step {i+1}/{len(config)}: {setup_function}({str(parameters)})")
                logger.error(f"Error details: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # raise Exception(f"Setup step {i+1} failed: {setup_function} - {e}") from e
                return False
        
        return True

    def _download_setup(self, files: List[Dict[str, str]]):
        """
        Args:
            files (List[Dict[str, str]]): files to download. lisf of dict like
              {
                "url": str, the url to download
                "path": str, the path on the VM to store the downloaded file
              }
        """
        for f in files:
            url: str = f["url"]
            path: str = f["path"]
            # Add process ID to make cache path unique per worker to avoid conflicts
            # worker_id = os.getpid()
            cache_path: str = os.path.join(self.cache_dir, "{:}_{:}".format(
                uuid.uuid5(uuid.NAMESPACE_URL, url),
                os.path.basename(path)))
            if not url or not path:
                raise Exception(f"Setup Download - Invalid URL ({url}) or path ({path}).")

            if not os.path.exists(cache_path):
                logger.info(f"Cache file not found, downloading from {url} to {cache_path}")
                max_retries = 3
                downloaded = False
                e = None
                for i in range(max_retries):
                    try:
                        logger.info(f"Download attempt {i+1}/{max_retries} for {url}")
                        response = requests.get(url, stream=True, timeout=120)  # Add 5 minute timeout
                        response.raise_for_status()
                        
                        # Get file size if available
                        total_size = int(response.headers.get('content-length', 0))
                        if total_size > 0:
                            logger.info(f"File size: {total_size / (1024*1024):.2f} MB")

                        downloaded_size = 0
                        with open(cache_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    if total_size > 0 and downloaded_size % (1024*1024) == 0:  # Log every MB
                                        progress = (downloaded_size / total_size) * 100
                                        logger.info(f"Download progress: {progress:.1f}%")
                        
                        logger.info(f"File downloaded successfully to {cache_path} ({downloaded_size / (1024*1024):.2f} MB)")
                        downloaded = True
                        break

                    except requests.RequestException as e:
                        logger.error(
                            f"Failed to download {url} caused by {e}. Retrying... ({max_retries - i - 1} attempts left)")
                        # Clean up partial download
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                if not downloaded:
                    logger.error(f"Failed to download {url}. No retries left.")
                    # raise requests.RequestException(f"Failed to download {url}. No retries left.")

            # send request to server to upload file with retry mechanism
            max_upload_retries = 3
            upload_retry_interval = 2
            uploaded = False
            last_error = None
            
            for attempt in range(max_upload_retries):
                try:
                    if attempt > 0:
                        logger.info(f"Upload retry attempt {attempt + 1}/{max_upload_retries} for {os.path.basename(path)}")
                    else:
                        logger.info(f"Uploading {os.path.basename(path)} to VM at {path}")
                    
                    logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")
                    
                    # Use context manager to ensure file is properly closed
                    with open(cache_path, "rb") as file_handle:
                        form = MultipartEncoder({
                            "file_path": path,
                            "file_data": (os.path.basename(path), file_handle)
                        })
                        headers = {"Content-Type": form.content_type}
                        logger.debug(form.content_type)
                        
                        response = requests.post(self.http_server + "/setup" + "/upload", headers=headers, data=form, timeout=120)  # 10 minute timeout for upload
                        
                    if response.status_code == 200:
                        logger.info(f"File uploaded successfully: {path}")
                        logger.debug("Upload response: %s", response.text)
                        uploaded = True
                        break
                    else:
                        last_error = f"HTTP Status code: {response.status_code}, Response: {response.text}"
                        logger.error(f"Failed to upload file {path}. Status code: {response.status_code}, Response: {response.text}")
                        if attempt < max_upload_retries - 1:
                            logger.info(f"Retrying upload in {upload_retry_interval} seconds...")
                            time.sleep(upload_retry_interval)
                        
                except requests.exceptions.RequestException as e:
                    last_error = str(e)
                    logger.error(f"An error occurred while trying to upload {path}: {e}")
                    if attempt < max_upload_retries - 1:
                        logger.info(f"Retrying upload in {upload_retry_interval} seconds...")
                        time.sleep(upload_retry_interval)
                    
            if not uploaded:
                error_msg = f"Failed to upload file {path} after {max_upload_retries} attempts. Last error: {last_error}"
                logger.error(error_msg)
                # raise requests.RequestException(error_msg)

    def _upload_file_setup(self, files: List[Dict[str, str]]):
        """
        Args:
            files (List[Dict[str, str]]): files to download. lisf of dict like
              {
                "local_path": str, the local path to the file to upload
                "path": str, the path on the VM to store the downloaded file
              }
        """
        for f in files:
            local_path: str = f["local_path"]
            path: str = f["path"]

            if not os.path.exists(local_path):
                logger.error(f"Setup Upload - Invalid local path ({local_path}).")
                return

            # send request to server to upload file with retry mechanism
            max_upload_retries = 3
            upload_retry_interval = 5
            uploaded = False
            last_error = None
            
            for attempt in range(max_upload_retries):
                try:
                    if attempt > 0:
                        logger.info(f"Upload retry attempt {attempt + 1}/{max_upload_retries} for {os.path.basename(path)}")
                    else:
                        logger.info(f"Uploading {os.path.basename(path)} to VM at {path}")
                    
                    logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")
                    
                    # Use context manager to ensure file is properly closed
                    with open(local_path, "rb") as file_handle:
                        form = MultipartEncoder({
                            "file_path": path,
                            "file_data": (os.path.basename(path), file_handle)
                        })
                        headers = {"Content-Type": form.content_type}
                        logger.debug(form.content_type)
                        
                        response = requests.post(self.http_server + "/setup" + "/upload", headers=headers, data=form, timeout=120)
                        
                    if response.status_code == 200:
                        logger.info(f"File uploaded successfully: {path}")
                        logger.debug("Upload response: %s", response.text)
                        uploaded = True
                        break
                    else:
                        last_error = f"HTTP Status code: {response.status_code}, Response: {response.text}"
                        logger.error(f"Failed to upload file {path}. Status code: {response.status_code}, Response: {response.text}")
                        if attempt < max_upload_retries - 1:
                            logger.info(f"Retrying upload in {upload_retry_interval} seconds...")
                            time.sleep(upload_retry_interval)
                        
                except requests.exceptions.RequestException as e:
                    last_error = str(e)
                    logger.error(f"An error occurred while trying to upload {path}: {e}")
                    if attempt < max_upload_retries - 1:
                        logger.info(f"Retrying upload in {upload_retry_interval} seconds...")
                        time.sleep(upload_retry_interval)
                    
            if not uploaded:
                error_msg = f"Failed to upload file {path} after {max_upload_retries} attempts. Last error: {last_error}"
                logger.error(error_msg)
                # Note: This method doesn't raise exceptions, just logs errors for backward compatibility

    def _change_wallpaper_setup(self, path: str):
        if not path:
            raise Exception(f"Setup Wallpaper - Invalid path ({path}).")

        payload = json.dumps({"path": path})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to change wallpaper
        try:
            response = requests.post(self.http_server + "/setup" + "/change_wallpaper", headers=headers, data=payload, timeout=120)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error("Failed to change wallpaper. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    def _tidy_desktop_setup(self, **config):
        raise NotImplementedError()

    def _open_setup(self, path: str):
        if not path:
            raise Exception(f"Setup Open - Invalid path ({path}).")

        payload = json.dumps({"path": path})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            # The server-side call is now blocking and can take time.
            # We set a timeout that is slightly longer than the server's timeout (1800s).
            response = requests.post(self.http_server + "/setup" + "/open_file", headers=headers, data=payload, timeout=120)
            response.raise_for_status()  # This will raise an exception for 4xx and 5xx status codes
            logger.info("Command executed successfully: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to open file '{path}'. An error occurred while trying to send the request or the server responded with an error: {e}")
            raise Exception(f"Failed to open file '{path}'. An error occurred while trying to send the request or the server responded with an error: {e}") from e

    def _launch_setup(self, command: Union[str, List[str]], shell: bool = False):
        if not command:
            raise Exception("Empty command to launch.")

        if not shell and isinstance(command, str) and len(command.split()) > 1:
            logger.warning("Command should be a list of strings. Now it is a string. Will split it by space.")
            command = command.split()
            
        if command[0] == "google-chrome" and self.use_proxy:
            command.append("--proxy-server=http://127.0.0.1:18888")  # Use the proxy server set up by _proxy_setup

        payload = json.dumps({"command": command, "shell": shell})
        headers = {"Content-Type": "application/json"}

        try:
            logger.info("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/launch")
            response = requests.post(self.http_server + "/setup" + "/launch", headers=headers, data=payload, timeout=120)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error("Failed to launch application. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    def _execute_setup(
            self,
            command: List[str],
            stdout: str = "",
            stderr: str = "",
            shell: bool = False,
            until: Optional[Dict[str, Any]] = None
    ):
        if not command:
            raise Exception("Empty command to launch.")

        until: Dict[str, Any] = until or {}
        terminates: bool = False
        nb_failings = 0

        def replace_screen_env_in_command(command):
            password = self.client_password
            width = self.screen_width
            height = self.screen_height
            width_half = str(width // 2)
            height_half = str(height // 2)
            new_command_list = []
            new_command = ""
            if isinstance(command, str):
                new_command = command.replace("{CLIENT_PASSWORD}", password)
                new_command = new_command.replace("{SCREEN_WIDTH_HALF}", width_half)
                new_command = new_command.replace("{SCREEN_HEIGHT_HALF}", height_half)
                new_command = new_command.replace("{SCREEN_WIDTH}", str(width))
                new_command = new_command.replace("{SCREEN_HEIGHT}", str(height))
                return new_command
            else:
                for item in command:
                    item = item.replace("{CLIENT_PASSWORD}", password)
                    item = item.replace("{SCREEN_WIDTH_HALF}", width_half)
                    item = item.replace("{SCREEN_HEIGHT_HALF}", height_half)
                    item = item.replace("{SCREEN_WIDTH}", str(width))
                    item = item.replace("{SCREEN_HEIGHT}", str(height))
                    new_command_list.append(item)
                return new_command_list
        command = replace_screen_env_in_command(command)
        payload = json.dumps({"command": command, "shell": shell})
        headers = {"Content-Type": "application/json"}

        while not terminates:
            try:
                response = requests.post(self.http_server + "/setup" + "/execute", headers=headers, data=payload, timeout=120)
                if response.status_code == 200:
                    results: Dict[str, str] = response.json()
                    if stdout:
                        with open(os.path.join(self.cache_dir, stdout), "w") as f:
                            f.write(results["output"])
                    if stderr:
                        with open(os.path.join(self.cache_dir, stderr), "w") as f:
                            f.write(results["error"])
                    logger.info("Command executed successfully: %s -> %s"
                                , " ".join(command) if isinstance(command, list) else command
                                , response.text
                                )
                else:
                    logger.error("Failed to launch application. Status code: %s", response.text)
                    results = None
                    nb_failings += 1
            except requests.exceptions.RequestException as e:
                logger.error("An error occurred while trying to send the request: %s", e)
                traceback.print_exc()

                results = None
                nb_failings += 1

            if len(until) == 0:
                terminates = True
            elif results is not None:
                terminates = "returncode" in until and results["returncode"] == until["returncode"] \
                             or "stdout" in until and until["stdout"] in results["output"] \
                             or "stderr" in until and until["stderr"] in results["error"]
            terminates = terminates or nb_failings >= 5
            if not terminates:
                time.sleep(0.3)

    def _execute_with_verification_setup(
            self,
            command: List[str],
            verification: Dict[str, Any] = None,
            max_wait_time: int = 10,
            check_interval: float = 1.0,
            shell: bool = False
    ):
        """Execute command with verification of results
        
        Args:
            command: Command to execute
            verification: Dict with verification criteria:
                - window_exists: Check if window with this name exists
                - command_success: Execute this command and check if it succeeds
            max_wait_time: Maximum time to wait for verification
            check_interval: Time between verification checks
            shell: Whether to use shell
        """
        if not command:
            raise Exception("Empty command to launch.")

        verification = verification or {}
        
        payload = json.dumps({
            "command": command, 
            "shell": shell,
            "verification": verification,
            "max_wait_time": max_wait_time,
            "check_interval": check_interval
        })
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.http_server + "/setup" + "/execute_with_verification", 
                                   headers=headers, data=payload, timeout=max_wait_time + 10)
            if response.status_code == 200:
                result = response.json()
                logger.info("Command executed and verified successfully: %s -> %s"
                            , " ".join(command) if isinstance(command, list) else command
                            , response.text
                            )
                return result
            else:
                logger.error("Failed to execute with verification. Status code: %s", response.text)
                raise Exception(f"Command verification failed: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)
            traceback.print_exc()
            raise Exception(f"Request failed: {e}")

    def _command_setup(self, command: List[str], **kwargs):
        self._execute_setup(command, **kwargs)

    def _sleep_setup(self, seconds: float):
        time.sleep(seconds)

    # -----------------------------
    # ScienceBoard compatibility helpers
    # -----------------------------
    def _opt_setup(self, depth: int):
        """ScienceBoard/OSWorld compatibility: set a11y MAX_DEPTH via /opt."""
        payload = json.dumps({"depth": int(depth)})
        headers = {"Content-Type": "application/json"}
        resp = requests.post(self.http_server + "/opt", headers=headers, data=payload, timeout=120)
        if resp.status_code != 200:
            raise Exception(f"Failed to set /opt depth={depth}: {resp.status_code} {resp.text}")

    def _write_file_setup(self, path: str, content: str):
        """Write a file via server /write (ScienceBoard VM server extension)."""
        payload = json.dumps({"path": path, "content": content})
        headers = {"Content-Type": "application/json"}
        resp = requests.post(self.http_server + "/write", headers=headers, data=payload, timeout=120)
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"Failed to write file {path}: {resp.status_code} {resp.text}")

    def _append_file_setup(self, path: str, content: str):
        """Append to a file via server /append (ScienceBoard VM server extension)."""
        payload = json.dumps({"path": path, "content": content})
        headers = {"Content-Type": "application/json"}
        resp = requests.post(self.http_server + "/append", headers=headers, data=payload, timeout=120)
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"Failed to append file {path}: {resp.status_code} {resp.text}")

    def _chimerax_cmd_setup(self, command: str, port: int = 8000):
        """Call ChimeraX remotecontrol via server /chimerax/run (ScienceBoard VM server extension)."""
        payload = json.dumps({"port": int(port), "command": command})
        headers = {"Content-Type": "application/json"}
        resp = requests.post(self.http_server + "/chimerax/run", headers=headers, data=payload, timeout=120)
        if resp.status_code != 200:
            raise Exception(f"Failed chimerax cmd={command!r}: {resp.status_code} {resp.text}")
        data = resp.json()
        if data.get("error") is not None:
            raise Exception(f"ChimeraX error for cmd={command!r}: {data.get('error')}")

    def _lean_prep_setup(self, func: str, libs=None, expr: str = None):
        """Prepare Lean file (/home/user/sci/Sci/Basic.lean) similarly to ScienceBoard's VMTask."""
        if not hasattr(self, "_scienceboard_lean_buf"):
            setattr(self, "_scienceboard_lean_buf", [])
        buf = getattr(self, "_scienceboard_lean_buf")

        base_path = "/home/user/sci/Sci/Basic.lean"

        def probe():
            if len(buf) == 0 or buf[-1] != "":
                buf.append("")

        if func == "import":
            libs = libs or []
            libs = [lib for lib in libs if lib != "Mathlib"]
            if libs:
                probe()
                buf.append(f"import {' '.join(libs)}")
        elif func == "open":
            libs = libs or []
            probe()
            buf.append(f"open {' '.join(libs)}")
            probe()
        elif func == "def":
            if expr:
                buf.append(str(expr))
        elif func == "query":
            if not isinstance(expr, str):
                raise Exception("Lean query expects expr string")
            header = expr.rstrip()
            if header.endswith("by sorry"):
                header = header[:-6].rstrip()
            probe()
            buf.append(header + "\n  sorry\n")
            content = "\n".join(buf)
            self._append_file_setup(path=base_path, content=content)
        else:
            raise Exception(f"Unknown lean_prep func: {func}")

    # ---- KAlgebra init calls (direct app server on port)
    def _kalgebra_tab_setup(self, port: int, index: int):
        resp = requests.post(f"http://{self.vm_ip}:{int(port)}/tab", json=int(index), timeout=120)
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"KAlgebra tab failed: {resp.status_code} {resp.text}")

    def _kalgebra_func_2d_setup(self, port: int, expr: str):
        resp = requests.post(f"http://{self.vm_ip}:{int(port)}/add/2d", data=str(expr), timeout=120)
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"KAlgebra func_2d failed: {resp.status_code} {resp.text}")

    def _kalgebra_func_3d_setup(self, port: int, expr: str):
        resp = requests.post(f"http://{self.vm_ip}:{int(port)}/add/3d", data=str(expr), timeout=120)
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"KAlgebra func_3d failed: {resp.status_code} {resp.text}")

    # ---- GrassGIS init calls (direct app server on port)
    def _grass_map_setup(self, port: int, grassdb: str, location: str, mapset: str):
        resp = requests.post(
            f"http://{self.vm_ip}:{int(port)}/init/map",
            json={"grassdb": grassdb, "location": location, "mapset": mapset},
            timeout=120,
        )
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"GrassGIS map failed: {resp.status_code} {resp.text}")

    def _grass_layer_setup(self, port: int, query: Dict[str, str]):
        resp = requests.post(
            f"http://{self.vm_ip}:{int(port)}/init/layer",
            json={"query": query},
            timeout=120,
        )
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"GrassGIS layer failed: {resp.status_code} {resp.text}")

    def _grass_scale_setup(self, port: int, scale: int):
        resp = requests.post(
            f"http://{self.vm_ip}:{int(port)}/init/scale",
            json={"scale": int(scale)},
            timeout=120,
        )
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"GrassGIS scale failed: {resp.status_code} {resp.text}")

    def _grass_cmd_setup(self, port: int):
        resp = requests.get(f"http://{self.vm_ip}:{int(port)}/init/cmd", timeout=120)
        if resp.status_code != 200 or resp.text.strip() != "OK":
            raise Exception(f"GrassGIS cmd init failed: {resp.status_code} {resp.text}")

    def _act_setup(self, action_seq: List[Union[Dict[str, Any], str]]):
        # TODO
        raise NotImplementedError()

    def _replay_setup(self, trajectory: str):
        """
        Args:
            trajectory (str): path to the replay trajectory file
        """

        # TODO
        raise NotImplementedError()

    def _activate_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False):
        if not window_name:
            raise Exception(f"Setup Open - Invalid path ({window_name}).")

        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            response = requests.post(self.http_server + "/setup" + "/activate_window", headers=headers, data=payload, timeout=120)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error(f"Failed to activate window {window_name}. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    def _close_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False):
        if not window_name:
            raise Exception(f"Setup Open - Invalid path ({window_name}).")

        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            response = requests.post(self.http_server + "/setup" + "/close_window", headers=headers, data=payload, timeout=120)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error(f"Failed to close window {window_name}. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    def _proxy_setup(self, client_password: str = ""):
        """Setup system-wide proxy configuration using proxy pool
        
        Args:
            client_password (str): Password for sudo operations, defaults to "password"
        """
        retry = 0
        while retry < MAX_RETRIES:
            try:
                _ = requests.get(self.http_server + "/terminal", timeout=120)
                break
            except:
                time.sleep(5)
                retry += 1
                logger.info(f"retry: {retry}/{MAX_RETRIES}")
            
            if retry == MAX_RETRIES:
                return False
            
        # Get proxy from global proxy pool
        proxy_pool = get_global_proxy_pool()
        current_proxy = proxy_pool.get_next_proxy()
        
        if not current_proxy:
            logger.error("No proxy available from proxy pool")
            raise Exception("No proxy available from proxy pool")
        
        # Format proxy URL
        proxy_url = proxy_pool._format_proxy_url(current_proxy)
        logger.info(f"Setting up proxy: {current_proxy.host}:{current_proxy.port}")
        
        # Configure system proxy environment variables  
        proxy_commands = [
            f"echo '{client_password}' | sudo -S bash -c \"apt-get update\"", ## TODO: remove this line if ami is already updated
            f"echo '{client_password}' | sudo -S bash -c \"apt-get install -y tinyproxy\"", ## TODO: remove this line if tinyproxy is already installed
            f"echo '{client_password}' | sudo -S bash -c \"echo 'Port 18888' > /tmp/tinyproxy.conf\"",
            f"echo '{client_password}' | sudo -S bash -c \"echo 'Allow 127.0.0.1' >> /tmp/tinyproxy.conf\"",
            f"echo '{client_password}' | sudo -S bash -c \"echo 'Upstream http {current_proxy.username}:{current_proxy.password}@{current_proxy.host}:{current_proxy.port}' >> /tmp/tinyproxy.conf\"",
            
            # CML commands to set environment variables for proxy
            f"echo 'export http_proxy={proxy_url}' >> ~/.bashrc",
            f"echo 'export https_proxy={proxy_url}' >> ~/.bashrc",
            f"echo 'export HTTP_PROXY={proxy_url}' >> ~/.bashrc",
            f"echo 'export HTTPS_PROXY={proxy_url}' >> ~/.bashrc",
        ]

        # Execute all proxy configuration commands
        for cmd in proxy_commands:
            try:
                self._execute_setup([cmd], shell=True)
            except Exception as e:
                logger.error(f"Failed to execute proxy setup command: {e}")
                proxy_pool.mark_proxy_failed(current_proxy)
                raise
        
        self._launch_setup(["tinyproxy -c /tmp/tinyproxy.conf -d"], shell=True)
        
        # Reload environment variables
        reload_cmd = "source /etc/environment"
        try:
            logger.info(f"Proxy setup completed successfully for {current_proxy.host}:{current_proxy.port}")
            proxy_pool.mark_proxy_success(current_proxy)
        except Exception as e:
            logger.error(f"Failed to reload environment variables: {e}")
            proxy_pool.mark_proxy_failed(current_proxy)
            raise

    # Chrome setup
    def _chrome_open_tabs_setup(self, urls_to_open: List[str]):
        host = self.vm_ip
        port = self.chromium_port  # fixme: this port is hard-coded, need to be changed from config file

        remote_debugging_url = f"http://{host}:{port}"
        logger.info("Connect to Chrome @: %s", remote_debugging_url)
        logger.debug("PLAYWRIGHT ENV: %s", repr(os.environ))
        for attempt in range(15):
            if attempt > 0:
                time.sleep(5)

            browser = None
            with sync_playwright() as p:
                try:
                    browser = p.chromium.connect_over_cdp(remote_debugging_url)
                    # break
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        # time.sleep(10)
                        continue
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e

                if not browser:
                    return

                logger.info("Opening %s...", urls_to_open)
                for i, url in enumerate(urls_to_open):
                    # Use the first context (which should be the only one if using default profile)
                    if i == 0:
                        context = browser.contexts[0]

                    page = context.new_page()  # Create a new page (tab) within the existing context
                    try:
                        page.goto(url, timeout=60000)
                    except:
                        logger.warning("Opening %s exceeds time limit", url)  # only for human test
                    logger.info(f"Opened tab {i + 1}: {url}")

                    if i == 0:
                        # clear the default tab
                        default_page = context.pages[0]
                        default_page.close()

                # Do not close the context or browser; they will remain open after script ends
                return browser, context

    def _chrome_close_tabs_setup(self, urls_to_close: List[str]):
        time.sleep(5)  # Wait for Chrome to finish launching

        host = self.vm_ip
        port = self.chromium_port  # fixme: this port is hard-coded, need to be changed from config file

        remote_debugging_url = f"http://{host}:{port}"
        with sync_playwright() as p:
            browser = None
            for attempt in range(15):
                try:
                    browser = p.chromium.connect_over_cdp(remote_debugging_url)
                    break
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        time.sleep(5)
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e

            if not browser:
                return

            for i, url in enumerate(urls_to_close):
                # Use the first context (which should be the only one if using default profile)
                if i == 0:
                    context = browser.contexts[0]

                for page in context.pages:

                    # if two urls are the same, close the tab
                    if compare_urls(page.url, url):
                        context.pages.pop(context.pages.index(page))
                        page.close()
                        logger.info(f"Closed tab {i + 1}: {url}")
                        break

            # Do not close the context or browser; they will remain open after script ends
            return browser, context

    # google drive setup
    def _googledrive_setup(self, **config):
        """ Clean google drive space (eliminate the impact of previous experiments to reset the environment)
        @args:
            config(Dict[str, Any]): contain keys
                settings_file(str): path to google drive settings file, which will be loaded by pydrive.auth.GoogleAuth()
                operation(List[str]): each operation is chosen from ['delete', 'upload']
                args(List[Dict[str, Any]]): parameters for each operation
            different args dict for different operations:
                for delete:
                    query(str): query pattern string to search files or folder in google drive to delete, please refer to
                        https://developers.google.com/drive/api/guides/search-files?hl=en about how to write query string.
                    trash(bool): whether to delete files permanently or move to trash. By default, trash=false, completely delete it.
                for mkdirs:
                    path(List[str]): the path in the google drive to create folder
                for upload:
                    path(str): remote url to download file
                    dest(List[str]): the path in the google drive to store the downloaded file
        """
        settings_file = config.get('settings_file', 'evaluation_examples/settings/googledrive/settings.yml')
        gauth = GoogleAuth(settings_file=settings_file)
        drive = GoogleDrive(gauth)

        def mkdir_in_googledrive(paths: List[str]):
            paths = [paths] if type(paths) != list else paths
            parent_id = 'root'
            for p in paths:
                q = f'"{parent_id}" in parents and title = "{p}" and mimeType = "application/vnd.google-apps.folder" and trashed = false'
                folder = drive.ListFile({'q': q}).GetList()
                if len(folder) == 0:  # not exists, create it
                    parents = {} if parent_id == 'root' else {'parents': [{'id': parent_id}]}
                    file = drive.CreateFile({'title': p, 'mimeType': 'application/vnd.google-apps.folder', **parents})
                    file.Upload()
                    parent_id = file['id']
                else:
                    parent_id = folder[0]['id']
            return parent_id

        for oid, operation in enumerate(config['operation']):
            if operation == 'delete':  # delete a specific file
                # query pattern string, by default, remove all files/folders not in the trash to the trash
                params = config['args'][oid]
                q = params.get('query', '')
                trash = params.get('trash', False)
                q_file = f"( {q} ) and mimeType != 'application/vnd.google-apps.folder'" if q.strip() else "mimeType != 'application/vnd.google-apps.folder'"
                filelist: GoogleDriveFileList = drive.ListFile({'q': q_file}).GetList()
                q_folder = f"( {q} ) and mimeType = 'application/vnd.google-apps.folder'" if q.strip() else "mimeType = 'application/vnd.google-apps.folder'"
                folderlist: GoogleDriveFileList = drive.ListFile({'q': q_folder}).GetList()
                for file in filelist:  # first delete file, then folder
                    file: GoogleDriveFile
                    if trash:
                        file.Trash()
                    else:
                        file.Delete()
                for folder in folderlist:
                    folder: GoogleDriveFile
                    # note that, if a folder is trashed/deleted, all files and folders in it will be trashed/deleted
                    if trash:
                        folder.Trash()
                    else:
                        folder.Delete()
            elif operation == 'mkdirs':
                params = config['args'][oid]
                mkdir_in_googledrive(params['path'])
            elif operation == 'upload':
                params = config['args'][oid]
                url = params['url']
                with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmpf:
                    response = requests.get(url, stream=True, timeout=120)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmpf.write(chunk)
                    tmpf.close()
                    paths = [params['path']] if params['path'] != list else params['path']
                    parent_id = mkdir_in_googledrive(paths[:-1])
                    parents = {} if parent_id == 'root' else {'parents': [{'id': parent_id}]}
                    file = drive.CreateFile({'title': paths[-1], **parents})
                    file.SetContentFile(tmpf.name)
                    file.Upload()
                return
            else:
                raise ValueError('[ERROR]: not implemented clean type!')

    def _login_setup(self, **config):
        """ Login to a website with account and password information.
        @args:
            config(Dict[str, Any]): contain keys
                settings_file(str): path to the settings file
                platform(str): platform to login, implemented platforms include:
                    googledrive: https://drive.google.com/drive/my-drive

        """
        host = self.vm_ip
        port = self.chromium_port

        remote_debugging_url = f"http://{host}:{port}"
        with sync_playwright() as p:
            browser = None
            for attempt in range(15):
                try:
                    browser = p.chromium.connect_over_cdp(remote_debugging_url)
                    break
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        time.sleep(5)
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e
            if not browser:
                return

            context = browser.contexts[0]
            platform = config['platform']

            if platform == 'googledrive':
                url = 'https://drive.google.com/drive/my-drive'
                page = context.new_page()  # Create a new page (tab) within the existing context
                try:
                    page.goto(url, timeout=60000)
                except:
                    logger.warning("Opening %s exceeds time limit", url)  # only for human test
                logger.info(f"Opened new page: {url}")
                settings = json.load(open(config['settings_file']))
                email, password = settings['email'], settings['password']

                try:
                    page.wait_for_selector('input[type="email"]', state="visible", timeout=3000)
                    page.fill('input[type="email"]', email)
                    page.click('#identifierNext > div > button')
                    page.wait_for_selector('input[type="password"]', state="visible", timeout=5000)
                    page.fill('input[type="password"]', password)
                    page.click('#passwordNext > div > button')
                    page.wait_for_load_state('load', timeout=5000)
                except TimeoutError:
                    logger.info('[ERROR]: timeout when waiting for google drive login page to load!')
                    return

            else:
                raise NotImplementedError

            return browser, context

    def _update_browse_history_setup(self, **config):
        cache_path = os.path.join(self.cache_dir, "history_new.sqlite")
        db_url = "https://drive.usercontent.google.com/u/0/uc?id=1Lv74QkJYDWVX0RIgg0Co-DUcoYpVL0oX&export=download" # google drive
        if not os.path.exists(cache_path):
                max_retries = 3
                downloaded = False
                e = None
                for i in range(max_retries):
                    try:
                        response = requests.get(db_url, stream=True, timeout=120)
                        response.raise_for_status()

                        with open(cache_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        logger.info("File downloaded successfully")
                        downloaded = True
                        break

                    except requests.RequestException as e:
                        logger.error(
                            f"Failed to download {db_url} caused by {e}. Retrying... ({max_retries - i - 1} attempts left)")
                if not downloaded:
                    raise requests.RequestException(f"Failed to download {db_url}. No retries left. Error: {e}")
        else:
            logger.info("File already exists in cache directory")
        # copy a new history file in the tmp folder
        db_path = cache_path

        history = config['history']

        for history_item in history:
            url = history_item['url']
            title = history_item['title']
            visit_time = datetime.now() - timedelta(seconds=history_item['visit_time_from_now_in_seconds'])

            # Chrome use ms from 1601-01-01 as timestamp
            epoch_start = datetime(1601, 1, 1)
            chrome_timestamp = int((visit_time - epoch_start).total_seconds() * 1000000)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                   INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden)
                   VALUES (?, ?, ?, ?, ?, ?)
               ''', (url, title, 1, 0, chrome_timestamp, 0))

            url_id = cursor.lastrowid

            cursor.execute('''
                   INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration)
                   VALUES (?, ?, ?, ?, ?, ?)
               ''', (url_id, chrome_timestamp, 0, 805306368, 0, 0))

            conn.commit()
            conn.close()

        logger.info('Fake browsing history added successfully.')

        controller = PythonController(self.vm_ip, self.server_port)

        # get the path of the history file according to the platform
        os_type = controller.get_vm_platform()

        if os_type == 'Windows':
            chrome_history_path = controller.execute_python_command(
                """import os; print(os.path.join(os.getenv('USERPROFILE'), "AppData", "Local", "Google", "Chrome", "User Data", "Default", "History"))""")[
                'output'].strip()
        elif os_type == 'Darwin':
            chrome_history_path = controller.execute_python_command(
                """import os; print(os.path.join(os.getenv('HOME'), "Library", "Application Support", "Google", "Chrome", "Default", "History"))""")[
                'output'].strip()
        elif os_type == 'Linux':
            if "arm" in platform.machine():
                chrome_history_path = controller.execute_python_command(
                    "import os; print(os.path.join(os.getenv('HOME'), 'snap', 'chromium', 'common', 'chromium', 'Default', 'History'))")[
                    'output'].strip()
            else:
                chrome_history_path = controller.execute_python_command(
                    "import os; print(os.path.join(os.getenv('HOME'), '.config', 'google-chrome', 'Default', 'History'))")[
                    'output'].strip()
        else:
            raise Exception('Unsupported operating system')

        # send request to server to upload file with retry mechanism
        max_upload_retries = 3
        upload_retry_interval = 2
        uploaded = False
        last_error = None
        
        for attempt in range(max_upload_retries):
            try:
                if attempt > 0:
                    logger.info(f"Upload retry attempt {attempt + 1}/{max_upload_retries} for Chrome history file")
                else:
                    logger.info("Uploading Chrome history file to VM")
                
                logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")
                
                # Use context manager to ensure file is properly closed
                with open(db_path, "rb") as file_handle:
                    form = MultipartEncoder({
                        "file_path": chrome_history_path,
                        "file_data": (os.path.basename(chrome_history_path), file_handle)
                    })
                    headers = {"Content-Type": form.content_type}
                    logger.debug(form.content_type)
                    
                    response = requests.post(self.http_server + "/setup" + "/upload", headers=headers, data=form, timeout=120)
                    
                if response.status_code == 200:
                    logger.info("Chrome history file uploaded successfully: %s", response.text)
                    uploaded = True
                    break
                else:
                    last_error = f"HTTP Status code: {response.status_code}, Response: {response.text}"
                    logger.error(f"Failed to upload Chrome history file. Status code: {response.status_code}, Response: {response.text}")
                    if attempt < max_upload_retries - 1:
                        logger.info(f"Retrying upload in {upload_retry_interval} seconds...")
                        time.sleep(upload_retry_interval)
                    
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.error(f"An error occurred while trying to upload Chrome history file: {e}")
                if attempt < max_upload_retries - 1:
                    logger.info(f"Retrying upload in {upload_retry_interval} seconds...")
                    time.sleep(upload_retry_interval)
                
        if not uploaded:
            error_msg = f"Failed to upload Chrome history file after {max_upload_retries} attempts. Last error: {last_error}"
            logger.error(error_msg)
            # Note: This method doesn't raise exceptions, just logs errors for backward compatibility

        self._execute_setup(["sudo chown -R user:user /home/user/.config/google-chrome/Default/History"], shell=True)
