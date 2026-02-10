import logging
import os
import platform
import time
import docker
import psutil
import requests
import uuid
from filelock import FileLock
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.docker.DockerProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 10


class PortAllocationError(Exception):
    pass


class DockerProvider(Provider):
    def __init__(self, region: str, api_base_url: Optional[str] = None, api_token: Optional[str] = None):
        # NOTE: release-clean version: API mode + local docker only (no remote deployment path).
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        # ScienceBoard app port (e.g. GrassGIS/KAlgebra/Celestia/ChimeraX app servers)
        # Convention: container listens on 8000, host maps to an available port.
        self.app_port = None
        self.container = None
        self.container_name = f"env-{uuid.uuid4().hex[:8]}"  # Generate unique container name
        self.container_id = None  # For API mode
        self.environment = {"DISK_SIZE": "8G", "RAM_SIZE": "2G", "CPU_CORES": "32"}  # Modify if needed
        
        # API configuration
        self.api_base_url = api_base_url
        self.api_token = api_token
        self.use_api = self.api_base_url is not None
        # Optional: request a specific backend image when launching via API.
        # When None/empty, do not send the `image` field and let backend default.
        self.api_image = None

        # Hostname to reach the launched VM from the runner (API backend host).
        self.api_host = None
        if self.use_api:
            try:
                self.api_host = urlparse(self.api_base_url).hostname
            except Exception:
                self.api_host = None

        # Local docker client is only needed in local mode.
        self.client = None if self.use_api else docker.from_env()

        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "docker_port_allocation.lck"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_used_ports(self):
        """Get all currently used ports (both system and Docker)."""
        # Get system ports
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())
        
        # Get Docker container ports
        docker_ports = set()
        for container in self.client.containers.list():
            ports = container.attrs['NetworkSettings']['Ports']
            if ports:
                for port_mappings in ports.values():
                    if port_mappings:
                        docker_ports.update(int(p['HostPort']) for p in port_mappings)
        
        return system_ports | docker_ports

    def _get_remote_ip(self) -> str:
        """Get the host IP used by the runner to reach the VM."""
        if self.use_api:
            return self.api_host or "localhost"
        return "localhost"
    
    def _launch_via_api(self) -> dict:
        """Launch a container via API. Returns container info including ports."""
        if not self.use_api:
            raise RuntimeError("API mode not configured")
        
        url = f"{self.api_base_url}/launch"
        # Allow API backend to choose image; keep default compatible.
        payload = {"token": self.api_token}
        if self.api_image:
            payload["image"] = self.api_image
        
        try:
            logger.info(f"Launching container via API: {url}")
            response = requests.post(url, json=payload, timeout=1200)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Container launched successfully via API")
            logger.info(f"  Container ID: {data['container_id']}")
            logger.info(f"  Container Name: {data['container_name']}")
            
            return data
            
        except requests.exceptions.Timeout:
            raise TimeoutError("API request timed out")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Could not connect to API at {self.api_base_url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"API HTTP Error: {e}")
        except Exception as e:
            raise RuntimeError(f"API error: {e}")
    
    def _stop_via_api(self, container_id: str, raise_on_error: bool = False) -> bool:
        """Stop a container via API.
        
        Args:
            container_id: Container ID to stop
            raise_on_error: If True, raise exception on failure instead of returning False
        
        Returns True if successful.
        """
        if not self.use_api:
            raise RuntimeError("API mode not configured")
        
        url = f"{self.api_base_url}/stop"
        payload = {
            "container_id": container_id,
            "token": self.api_token
        }
        
        try:
            logger.info(f"Stopping container via API: {container_id}")
            response = requests.post(url, json=payload, timeout=1200)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Container stopped successfully via API: {data.get('message')}")
            return True
            
        except requests.exceptions.Timeout as e:
            logger.warning("API stop request timed out")
            if raise_on_error:
                raise RuntimeError(f"Failed to stop container {container_id}: Timeout") from e
            return False
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Could not connect to API at {self.api_base_url}")
            if raise_on_error:
                raise RuntimeError(f"Failed to stop container {container_id}: Connection error") from e
            return False
        except Exception as e:
            logger.warning(f"API stop error: {e}")
            if raise_on_error:
                raise RuntimeError(f"Failed to stop container {container_id}: {e}") from e
            return False

    def _get_available_port(self, start_port: int) -> int:
        """Find next available port starting from start_port."""
        used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = 120):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        
        def check_screenshot():
            try:
                if self.use_api:
                    # Use API backend host for connection
                    remote_ip = self._get_remote_ip()
                    url = f"http://{remote_ip}:{self.server_port}/screenshot"
                else:
                    # Use localhost for local connection
                    url = f"http://localhost:{self.server_port}/screenshot"
                
                response = requests.get(url, timeout=(10, 10))
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Error checking screenshot: {e}")
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        # Output container/pod name when VM fails to become ready
        if self.use_api:
            logger.error(f"VM failed to become ready within timeout period. Container ID: {self.container_id}")
            raise TimeoutError(f"VM failed to become ready within timeout period. Container ID: {self.container_id}")
        else:
            container_name = self.container.name if self.container else "Unknown"
            logger.error(f"VM failed to become ready within timeout period. Container name: {container_name}")
            raise TimeoutError(f"VM failed to become ready within timeout period. Container name: {container_name}")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        if self.use_api:
            # Use API mode
            self._start_emulator_api(path_to_vm, headless, os_type)
        else:
            # Use local Docker execution (original logic)
            self._start_emulator_local(path_to_vm, headless, os_type)
    
    def _start_emulator_api(self, path_to_vm: str, headless: bool, os_type: str):
        """Start emulator using API mode."""
        try:
            logger.info(f"Starting container via API")
            
            # Launch container via API
            data = self._launch_via_api()
            
            # Store container info
            self.container_id = data['container_id']
            self.container_name = data['container_name']
            
            # Extract port mappings from API response
            # Expected format: ports['8006/tcp'] = {'host_port': 12345, 'url': 'http://...'}
            ports = data.get('ports', {})
            
            # Parse ports from API response
            for container_port, port_info in ports.items():
                port_num = int(container_port.split('/')[0])
                host_port = int(port_info['host_port'])
                
                if port_num == 8006:
                    self.vnc_port = host_port
                elif port_num == 5000:
                    self.server_port = host_port
                elif port_num == 9222:
                    self.chromium_port = host_port
                elif port_num == 8080:
                    self.vlc_port = host_port
                elif port_num == 8000:
                    # ScienceBoard app server port (KAlgebra/GrassGIS/Celestia/ChimeraX)
                    self.app_port = host_port
            
            logger.info(f"Started container via API with ports - "
                       f"VNC: {self.vnc_port}, Server: {self.server_port}, "
                       f"Chrome: {self.chromium_port}, VLC: {self.vlc_port}, App: {self.app_port}")
            
            # Wait for VM to be ready
            self._wait_for_vm_ready()
            logger.info(f"Container started successfully via API")
            
        except Exception as e:
            # Clean up failed container before re-raising
            logger.error(f"Failed to start container via API: {e}")
            if self.container_id:
                try:
                    self._stop_via_api(self.container_id, raise_on_error=False)
                    logger.info("Cleaned up failed container")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup container: {cleanup_error}")
            raise

    def _start_emulator_local(self, path_to_vm: str, headless: bool, os_type: str):
        """Start emulator using local Docker execution (original logic)."""
        # Use a single lock for all port allocation and container startup
        lock = FileLock(str(self.lock_file), timeout=LOCK_TIMEOUT)
        
        try:
            with lock:
                # Allocate all required ports
                self.vnc_port = self._get_available_port(8006)
                self.server_port = self._get_available_port(5000)
                self.chromium_port = self._get_available_port(9222)
                self.vlc_port = self._get_available_port(8080)
                # ScienceBoard app server port mapping (container 8000 -> host app_port)
                self.app_port = self._get_available_port(8000)

                # Start container while still holding the lock
                # Check if KVM is available
                devices = []
                if os.path.exists("/dev/kvm"):
                    devices.append("/dev/kvm")
                    logger.info("KVM device found, using hardware acceleration")
                else:
                    self.environment["KVM"] = "N"
                    logger.warning("KVM device not found, running without hardware acceleration (will be slower)")

                self.container = self.client.containers.run(
                    "happysixd/osworld-docker",
                    environment=self.environment,
                    cap_add=["NET_ADMIN"],
                    devices=devices,
                    volumes={
                        os.path.abspath(path_to_vm): {
                            "bind": "/System.qcow2",
                            "mode": "ro"
                        }
                    },
                    ports={
                        8006: self.vnc_port,
                        5000: self.server_port,
                        9222: self.chromium_port,
                        8080: self.vlc_port,
                        8000: self.app_port,
                    },
                    detach=True
                )

            logger.info(
                f"Started container with ports - VNC: {self.vnc_port}, "
                f"Server: {self.server_port}, Chrome: {self.chromium_port}, "
                f"VLC: {self.vlc_port}, App: {self.app_port}"
            )

            # Wait for VM to be ready
            self._wait_for_vm_ready()

        except Exception as e:
            # Clean up if anything goes wrong
            container_name = self.container.name if self.container else "Unknown"
            logger.error(f"Local Docker container '{container_name}' failed to start: {e}")
            if self.container:
                try:
                    self.container.stop()
                    self.container.remove()
                except:
                    pass
            raise RuntimeError(f"Local Docker container '{container_name}' failed to start: {e}")

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        
        if self.use_api:
            # Return API backend host address
            remote_ip = self._get_remote_ip()
            # Include app_port as the last field when available (ScienceBoard app servers)
            if self.app_port:
                return f"{remote_ip}:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}:{self.app_port}"
            return f"{remote_ip}:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"
        else:
            # Return localhost for local execution
            if self.app_port:
                return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}:{self.app_port}"
            return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Docker provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        # Note: region parameter is ignored for Docker provider
        # but kept for interface consistency with other providers
        
        # Stop via API if using API mode
        if self.use_api and self.container_id:
            try:
                logger.info(f"Stopping container via API: {self.container_id}")
                success = self._stop_via_api(self.container_id)
                if success:
                    # Only clear state if stop was successful
                    self.container_id = None
                    self.server_port = None
                    self.vnc_port = None
                    self.chromium_port = None
                    self.vlc_port = None
                    self.app_port = None
                else:
                    logger.warning(f"Failed to stop container {self.container_id}, state not cleared")
            except Exception as e:
                logger.error(f"Error stopping container via API: {e}")
                # Don't clear state on error so we can retry later
            return
        
        # Stop local container if exists
        if self.container:
            logger.info("Stopping local container...")
            try:
                self.container.stop()
                self.container.remove()
                time.sleep(WAIT_TIME)
            except Exception as e:
                logger.error(f"Error stopping local container: {e}")
            finally:
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None
                self.app_port = None
