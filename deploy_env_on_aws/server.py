import traceback
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import docker
from docker.errors import DockerException, NotFound
import socket
import time
from typing import Optional, Dict, Literal
import logging
import requests
import threading
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Docker Container Manager")

SERVER_HOST = "127.0.0.1"

# Port range configuration
PORT_RANGE_START = 20000  # Start of port range
PORT_RANGE_END = 25000  # End of port range

# Global port forwarding management
allocated_ports: Dict[str, Dict] = {}  # container_id -> {container_port: forwarding_info}

# Configuration - Update these values
DOCKER_IMAGE = "happysixd/osworld-docker"
VALID_TOKEN = "3eb40e28-8f12-4b95-b9dd-743ea98334a9"
CONTAINER_ENVIRONMENT = {"DISK_SIZE": "8G", "RAM_SIZE": "2G", "CPU_CORES": "16"}
CONTAINER_COMMAND = None
CONTAINER_CAP_ADD = ["NET_ADMIN"]
CONTAINER_DEVICES = ["/dev/kvm"]

# Initialize Docker client
try:
    docker_client = docker.from_env(timeout=1200)
except DockerException as e:
    logger.error(f"Failed to connect to Docker: {e}")
    docker_client = None


def tcp_forward_with_random_port(
    dest_host: str,
    dest_port: int,
    stop_event: threading.Event,
    ready_event: threading.Event,
    port_container: dict,
    max_retries: int = 100,
):
    """
    Atomically find an available port and start TCP forwarding.
    Sets ready_event and stores bound port in port_container after successful binding.
    Runs until stop_event is set.
    """
    server_socket = None
    bound_port = None

    # Try to bind to a random port
    for _ in range(max_retries):
        port = random.randint(PORT_RANGE_START, PORT_RANGE_END - 1)
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.settimeout(1.0)  # Timeout to check stop_event periodically
            server_socket.bind(("0.0.0.0", port))
            server_socket.listen(5)
            bound_port = port
            logger.info(f"Started TCP forwarding: 0.0.0.0:{bound_port} -> {dest_host}:{dest_port}")

            # Signal that port is bound and ready
            port_container["port"] = bound_port
            ready_event.set()
            break
        except Exception:
            if server_socket:
                server_socket.close()
                server_socket = None
            continue  # Try another port

    if server_socket is None or bound_port is None:
        logger.error(f"Failed to bind to any port after {max_retries} attempts")
        port_container["port"] = None
        ready_event.set()
        return

    # Start forwarding loop
    try:
        while not stop_event.is_set():
            try:
                client_socket, client_addr = server_socket.accept()
                if stop_event.is_set():
                    client_socket.close()
                    break

                # Handle connection in a separate thread
                forward_thread = threading.Thread(
                    target=handle_tcp_connection, args=(client_socket, dest_host, dest_port), daemon=True
                )
                forward_thread.start()
            except socket.timeout:
                continue  # Check stop_event again
            except Exception as e:
                if not stop_event.is_set():
                    logger.error(f"Error accepting connection on port {bound_port}: {e}")
                break
    except Exception as e:
        logger.error(f"Error in TCP forwarding on port {bound_port}: {e}")
    finally:
        server_socket.close()
        logger.info(f"Stopped TCP forwarding on port {bound_port}")


def handle_tcp_connection(client_socket: socket.socket, dest_host: str, dest_port: int):
    """Handle a single TCP connection by forwarding data bidirectionally."""
    try:
        # Connect to destination
        dest_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dest_socket.connect((dest_host, dest_port))

        # Forward data bidirectionally
        def forward_data(source: socket.socket, destination: socket.socket):
            try:
                while True:
                    data = source.recv(4096)
                    if not data:
                        break
                    destination.sendall(data)
            except Exception:
                pass
            finally:
                source.close()
                destination.close()

        # Start two threads for bidirectional forwarding
        client_to_dest = threading.Thread(target=forward_data, args=(client_socket, dest_socket), daemon=True)
        dest_to_client = threading.Thread(target=forward_data, args=(dest_socket, client_socket), daemon=True)

        client_to_dest.start()
        dest_to_client.start()

        client_to_dest.join()
        dest_to_client.join()
    except Exception as e:
        logger.error(f"Error handling TCP connection to {dest_host}:{dest_port}: {e}")
        client_socket.close()


def get_container_ip(container) -> Optional[str]:
    """Get the IP address of a container."""
    try:
        container.reload()
        networks = container.attrs["NetworkSettings"]["Networks"]
        for network_name, network_info in networks.items():
            ip = network_info.get("IPAddress")
            if ip:
                return ip
        return None
    except Exception as e:
        logger.error(f"Error getting container IP: {e}")
        return None


class LaunchDockerRequest(BaseModel):
    token: str
    image: Literal["osworld", "scienceboard"] = "osworld"


class StopDockerRequest(BaseModel):
    container_id: str
    token: str


def check_container_port_available(container_ip: str, port: int, timeout: int = 1) -> bool:
    """Check if container port screenshot endpoint is accessible."""
    try:
        url = f"http://{container_ip}:{port}/screenshot"
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        return False


def wait_for_container_ready(container_ip: str, max_wait: int = 300, check_interval: int = 2) -> bool:
    """Wait for container port 5000 screenshot endpoint to become available."""
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if check_container_port_available(container_ip, 5000):
            logger.info(f"Container {container_ip}:5000 /screenshot endpoint is ready")
            return True

        logger.info(
            f"Waiting for container {container_ip}:5000 /screenshot endpoint... ({int(time.time() - start_time)}s)"
        )
        time.sleep(check_interval)

    return False


def setup_port_forwarding(container_id: str, container_ip: str, container_ports: list) -> Dict[int, int]:
    """
    Setup port forwarding for a container.
    Returns a dict mapping container_port -> host_port.
    """
    port_mappings = {}
    forwarding_info = {}

    for container_port in container_ports:
        # Create stop event and port ready event
        stop_event = threading.Event()
        port_ready_event = threading.Event()
        host_port_container = {"port": None}

        # Start forwarding thread
        forward_thread = threading.Thread(
            target=tcp_forward_with_random_port,
            args=(container_ip, container_port, stop_event, port_ready_event, host_port_container),
            daemon=True,
        )
        forward_thread.start()

        # Wait for port to be bound (with timeout)
        if not port_ready_event.wait(timeout=60):
            cleanup_port_forwarding(container_id)
            raise Exception(f"Failed to bind port for container port {container_port}")

        host_port = host_port_container["port"]
        if host_port is None:
            cleanup_port_forwarding(container_id)
            raise Exception(f"No available ports in range for container port {container_port}")

        port_mappings[container_port] = host_port
        forwarding_info[container_port] = {"host_port": host_port, "thread": forward_thread, "stop_event": stop_event}

    # Store forwarding info globally
    allocated_ports[container_id] = forwarding_info
    logger.info(f"Setup port forwarding for container {container_id}: {port_mappings}")

    return port_mappings


def cleanup_port_forwarding(container_id: str):
    """
    Clean up port forwarding for a container.
    Stops all forwarding threads and removes the container from tracking.
    """
    if container_id not in allocated_ports:
        return

    forwarding_info = allocated_ports[container_id]

    logger.info(f"Cleaning up port forwarding for container {container_id}")

    for container_port, info in forwarding_info.items():
        # Signal thread to stop
        info["stop_event"].set()
        logger.info(
            f"Stopped forwarding for container {container_id} port {container_port} (host port {info['host_port']})"
        )

    # Remove container from allocated_ports
    del allocated_ports[container_id]

    logger.info(f"Cleaned up port forwarding for container {container_id}")


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Docker Container Manager API",
        "allocated_containers": len(allocated_ports),
    }


@app.post("/launch")
def launch_docker(request: LaunchDockerRequest):
    """
    Launch a Docker container with specified configuration.
    Sets up Python-based TCP forwarding from host ports to container ports.
    Returns container info once ports are available.
    """
    if docker_client is None:
        raise HTTPException(status_code=500, detail="Docker client not initialized")

    # Verify token
    if request.token != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    container_name = uuid.uuid4().hex
    container = None

    try:
        # Container configuration (no port mapping)
        container_config = {
            "name": container_name,
            "image": DOCKER_IMAGE,
            "detach": True,
            "auto_remove": True,  # Container will be auto-deleted after stop
            "cap_add": CONTAINER_CAP_ADD,
            "devices": CONTAINER_DEVICES,
        }

        # Add optional parameters if configured
        if CONTAINER_ENVIRONMENT:
            container_config["environment"] = CONTAINER_ENVIRONMENT

        CONTAINER_VOLUMES = {"/srv/share/osworld-images/Ubuntu2.qcow2": {"bind": "/System.qcow2", "mode": "ro"}}
        if request.image == "scienceboard":
            CONTAINER_VOLUMES = {
                "/srv/share/osworld-images/scienceboard/Ubuntu2.qcow2": {"bind": "/System.qcow2", "mode": "ro"}
            }
        if CONTAINER_VOLUMES:
            container_config["volumes"] = CONTAINER_VOLUMES

        if CONTAINER_COMMAND:
            container_config["command"] = CONTAINER_COMMAND

        logger.info(f"Launching container with image: {DOCKER_IMAGE}")

        # Run the container without port mapping
        container = docker_client.containers.run(**container_config)
        container_id = container.id

        logger.info(f"Container started with ID: {container_id}")

        # Get container IP
        container_ip = get_container_ip(container)
        if not container_ip:
            raise Exception("Failed to get container IP address")

        logger.info(f"Container IP: {container_ip}")

        # Wait for container to be ready (port 5000 /screenshot endpoint)
        logger.info("Waiting for container to be ready...")
        container_ready = wait_for_container_ready(container_ip, max_wait=150)

        if not container_ready:
            logger.warning("Container did not become ready within timeout")
            raise HTTPException(
                status_code=504, detail="Container port 5000 /screenshot endpoint not available in time"
            )

        # Setup port forwarding for required ports
        container_ports = [5000, 8006, 9222, 8080, 8000]
        port_mappings = setup_port_forwarding(container_id, container_ip, container_ports)

        # Prepare response
        response = {
            "container_id": container_id,
            "container_name": container.name,
            "status": "running",
            "container_ip": container_ip,
            "ports": {},
            "image": DOCKER_IMAGE,
            "created": container.attrs.get("Created"),
        }

        # Add port information
        for container_port, host_port in port_mappings.items():
            response["ports"][str(container_port)] = {
                "host_port": host_port,
                "url": f"http://{SERVER_HOST}:{host_port}",
            }

        logger.info(f"Container {container_id} launched successfully with port forwarding: {port_mappings}")
        return response

    except docker.errors.ImageNotFound:
        traceback.print_exc()
        raise HTTPException(status_code=404, detail=f"Image '{DOCKER_IMAGE}' not found")
    except docker.errors.APIError as e:
        traceback.print_exc()
        logger.error(f"Docker API error: {e}")
        # Cleanup
        if container:
            try:
                cleanup_port_forwarding(container.id)
                container.kill()
            except Exception:
                pass
        raise HTTPException(status_code=504, detail=f"Docker API error: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Unexpected error launching container: {e}")
        # Cleanup
        if container:
            try:
                cleanup_port_forwarding(container.id)
                logger.info(f"Stopping container {container_name} due to launch error")
                container.kill()
                logger.info(f"Container {container_name} stopped successfully after launch error")
            except Exception as stop_error:
                logger.warning(f"Failed to stop container {container_name} after launch error: {stop_error}")
        raise HTTPException(status_code=500, detail=f"Error launching container: {str(e)}")


@app.post("/stop")
def stop_docker(request: StopDockerRequest):
    """
    Stop a Docker container by container ID.
    Cleans up port forwarding and the container will be automatically removed due to auto_remove flag.
    """
    if docker_client is None:
        raise HTTPException(status_code=500, detail="Docker client not initialized")

    # Verify token
    if request.token != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        logger.info(f"Stopping container: {request.container_id}")

        # Get the container
        container = docker_client.containers.get(request.container_id)

        container_image = container.image.tags[0] if container.image.tags else None

        # Match against DOCKER_IMAGE (handle cases with/without tags)
        if container_image and (container_image == DOCKER_IMAGE or container_image.startswith(f"{DOCKER_IMAGE}:")):
            # Clean up port forwarding first
            cleanup_port_forwarding(container.id)

            # Stop the container, retry three times
            for i in range(3):
                try:
                    container.kill()
                    break
                except Exception as e:
                    logger.warning(f"Failed to stop container {request.container_id} on attempt {i+1}: {e}")
                    if i == 2:
                        raise  # Re-raise if all attempts fail

            logger.info(f"Container {request.container_id} stopped successfully")

        return {
            "status": "stopped",
            "container_id": request.container_id,
            "message": "Container stopped and removed successfully",
        }

    except NotFound:
        raise HTTPException(status_code=404, detail=f"Container '{request.container_id}' not found")
    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error stopping container: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping container: {str(e)}")


@app.post("/stop_all")
def stop_all_containers(request: LaunchDockerRequest):
    """
    Stop all containers running with the configured DOCKER_IMAGE.
    Cleans up port forwarding for all containers.
    Requires authentication token.
    """
    if docker_client is None:
        raise HTTPException(status_code=500, detail="Docker client not initialized")

    # Verify token
    if request.token != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        logger.info(f"Stopping all containers with image: {DOCKER_IMAGE}")

        # Get all running containers
        all_containers = docker_client.containers.list()

        stopped_containers = []
        failed_containers = []

        for container in all_containers:
            # Check if container is running the target image
            container_image = container.image.tags[0] if container.image.tags else None

            # Match against DOCKER_IMAGE (handle cases with/without tags)
            if container_image and (container_image == DOCKER_IMAGE or container_image.startswith(f"{DOCKER_IMAGE}:")):
                try:
                    logger.info(f"Stopping container {container.id} ({container.name})")
                    # Clean up port forwarding
                    cleanup_port_forwarding(container.id)
                    # Stop container
                    container.kill()
                    stopped_containers.append(
                        {
                            "id": container.id,
                            "name": container.name,
                            "image": container_image,
                            "created": container.attrs.get("Created"),
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to stop container {container.id}: {e}")
                    failed_containers.append({"id": container.id, "name": container.name, "error": str(e)})

        logger.info(f"Stopped {len(stopped_containers)} containers")

        return {
            "status": "completed",
            "image": DOCKER_IMAGE,
            "stopped_count": len(stopped_containers),
            "failed_count": len(failed_containers),
            "stopped_containers": stopped_containers,
            "failed_containers": failed_containers,
            "message": f"Stopped {len(stopped_containers)} container(s) with image {DOCKER_IMAGE}",
        }

    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error stopping containers: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping containers: {str(e)}")


@app.post("/list_all")
def list_containers(request: LaunchDockerRequest):
    """List all running containers with the configured DOCKER_IMAGE. Requires authentication token."""
    if docker_client is None:
        raise HTTPException(status_code=500, detail="Docker client not initialized")

    # Verify token
    if request.token != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        containers = docker_client.containers.list()
        container_list = []

        for container in containers:
            # Check if container is running the target image
            container_image = container.image.tags[0] if container.image.tags else None

            # Match against DOCKER_IMAGE (handle cases with/without tags)
            if container_image and (container_image == DOCKER_IMAGE or container_image.startswith(f"{DOCKER_IMAGE}:")):
                container_list.append(
                    {
                        "id": container.id,
                        "name": container.name,
                        "image": container_image,
                        "status": container.status,
                        "ports": container.ports,
                        "created": container.attrs.get("Created"),
                    }
                )

        return {"image": DOCKER_IMAGE, "count": len(container_list), "containers": container_list}

    except Exception as e:
        logger.error(f"Error listing containers: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing containers: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10473)
