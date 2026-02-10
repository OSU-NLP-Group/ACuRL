This document provides a complete, end-to-end deployment guide for setting up the server on AWS EC2, including instance provisioning, storage configuration, OSWorld image preparation, and server launch.

# Config EC2 Instances

## Launch EC2 Instances
Navigate to: AWS Console → EC2 → Instances → Launch an instance

![aws_ec2_1](../figures/aws_ec2_1.png)
![aws_ec2_2](../figures/aws_ec2_2.png)

## Networking & Security Group
Allow all inbound ports from trusted IPs

# Set up EC2 Instances

## Connect to the Instance
```
ssh ubuntu@ip
```

## Docker Installation
```
curl -fsSL https://get.docker.com -o install-docker.sh
sudo sh install-docker.sh
```

## Enable Docker for Non-root User
```
sudo usermod -aG docker ubuntu
newgrp docker
```

## Python Environment Setup (Miniconda)

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

## Upload Server Code 
Upload the server.py, gunicorn.conf.py and requirements.txt files

## Install Dependencies
```
pip install -r requirements.txt
```

## Download Environment Image and unzip
```
sudo mkdir -p /srv/share/osworld-images/scienceboard
```

```
wget -c "https://huggingface.co/datasets/xuetianci99/ACuRL_files/resolve/main/Ubuntu2.qcow2.zip"
wget -c "https://huggingface.co/datasets/xuetianci99/ACuRL_files/resolve/main/Ubuntu2_scienceboard.qcow2.zip"
```

Unzip files and put the images into corresponding paths
```text
/srv/share/osworld-images/
├── Ubuntu2.qcow2
└── scienceboard/
    └── Ubuntu2.qcow2
```

## System Limits and Kernel Tuning
```
sudo nano /etc/sysctl.conf
```

Append:
```
fs.file-max = 2147480000
fs.inotify.max_user_instances = 131072
fs.inotify.max_user_watches = 1048576
```

Apply:
```
sudo sysctl -p
```

## Launch the Server
```
gunicorn -c gunicorn.conf.py server:app > tmp.txt 2>&1
```

# List all the environments
```
curl -X POST http://IP:10473/list_all \
  -H "Content-Type: application/json" \
  -d '{"token": "3eb40e28-8f12-4b95-b9dd-743ea98334a9"}' \
  | jq '.containers | length'
```

# Stop all the environments
```
curl -X POST http://IP:10473/stop_all \
  -H "Content-Type: application/json" \
  -d '{"token": "3eb40e28-8f12-4b95-b9dd-743ea98334a9"}' \
  | jq
```

# Create one environments
```
curl -X POST http://IP:10473/launch \
  -H "Content-Type: application/json" \
  -d '{
    "token": "3eb40e28-8f12-4b95-b9dd-743ea98334a9",
    "image": "scienceboard"
  }'