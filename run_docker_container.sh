#!/bin/bash

# Configure display for GUI applications
export DISPLAY=:0
xhost +local:root
xhost +local:docker

# Define container configuration
IMAGE_NAME="picar:latest"
CONTAINER_NAME="picar-container"

# Check if the container exists
if ! sudo docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "Error: Container '${CONTAINER_NAME}' does not exist. Please run build_and_run.sh first."
    exit 1
fi

# Check if the container is already running
if sudo docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "Container '${CONTAINER_NAME}' is already running."
else
    echo "Starting the container..."
    sudo docker start -i $CONTAINER_NAME
fi
