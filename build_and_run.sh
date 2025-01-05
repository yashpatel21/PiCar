#!/bin/bash

# Define image and container names
IMAGE_NAME="picar:latest"
CONTAINER_NAME="picar-container"

# Build the image
echo "Building the image..."
sudo docker build -t $IMAGE_NAME -f docker/Dockerfile .
# Remove existing container if it exists
echo "Removing existing container (if any)..."
sudo docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Run the container
echo "Running the container..."
sudo docker run --name $CONTAINER_NAME -it \
    --device /dev/bus/usb \
    --device /dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    $IMAGE_NAME

# Clean up dangling images and stopped containers
echo "Cleaning up dangling images and stopped containers..."
sudo docker container prune -f
sudo docker image prune -f