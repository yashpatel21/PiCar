#!/bin/bash

# Configure display for GUI applications
export DISPLAY=:0
xhost +local:root
xhost +local:docker

# Define container configuration
IMAGE_NAME="picar:latest"
CONTAINER_NAME="picar-container"

# Build the Docker image
echo "Building the image..."
sudo docker build -t $IMAGE_NAME -f docker/Dockerfile .

# Check if a container with the same name exists
if sudo docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "Stopping and removing existing container..."
    sudo docker rm -f $CONTAINER_NAME
fi

# Run the container with hardware access
echo "Running the container..."
sudo docker run --name $CONTAINER_NAME -it \
    --device /dev/bus/usb \
    --device /dev/video0:/dev/video0 \
    --device /dev/vchiq:/dev/vchiq \
    --device /dev/dri:/dev/dri \
    --device /dev/i2c-1:/dev/i2c-1 \
    --device /dev/spidev0.0:/dev/spidev0.0 \
    --device /dev/spidev0.1:/dev/spidev0.1 \
    --device /dev/gpiomem:/dev/gpiomem \
    --device /dev/mem:/dev/mem \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /run/udev:/run/udev:ro \
    -v "$(pwd)/src:/app/src" \
    --network host \
    --privileged \
    --ipc=host \
    $IMAGE_NAME

# Cleanup
echo "Cleaning up dangling images and stopped containers..."
sudo docker container prune -f
sudo docker image prune -f