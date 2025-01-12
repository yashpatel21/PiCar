#!/bin/bash

# Configuration
FIRMWARE_URL="https://github.com/google-coral/libedgetpu/raw/master/driver/usb/apex_latest_single_ep.bin"
FIRMWARE_FILE="/app/apex_latest_single_ep.bin"
EXPECTED_DEVICE_ID="18d1:9302"
INITIAL_DEVICE_ID="1a6e:089a"

echo "Starting Edge TPU initialization..."

# Download firmware if needed
if [ ! -f "$FIRMWARE_FILE" ]; then
    echo "Downloading Edge TPU firmware..."
    wget -O "$FIRMWARE_FILE" "$FIRMWARE_URL" || {
        echo "Error: Failed to download firmware"
        exit 1
    }
fi

# Function to check USB devices
check_usb_device() {
    lsusb | grep -q "$1"
}

echo "Checking Coral USB device status..."

if check_usb_device "$EXPECTED_DEVICE_ID"; then
    echo "Edge TPU device already configured correctly"
elif check_usb_device "$INITIAL_DEVICE_ID"; then
    echo "Flashing Edge TPU firmware..."
    if dfu-util -D "$FIRMWARE_FILE" -d "$INITIAL_DEVICE_ID" -R; then
        echo "Firmware flashed successfully"
    else
        echo "Error: Firmware flashing failed"
        exit 1
    fi
else
    echo "Warning: No Edge TPU device detected"
    exit 0
fi