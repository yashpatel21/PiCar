#!/bin/bash

# Variables
FIRMWARE_FILE="/app/apex_latest_single_ep.bin"
EXPECTED_DEVICE_ID="18d1:9302"
INITIAL_DEVICE_ID="1a6e:089a"

# Function to check USB devices
check_usb_device() {
    lsusb | grep -q "$1"
}

echo "Checking Coral USB device status..."

# Check if the device is already flashed
if check_usb_device "$EXPECTED_DEVICE_ID"; then
    echo "Coral USB device is already flashed with the correct firmware ($EXPECTED_DEVICE_ID). Skipping firmware flashing."
else
    # Check if the device needs flashing
    if check_usb_device "$INITIAL_DEVICE_ID"; then
        echo "Coral USB device requires firmware flashing. Proceeding..."
        # Attempt to flash the firmware
        if dfu-util -D "$FIRMWARE_FILE" -d "$INITIAL_DEVICE_ID" -R; then
            echo "Firmware flashed successfully."
        else
            echo "Failed to flash firmware. Exiting."
            exit 1
        fi
    else
        echo "Coral USB device not detected. Ensure it is connected and try again."
        exit 1
    fi
fi
