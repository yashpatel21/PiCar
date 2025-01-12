#!/bin/bash

# Exit on any error
set -e

# Initialize container environment
echo "Initializing container environment..."
if ! /app/init_container.sh; then
    echo "Error: Container initialization failed"
    exit 1
fi

# Initialize Edge TPU
echo "Initializing Edge TPU..."
/app/fix_coral_usb.sh || echo "Warning: Edge TPU initialization failed but continuing..."

# Activate Python virtual environment
echo "Activating Python virtual environment..."
source /app/venv/bin/activate

# Execute container command
echo "Container setup complete..."
exec "$@"