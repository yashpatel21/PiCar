#!/bin/bash

# Run the Coral USB firmware fix
/app/fix_coral_usb.sh

# Add other initialization commands here if needed
echo "Running additional initialization tasks..."

# Execute any additional commands provided to the container
exec "$@"
