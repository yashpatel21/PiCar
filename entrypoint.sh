#!/bin/bash

# Run the interface setup
/app/setup_interfaces.sh

# Run the Coral USB firmware fix
/app/fix_coral_usb.sh

# Activate venv
source /app/venv/bin/activate

# Execute any additional commands provided to the container
exec "$@"