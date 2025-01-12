#!/bin/bash

# Exit on error
set -e

echo "Starting container initialization..."

# === Display Environment Setup ===
mkdir -p /tmp/runtime-root
chmod 700 /tmp/runtime-root

# === Hardware Interface Setup ===
# Load necessary kernel modules without failing if already loaded
echo "Configuring hardware interfaces..."
modprobe i2c-dev 2>/dev/null || true
modprobe spi-bcm2835 2>/dev/null || true

# Create hardware access groups if they don't exist
for group in i2c gpio spi; do
    getent group $group || groupadd -f $group
done

# Add root to hardware groups
usermod -aG i2c,gpio,spi root

# Configure device permissions with detailed logging
configure_device() {
    local device=$1
    local group=$2
    local perms=$3
    
    if [ -e "$device" ]; then
        echo "Configuring $device..."
        chown root:$group $device
        chmod $perms $device
    else
        echo "Warning: $device not found"
    fi
}

# Configure all hardware interfaces
configure_device "/dev/gpiomem" "gpio" "660"
configure_device "/dev/gpiochip0" "gpio" "660"
configure_device "/dev/i2c-1" "i2c" "660"
configure_device "/dev/spidev0.0" "spi" "660"
configure_device "/dev/spidev0.1" "spi" "660"

# === Robot Setup ===
echo "Setting up PiCar-X configuration..."
mkdir -p /opt/picar-x
touch /opt/picar-x/picar-x.conf
chmod 777 /opt/picar-x/picar-x.conf

# Initialize GPIO system
echo "Initializing GPIO system..."
mkdir -p /tmp/lgpio
chmod 777 /tmp/lgpio
pigpiod 2>/dev/null || echo "Note: pigpiod already running"

echo "Container environment initialization completed successfully"