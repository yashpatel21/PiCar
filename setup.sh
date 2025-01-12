#!/bin/bash

# Helper functions for status messages and error checking
print_status() {
    echo "==> $1"
}

check_error() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Helper function to safely add kernel modules without duplicates
add_module_if_missing() {
    local module=$1
    if ! grep -q "^$module\$" /etc/modules; then
        sh -c "echo '$module' >> /etc/modules"
    fi
}

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
}

print_status "Beginning Raspberry Pi setup for PiCar project..."

# System updates and essential packages
print_status "Updating system packages..."
apt-get update && apt-get upgrade -y
check_error "Failed to update system packages"

print_status "Installing essential utilities..."
apt-get install -y git curl wget ca-certificates
check_error "Failed to install essential utilities"

# Hardware interface setup
print_status "Enabling I2C interface..."
raspi-config nonint do_i2c 0
check_error "Failed to enable I2C"
add_module_if_missing "i2c-dev"

print_status "Enabling SPI interface..."
raspi-config nonint do_spi 0
check_error "Failed to enable SPI"
add_module_if_missing "spi-bcm2835"

# Docker installation preparation
print_status "Removing any conflicting Docker packages..."
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
    apt-get remove -y $pkg 2>/dev/null || true
done

# Docker repository setup
print_status "Setting up Docker repository..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository to system
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker and related packages
print_status "Updating package list with Docker repository..."
apt-get update
check_error "Failed to update package list"

print_status "Installing Docker packages..."
apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
check_error "Failed to install Docker packages"

# Docker user setup
print_status "Adding user to docker group..."
usermod -aG docker $SUDO_USER
check_error "Failed to add user to docker group"

# Docker service configuration
print_status "Enabling Docker service..."
systemctl enable docker
systemctl start docker
check_error "Failed to enable Docker service"

# Load kernel modules
print_status "Loading kernel modules..."
modprobe i2c-dev
modprobe spi-bcm2835

# X11 configuration for GUI applications
print_status "Setting up X11 permissions..."
if ! grep -q "xhost +local:docker" "/home/$SUDO_USER/.bashrc"; then
    echo 'xhost +local:docker' >> "/home/$SUDO_USER/.bashrc"
fi

# Clean up duplicate module entries if any exist
print_status "Cleaning up module configuration..."
sort -u /etc/modules > /etc/modules.tmp && mv /etc/modules.tmp /etc/modules

# Display completion message and next steps
print_status "Installation complete!"

# Prompt for reboot
read -p "Would you like to reboot now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Rebooting system..."
    reboot
fi