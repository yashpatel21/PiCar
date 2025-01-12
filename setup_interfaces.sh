#!/bin/bash

# Set up GPIO permissions
if [ -e /dev/gpiomem ]; then
    chown root:gpio /dev/gpiomem
    chmod 660 /dev/gpiomem
fi

if [ -e /dev/gpiochip0 ]; then
    chown root:gpio /dev/gpiochip0
    chmod 660 /dev/gpiochip0
fi

# Set up I2C permissions
if [ -e /dev/i2c-1 ]; then
    chown root:i2c /dev/i2c-1
    chmod 660 /dev/i2c-1
fi

# Set up SPI permissions
if [ -e /dev/spidev0.0 ]; then
    chown root:spi /dev/spidev0.0
    chmod 660 /dev/spidev0.0
fi

if [ -e /dev/spidev0.1 ]; then
    chown root:spi /dev/spidev0.1
    chmod 660 /dev/spidev0.1
fi

# Create and configure PiCar-X directory if it doesn't exist
if [ ! -d /opt/picar-x ]; then
    mkdir -p /opt/picar-x
    touch /opt/picar-x/picar-x.conf
    chmod 777 /opt/picar-x/picar-x.conf
fi

# Start the pigpio daemon for GPIO access
pigpiod 2>/dev/null || echo "Note: pigpiod already running"

echo "Interface setup completed - all device permissions and services configured"