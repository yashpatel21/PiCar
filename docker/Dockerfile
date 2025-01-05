# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies, Coral Edge TPU runtime, and configure udev rules
RUN apt-get update && apt-get install -y --no-install-recommends \
	gnupg \
	curl \
	usbutils \
	udev \
	sudo \
	&& curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg && \
	echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
	apt-get update && \
	apt-get install -y libedgetpu1-std \
	edgetpu-examples \
	&& \
	echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTR{idProduct}=="9302", MODE="0666"' > /etc/udev/rules.d/CORALUSB

# Default command
CMD ["bash"]
