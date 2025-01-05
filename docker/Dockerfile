# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # General utilities
    gnupg git nano wget curl sudo \
    # USB tools and udev for Coral Edge TPU
    usbutils udev dfu-util \
    # OpenCV and GUI dependencies
    libopencv-dev python3-opencv \
    libx11-xcb1 libxcb-util1 libxrender1 \
    libqt5gui5 libqt5widgets5 libqt5x11extras5 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Coral Edge TPU runtime
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y libedgetpu1-std && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl && \
    pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_aarch64.whl && \
    pip uninstall -y numpy && \
    pip install "numpy<2"

# Copy all project files into the container
COPY . /app

# Download the Coral USB firmware file
RUN wget -O /app/apex_latest_single_ep.bin https://github.com/google-coral/libedgetpu/raw/master/driver/usb/apex_latest_single_ep.bin

# Ensure necessary scripts are executable
RUN chmod +x /app/fix_coral_usb.sh /app/entrypoint.sh

# Set the entrypoint for initialization tasks
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["bash"]
