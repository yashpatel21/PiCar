# Use the official Python 3.9 bookworm image as the base
FROM python:3.9-bookworm

# Set environment variables for non-interactive installations and virtual environment
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Update sources.list and remove redundant entries
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://security.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    awk '!seen[$0]++' /etc/apt/sources.list > /etc/apt/sources.list.clean && \
    mv /etc/apt/sources.list.clean /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/debian.sources && \
    apt-get update

# Install system utilities and libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # General utilities
    gnupg git nano wget curl sudo usbutils \
    # Coral Edge TPU dependencies
    udev dfu-util libx11-xcb1 libxcb-util1 \
    # OpenCV dependencies
    libxrender1 libqt5gui5 libqt5widgets5 libqt5x11extras5 \
    mesa-utils libgl1 libgl1-mesa-dri libgl1-mesa-glx libgles2-mesa libegl1-mesa \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Coral Edge TPU runtime
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y libedgetpu1-std && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . /app

# Set up a Python virtual environment and install dependencies
RUN /usr/local/bin/python3 -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir -r /app/requirements.txt

# Download Coral USB firmware
RUN wget -O /app/apex_latest_single_ep.bin https://github.com/google-coral/libedgetpu/raw/master/driver/usb/apex_latest_single_ep.bin

# Ensure scripts are executable
RUN chmod +x /app/fix_coral_usb.sh /app/entrypoint.sh

# Set the entrypoint for initialization tasks
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["bash"]
