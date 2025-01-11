# Use the official Python 3.9 bookworm image as the base
FROM python:3.9-bookworm

# Set environment variables for non-interactive installations and virtual environment
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set environment variables for GTK
ENV NO_AT_BRIDGE=1
ENV GDK_SYNCHRONIZE=1
ENV GDK_BACKEND=x11
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# Set the working directory in the container
WORKDIR /app

# Add the Raspberry Pi repository
RUN echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E

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
    gnupg git nano wget curl sudo usbutils x11-xserver-utils \
    # GUI dependencies
    libgl1-mesa-glx libx11-xcb1 libxcb-util1 libqt5gui5 libqt5widgets5 libqt5x11extras5 \
    # GTK dependencies
    python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-gobject-2.0 \
    libgirepository1.0-dev libcairo2-dev pkg-config python3-dev \
    gobject-introspection libgtk-3-dev \
    # Coral Edge TPU dependencies
    udev dfu-util \
    # OpenCV dependencies
    mesa-utils libgl1 libgl1-mesa-dri libgl1-mesa-glx libgles2-mesa libegl1-mesa \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    # Picamera2 and libcamera dependencies
    libcap-dev libatlas-base-dev ffmpeg libopenjp2-7 libcamera-dev \
    libkms++-dev libfmt-dev libdrm-dev && \
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
    $VIRTUAL_ENV/bin/pip install --no-cache-dir wheel && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir -r /app/requirements.txt

# Download Coral USB firmware
RUN wget -O /app/apex_latest_single_ep.bin https://github.com/google-coral/libedgetpu/raw/master/driver/usb/apex_latest_single_ep.bin

# Ensure scripts are executable
RUN chmod +x /app/fix_coral_usb.sh /app/entrypoint.sh

# Create runtime directory for GTK
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# Set the entrypoint for initialization tasks
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["bash"]