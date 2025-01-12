# Base Python image
FROM python:3.9-bookworm

# Core environment configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/venv
ENV PATH="/usr/local/bin:${VIRTUAL_ENV}/bin:$PATH"

# GUI configuration
ENV NO_AT_BRIDGE=1 \
    GDK_SYNCHRONIZE=1 \
    GDK_BACKEND=x11 \
    XDG_RUNTIME_DIR=/tmp/runtime-root

# GPIO configuration
ENV LG_WD=/tmp/lgpio/

WORKDIR /app

# Configure package repositories
RUN echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E && \
    echo "deb [arch=arm64] https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb http://deb.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://security.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    apt-get update || true && \
    apt-get install -y --fix-missing apt-transport-https && \
    apt-get update

# System package installation
RUN apt-get install -y --no-install-recommends \
    # Basic system utilities
    gnupg git nano wget curl sudo usbutils x11-xserver-utils \
    # GUI and display dependencies
    libgl1-mesa-glx libx11-xcb1 libxcb-util1 libqt5gui5 libqt5widgets5 libqt5x11extras5 \
    python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-gobject-2.0 \
    libgirepository1.0-dev libcairo2-dev pkg-config python3-dev \
    gobject-introspection libgtk-3-dev \
    # Hardware interface support
    udev dfu-util \
    # OpenCV and image processing dependencies
    mesa-utils libgl1 libgl1-mesa-dri libgl1-mesa-glx libgles2-mesa libegl1-mesa \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    # Camera system dependencies
    libcap-dev libatlas-base-dev ffmpeg libopenjp2-7 libcamera-dev \
    libkms++-dev libfmt-dev libdrm-dev \
    # Robot hardware interfaces
    i2c-tools libsdl2-dev libsdl2-mixer-dev portaudio19-dev python3-smbus \
    # GPIO and hardware control
    python3-rpi.gpio python3-pigpio pigpio \
    # Edge TPU runtime
    libedgetpu1-std && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create and configure Python virtual environment
RUN /usr/local/bin/python3 -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/python -m ensurepip --upgrade && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Setup getlogin override for PiCar-X compatibility
RUN mkdir -p $VIRTUAL_ENV/lib/python3.9/site-packages && \
    echo 'import os\ndef mock_getlogin():\n    return "root"\nos.getlogin = mock_getlogin' > \
    $VIRTUAL_ENV/lib/python3.9/site-packages/sitecustomize.py

# Copy codebase
COPY . /app

# Install Python packages with improved error handling
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -v -r /app/requirements.txt || \
    ($VIRTUAL_ENV/bin/pip install --no-cache-dir -v --no-deps -r /app/requirements.txt && \
     $VIRTUAL_ENV/bin/pip check)

# Set script permissions
RUN chmod +x /app/fix_coral_usb.sh /app/entrypoint.sh /app/init_container.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]