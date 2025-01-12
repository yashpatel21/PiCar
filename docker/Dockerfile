# Base Python image for ARM compatibility
FROM python:3.9-bookworm

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV NO_AT_BRIDGE=1
ENV GDK_SYNCHRONIZE=1
ENV GDK_BACKEND=x11
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV LG_WD=/tmp/lgpio/

WORKDIR /app

# Add Raspberry Pi repository for hardware support packages
RUN echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E

# Configure Debian repositories
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://security.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    awk '!seen[$0]++' /etc/apt/sources.list > /etc/apt/sources.list.clean && \
    mv /etc/apt/sources.list.clean /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/debian.sources && \
    apt-get update

# Create groups for hardware access
RUN groupadd -f i2c && \
    groupadd -f gpio && \
    groupadd -f spi && \
    usermod -aG i2c,gpio,spi root

# Install dependencies for all system components
RUN apt-get update && apt-get install -y --no-install-recommends \
    # System utilities
    gnupg git nano wget curl sudo usbutils x11-xserver-utils \
    # GUI libraries
    libgl1-mesa-glx libx11-xcb1 libxcb-util1 libqt5gui5 libqt5widgets5 libqt5x11extras5 \
    python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-gobject-2.0 \
    libgirepository1.0-dev libcairo2-dev pkg-config python3-dev \
    gobject-introspection libgtk-3-dev \
    # Hardware support
    udev dfu-util \
    # OpenCV libraries
    mesa-utils libgl1 libgl1-mesa-dri libgl1-mesa-glx libgles2-mesa libegl1-mesa \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    # Camera support
    libcap-dev libatlas-base-dev ffmpeg libopenjp2-7 libcamera-dev \
    libkms++-dev libfmt-dev libdrm-dev \
    # Robot hardware control
    i2c-tools espeak libsdl2-dev libsdl2-mixer-dev portaudio19-dev python3-smbus \
    # GPIO control
    python3-rpi.gpio python3-pigpio pigpio && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Edge TPU support
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y libedgetpu1-std && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup PiCar-X config
RUN mkdir -p /opt/picar-x && \
    touch /opt/picar-x/picar-x.conf && \
    chmod 777 /opt/picar-x/picar-x.conf

# First create and activate the virtual environment
RUN /usr/local/bin/python3 -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir wheel

# Now we can create the sitecustomize.py in the virtual environment
RUN mkdir -p $VIRTUAL_ENV/lib/python3.9/site-packages && \
    echo 'import os\n\
def mock_getlogin():\n\
    return "root"\n\
os.getlogin = mock_getlogin' > $VIRTUAL_ENV/lib/python3.9/site-packages/sitecustomize.py

# Copy project files and install Python packages
COPY . /app
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -r /app/requirements.txt

# Download TPU firmware
RUN wget -O /app/apex_latest_single_ep.bin https://github.com/google-coral/libedgetpu/raw/master/driver/usb/apex_latest_single_ep.bin

# Set script permissions
RUN chmod +x /app/fix_coral_usb.sh /app/entrypoint.sh /app/setup_interfaces.sh

# Setup GTK runtime
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

RUN mkdir -p /tmp/lgpio && chmod 777 /tmp/lgpio

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]