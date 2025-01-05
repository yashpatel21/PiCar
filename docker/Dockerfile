# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies, GPG, and the Coral Edge TPU runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
	gnupg \
	curl \
	usbutils \
	&& curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg && \
	echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
	apt-get update && \
	apt-get install -y libedgetpu1-std

CMD [ "bash" ]