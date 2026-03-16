FROM ghcr.io/selkies-project/nvidia-glx-desktop:22.04

ARG NVIDIA_DRIVER_VERSION=575.64.03
ARG NVIDIA_DRIVER_ARCH=x86_64

ENV DEBIAN_FRONTEND=noninteractive
ENV VISIONARY_PORT=3000
ENV CHROME_FLAGS="--no-first-run --no-default-browser-check --disable-dev-shm-usage --ignore-gpu-blocklist --enable-features=Vulkan,UnsafeWebGPU --use-vulkan --enable-unsafe-webgpu --ozone-platform=x11"
ENV BUN_INSTALL=/opt/bun
ENV NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION}

USER root

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    git \
    gnupg \
    python3 \
    python3-pip \
    libnss3 \
    libgbm1 \
    libasound2 \
    libxss1 \
    libgtk-3-0 \
    libxshmfence1 \
    libglu1-mesa \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /etc/apt/keyrings/google-chrome.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o "/tmp/NVIDIA-Linux-${NVIDIA_DRIVER_ARCH}-${NVIDIA_DRIVER_VERSION}.run" \
      "https://international.download.nvidia.com/XFree86/Linux-${NVIDIA_DRIVER_ARCH}/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-${NVIDIA_DRIVER_ARCH}-${NVIDIA_DRIVER_VERSION}.run" || \
    curl -fsSL -o "/tmp/NVIDIA-Linux-${NVIDIA_DRIVER_ARCH}-${NVIDIA_DRIVER_VERSION}.run" \
      "https://international.download.nvidia.com/tesla/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-${NVIDIA_DRIVER_ARCH}-${NVIDIA_DRIVER_VERSION}.run"

RUN curl -fsSL https://bun.sh/install | bash

RUN install -m 0755 /opt/bun/bin/bun /usr/local/bin/bun

ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /workspace

COPY start.sh /opt/visionary/autostart-visionary.sh
COPY visionary.desktop /etc/xdg/autostart/visionary.desktop
RUN chmod +x /opt/visionary/autostart-visionary.sh && \
    sed -i 's|Exec=.*|Exec=/opt/visionary/autostart-visionary.sh|' /etc/xdg/autostart/visionary.desktop

EXPOSE 8080 3000
