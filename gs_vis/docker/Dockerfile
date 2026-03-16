FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    git \
    gnupg \
    unzip \
    x11vnc \
    xvfb \
    fluxbox \
    dbus-x11 \
    xterm \
    x11-xserver-utils \
    xserver-xorg-core \
    xserver-xorg-video-dummy \
    novnc \
    websockify \
    python3 \
    python3-pip \
    libnss3 \
    libgbm1 \
    libasound2 \
    libxss1 \
    libgtk-3-0 \
    libxshmfence1 \
    libglu1-mesa \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /etc/apt/keyrings/google-chrome.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bun.sh/install | bash

ENV PATH="/root/.bun/bin:${PATH}"
ENV DISPLAY=:1
ENV VNC_PORT=5901
ENV NOVNC_PORT=6080
ENV VISIONARY_PORT=3000
ENV CHROME_FLAGS="--no-first-run --no-default-browser-check --disable-dev-shm-usage --disable-software-rasterizer --ignore-gpu-blocklist --enable-features=Vulkan,UnsafeWebGPU --use-vulkan --enable-unsafe-webgpu --ozone-platform=x11"

WORKDIR /workspace

COPY xorg-dummy.conf /etc/X11/xorg.conf
COPY start.sh /opt/visionary/start.sh
RUN chmod +x /opt/visionary/start.sh

EXPOSE 3000 5901 6080

CMD ["/opt/visionary/start.sh"]
