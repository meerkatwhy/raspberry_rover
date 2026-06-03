if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run this script with sudo." >&2
  exit 1
fi

set -e

# Picamera2 depencencies
apt install -y libcap-dev
apt install -y python3-kms++

# OpenCV dependencies
apt install -y libgl1

# PyAudio dependencies
apt install -y portaudio19-dev

# Install uv and python packages
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv --system-site-packages
uv sync
