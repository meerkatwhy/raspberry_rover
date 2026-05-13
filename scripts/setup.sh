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

# Install pip and python packages
apt install python3-pip -y
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

