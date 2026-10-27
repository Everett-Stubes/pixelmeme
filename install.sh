#!/usr/bin/env bash
# One-shot setup for a Raspberry Pi 3 running Raspberry Pi OS (Bookworm).
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> system packages"
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev python3-pip \
  libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0 \
  libatlas-base-dev

echo "==> python environment"
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# The writer: a language model, on this board, with no network at runtime.
# There is no prebuilt aarch64 wheel, so llama-cpp-python compiles here --
# ten minutes or so on a Pi 3 -- and nothing in this block is fatal. If any
# of it fails the broadcast runs off the lexicon generator exactly as it did
# before, and says so once at startup.
#
# Which model depends on the board, and `--auto` works it out from the memory
# that is actually free rather than from the board name:
#
#   1GB Pi 3   -> Qwen2.5-0.5B Q4, 398MB
#   2GB+ Pi 4  -> Llama 3.2 1B Q4, 808MB
#
# A 1B at Q4 needs about 1GB resident, which a Pi 3 does not have; the app
# checks the file's own size against free memory at startup and refuses
# rather than letting the board swap to the SD card.
echo "==> on-device writer (llama-cpp-python, compiles from source)"
if pip install "llama-cpp-python>=0.3" && python tools/probe_llm.py; then
  echo "==> the model"
  python tools/get_model.py --auto ||     echo "    failed; run 'python tools/get_model.py --auto' when convenient"
else
  # Installing is not the same as working: a wheel can import and then die
  # inside llama_backend_init. probe_llm.py is what tells the difference.
  echo "    no working build here; the lexicon generator writes the show"
fi

echo
echo "Done. Next:"
echo "  ./run.sh --windowed --debug      # check it works"
echo
echo "Optional: bake the idle clip (on a desktop, not the Pi):"
echo "  pip install opencv-python"
echo "  python tools/prepare_video.py your-climbing-clip.mp4 --seconds 45"
echo
echo "To run it on boot:"
echo "  sudo cp systemd/pixelmeme.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload && sudo systemctl enable --now pixelmeme"
