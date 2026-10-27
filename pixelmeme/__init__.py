"""pixelmeme -- a vertical, always-on pixel meme machine."""

import os

# pygame prints a community banner to stdout on import. Set before any
# submodule imports it, which is why this lives in the package __init__:
# on the Pi every line here ends up in the systemd journal.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

__version__ = "1.0.0"
