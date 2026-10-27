"""Configuration: defaults in code, overridden by config.yaml, overridden by
command-line flags."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "display": {
        # Internal render size, portrait. Everything is drawn at this size and
        # then blown up with nearest-neighbour, which is where the pixels come
        # from. 90x160 is 9:16 and divides 1080x1920 exactly by 12. Go to
        # 72x128 for chunkier pixels at the cost of subtitle room.
        "base_width": 108,
        "base_height": 192,
        "fullscreen": True,
        "fps": 20,
        "hide_cursor": True,
        # "auto" keeps the aspect ratio and letterboxes; "stretch" fills.
        "fit": "auto",
        # Rotate the rendered image by 90/180/270 if the panel itself is
        # landscape and you cannot rotate it in the OS.
        "rotate": 0,
        "scanlines": False,
        "dither": 0.55,
        "palette": "nyc",
        # Dithering for the flat-shaded 3D palette. Must be 0: a flat lit
        # face is one colour, and an ordered dither on one colour alternates
        # half its pixels into the neighbouring palette entry, which comes
        # out as a checkerboard across people's faces. The gradients that
        # dithering exists for are all in the 2D climbing scene, which uses
        # a different palette and keeps the full amount.
        "dither_flat": 0.0,
        "idle_palette": "stone",
        # Re-asserts the X11 no-blanking settings every N seconds, in case
        # something (a desktop session, a screensaver daemon) turns them back
        # on while we are running.
        "keepalive_seconds": 60,
    },
    "stage": {
        # Subtitle reveal rate, characters per second.
        "type_speed": 26.0,
    },
    "cutaway": {
        # The climb is a scheduled segment, like a station break, not a
        # fallback for anything. Every so often the broadcast cuts to it for
        # a fixed stretch and then cuts back.
        #
        # A break has to be shorter than the thing it interrupts. At five
        # minutes against a two-and-a-half-minute minimum the wall took 62%
        # of the running time, and since the opener is only twenty-five
        # seconds the first real break was twelve times longer than the one
        # the viewer had already been taught to expect -- which does not
        # read as a station break, it reads as the show never coming back.
        # Cut to the climb after about this many scenes.
        "every_scenes": 5.0,
        # The climb shown at startup, before the sitcom has run at all.
        "opening_seconds": 25.0,
        "duration_minutes": 1.0,
        # Guard rails in seconds, so a run of very short scenes cannot cut
        # away constantly and a very long one still eventually breaks. The
        # floor has to stay below what every_scenes takes to play, or it is
        # the only thing that ever decides and the cut lands mid-scene.
        "min_live_seconds": 150.0,
        "max_live_seconds": 1500.0,
    },
    "llm": {
        # Text generation on the device: Llama 3.2 1B Instruct, GGUF, through
        # llama-cpp-python. This is the writer; the lexicon generator in
        # dialogue.py is the floor under it and covers every frame the model
        # is not ready for. On by default, and harmless when the file is
        # absent -- startup says so once and the show runs off the lexicon.
        "enabled": True,
        # A list, tried in order: the first file that is present AND fits the
        # memory actually free is the one that runs. One config for every
        # board -- the Pi only ever has the 0.5B on disk, the desktop has the
        # 1B, and neither needs to know about the other.
        "model_path": [
            "assets/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "assets/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        ],
        # Prompt plus four few-shot turns plus the answer. 768 is enough and
        # the KV cache is the part that costs memory on a small board.
        "n_ctx": 768,
        # 0 = one per core, less one kept free for the render loop.
        "threads": 0,
        "max_tokens": 170,            # six short turns and the stop
        # Measured on this model, not guessed: 0.9 produced "THEY ARE
        # DOUBTFIRED IN THEIR OWN SERVICES" and 1.05 rambled, while 0.75
        # stayed on the subject and still gave 20 unique lines out of 20
        # -- so the earlier worry that a low temperature would make it
        # recite the few-shot examples back was simply wrong.
        "temperature": 0.75,
        "top_p": 0.95,
        "repeat_penalty": 1.15,
        # Six, not four. The frame rate is costed by how much of the
        # time the model is generating at all, not by how long any one
        # call takes, so a longer exchange is most of a free lunch: the
        # same share of the lines for a third fewer calls, and one
        # prompt evaluation instead of two.
        "max_lines_per_call": 6,
        # How often an exchange is the weird one instead of the small one.
        "existential_chance": 0.18,
        # The share of calls spent on the two bits the lexicon used to own.
        # Both are cheap to over-supply: the writer already throws away more
        # than half of what it writes, because the show has nowhere near
        # enough airtime for it, so spending a fifth of that surplus on
        # stand-up costs nothing anybody can see.
        "standup_chance": 0.20,
        # A slip offered to the model instead of inserted after it. Higher
        # than the director's old 0.12 because a call is not an exchange:
        # most of what is written is never used.
        "slip_chance": 0.20,
        # Floor between generations, not a target. At 15 the model wrote
        # only a quarter of the lines on a desktop where a generation
        # takes five seconds -- the floor, not the model, was what the
        # lexicon was covering for. On a slower board the model is still
        # the limit and the floor never binds.
        "min_interval_seconds": 6.0,
        # Spare memory required on top of the model file itself: context,
        # compute buffers, the renderer and the OS. A model only loads if
        # the free memory covers its own size plus this, so the check
        # follows whatever model is on disk instead of being a number
        # somebody has to maintain.
        #
        # Measured resident, app included: the 808MB 1B at Q4 reaches
        # 958MB (150MB over the file) and the 398MB 0.5B reaches 526MB
        # (129MB over). 220 is deliberately above both -- this box runs
        # unattended for weeks, and the failure mode of being too tight
        # is the kernel dropping page cache and the SD card thrashing,
        # which costs the frame rate rather than raising an error.
        "ram_headroom_mb": 220.0,
        # Keep the few-shot examples fixed so the prompt prefix never
        # changes and llama.cpp can reuse the KV cache. On the 1B this took
        # a warm call from 4.9s to 2.7s; on a Pi, where evaluating the
        # prompt is most of the cost, it matters a great deal more.
        "stable_prompt": True,
    },
    "cloud": {
        # Off by default, and a separate thing from `llm` above: that one
        # runs on the device, this one is an API call. If both are on, this
        # wins -- it is the more deliberate switch of the two.
        "enabled": False,
        "model": "claude-opus-5",
        "effort": "low",
        "timeout_seconds": 12.0,
        # Floor on how often we are willing to call the API.
        "min_interval_seconds": 20.0,
        "max_lines_per_call": 6,
    },
    "idle": {
        # Baked pixel video (see tools/prepare_video.py). If it is missing we
        # fall back to a generated climbing scene, so the app still runs.
        "video": "assets/idle/honnold.pxv.npz",
    },
    "debug": {
        "overlay": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self._data[section]

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve_path(self, path: str | os.PathLike[str] | None) -> Path | None:
        """Project-root-relative paths, so the app works from any cwd."""
        if not path:
            return None
        p = Path(path)
        return p if p.is_absolute() else (ROOT / p)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def load(config_file: str | os.PathLike[str] | None = None) -> Config:
    data = copy.deepcopy(DEFAULTS)
    candidate = Path(config_file) if config_file else (ROOT / "config.yaml")
    if candidate.exists():
        try:
            import yaml

            with candidate.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            data = _deep_merge(data, loaded)
        except Exception as exc:  # pragma: no cover - config is best-effort
            print(f"[config] ignoring {candidate}: {exc}")
    return Config(data)
