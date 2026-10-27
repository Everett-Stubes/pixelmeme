"""The window, the upscale, and the part that stops the screen going to sleep.

Everything is drawn into a small portrait surface and blown up with
nearest-neighbour at the very end. Drawing small is also what makes a Pi 3
able to keep up: the expensive operations all run on ~32k pixels.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import numpy as np
import pygame

from . import effects
from .palette import Quantizer

# Commands that disable blanking, DPMS and the screensaver. Each is tried
# independently; on a Pi running KMSDRM without X none of them exist, and
# that is fine -- there is no X screensaver to disable.
_NOBLANK_COMMANDS = (
    ("xset", "s", "off"),
    ("xset", "s", "noblank"),
    ("xset", "-dpms"),
    ("setterm", "--blank", "0"),
    ("setterm", "--powerdown", "0"),
)


# Windows equivalent of xset -dpms. The flags persist for this thread until
# they are cleared, which is why shutdown() puts them back.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def _windows_keep_awake(awake: bool) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        # argtypes set explicitly: ES_CONTINUOUS is 0x80000000, which is
        # larger than a signed int, and ctypes' default conversion for an
        # unprototyped call is not something to rely on.
        fn = ctypes.windll.kernel32.SetThreadExecutionState
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_uint32

        state = _ES_CONTINUOUS
        if awake:
            state |= _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
        fn(state)
        return True
    except Exception:
        return False


def disable_screen_blanking(verbose: bool = False) -> list[str]:
    """Best-effort. Returns the names of whatever actually worked."""
    if os.name == "nt":
        return ["SetThreadExecutionState"] if _windows_keep_awake(True) else []
    ran = []
    for cmd in _NOBLANK_COMMANDS:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(
                cmd,
                check=False,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            ran.append(" ".join(cmd))
        except Exception as exc:
            if verbose:
                print(f"[display] {' '.join(cmd)} failed: {exc}")
    return ran


# Both of this show's looks are flat: the sitcom is flat-shaded polygons and
# the climb is cel art off a fixed ramp. Neither has a gradient anywhere in
# it, so neither has anything a dither could smooth -- all an ordered dither
# can do to a region that is meant to be one solid colour is alternate it
# between the two palette entries it falls between, which is a grid of dots
# across a whole face or a whole bed of rock.
#
# `display.dither` therefore only bites if you select a palette outside this
# set, where the art may well be gradients.
_FLAT_PALETTES = {"nyc", "stone"}


def _dither_for(name: str, display_cfg) -> float:
    if name in _FLAT_PALETTES:
        return float(display_cfg.get("dither_flat", 0.08))
    return float(display_cfg["dither"])


class Display:
    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg["display"]
        self.base_w = int(d["base_width"])
        self.base_h = int(d["base_height"])
        self.fps = int(d["fps"])
        self.rotate = int(d["rotate"]) % 360
        self.fit = d["fit"]
        self.want_scanlines = bool(d["scanlines"])
        self._keepalive_period = float(d["keepalive_seconds"])
        self._last_keepalive = 0.0

        # SDL picks a driver before set_mode; on a headless Pi booted to a
        # console, KMSDRM is the one that works.
        if os.name == "posix" and not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")
        # Never let SDL idle the screen on our behalf.
        os.environ.setdefault("SDL_VIDEO_ALLOW_SCREENSAVER", "0")

        pygame.display.init()
        pygame.font.quit()  # we use our own bitmap font; no TTF needed

        info = pygame.display.Info()
        flags = pygame.FULLSCREEN if d["fullscreen"] else 0
        size = (0, 0) if d["fullscreen"] else self._windowed_size(info)
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("pixelmeme")
        self.screen_w, self.screen_h = self.screen.get_size()

        if d["hide_cursor"]:
            pygame.mouse.set_visible(False)
        try:
            pygame.display.set_allow_screensaver(False)
        except Exception:
            pass

        # The surface every scene draws onto.
        self.base = pygame.Surface((self.base_w, self.base_h))
        self._quantized = pygame.Surface((self.base_w, self.base_h))

        self.quantizers = {
            name: Quantizer(name, _dither_for(name, d))
            for name in {d["palette"], d["idle_palette"]}
        }
        self.quant = self.quantizers[d["palette"]]

        self._dest_rect = self._compute_dest()
        self._scanlines = (
            effects.make_scanline_overlay(
                (self.screen_w, self.screen_h), gap=max(2, self.scale_y * 2), alpha=60
            )
            if self.want_scanlines
            else None
        )

        self.clock = pygame.time.Clock()
        disable_screen_blanking()
        self._last_keepalive = time.monotonic()

    # -- geometry ------------------------------------------------------------

    def _windowed_size(self, info) -> tuple[int, int]:
        scale = max(1, min(info.current_w // self.base_w, info.current_h // self.base_h))
        scale = min(scale, 4)
        return (self.base_w * scale, self.base_h * scale)

    @property
    def scale_y(self) -> int:
        return max(1, self._dest_rect.height // self.base_h)

    def _rotated_size(self) -> tuple[int, int]:
        if self.rotate in (90, 270):
            return (self.base_h, self.base_w)
        return (self.base_w, self.base_h)

    def _compute_dest(self) -> pygame.Rect:
        src_w, src_h = self._rotated_size()
        if self.fit == "stretch":
            return pygame.Rect(0, 0, self.screen_w, self.screen_h)

        if self.fit == "fill":
            # Largest size that still fits, fractional scale allowed. Uses the
            # whole screen height at the cost of slightly uneven pixels --
            # worth it on a panel whose height is not a multiple of the base.
            ratio = min(self.screen_w / src_w, self.screen_h / src_h)
        else:
            # "auto": prefer a whole-number scale factor, because fractional
            # nearest-neighbour makes some pixels wider than others and you
            # can see it. Falls back to fractional on a screen too small for
            # even 1x.
            ratio = min(self.screen_w // src_w, self.screen_h // src_h)
            if ratio < 1:
                ratio = min(self.screen_w / src_w, self.screen_h / src_h)

        w, h = int(src_w * ratio), int(src_h * ratio)
        return pygame.Rect(
            (self.screen_w - w) // 2, (self.screen_h - h) // 2, w, h
        )

    # -- per-frame -----------------------------------------------------------

    def use_palette(self, name: str) -> None:
        if name not in self.quantizers:
            self.quantizers[name] = Quantizer(
                name, _dither_for(name, self.cfg["display"])
            )
        self.quant = self.quantizers[name]

    def present(self) -> None:
        """Quantise, rotate, upscale, show."""
        src = pygame.surfarray.pixels3d(self.base)
        quantized = self.quant.apply(np.asarray(src))
        del src
        pygame.surfarray.blit_array(self._quantized, quantized)

        frame = self._quantized
        if self.rotate:
            frame = pygame.transform.rotate(frame, self.rotate)

        # transform.scale is nearest-neighbour. smoothscale is the one that
        # would blur the pixels away, so it is never used here.
        if self._dest_rect.size != frame.get_size():
            frame = pygame.transform.scale(frame, self._dest_rect.size)

        if self._dest_rect.size != (self.screen_w, self.screen_h):
            self.screen.fill((0, 0, 0))
        self.screen.blit(frame, self._dest_rect.topleft)
        if self._scanlines is not None:
            self.screen.blit(self._scanlines, (0, 0))
        pygame.display.flip()

    def tick(self) -> float:
        """Advance the frame clock and periodically re-assert no-blanking.

        Returns delta time in seconds.
        """
        dt = self.clock.tick(self.fps) / 1000.0
        now = time.monotonic()
        if (
            self._keepalive_period > 0
            and now - self._last_keepalive >= self._keepalive_period
        ):
            disable_screen_blanking()
            self._last_keepalive = now
        return dt

    def toggle_fullscreen(self) -> None:
        pygame.display.toggle_fullscreen()
        self.screen_w, self.screen_h = self.screen.get_size()
        self._dest_rect = self._compute_dest()
        if self.want_scanlines:
            self._scanlines = effects.make_scanline_overlay(
                (self.screen_w, self.screen_h), gap=max(2, self.scale_y * 2), alpha=60
            )

    def shutdown(self) -> None:
        # Hand normal power management back; on a test PC you do not want to
        # have silently disabled the user's screen timeout.
        _windows_keep_awake(False)
        pygame.display.quit()
