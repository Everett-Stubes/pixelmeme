"""The audience.

A sitcom is a room with an audience in it, and the audience is half the
format. Every exchange here has a button, and the button gets a laugh --
generously, and sometimes for nothing, which is exactly how a real laugh
track behaves.

It is a caption, and only ever a caption. Nothing in this project makes or
receives a sound: no microphone, no speaker, no mixer. The laugh is a strip
that rides over the bottom of the set rather than a beat of its own, so the
scene keeps playing underneath and the laugh can overlap the next line the
way it does on television.
"""

from __future__ import annotations

import math
import random

import pygame

from . import pixelfont as pf
from ..brain import dialogue

# How long a laugh stays on screen, by strength.
_SHORT, _LONG = 1.1, 1.9

# Warm, slightly too bright -- it should read as an overlay from somewhere
# else in the building, not as part of the room.
_INK = (240, 186, 62)
_PAD = (16, 12, 10)


class LaughTrack:
    def __init__(self, rng: random.Random | None = None, width: int = 108):
        self.rng = rng or random.Random()
        # The strip is one line and does not wrap, so at a small base
        # resolution the longer labels simply are not available. Filtering
        # here beats shortening the writing to fit the worst case.
        self.pool = [s for s in dialogue.LAUGHS
                     if pf.text_width(s, 1) <= width - 4] or ["[ LAUGHTER ]"]
        self.t = 0.0
        self.remaining = 0.0
        self.label = ""
        self.strength = 0.0
        self.fired = 0

    def fire(self, strength: float = 1.0) -> None:
        strength = max(0.0, min(1.0, float(strength)))
        big = [s for s in self.pool if "BIG" in s or "APPLAUSE" in s]
        pool = big if (big and strength > 0.85 and self.rng.random() < 0.5) \
            else self.pool
        self.label = self.rng.choice(pool)
        self.strength = strength
        self.remaining = _SHORT + (_LONG - _SHORT) * strength
        self.fired += 1

    @property
    def active(self) -> bool:
        return self.remaining > 0.0

    def update(self, dt: float) -> None:
        self.t += dt
        self.remaining = max(0.0, self.remaining - dt)

    def draw(self, surface: pygame.Surface, width: int, bottom: int) -> None:
        """Strip sits with its baseline at `bottom`, over the set."""
        if not self.active:
            return
        # Flicker on the way out: the audience dies down rather than stopping.
        fade = min(1.0, self.remaining / 0.45)
        if fade < 1.0 and math.sin(self.t * 30.0) < -fade:
            return

        h = pf.line_height(1, 0) + 4
        y = bottom - h
        surface.fill(_PAD, pygame.Rect(0, y, width, h))
        pf.draw_centered(surface, self.label, width // 2, y + 2, 1, _INK, None)
