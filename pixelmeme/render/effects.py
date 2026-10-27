"""Frame-level abuse of the image: shake, zoom, glitch, deep-fry.

All of these run on the small base surface (a few tens of thousands of
pixels), which is why they are affordable on a Pi 3.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame


def shake(t: float, amount: float, seed: float = 0.0) -> tuple[int, int]:
    """Deterministic jitter offset in whole pixels."""
    if amount <= 0:
        return (0, 0)
    dx = math.sin(t * 37.0 + seed) * math.cos(t * 13.0 + seed * 2.0)
    dy = math.sin(t * 29.0 + seed * 3.0) * math.cos(t * 17.0 + seed)
    return (int(round(dx * amount)), int(round(dy * amount)))


def pulse(t: float, speed: float = 2.0, lo: float = 0.0, hi: float = 1.0) -> float:
    """A 0..1 sine remapped to [lo, hi]."""
    return lo + (hi - lo) * (0.5 + 0.5 * math.sin(t * speed * math.tau))


def zoom(surface: pygame.Surface, factor: float) -> pygame.Surface:
    """Scale about the centre and crop back to the original size.

    factor > 1 zooms in. Nearest-neighbour, so zooming makes the pixels
    bigger rather than blurrier.
    """
    if abs(factor - 1.0) < 1e-3:
        return surface
    w, h = surface.get_size()
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    scaled = pygame.transform.scale(surface, (nw, nh))
    out = pygame.Surface((w, h))
    out.blit(scaled, ((w - nw) // 2, (h - nh) // 2))
    return out


def crunch(surface: pygame.Surface, divisor: int) -> pygame.Surface:
    """Throw away resolution and put it back. The deep-fried look.

    divisor=1 is a no-op; 3 or 4 gets genuinely illegible, which is the joke.
    """
    if divisor <= 1:
        return surface
    w, h = surface.get_size()
    small = pygame.transform.scale(surface, (max(1, w // divisor), max(1, h // divisor)))
    return pygame.transform.scale(small, (w, h))


def glitch_rows(
    surface: pygame.Surface,
    intensity: float,
    rng: random.Random | None = None,
    bands: int = 5,
) -> None:
    """Shift horizontal bands sideways, in place."""
    if intensity <= 0:
        return
    rng = rng or random
    w, h = surface.get_size()
    max_shift = max(1, int(w * 0.25 * intensity))
    for _ in range(bands):
        band_h = rng.randint(1, max(2, h // 24))
        y = rng.randint(0, max(0, h - band_h))
        shift = rng.randint(-max_shift, max_shift)
        if shift == 0:
            continue
        strip = surface.subsurface(pygame.Rect(0, y, w, band_h)).copy()
        surface.fill((0, 0, 0), pygame.Rect(0, y, w, band_h))
        surface.blit(strip, (shift, y))
        # Wrap the band around so there is no black gap.
        surface.blit(strip, (shift - w if shift > 0 else shift + w, y))


def chromatic_shift(surface: pygame.Surface, px: int) -> None:
    """Pull the red and blue channels apart, in place."""
    if px == 0:
        return
    arr = pygame.surfarray.pixels3d(surface)
    arr[..., 0] = np.roll(arr[..., 0], px, axis=0)
    arr[..., 2] = np.roll(arr[..., 2], -px, axis=0)
    del arr  # releases the surface lock


def saturate(surface: pygame.Surface, amount: float) -> None:
    """Push colours away from grey, in place. amount 0 = unchanged."""
    if amount <= 0:
        return
    arr = pygame.surfarray.pixels3d(surface)
    work = arr.astype(np.int16)
    luma = (
        work[..., 0] * 77 + work[..., 1] * 151 + work[..., 2] * 28
    ) >> 8
    work = luma[..., None] + ((work - luma[..., None]) * (1.0 + amount)).astype(np.int16)
    arr[:] = np.clip(work, 0, 255).astype(np.uint8)
    del arr


def flash(surface: pygame.Surface, colour: tuple[int, int, int], alpha: int) -> None:
    """Tint the whole frame. Used for strobing on loud audio."""
    if alpha <= 0:
        return
    layer = pygame.Surface(surface.get_size())
    layer.fill(colour)
    layer.set_alpha(min(255, alpha))
    surface.blit(layer, (0, 0))


def noise_speckle(
    surface: pygame.Surface,
    count: int,
    colour: tuple[int, int, int] = (255, 255, 255),
    rng: random.Random | None = None,
) -> None:
    """Scatter single bright pixels. Cheap VHS grain."""
    if count <= 0:
        return
    rng = rng or random
    w, h = surface.get_size()
    for _ in range(count):
        surface.set_at((rng.randrange(w), rng.randrange(h)), colour)


def make_scanline_overlay(size: tuple[int, int], gap: int, alpha: int) -> pygame.Surface:
    """Pre-rendered dark lines, applied after upscaling.

    Applied at output resolution on purpose: at base resolution it would eat
    half the actual image.
    """
    overlay = pygame.Surface(size, pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 0))
    line = pygame.Surface((size[0], 1), pygame.SRCALPHA)
    line.fill((0, 0, 0, alpha))
    for y in range(0, size[1], max(2, gap)):
        overlay.blit(line, (0, y))
    return overlay
