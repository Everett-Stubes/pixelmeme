"""What shows when the room is quiet: Alex Honnold, free soloing, forever.

Two implementations. If a baked pixel video exists (see
tools/prepare_video.py) it plays that. If not -- which is the out-of-the-box
case, because the clip is not ours to ship -- it generates an endless
pixel-art free solo instead, so the screen is never blank.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pygame

from . import effects, sprites


class PixelVideo:
    """Plays a .pxv.npz of palette-indexed frames."""

    def __init__(self, path: Path, size: tuple[int, int]):
        data = np.load(path)
        self.frames: np.ndarray = data["frames"]  # (n, w, h) uint8 indices
        self.palette: np.ndarray = data["palette"]  # (k, 3) uint8
        self.fps = float(data["fps"]) if "fps" in data else 12.0
        self.count = int(self.frames.shape[0])
        if self.count == 0:
            raise ValueError(f"{path} contains no frames")

        self.native_size = (int(self.frames.shape[1]), int(self.frames.shape[2]))
        self.size = size
        self._surface = pygame.Surface(self.native_size)
        self._needs_scale = self.native_size != size
        self._t = 0.0
        self.index = 0

    def update(self, dt: float) -> None:
        self._t += dt
        self.index = int(self._t * self.fps) % self.count

    def draw(self, target: pygame.Surface) -> None:
        rgb = self.palette[self.frames[self.index]]
        pygame.surfarray.blit_array(self._surface, rgb)
        if self._needs_scale:
            target.blit(pygame.transform.scale(self._surface, self.size), (0, 0))
        else:
            target.blit(self._surface, (0, 0))


# Dark enough to read against the valley green, and on the palette.
_BIRD = (44, 40, 44)


class ProceduralClimb:
    """Honnold on a rock pillar, with an alpine valley and village visible in
    the exposure on either side of him -- a narrower climbing surface than a
    full-width wall, specifically so there is room beside it for the view.

    The wall tiles vertically, so scrolling it never repeats visibly and never
    runs out. The backdrop is rendered once and held fixed: at that distance
    nothing in it would appear to move during a single pitch anyway.
    """

    def __init__(self, size: tuple[int, int], seed: int = 7):
        self.w, self.h = size
        self.tile_h = self.h * 2
        # The figure is about twelve units tall and six wide. Five pixels to
        # the unit puts him at sixty pixels on a 192-pixel frame, which is
        # the smallest he can be and still have a face.
        self.scale = max(3, min(6, self.w // 20))
        # Wider than it looks: the silhouette is cut back by up to a fifth on
        # each side, so the pillar has to start wide enough that the pinched
        # sections are still wide enough to stand on.
        # 0.62, not 0.78. At the old width the pillar was most of the frame
        # and what was left either side was a sliver -- which is the one
        # thing a free-solo shot cannot afford, because the drop is the
        # subject. Narrower also means the silhouette's bulges and pinches
        # happen against sky instead of against the frame edge.
        self.pillar_w = max(16, int(self.w * 0.62))
        self.pillar_x = (self.w - self.pillar_w) // 2
        self.wall = sprites.make_wall(self.pillar_w, self.tile_h, seed)
        self.backdrop = sprites.make_alps_backdrop(self.w, self.h, seed + 1)

        self.scroll = 0.0
        self.t = 0.0
        self.rng = random.Random(seed)
        # Speed and height for each bird. All of them below the horizon the
        # backdrop drew, which is the whole point of them.
        horizon = int(self.h * 0.52)
        self._birds = [
            (self.rng.uniform(3.0, 7.0),
             self.rng.randrange(horizon + 4, max(horizon + 6, self.h - 8)))
            for _ in range(3)
        ]
        self.moves = sprites.ClimbSequence(self.rng)

    def update(self, dt: float) -> None:
        self.t += dt
        # The wall scrolls by exactly the height the sequence says he just
        # climbed, converted at the scale he is drawn at -- so a body unit
        # of climbing is a body unit of rock going past, and the two cannot
        # drift. Previously the wall ran on a speed per move plus a slow
        # sine of its own, which meant it slid at a constant rate through a
        # press while he accelerated and decelerated under it, and kept
        # sliding during a shake-out when he was not moving at all.
        before = self.moves.climbed
        self.moves.update(dt)
        gained = (self.moves.climbed - before) * self.scale
        self.scroll = (self.scroll + gained) % self.tile_h

    def draw(self, target: pygame.Surface) -> None:
        target.blit(self.backdrop, (0, 0))

        # Birds, and specifically birds BELOW him. Three dark pixels in a
        # shallow V, tracking slowly across the valley. It is the cheapest
        # altitude cue there is: a bird in the sky above a figure says
        # nothing, and the same bird seen from above says the ground is a
        # long way down. They are drawn over the backdrop and under the
        # rock, so the pillar occludes them and they read as being out in
        # the open air rather than painted on the glass.
        for i, (speed, lane) in enumerate(self._birds):
            bx = int((self.t * speed + i * 37) % (self.w + 16)) - 8
            by = int(lane + math.sin(self.t * 0.6 + i) * 1.5)
            target.fill(_BIRD, pygame.Rect(bx, by, 1, 1))
            target.fill(_BIRD, pygame.Rect(bx - 1, by - 1, 1, 1))
            target.fill(_BIRD, pygame.Rect(bx + 1, by - 1, 1, 1))

        # Scrolling the rock DOWN the screen (new stone entering at the top,
        # old stone exiting at the bottom) is what reads as climbing UP --
        # the world should descend past the camera as the camera ascends.
        offset = int(self.scroll)
        target.blit(self.wall, (self.pillar_x, offset))
        target.blit(self.wall, (self.pillar_x, offset - self.tile_h))

        # Grain on the rock, applied here rather than over the finished frame
        # so it lands on the wall and not on the climber's face. Warm, and
        # only a few pixels: grey specks on sandstone read as dirt on the
        # lens rather than as grain in the rock.
        effects.noise_speckle(target, 5, (236, 200, 158), self.rng)

        # Drift sideways a little, as if following a crack system. Clamped by
        # his own width *and* by how far the jagged edge can bite in, so he
        # never ends up hanging off the skyline.
        half_range = max(
            1.0,
            self.pillar_w / 2 - self.scale * 3.4 - self.pillar_w * 0.19,
        )
        sway = math.sin(self.t * 0.17) * half_range
        x = int(self.w / 2 + sway)
        y = int(self.h * 0.42)
        phase = self.t * 0.6

        sprites.draw_climber(target, x, y, self.scale, phase,
                             pose=self.moves.pose())

        # Chalk dust, falling away below his hands. Gravity does not care
        # which way the camera is scrolling.
        for i in range(3):
            d = (self.t * 30 + i * 37) % (self.h * 0.5)
            px = x + int(math.sin(self.t * 2 + i) * 6)
            target.fill(
                (236, 238, 242),
                pygame.Rect(px, y + 20 + int(d), 1, 1),
            )


class IdleScene:
    """Wraps whichever backend is available.

    Nothing is overlaid on it. There was a title card here; a label on top of
    the picture is the one thing that stops a shot like this reading as a
    shot, and the segment explains itself.
    """

    def __init__(self, cfg, size: tuple[int, int]):
        self.w, self.h = size
        self.t = 0.0
        self.backend_name = "procedural"
        self.backend: PixelVideo | ProceduralClimb

        video_path = cfg.resolve_path(cfg.get("idle.video"))
        if video_path is not None and video_path.exists():
            try:
                self.backend = PixelVideo(video_path, size)
                self.backend_name = f"video:{video_path.name}"
            except Exception as exc:
                print(f"[idle] {video_path.name} unusable ({exc}); generating instead")
                self.backend = ProceduralClimb(size)
        else:
            self.backend = ProceduralClimb(size)

        self._scratch = pygame.Surface(size)

    def update(self, dt: float) -> None:
        self.t += dt
        self.backend.update(dt)

    def draw(self, target: pygame.Surface, listening: bool = True) -> None:
        s = self._scratch
        self.backend.draw(s)

        # A baked clip gets its grain here; the procedural climb adds its own
        # underneath the climber instead.
        if isinstance(self.backend, PixelVideo):
            effects.noise_speckle(s, 6, (208, 204, 194))

        target.blit(s, (0, 0))
