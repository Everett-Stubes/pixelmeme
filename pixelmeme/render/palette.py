"""Colour quantisation: everything on screen goes through here, so everything
on screen is made of a small fixed set of chunky colours."""

from __future__ import annotations

import numpy as np

# Palettes are deliberately small. A small palette is most of what makes an
# image read as "pixel art" rather than "a photo with big pixels".
PALETTES: dict[str, list[tuple[int, int, int]]] = {
    # Blown-out meme colours: hot orange, cyan, magenta, nuclear white.
    "fried": [
        (0, 0, 0), (28, 20, 36), (68, 36, 52), (120, 36, 36),
        (220, 40, 30), (255, 110, 0), (255, 190, 0), (255, 245, 120),
        (255, 255, 255), (170, 170, 190), (90, 100, 130), (30, 60, 120),
        (0, 140, 220), (0, 230, 230), (210, 20, 180), (120, 0, 160),
    ],
    # Sandstone / sky / chalk, plus alpine meadow and terracotta for the
    # Swiss-village backdrop behind the climbing loop, and a red for the
    # climber's shirt. The sandstone ramp needs enough steps of its own or
    # the rock bands instead of reading as rock.
    # The climb. The eight browns are the sandstone ramp in sprites.py, to
    # the value -- the rock is drawn from this list rather than from its own
    # numbers, so there is nothing for the quantiser to land between. Every
    # one of the eight used to miss by two to twelve per channel, and on a
    # flat bed of rock that is a visible grid of dots, not a gradient.
    "stone": [
        (0, 0, 0), (18, 20, 26), (44, 40, 44), (70, 64, 62),
        (104, 98, 92), (140, 134, 126), (176, 170, 160), (208, 204, 194),
        (236, 238, 242),
        (62, 40, 26), (86, 56, 38), (112, 76, 50), (140, 98, 66),
        (168, 122, 84), (194, 148, 106), (216, 174, 132), (234, 200, 160),
        (60, 96, 140), (110, 160, 210), (190, 220, 245),
        (70, 110, 70), (130, 150, 90), (150, 80, 58),
        (206, 74, 62), (168, 48, 42), (116, 32, 30),
    ],
    # Four-tone handheld green.
    "gameboy": [
        (15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15),
    ],
    # A warm interior that has gone slightly wrong. Creams and wood against
    # purple shadows and a sickly teal. This is the default for the stage.
    "sitcom": [
        (10, 8, 14), (28, 22, 34), (52, 38, 46), (84, 54, 52),
        (120, 72, 58), (160, 104, 66), (198, 146, 92), (230, 196, 140),
        (248, 238, 210), (60, 70, 78), (86, 110, 112), (120, 158, 152),
        (170, 54, 48), (226, 118, 46), (240, 186, 62), (96, 62, 118),
    ],
    # For the 3D set. Flat shading needs real tonal steps -- a light, mid and
    # dark of every surface -- or the quantiser collapses the lighting and
    # every polygon on a box comes out the same colour. Hence the ramps, and
    # hence more entries than the 2D palettes needed.
    # Derived from the materials the sets and the cast actually use,
    # weighted by how much of the frame each one covers, with a ramp of
    # tones laid down for each in weight order (see
    # tools/make_palette.py). A flat-shaded surface is its material
    # times a lambert term, so a material without its own ramp has its
    # facets land on entries of different hue -- which is how one wall
    # came out grey on one plane and brown on the next, and how the
    # tallest of them turned into a single grey silhouette.
    "nyc": [
        (0, 0, 0), (10, 8, 14), (26, 24, 30),
        (248, 238, 210), (236, 238, 242), (120, 158, 152),
        (240, 186, 62), (72, 40, 40), (47, 69, 92),
        (66, 96, 128), (84, 122, 163), (104, 150, 200),
        (79, 73, 47), (110, 102, 66), (140, 130, 84),
        (172, 160, 104), (103, 98, 88), (145, 137, 123),
        (184, 174, 156), (226, 214, 192), (91, 53, 28),
        (127, 74, 39), (161, 94, 50), (198, 116, 62),
        (103, 83, 66), (144, 117, 92), (182, 148, 117),
        (224, 182, 144), (218, 174, 134), (150, 126, 102),
        (191, 160, 130), (234, 196, 160), (21, 15, 12),
        (46, 33, 27), (59, 45, 33), (75, 57, 42),
        (92, 70, 52), (122, 106, 76), (57, 33, 50),
        (81, 46, 70), (102, 58, 89), (126, 72, 110),
        (28, 34, 52), (39, 48, 73), (62, 76, 114),
        (44, 40, 50), (58, 54, 62), (61, 72, 90),
        (78, 91, 114), (96, 112, 140), (113, 95, 73),
        (158, 132, 102), (55, 39, 27), (120, 86, 60),
        (98, 88, 72), (137, 123, 101), (174, 156, 128),
        (214, 192, 158), (47, 24, 21), (66, 34, 29),
        (84, 44, 37), (104, 54, 46), (88, 64, 43),
        (112, 81, 55), (138, 100, 68), (94, 92, 85),
        (132, 128, 119), (168, 163, 151), (206, 200, 186),
        (208, 186, 156), (84, 55, 40), (104, 68, 50),
        (37, 29, 34), (75, 50, 36), (92, 58, 50),
        (45, 50, 59), (57, 63, 75), (70, 78, 92),
        (112, 70, 66), (91, 82, 69),
    ],
    # Washed-out VHS.
    "vhs": [
        (0, 0, 0), (26, 16, 32), (56, 28, 60), (96, 40, 78),
        (140, 52, 88), (184, 76, 92), (214, 112, 108), (236, 156, 140),
        (248, 200, 180), (255, 240, 226), (40, 70, 92), (60, 116, 136),
        (96, 164, 170), (150, 206, 198), (206, 236, 226), (255, 255, 255),
    ],
}

# 4x4 ordered dither. Cheap, deterministic, and it looks like 1995.
_BAYER4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float32,
)


class Quantizer:
    """Maps arbitrary RGB to a fixed palette via a precomputed 32x32x32 LUT.

    The LUT costs a few hundred milliseconds to build once and makes the
    per-frame cost a single fancy-index, which is what makes this viable on a
    Pi 3.
    """

    def __init__(self, palette_name: str = "fried", dither: float = 0.55):
        self.name = palette_name
        self.palette = np.array(PALETTES[palette_name], dtype=np.uint8)
        self.dither = float(dither)
        self._lut = self._build_lut(self.palette)
        self._dither_cache: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _build_lut(palette: np.ndarray) -> np.ndarray:
        levels = np.arange(32, dtype=np.int32) * 8 + 4
        r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
        grid = np.stack([r, g, b], axis=-1).reshape(-1, 3).astype(np.int32)

        best = np.zeros(grid.shape[0], dtype=np.uint8)
        best_dist = np.full(grid.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
        for i, colour in enumerate(palette.astype(np.int32)):
            delta = grid - colour
            # Rough luma weighting; greens matter most to the eye.
            dist = (delta[:, 0] ** 2) * 3 + (delta[:, 1] ** 2) * 6 + (delta[:, 2] ** 2)
            closer = dist < best_dist
            best_dist[closer] = dist[closer]
            best[closer] = i
        return best.reshape(32, 32, 32)

    def _dither_field(self, w: int, h: int) -> np.ndarray:
        key = (w, h)
        field = self._dither_cache.get(key)
        if field is None:
            tiles = np.tile(_BAYER4, (w // 4 + 1, h // 4 + 1))[:w, :h]
            field = (tiles / 16.0 - 0.46875) * 16.0
            field = field[:, :, None].astype(np.float32)
            self._dither_cache[key] = field
        return field

    def indices(self, rgb: np.ndarray) -> np.ndarray:
        """Palette indices for (w, h, 3) uint8 input.

        Used directly when baking video, where storing one byte per pixel
        instead of three is the whole point.
        """
        if self.dither > 0.0:
            field = self._dither_field(rgb.shape[0], rgb.shape[1]) * self.dither
            work = np.clip(rgb.astype(np.float32) + field, 0, 255).astype(np.uint8)
        else:
            work = rgb
        return self._lut[work[..., 0] >> 3, work[..., 1] >> 3, work[..., 2] >> 3]

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (w, h, 3) uint8 as returned by pygame.surfarray.array3d."""
        return self.palette[self.indices(rgb)]

    def nearest(self, colour: tuple[int, int, int]) -> tuple[int, int, int]:
        """Snap a single colour to the palette, for drawing primitives."""
        r, g, b = (int(c) & 0xFF for c in colour)
        i = int(self._lut[r >> 3, g >> 3, b >> 3])
        return tuple(int(c) for c in self.palette[i])

    def pick(self, i: int) -> tuple[int, int, int]:
        return tuple(int(c) for c in self.palette[i % len(self.palette)])
