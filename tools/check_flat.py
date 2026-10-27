"""Check that nothing flat comes out checkered.

Both of this show's looks are flat: the sitcom is flat-shaded polygons, the
climb is cel art off a fixed eight-step ramp. A region that is meant to be
one solid colour has nothing a dither could smooth -- all an ordered dither
can do to it is alternate it between the two palette entries it falls
between, which on screen is a grid of dots across a whole face or a whole
bed of rock.

So: render real frames of both, group the pixels by their raw colour, and
fail if any large region of one raw colour comes out as more than one
palette colour. Run after touching a palette, a material, or the dither
settings. Nothing reads this at runtime.

This has caught the same bug twice, in two unrelated places -- skin on the
cast, and the sandstone -- which is why it is in the repo rather than in a
scratch file.
"""
import os
import sys
import collections

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

pygame.init()

from pixelmeme import config as C
from pixelmeme.render.stage import Stage
from pixelmeme.render.idle import ProceduralClimb
from pixelmeme.render.display import _dither_for
from pixelmeme.render.palette import Quantizer

W, H = 108, 192
MIN_REGION = 60      # pixels of one raw colour before it is worth reporting

cfg = C.load(None)
fails = []


def split_report(label, palette_name, frames):
    """How many large single-colour regions the quantiser breaks in two."""
    dither = _dither_for(palette_name, cfg["display"])
    q = Quantizer(palette_name, dither)
    split = collections.Counter()
    total = 0
    for raw in frames:
        flat = raw.reshape(-1, 3).astype(np.int32)
        quant = q.apply(raw).reshape(-1, 3).astype(np.int32)
        keys = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
        qk = (quant[:, 0] << 16) | (quant[:, 1] << 8) | quant[:, 2]
        order = np.argsort(keys, kind="stable")
        ks, qs = keys[order], qk[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(ks)]):
            if hi - lo < MIN_REGION:
                continue
            total += 1
            if len(np.unique(qs[lo:hi])) > 1:
                c = int(ks[lo])
                split[(c >> 16, (c >> 8) & 255, c & 255)] += hi - lo
    print(f"{label} (palette {palette_name}, dither {dither}): "
          f"{len(split)} of {total} large flat regions split")
    for colour, n in split.most_common(6):
        print(f"    {colour} -> more than one entry, {n} px")
    if split:
        fails.append(f"{label}: {len(split)} flat colours checkerboard "
                     f"({', '.join(str(c) for c in list(split)[:3])})")


# -- the sitcom ---------------------------------------------------------------
stage = Stage(cfg, (W, H), seed=5)
surf = pygame.Surface((W, H))
frames = []
for _ in range(60):
    stage.update(0.3)
    stage.draw(surf)
    frames.append(np.asarray(pygame.surfarray.pixels3d(surf)).copy())
split_report("sitcom", cfg.get("display.palette"), frames)

# -- the climb ----------------------------------------------------------------
climb = ProceduralClimb((W, H), 7)
frames = []
for _ in range(30):
    for _ in range(8):
        climb.update(0.05)
    climb.draw(surf)
    frames.append(np.asarray(pygame.surfarray.pixels3d(surf)).copy())
split_report("climb", cfg.get("display.idle_palette"), frames)

# -- the climb's own ramps must be palette entries, exactly -------------------
# Snapping to the nearest entry is not enough: a ramp step that misses is a
# step the dither would split the moment anybody turned dithering back on,
# and it is also not the colour the art says it is.
from pixelmeme.render import sprites

exact = Quantizer(cfg.get("display.idle_palette"), 0.0)
off = []
for i, c in enumerate(sprites._SANDSTONE):
    c = tuple(int(v) for v in c)
    if exact.nearest(c) != c:
        off.append(f"_SANDSTONE[{i}] {c}")
for name in ("_SKIN", "_SKIN_LIT", "_SKIN_DARK", "_SHIRT", "_SHIRT_LIT",
             "_SHIRT_DARK", "_PANTS", "_PANTS_DARK", "_INK", "_CHALK",
             "_BAG", "_RUBBER"):
    c = getattr(sprites, name)
    if exact.nearest(c) != c:
        off.append(f"{name} {c}")
print(f"climb ramps: {len(off)} colours not in the palette")
for o in off:
    print("    ", o)
if off:
    fails.append(f"{len(off)} climb colours are not palette entries")

print("FAIL" if fails else "PASS")
for f in fails:
    print("  ", f)
sys.exit(1 if fails else 0)
