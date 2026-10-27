"""Derive the 3D palette from the materials the broadcast actually uses.

A hand-picked palette cannot keep up: a flat-shaded surface is its material
times a lambert term, so every material needs its own ramp or two facets of
one object land on entries of different hue. The apartment wall was coming
out grey on one plane and brown on the next; Kramer's whole body collapsed
to one grey; a plum top and a navy skirt both quantised to the same slate.

So: collect every material, weight it by how much of the frame it covers,
and lay down a ramp of tones for each in weight order until the palette is
full. Then the nearest entry to any lit facet is always on that material's
own ramp.

Run after changing any material, in either the sets or the cast, and commit
the result. Nothing reads this at runtime.
"""
import os, sys, collections
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, "C:/Users/evere/pixelmeme")
import pygame; pygame.init()
import numpy as np
from pixelmeme.render import r3d, world, actors as A

AMBIENT = 0.46
# The lambert terms a facet actually takes, sampled across the range.
TONES = tuple(AMBIENT + (1 - AMBIENT) * k for k in (0.0, 0.34, 0.66, 1.0))
LIMIT = 80
CAST_SHARE = 50     # entries reserved for the people, not the rooms

# Every material a face is built from, including the ones derived in the
# Actor constructor rather than declared in the archetype. Leaving those out
# is how a nose and a brow ridge ended up quantising to the same entry as
# the cheek they stand on, which undoes the whole point of building them.
#
# In order of how much it costs to lose. Taken actor by actor instead, the
# first one down the list spent the entire cast budget on his own twelve
# materials and everybody after him fell back on whatever was already there:
# a plum waistcoat came out brick, an orange shirt came out khaki.
CAST_KEYS = ("shirt", "skin", "hair", "jacket", "trousers", "band",
             "collar", "skin_warm", "shade", "shoes", "deep", "brow")

area = collections.Counter()
cast = A.make_sitcom_cast(1)
for name in world.SET_NAMES:
    marks = world.marks_for(name)
    for k, a in enumerate(cast):
        a.x, a.z = marks[k % len(marks)]
        a.state = A.IDLE
    sc = r3d.Scene(); world.build_set(sc, name, 3.0, 5)
    for b in cast:
        b.build_geometry(sc)
    V = np.asarray(sc.verts, np.float32); F = np.asarray(sc.faces, np.int32)
    for i, col in enumerate(sc.colours):
        p = V[F[i]]
        e1, e2 = p[1] - p[0], p[2] - p[0]
        area[tuple(int(c) for c in col)] += float(np.linalg.norm(np.cross(e1, e2)))

# Text and UI colours the stage draws on top; they must survive too.
FIXED = [
    (0, 0, 0), (10, 8, 14), (26, 24, 30), (248, 238, 210), (236, 238, 242),
    (120, 158, 152), (240, 186, 62), (72, 40, 40),
]


def dist(a, b):
    dr, dg, db = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return dr * dr * 3 + dg * dg * 6 + db * db


palette = []


def add(c, spacing=260, cap=None):
    c = tuple(max(0, min(255, int(v))) for v in c)
    if cap is not None and len(palette) >= cap:
        return False
    if all(dist(c, p) > spacing for p in palette):
        palette.append(c)
        return True
    return False


for c in FIXED:
    add(c)

# The cast gets its ramps first. Weighting purely by screen area hands the
# whole palette to walls and floors, and then the four people everybody is
# actually looking at share whatever grey is left -- which is how a blue
# shirt and a purple top ended up the same colour.
for key in CAST_KEYS:
    for actor in cast:
        material = getattr(actor, key, None)
        if material is None:
            continue
        for k in TONES:
            add(tuple(v * k for v in material), spacing=520, cap=CAST_SHARE)
print(f"{len(palette)} entries after the cast")

for colour, _w in area.most_common():
    for k in TONES:
        if len(palette) >= LIMIT:
            break
        add(tuple(v * k for v in colour))
    if len(palette) >= LIMIT:
        break

print(f"{len(palette)} entries from {len(area)} materials")

# Each entry already carries its own trailing comma, so the rows are joined
# on a bare newline. Joining on ",\n" is how the generated file came out
# with a double comma at the end of every row and refused to import.
body = "\n".join(
    "        " + " ".join(f"({r}, {g}, {b})," for r, g, b in palette[i:i + 3])
    for i in range(0, len(palette), 3)
)

import pathlib
p = pathlib.Path("C:/Users/evere/pixelmeme/pixelmeme/render/palette.py")
s = p.read_text(encoding="utf-8")
start = s.index('    "nyc": [')
end = s.index("    ],", start) + len("    ],")
new = ('    # Derived from the materials the sets and the cast actually use,\n'
       '    # weighted by how much of the frame each one covers, with a ramp of\n'
       '    # tones laid down for each in weight order (see\n'
       '    # tools/make_palette.py). A flat-shaded surface is its material\n'
       '    # times a lambert term, so a material without its own ramp has its\n'
       '    # facets land on entries of different hue -- which is how one wall\n'
       '    # came out grey on one plane and brown on the next, and how the\n'
       "    # tallest of them turned into a single grey silhouette.\n"
       '    "nyc": [\n' + body + "\n    ],")
p.write_text(s[:start] + new + s[end:], encoding="utf-8")
print("palette written")
