"""The sets, as 3D geometry.

Everything here emits polygons into an r3d.Scene in world space. One unit is
roughly a metre. The floor is y=0, the back wall is at negative z, and the
camera sits out past positive z looking in -- a sitcom set with the fourth
wall missing, which is what it is.

The rooms are narrow on purpose, and more narrow than they used to be. The
frame is portrait, so the *horizontal* field of view is about 33 degrees:
at the distance a person has to stand to fit head-to-knee in shot, the lens
only sees about three metres of width. Walls five metres apart are not
"spacious", they are invisible -- they never enter the frustum at all, and
the room collapses into one flat slab of back wall with furniture pinned to
it. Bringing them in to roughly two metres is what puts a corner in shot.

Depth here comes from three things, in this order of effect:

1. **Things that run away from the lens.** Floorboard seams, a cornice, a
   baseboard, a counter, a row of booths, a line of pendant lamps. Parallel
   lines converging is the whole of perspective; a room with nothing running
   in z is a painting no matter how much furniture is standing in it.
2. **Aerial perspective.** Each set declares a fog colour and range and the
   renderer mixes faces toward it with distance, so the far wall is not lit
   exactly as brightly as the chair in front of it.
3. **Staging in z.** Things at four or five distinct depths, so the
   perspective divide gives them visibly different scales.
"""

from __future__ import annotations

import math
import random

from . import r3d

SET_NAMES = ("APARTMENT", "STANDUP", "DINER", "HALLWAY")

# Where the cast stands. Staged in DEPTH, not in a line across the frame:
# a portrait shot has vertical room and no horizontal room, so the far
# characters sit higher and smaller and the near ones lower and larger.
# Every mark is clear of the furniture on its set.
MARKS = {
    "APARTMENT": ((-0.62, -1.30), (0.58, -2.05), (-0.18, -2.70), (0.95, -3.00)),
    "STANDUP": ((-0.15, -1.45), (-0.95, -2.35), (0.85, -2.50), (0.25, -3.05)),
    "DINER": ((-0.62, -1.15), (0.62, -1.85), (-0.20, -2.35), (1.00, -2.20)),
    "HALLWAY": ((-0.40, -1.30), (0.48, -2.30), (-0.32, -3.40), (0.46, -4.40)),
}

# (colour, near, far, strength) per set, handed to r3d.render. The colour is
# the set's own haze -- a warm room fades warm, a cold hallway fades cold --
# and the range runs from about where the nearest actor stands to a little
# past the back wall.
FOG = {
    "APARTMENT": ((96, 76, 58), 2.4, 7.6, 0.44),
    "STANDUP": ((34, 23, 25), 1.8, 6.2, 0.56),
    "DINER": ((78, 72, 74), 2.2, 7.2, 0.42),
    "HALLWAY": ((44, 33, 28), 2.0, 9.8, 0.66),
}

# After dark. Every room has two dressings and the show moves between them,
# so coming back to the apartment is coming back at a different hour rather
# than cutting to the identical frame it left.
FOG_NIGHT = {
    "APARTMENT": ((46, 28, 20), 2.0, 6.8, 0.62),
    "STANDUP": ((26, 17, 20), 1.8, 6.2, 0.62),
    "DINER": ((40, 38, 46), 2.0, 7.0, 0.56),
    "HALLWAY": ((28, 21, 18), 1.8, 9.4, 0.74),
}


def marks_for(name: str):
    return MARKS.get(name, MARKS["APARTMENT"])


def night(seed: int) -> bool:
    """Which of a set's two dressings this visit gets.

    Taken off the seed the stage already draws fresh on every move, so
    nothing new has to be threaded through to choose it.
    """
    return bool(seed & 1)


def fog_for(name: str, seed: int = 0):
    table = FOG_NIGHT if night(seed) else FOG
    return table.get(name, table["APARTMENT"])


def _boards(sc: r3d.Scene, x0, x1, z0, z1, y, floor, seam, step=0.46,
            band=1.35):
    """A floor with plank seams running away from the camera.

    Tiling the floor into squares, which is what this used to do, produces
    no line that converges: every edge is either horizontal across the
    frame or lost in the tile next to it. Thin seams running in z cost
    about what the tiling cost and they are the single strongest depth cue
    on any of these sets.

    Banded in depth for the same reason `plane_y` is -- a seam that runs
    from the back wall to past the lens crosses the near plane and is
    dropped whole, taking the entire floor with it.
    """
    sc.plane_y(x0, x1, z0, z1, y, floor, step=band)
    n = max(1, int(math.ceil((z1 - z0) / band)))
    dz = (z1 - z0) / n
    for i in range(1, int((x1 - x0) / step)):
        x = x0 + i * step
        for j in range(n):
            az = z0 + j * dz
            sc.quad_y(x - 0.013, x + 0.013, az, az + dz, y + 0.004, seam)


# -- the apartment -----------------------------------------------------------

_WALL = (178, 150, 116)
_WALL_B = (158, 132, 102)
_FLOOR = (138, 100, 68)
_SEAM = (104, 74, 50)
_TRIM = (206, 190, 164)

_AP_X = 1.95            # side walls, in far enough to actually be in shot
_AP_CEIL = 2.46         # and a ceiling low enough to be in it too
_AP_BACK = -3.6


def _apartment(sc: r3d.Scene, t: float, seed: int) -> None:
    rng = random.Random(seed ^ int(t / 13))
    X, CY, BZ = _AP_X, _AP_CEIL, _AP_BACK
    dark = night(seed)
    # Lamplight rather than daylight: warmer, a stop and a half down, and
    # the ceiling bounces a lot less of it.
    wall = (142, 96, 60) if dark else _WALL
    wall_b = (118, 78, 50) if dark else _WALL_B
    floor = (86, 58, 40) if dark else _FLOOR
    trim = (176, 146, 110) if dark else _TRIM
    ceiling = (130, 100, 72) if dark else (208, 186, 156)

    # Shell. Only the inward faces are ever seen, so they are single quads
    # rather than boxes. Layer 0 keeps them behind everything in the room.
    sc.layer = 0
    _boards(sc, -X, X, BZ, 2.6, 0.0, floor, _SEAM)
    sc.quad_z(-X, X, 0.0, CY, BZ, wall)             # back wall
    sc.plane_x(BZ, 3.0, 0.0, CY, -X, wall_b)        # left wall
    sc.plane_x(BZ, 3.0, 0.0, CY, X, wall_b)         # right wall
    sc.plane_y(-X, X, BZ, 1.6, CY, ceiling)         # ceiling
    sc.layer = 1

    # Cornice and baseboard, running the length of both side walls and
    # across the back. Six boxes, and they are what turn three grey planes
    # into a room: every one of them converges.
    for side in (-1, 1):
        sc.run_z(side * (X - 0.05), CY - 0.07, BZ, 2.4, (0.10, 0.14),
                 trim, faces="xy")
        sc.run_z(side * (X - 0.04), 0.07, BZ, 2.4, (0.08, 0.14),
                 (96, 70, 54), faces="xy")
    sc.box((0.0, CY - 0.07, BZ + 0.06), (2 * X, 0.14, 0.12), trim, faces="zy")
    sc.box((0.0, 0.07, BZ + 0.05), (2 * X, 0.14, 0.10), (96, 70, 54),
           faces="zy")

    # A ceiling rose and a shade, so the ceiling is a surface with
    # something on it rather than a blank band across the top of the frame.
    sc.revolve((0.30, CY - 0.26, -1.90),
               [(0.0, 0.26), (0.10, 0.20), (0.22, 0.05)],
               (248, 232, 186) if dark else (206, 192, 160), 8, caps="b")
    sc.tube((0.30, CY, -1.90), (0.30, CY - 0.06, -1.90), 0.05, 0.04,
            (182, 164, 140), 6, caps="b")

    # Window onto the airshaft, back wall left of centre. Built with a
    # reveal -- jambs with real thickness standing proud of the wall --
    # because a window painted flat on a wall is the thing that gives a set
    # away as a backdrop faster than anything else in the room.
    wx0, wx1, wy0, wy1 = -1.72, -0.52, 0.92, 2.00
    # By day the airshaft is a grey slot with the odd lit window across it.
    # By night it is black with most of them burning, which is the single
    # biggest difference between the two dressings and costs nothing.
    sc.quad_z(wx0, wx1, wy0, wy1, BZ + 0.16,
              (22, 26, 40) if dark else (52, 60, 82))
    for i in range(4):
        if rng.random() < (0.85 if dark else 0.5):
            x = wx0 + 0.08 + i * 0.28
            sc.quad_z(x, x + 0.16, 1.14 + (i % 2) * 0.38,
                      1.52 + (i % 2) * 0.38, BZ + 0.17,
                      (248, 214, 128) if dark else (226, 190, 110))
    for x in (wx0, wx1):                                   # side reveals
        sc.box((x, (wy0 + wy1) * 0.5, BZ + 0.09), (0.07, wy1 - wy0, 0.20),
               trim, faces="xy")
    sc.box(((wx0 + wx1) * 0.5, wy1, BZ + 0.09), (wx1 - wx0, 0.07, 0.20),
           trim, faces="zy")
    sc.box(((wx0 + wx1) * 0.5, wy0, BZ + 0.10), (wx1 - wx0 + 0.14, 0.09, 0.26),
           trim, faces="zy")                              # sill, proud
    sc.box(((wx0 + wx1) * 0.5, (wy0 + wy1) * 0.5, BZ + 0.17),
           (0.05, wy1 - wy0, 0.03), trim, faces="zx")     # mullion

    # The front door, back wall right -- where Kramer comes through. The
    # architrave stands off the wall so the doorway has an edge.
    dx, dy = 1.28, 1.02
    sc.quad_z(dx - 0.46, dx + 0.46, 0.0, 2.04, BZ + 0.08, (112, 74, 54))
    sc.box((dx, dy, BZ + 0.10), (0.78, 1.80, 0.03), (132, 90, 64), faces="z")
    for s in (-1, 1):
        sc.box((dx + s * 0.52, 1.04, BZ + 0.07), (0.09, 2.14, 0.16),
               trim, faces="xy")
    sc.box((dx, 2.12, BZ + 0.07), (1.13, 0.09, 0.16), trim, faces="zy")
    sc.tube((dx - 0.32, 1.00, BZ + 0.12), (dx - 0.32, 1.00, BZ + 0.20),
            0.035, 0.03, (214, 178, 74), 5, caps="t")

    # The bicycle on the wall.
    bx, by = 0.30, 1.78
    sc.ring((bx - 0.28, by, BZ + 0.13), 0.25, 0.055, (44, 42, 48))
    sc.ring((bx + 0.30, by, BZ + 0.13), 0.25, 0.055, (44, 42, 48))
    sc.box((bx + 0.01, by + 0.06, BZ + 0.14), (0.52, 0.05, 0.05),
           (176, 60, 52), faces="zy")
    sc.box((bx - 0.10, by - 0.02, BZ + 0.14), (0.05, 0.32, 0.05),
           (176, 60, 52), faces="zx")
    sc.box((bx + 0.18, by + 0.04, BZ + 0.14), (0.05, 0.28, 0.05),
           (176, 60, 52), faces="zx")

    # Couch against the back wall, under the window. Cushions sag outward
    # at the top and the arms are rolled, because upholstery has no edges.
    cx = -1.06
    sc.frustum((cx, 0.20, -3.08), (1.52, 0.70), (1.62, 0.80), 0.40,
               (84, 96, 112), caps="t")
    for i in (-1, 1):
        sc.frustum((cx + i * 0.40, 0.47, -3.10), (0.72, 0.74), (0.78, 0.80),
                   0.18, (96, 108, 124), caps="t")
    sc.frustum((cx, 0.62, -3.36), (1.58, 0.20), (1.50, 0.16), 0.48,
               (104, 116, 132), lean=-0.06, caps="t")
    for i in (-1, 1):
        sc.tube((cx + i * 0.77, 0.52, -3.42), (cx + i * 0.77, 0.52, -2.76),
                0.15, 0.15, (110, 122, 138), 6, caps="tb")

    # Kitchen counter, left, running a metre and a half *in depth*. It used
    # to sit square to the lens, which made it another flat band; run along
    # z it is the longest converging line on the set.
    kx = -X + 0.52
    sc.box((kx, 0.46, -1.55), (1.04, 0.92, 1.70), (150, 118, 86),
           faces="xz")
    sc.box((kx, 0.95, -1.55), (1.14, 0.07, 1.82), (214, 206, 190))
    sc.box((kx - 0.02, 0.09, -1.55), (0.96, 0.18, 1.74), (118, 90, 64),
           faces="xz")                                      # toe kick
    # Upper cabinets, likewise running in z.
    sc.box((-X + 0.20, 1.86, -2.00), (0.36, 0.70, 1.30), (158, 126, 92),
           faces="xyz")

    # Fridge, back-left corner.
    sc.box((-X + 0.44, 0.86, -3.24), (0.78, 1.72, 0.70), (216, 212, 200))
    sc.box((-X + 0.82, 1.22, -3.24), (0.03, 0.46, 0.05), (120, 118, 112),
           faces="x")

    # A rug under the coffee table: another plane in depth, with a border,
    # for three polygons.
    sc.quad_y(-1.76, -0.16, -2.70, -1.30, 0.012, (112, 76, 72))
    sc.quad_y(-1.68, -0.24, -2.62, -1.38, 0.016, (146, 104, 92))

    # Coffee table in front of the couch: turned legs, a top with a lip.
    tx, tz = -0.96, -2.05
    sc.frustum((tx, 0.34, tz), (0.92, 0.50), (0.96, 0.54), 0.06, (122, 84, 58))
    for ddx, ddz in ((-0.38, -0.18), (0.38, -0.18), (-0.38, 0.18), (0.38, 0.18)):
        sc.tube((tx + ddx, 0.31, tz + ddz), (tx + ddx, 0.0, tz + ddz),
                0.033, 0.021, (104, 70, 50), 5)
    sc.box((tx - 0.18, 0.40, tz), (0.28, 0.05, 0.20), (206, 178, 96))
    sc.box((tx - 0.15, 0.44, tz + 0.03), (0.24, 0.04, 0.17), (176, 98, 74))
    sc.tube((tx + 0.26, 0.37, tz - 0.06), (tx + 0.26, 0.49, tz - 0.06),
            0.05, 0.05, (214, 206, 190), 7, caps="t")

    # Pictures on the right-hand wall, hung at two depths so the wall has a
    # rhythm running away from the lens rather than one object on it.
    for z0, z1, y0, y1, col in ((-2.75, -2.15, 1.42, 1.98, (206, 170, 90)),
                                (-1.70, -1.20, 1.28, 1.72, (120, 132, 150))):
        sc.box((X - 0.05, (y0 + y1) * 0.5, (z0 + z1) * 0.5),
               (0.06, y1 - y0 + 0.08, z1 - z0 + 0.08), (88, 64, 48),
               faces="xy")
        sc.quad_x(z0, z1, y0, y1, X - 0.09, col)

    # Standing lamp in the back-right corner: a tapered shade on a turned
    # pole, which is the one object in the room that has to look round.
    lx, lz = X - 0.34, -2.90
    sc.tube((lx, 0.04, lz), (lx, 1.50, lz), 0.034, 0.027, (72, 66, 62), 6)
    sc.revolve((lx, 1.50, lz), [(0.0, 0.14), (0.20, 0.20), (0.34, 0.24)],
               (246, 226, 168) if dark else (196, 180, 140), 8, caps="")
    sc.revolve((lx, 0.0, lz), [(0.0, 0.19), (0.05, 0.17)], (72, 66, 62), 6)

    # Cereal boxes and a carton on the counter.
    for i, col in enumerate(((198, 98, 62), (218, 176, 86), (120, 140, 176))):
        sc.box((kx - 0.18 + (i % 2) * 0.22, 1.13, -2.10 + i * 0.26),
               (0.16, 0.28, 0.11), col)
    sc.frustum((kx + 0.14, 1.14, -0.95), (0.13, 0.13), (0.10, 0.10), 0.32,
               (226, 222, 210))

    # A shelf of books above the counter, running in z with it.
    sc.box((-X + 0.22, 1.40, -1.55), (0.32, 0.06, 1.20), (132, 94, 62))
    for i in range(5):
        col = ((176, 72, 58), (96, 118, 150), (206, 176, 96),
               (112, 128, 92), (150, 110, 150))[i]
        sc.box((-X + 0.22, 1.58, -2.05 + i * 0.22), (0.24, 0.30, 0.15), col,
               faces="xz")


# -- the stand-up spot -------------------------------------------------------


def _standup(sc: r3d.Scene, t: float, seed: int) -> None:
    # Two rooms: the brick one, and the one with the curtain behind the mic.
    # Both are real comedy clubs and they look nothing like each other,
    # which is the point of having two.
    curtain = night(seed)
    brick, mortar = ((104, 42, 52), (74, 32, 40)) if curtain else \
                    ((132, 68, 56), (92, 58, 50))
    X, BZ = 1.90, -3.30
    sc.layer = 0
    sc.plane_y(-X, X, BZ, 2.6, 0.0, (44, 44, 52))
    sc.quad_z(-X, X, 0.0, 2.9, BZ, mortar)
    # Side walls of brick as well, so the back wall has a corner to meet.
    for side in (-1, 1):
        sc.plane_x(BZ, 1.2, 0.0, 2.7, side * X, (104, 54, 46))
    sc.plane_y(-X, X, BZ, 1.0, 2.70, (46, 36, 42))
    sc.layer = 1

    # A lighting bar with three cans on it, hung over the front of the
    # stage. The top of this frame was an eighth of the picture in flat
    # black; a club ceiling is dark, but it is dark with things hanging
    # from it, and those things are what aim the eye at the stage.
    sc.box((0.0, 2.58, -1.35), (2 * X, 0.07, 0.07), (62, 58, 62), faces="zy")
    for i, lx in enumerate((-1.05, 0.0, 1.05)):
        sc.tube((lx, 2.58, -1.35), (lx, 2.40, -1.40), 0.035, 0.028,
                (48, 46, 50), 5)
        sc.revolve((lx, 2.22, -1.44), [(0.0, 0.085), (0.13, 0.105),
                                       (0.18, 0.100)],
                   (236, 206, 140) if i == 1 else (74, 70, 72), 6, caps="b")

    # Brick courses, offset every other row. Drawn as mortar on brick rather
    # than brick on mortar: one quad of wall, then the horizontal courses,
    # then staggered vertical joints on alternate rows. Drawing every brick
    # as its own quad cost 265 polygons for a wall forty pixels tall.
    sc.quad_z(-X, X, 0.0, 2.4, BZ + 0.04, brick)
    if curtain:
        # Vertical folds instead of courses: the same number of quads, and
        # the only thing that has to change for the room to read as a
        # different room.
        x = -X
        while x < X:
            w = 0.16
            sc.quad_z(x, x + w * 0.5, 0.0, 2.4, BZ + 0.05, mortar)
            sc.quad_z(x + w * 0.5, x + w, 0.0, 2.4, BZ + 0.055,
                      (134, 58, 68))
            x += w
        sc.box((0.0, 2.44, BZ + 0.07), (2 * X, 0.14, 0.10), (88, 70, 46),
               faces="zy")      # pelmet
    else:
        bh = 0.155
        rows = int(2.4 / bh)
        for r in range(1, rows):
            y = r * bh
            sc.quad_z(-X, X, y - 0.013, y + 0.013, BZ + 0.05, mortar)
            if r % 2:
                continue
            x = -X + (0.16 if r % 4 else 0.0)
            while x < X:
                sc.quad_z(x - 0.013, x + 0.013, y, y + bh, BZ + 0.05, mortar)
                x += 0.60

    # A pool of light, built out of the wall's own colour in a few steps
    # rather than as one hot rectangle -- a single bright quad reads as a
    # sheet of plywood nailed up behind him.
    for i, (half, lo, hi) in enumerate(
            ((1.15, 0.15, 2.35), (0.88, 0.40, 2.10), (0.58, 0.70, 1.84))):
        k = 1.26 + i * 0.20
        sc.quad_z(-half, half, lo, hi, BZ + 0.06 + i * 0.006,
                  tuple(min(255, int(c * k)) for c in brick))

    # The stage: a low platform whose lip runs away from the lens, and the
    # one line on this set that converges.
    sc.quad_y(-X, X, BZ, -1.05, 0.11, (62, 54, 56))
    sc.box((0.0, 0.055, -1.05), (2 * X, 0.11, 0.06), (92, 80, 76), faces="zy")
    for side in (-1, 1):
        sc.run_z(side * (X - 0.03), 0.055, BZ, -1.05, (0.06, 0.11),
                 (80, 70, 68), faces="xy")
    for i, (half, near) in enumerate(((1.2, -1.3), (0.88, -1.7))):
        sc.quad_y(-half, half, BZ + 0.1, near, 0.118 + i * 0.004,
                  (78 + i * 16, 66 + i * 13, 60 + i * 11))

    # Mic stand, a little stage-left of centre.
    sc.tube((-0.15, 0.15, -2.45), (-0.15, 1.42, -2.45), 0.027, 0.021,
            (120, 122, 128), 6)
    sc.revolve((-0.15, 1.42, -2.45), [(0.0, 0.035), (0.09, 0.055),
                                      (0.16, 0.038)], (36, 36, 42), 6)
    sc.revolve((-0.15, 0.11, -2.45), [(0.0, 0.21), (0.04, 0.19)],
               (52, 52, 60), 7)
    # Stool: a round seat on splayed legs.
    sc.revolve((0.72, 0.60, -2.55), [(0.0, 0.21), (0.05, 0.23), (0.08, 0.22)],
               (120, 80, 58), 7)
    for ddx, ddz in ((-0.15, -0.15), (0.15, -0.15), (-0.15, 0.15), (0.15, 0.15)):
        sc.tube((0.72 + ddx * 0.8, 0.60, -2.55 + ddz * 0.8),
                (0.72 + ddx * 1.3, 0.11, -2.55 + ddz * 1.3),
                0.027, 0.021, (90, 60, 44), 5)

    # The room. Two tables with a candle each and four heads in silhouette,
    # between the lens and the stage -- the shot a comedy club is always
    # shot from. Nothing says depth like something out of focus-near being
    # dark and enormous while the subject is small and lit.
    for hx, hz, hh in ((-0.95, -0.35, 0.44), (0.80, -0.20, 0.40),
                       (-0.30, 0.35, 0.47), (1.05, 0.55, 0.43)):
        sc.revolve((hx, hh, hz), [(0.0, 0.105), (0.10, 0.118), (0.19, 0.075)],
                   (18, 14, 18), 7)
        sc.revolve((hx, hh - 0.30, hz), [(0.0, 0.30), (0.26, 0.22),
                                         (0.32, 0.12)], (14, 11, 15), 7,
                   caps="")
    for tx, tz in ((-0.62, 0.05), (0.95, 0.20)):
        sc.revolve((tx, 0.62, tz), [(0.0, 0.30), (0.03, 0.30)],
                   (40, 30, 34), 8)
        sc.tube((tx, 0.0, tz), (tx, 0.62, tz), 0.05, 0.04, (30, 24, 28), 5)
        sc.tube((tx, 0.65, tz), (tx, 0.74, tz), 0.025, 0.022,
                (210, 160, 80), 5, caps="t")


# -- the coffee shop ---------------------------------------------------------


def _diner(sc: r3d.Scene, t: float, seed: int) -> None:
    rng = random.Random(seed ^ int(t / 9))
    X, CY, BZ = 1.95, 2.42, -3.40
    dark = night(seed)
    back = (150, 126, 92) if dark else (196, 166, 120)
    side = (138, 116, 84) if dark else (180, 150, 108)
    street = (26, 30, 46) if dark else (54, 62, 80)
    pale = (176, 172, 160) if dark else (206, 200, 186)
    sc.layer = 0
    sc.quad_z(-X, X, 0.0, CY, BZ, back)
    sc.plane_x(BZ, 3.0, 0.0, CY, -X, side)
    sc.plane_x(BZ, 3.0, 0.0, CY, X, side)
    sc.plane_y(-X, X, BZ, 1.4, CY, (212, 196, 170))

    # Chequerboard floor: one pale quad, then only the dark squares on top.
    # Half the polygons of drawing both colours, which buys tiles small
    # enough to read as a floor rather than as a chessboard.
    sc.plane_y(-X, X, BZ, 2.6, 0.0, pale)
    cell = 0.52
    nx = int(2 * X / cell) + 1
    nz = int(6.4 / cell) + 1
    for ix in range(nx):
        for iz in range(nz):
            if (ix + iz) % 2:
                continue
            x = -X + ix * cell
            z = BZ + iz * cell
            sc.quad_y(x, min(x + cell, X), z, min(z + cell, 3.0), 0.004,
                      (70, 78, 92))
    sc.layer = 1

    # Cornice and a chair rail, both walls, both running in z.
    for side in (-1, 1):
        sc.run_z(side * (X - 0.05), CY - 0.08, BZ, 2.4, (0.10, 0.16),
                 (222, 212, 192), faces="xy")
        sc.run_z(side * (X - 0.04), 0.84, BZ, 2.4, (0.08, 0.07),
                 (150, 118, 86), faces="xy")

    # Counter along the back, with round stools on chrome posts.
    sc.frustum((0.0, 0.52, -2.80), (3.5, 0.60), (3.5, 0.72), 1.04,
               (108, 70, 52), caps="")
    sc.frustum((0.0, 1.07, -2.80), (3.7, 0.84), (3.66, 0.80), 0.08,
               (198, 190, 172))
    for i in range(4):
        x = -1.3 + i * 0.86
        sc.tube((x, 0.0, -2.12), (x, 0.66, -2.12), 0.048, 0.038,
                (150, 152, 160), 6)
        sc.revolve((x, 0.66, -2.12), [(0.0, 0.17), (0.06, 0.19),
                                      (0.10, 0.175)], (168, 62, 54), 7)

    # Booths down the left wall at three depths. This is the set's spine:
    # three identical objects at three distances, which is perspective shown
    # rather than implied, and it is what the room was missing.
    for i in range(3):
        z = -2.45 + i * 1.08
        sc.frustum((-X + 0.58, 0.42, z), (1.04, 0.90), (1.08, 0.94), 0.12,
                   (150, 70, 60))
        sc.frustum((-X + 0.10, 0.86, z), (0.14, 1.00), (0.12, 1.00), 0.88,
                   (160, 76, 64), caps="t")
        sc.box((-X + 0.72, 0.72, z), (0.80, 0.05, 0.86), (206, 198, 180))
        sc.tube((-X + 0.72, 0.0, z), (-X + 0.72, 0.70, z), 0.05, 0.04,
                (120, 118, 112), 5)

    # Street window above the booths, mullions at regular depth.
    sc.plane_x(-2.90, 0.60, 1.10, 2.10, -X + 0.04, street)
    for i in range(5):
        z = -2.90 + i * 0.70
        sc.box((-X + 0.07, 1.60, z), (0.05, 1.04, 0.06), (214, 206, 186),
               faces="xz")
        if rng.random() < (0.8 if dark else 0.55):
            sc.quad_x(z + 0.08, z + 0.56, 1.28, 1.82, -X + 0.05,
                      (236, 128, 96) if dark else (226, 186, 100))
    sc.run_z(-X + 0.09, 1.08, -2.90, 0.60, (0.09, 0.07), (214, 206, 186),
             faces="xy")

    # Pendant lamps over the counter at three depths, swinging slightly.
    for i, z in enumerate((-2.50, -1.55, -0.60)):
        x = -0.70 + i * 0.62
        sway = math.sin(t * 0.6 + i) * 0.035
        sc.tube((x, CY, z), (x + sway, CY - 0.44, z), 0.016, 0.016,
                (50, 46, 44), 4)
        sc.revolve((x + sway, CY - 0.60, z),
                   [(0.0, 0.17), (0.10, 0.11), (0.16, 0.055)],
                   (248, 212, 130) if dark else (214, 180, 104), 8, caps="b")

    # A table out in the room and well off to one side, so something is
    # big and near without standing in front of whoever is talking.
    sc.revolve((1.52, 0.70, -0.45), [(0.0, 0.42), (0.04, 0.42)],
               (182, 160, 130), 9)
    sc.tube((1.52, 0.0, -0.45), (1.52, 0.70, -0.45), 0.07, 0.05,
            (110, 104, 96), 6)
    sc.revolve((1.52, 0.72, -0.45), [(0.0, 0.055), (0.09, 0.062)],
               (226, 222, 210), 7)


# -- the hallway -------------------------------------------------------------


def _hallway(sc: r3d.Scene, t: float, seed: int) -> None:
    X, CY, BZ = 1.42, 2.40, -7.2
    # Half the strips out, which in a corridor this long changes the whole
    # feel of it without moving a single wall.
    dim = night(seed)
    sc.layer = 0
    _boards(sc, -X, X, BZ, 2.6, 0.0, (120, 86, 60), (86, 60, 42), step=0.40)
    sc.plane_y(-X, X, BZ, 1.4, CY, (214, 192, 158))
    sc.plane_x(BZ, 3.0, 0.0, CY, -X, (176, 148, 114))
    sc.plane_x(BZ, 3.0, 0.0, CY, X, (176, 148, 114))
    sc.quad_z(-X, X, 0.0, CY, BZ, (150, 124, 94))
    sc.layer = 1

    # A runner down the middle of the floor, and trim top and bottom on
    # both walls: four more lines converging on the vanishing point.
    sc.quad_y(-0.62, 0.62, BZ + 0.2, 2.6, 0.015, (112, 70, 66))
    sc.quad_y(-0.54, 0.54, BZ + 0.3, 2.5, 0.019, (142, 96, 88))
    for side in (-1, 1):
        sc.run_z(side * (X - 0.04), 0.08, BZ, 2.6, (0.08, 0.16),
                 (96, 70, 54), faces="xy")
        sc.run_z(side * (X - 0.05), CY - 0.08, BZ, 2.6, (0.10, 0.16),
                 (198, 180, 150), faces="xy")
        sc.run_z(side * (X - 0.04), 1.02, BZ, 2.6, (0.07, 0.06),
                 (140, 112, 84), faces="xy")

    # Doors receding down both walls, with architraves that stand proud.
    for i in range(5):
        z = -0.9 - i * 1.45
        for side in (-1, 1):
            sc.quad_x(z - 0.42, z + 0.42, 0.0, 2.02, side * (X - 0.03),
                      (104, 68, 50))
            sc.box((side * (X - 0.08), 1.01, z), (0.08, 2.14, 0.98),
                   (176, 152, 122), faces="xy")
            sc.quad_x(z - 0.36, z + 0.36, 0.1, 1.94, side * (X - 0.10),
                      (118, 78, 56))
            sc.tube((side * (X - 0.14), 0.98, z + side * 0.28),
                    (side * (X - 0.22), 0.98, z + side * 0.28),
                    0.035, 0.03, (206, 172, 70), 5, caps="t")

    # Strip lights in the ceiling.
    for i in range(4):
        dead = dim and i % 2 == 1
        if dead or (int(t * 6) % 23 == 0 and i == 1):
            z = -1.3 - i * 1.7
            sc.box((0.0, CY - 0.06, z), (0.46, 0.07, 0.62), (96, 92, 84),
                   faces="xz")
            continue
        z = -1.3 - i * 1.7
        sc.quad_y(-0.36, 0.36, z - 0.26, z + 0.26, CY - 0.02,
                  (208, 202, 176) if dim else (236, 228, 196))
        sc.box((0.0, CY - 0.06, z), (0.46, 0.07, 0.62), (170, 164, 150),
               faces="xz")

    # A radiator under the near door on the left: something solid at a known
    # size, which is what tells the eye how long the corridor is.
    for i in range(5):
        sc.tube((-X + 0.14, 0.12, -2.6 + i * 0.09),
                (-X + 0.14, 0.62, -2.6 + i * 0.09),
                0.035, 0.035, (196, 186, 170), 5)


_SETS = {
    "APARTMENT": _apartment,
    "STANDUP": _standup,
    "DINER": _diner,
    "HALLWAY": _hallway,
}


def build_set(scene: r3d.Scene, name: str, t: float, seed: int) -> None:
    _SETS.get(name, _apartment)(scene, t, seed)


# -- cameras -----------------------------------------------------------------

# Per-set framings. The wide establishes the room; the medium is a cut to
# whoever is talking; the close is a cut to their face. Everything is a cut,
# never a move, which is both cheaper and closer to how the reference shoots.
_WIDE = {
    "APARTMENT": ((0.52, 1.50, 2.35), (-0.10, 1.26, -2.90), 46.0),
    "STANDUP": ((0.30, 1.52, 2.45), (-0.10, 1.26, -2.90), 46.0),
    "DINER": ((0.52, 1.50, 2.25), (-0.10, 1.26, -2.80), 46.0),
    "HALLWAY": ((0.22, 1.48, 2.10), (0.0, 1.22, -5.4), 46.0),
}


def wide_camera(set_name: str, size) -> r3d.Camera:
    pos, target, fov = _WIDE.get(set_name, _WIDE["APARTMENT"])
    return r3d.Camera(pos, target, size, fov)


def medium_camera(set_name: str, size, at, side: float = 1.0) -> r3d.Camera:
    """A closer framing on a mark, from off to one side."""
    x, z = at
    # Aimed at chest height, which puts faces in the upper third where a
    # television camera would put them.
    pos = (x + 0.72 * side, 1.52, z + 2.25)
    target = (x, 1.30, z)
    return r3d.Camera(pos, target, size, 40.0)


def close_camera(set_name: str, size, at, height: float,
                 side: float = 1.0) -> r3d.Camera:
    """A cut to somebody's face.

    The medium puts a head about sixteen pixels tall, which is enough for a
    silhouette and not enough for a face. This comes in to a little over a
    metre and three-quarters and frames eyeline to collar, which is where
    the brow, the nose and the set of the jaw actually get to do something.
    Shot from slightly below, as a sitcom does.
    """
    x, z = at
    eye = height * 0.93
    pos = (x + 0.58 * side, eye - 0.10, z + 1.42)
    target = (x, eye - 0.02, z)
    return r3d.Camera(pos, target, size, 38.0)
