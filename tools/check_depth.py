"""Check that the sets still have the depth they are supposed to have.

Run it after touching anything in render/world.py or render/r3d.py; it exits
non-zero and says what broke. Nothing reads this at runtime.

Three things are checked, all of which were broken and none of which is
visible from reading the code:

1. **No holes.** A face that crosses the near plane is dropped whole rather
   than clipped, so any floor, ceiling or wall that runs from the back of
   the room to past the lens simply is not drawn. That put a band of raw
   void across the bottom of every wide and down the edge of the hallway.
   Every room-shaped set must fill its frame from every camera it uses.

2. **Perspective is actually in shot.** A room reads flat unless something
   in it converges. Counted as: distinct depths among the faces that get
   drawn, and the presence of geometry at the left and right edges of the
   frame -- the side walls used to be outside the frustum entirely.

3. **Solids are wound outward.** Getting this backwards does not look like
   a hole, it looks like the inside of things drawing over the outside, so
   nothing short of a winding check catches it.
"""
import os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame; pygame.init()
import numpy as np
from pixelmeme.render import r3d, world, actors as A

fails = []
VOID = (26, 24, 30)
W, H = 108, 149
ROOMS = ("APARTMENT", "DINER", "HALLWAY")   # STANDUP is a dark club by design

# -- 1. winding ---------------------------------------------------------------
for name, build in (
    ("box", lambda sc: sc.box((0, 0, 0), (1, 1, 1), (200, 100, 100))),
    ("frustum", lambda sc: sc.frustum((0, 0, 0), (1, 1), (0.6, 0.6), 1.0,
                                      (200, 100, 100))),
    ("tube", lambda sc: sc.tube((0, -0.5, 0), (0, 0.5, 0), 0.5, 0.3,
                                (200, 100, 100), 8, caps="tb")),
    ("revolve", lambda sc: sc.revolve((0, -0.5, 0),
                                      [(0, 0.5), (0.5, 0.55), (1.0, 0.2)],
                                      (200, 100, 100), 8)),
    ("revolve/squash", lambda sc: sc.revolve((0, -0.5, 0),
                                             [(0, 0.4), (0.6, 0.5), (1.0, 0.1)],
                                             (200, 100, 100), 9, roll=0.4,
                                             squash=1.3)),
):
    sc = r3d.Scene(); build(sc)
    V = np.asarray(sc.verts, np.float32); F = np.asarray(sc.faces, np.int32)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    outward = np.einsum("ij,ij->i", n, V[F].mean(axis=1) - V.mean(axis=0))
    bad = int((outward < 0).sum())
    print(f"{name:16s} {len(F):3d} faces, {bad} wound inward")
    if bad:
        fails.append(f"{name}: {bad}/{len(F)} faces wound inward")

# -- 2 & 3. every camera on every set ------------------------------------------
cast = A.make_sitcom_cast(1)


def shoot(name, cam):
    sc = r3d.Scene()
    world.build_set(sc, name, 3.0, 5)
    marks = world.marks_for(name)
    for k, act in enumerate(cast):
        act.x, act.z = marks[k % len(marks)]
        act.state = A.IDLE
        act.yaw = 0.3 * (1 if k % 2 else -1)
        act.build_geometry(sc, 1)
    s = pygame.Surface((W, H)); s.fill(VOID)
    drawn = r3d.render(sc, s, cam, ambient=0.46, fog=world.fog_for(name))
    arr = np.asarray(pygame.surfarray.pixels3d(s)).copy()
    return arr, drawn


for name in world.SET_NAMES:
    marks = world.marks_for(name)
    cams = [("wide", world.wide_camera(name, (W, H)))]
    for i, m in enumerate(marks):
        cams.append((f"med{i}", world.medium_camera(name, (W, H), m,
                                                    1.0 if i % 2 else -1.0)))
        cams.append((f"close{i}", world.close_camera(name, (W, H), m,
                                                     cast[i].height,
                                                     1.0 if i % 2 else -1.0)))
    for tag, cam in cams:
        arr, drawn = shoot(name, cam)
        void = (arr == np.asarray(VOID)).all(axis=2)
        frac = void.mean()
        # Columns at the extreme edges: the side walls used to fall outside
        # the frustum altogether, so the room had no corner in any shot.
        edge = void[:3, :].mean() + void[-3:, :].mean()
        if frac > 0.02:
            fails.append(f"{name}/{tag}: {100*frac:.1f}% of the frame is void")
        if name in ROOMS and edge > 0.10:
            fails.append(f"{name}/{tag}: {100*edge/2:.0f}% of the frame edge "
                         f"is void -- no wall in shot")
        if drawn < 120:
            fails.append(f"{name}/{tag}: only {drawn} polygons drawn")
    print(f"{name:10s} {len(cams)} framings checked")

# -- depth spread: how much of the frame is at a different distance -----------
for name in world.SET_NAMES:
    sc = r3d.Scene()
    world.build_set(sc, name, 3.0, 5)
    cam = world.wide_camera(name, (W, H))
    V = np.asarray(sc.verts, np.float32)
    Vc = (V - cam.pos) @ cam.basis
    d = -Vc[:, 2]
    d = d[(d > 0.1) & (d < 20)]
    # Faces spread over depth, not all pinned to one plane at the back.
    spread = float(np.percentile(d, 90) - np.percentile(d, 10))
    print(f"{name:10s} geometry spans {spread:.1f} m of depth")
    if spread < 2.5:
        fails.append(f"{name}: only {spread:.1f} m of depth in the set")

print("FAIL" if fails else "PASS")
for f in fails:
    print("  ", f)
sys.exit(1 if fails else 0)
