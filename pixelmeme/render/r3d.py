"""A tiny software 3D renderer.

Flat-shaded convex polygons, a perspective camera, and the painter's
algorithm. No z-buffer, no textures, no shading beyond three lambert terms.
That is not a limitation to work around -- it is close to the renderer a
1996 console had, and it is why the output looks the way it does.

Everything a frame needs goes into a `Scene` as polygons in world space, then
`render()` transforms the whole vertex buffer in one numpy matmul, sorts the
faces back-to-front, and fills them. Six or seven hundred quads a frame at
108x192, which is nothing.

Solids (boxes, frusta, tubes, revolves) are wound outward and marked
cullable, so their back faces are dropped before anything is filled -- about
half of every closed shape, which is what pays for the extra geometry.
Single-sided surfaces (a floor, a wall, an open band of hair) are not
marked, are never culled, and so cannot vanish when seen from behind. Face
normals are flipped toward the camera before shading either way, so lighting
is correct regardless of winding.

Winding still has to be right, and getting it wrong does not look like a
hole. Culling then keeps the far side of a solid and drops the near side, so
whatever is inside draws over whatever is outside: a skull drew straight
through the hair covering it. `tools/check_depth.py` checks every primitive.

Two things exist here purely to give a flat-shaded room a sense of volume.
Lighting is a key, a fill and a floor bounce, because with a single key
every plane turned away from it lands on the same flat ambient and the
corner where two walls meet disappears. And `render` takes a `fog`, which
mixes faces toward a colour with distance, because without it the far wall
is lit exactly as brightly as the chair in front of it.

Anything that spans the room in depth -- a floor, a ceiling, a side wall, a
cornice -- must be emitted in bands (`plane_y`, `plane_x`, `run_z`). A face
that crosses the near plane is dropped whole rather than clipped, so a
single quad running past the lens is not drawn at all.
"""

from __future__ import annotations

import math

import numpy as np
import pygame


class Camera:
    def __init__(
        self,
        pos,
        target,
        size: tuple[int, int],
        fov_deg: float = 60.0,
        up=(0.0, 1.0, 0.0),
        near: float = 0.08,
    ):
        self.pos = np.asarray(pos, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        self.near = near

        forward = target - self.pos
        forward /= max(1e-6, float(np.linalg.norm(forward)))
        right = np.cross(forward, np.asarray(up, dtype=np.float32))
        right /= max(1e-6, float(np.linalg.norm(right)))
        true_up = np.cross(right, forward)

        # Columns, so world->camera is a single `(V - pos) @ basis`.
        self.basis = np.stack([right, true_up, -forward], axis=1).astype(np.float32)

        w, h = size
        self.cx = w * 0.5
        self.cy = h * 0.5
        self.focal = (h * 0.5) / math.tan(math.radians(fov_deg) * 0.5)

    def lerp(self, other: "Camera", t: float, size, fov_deg) -> "Camera":
        """Blend two cameras, for a move that is not a hard cut."""
        t = max(0.0, min(1.0, t))
        pos = self.pos * (1 - t) + other.pos * t
        # Reconstruct a target a unit ahead of each camera and blend that.
        f_a = -self.basis[:, 2]
        f_b = -other.basis[:, 2]
        fwd = f_a * (1 - t) + f_b * t
        return Camera(pos, pos + fwd, size, fov_deg)


class Scene:
    """A frame's worth of polygons, in world space."""

    __slots__ = ("verts", "faces", "colours", "layers", "culls", "layer",
                 "_n", "_yaw")

    def __init__(self):
        self.verts: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int, int]] = []
        self.colours: list[tuple[int, int, int]] = []
        self.layers: list[int] = []
        # Per face: may this one be dropped when it faces away? True only for
        # faces that belong to a closed solid, where something else is always
        # in front of them.
        self.culls: list[bool] = []
        # Painter's sorting is per-layer. A floor is one enormous polygon
        # whose centroid can easily be nearer the camera than a person
        # standing on it, which draws the floor over their legs. Putting the
        # room shell in layer 0 and everything in it in layer 1 fixes that
        # class of bug outright, without subdividing anything.
        self.layer = 1
        self._n = 0
        # An optional yaw about a vertical axis, applied to every point that
        # goes in while it is set. See `set_yaw`.
        self._yaw: tuple[float, float, float, float] | None = None

    def set_yaw(self, angle: float = 0.0, origin=(0.0, 0.0, 0.0)) -> None:
        """Turn everything emitted from here on about a vertical axis.

        Every primitive in this file is built square to the world axes, so
        without this nothing can face anywhere but straight down the lens.
        Four people standing in a row all facing the camera is a police
        lineup, not a scene: they never turn to each other, the heads only
        ever show one facet, and no amount of geometry on a face reads as
        three-dimensional when the face is never seen from an angle.

        One cosine and one sine per point, only while a yaw is set, is a
        great deal cheaper than giving every primitive its own basis.
        """
        if not angle:
            self._yaw = None
        else:
            self._yaw = (math.cos(angle), math.sin(angle), origin[0], origin[2])

    def poly(self, points, colour, cull: bool = False) -> None:
        """A convex polygon of 3 or 4 points.

        Winding does not matter for shading. It matters for `cull`: pass True
        only from a closed solid whose faces are wound outward.
        """
        if self._yaw is not None:
            c, s, ox, oz = self._yaw
            points = [(ox + (px - ox) * c + (pz - oz) * s, py,
                       oz - (px - ox) * s + (pz - oz) * c)
                      for px, py, pz in points]
        n = len(points)
        self.verts.extend(points)
        base = self._n
        if n == 3:
            # Pad to a quad by repeating the last vertex; the degenerate edge
            # costs nothing and keeps the face array rectangular.
            self.faces.append((base, base + 1, base + 2, base + 2))
        else:
            self.faces.append((base, base + 1, base + 2, base + 3))
        self.colours.append(colour)
        self.layers.append(self.layer)
        self.culls.append(cull)
        self._n += n

    def box(self, center, size, colour, faces: str = "xyz") -> None:
        """An axis-aligned box, wound outward. `faces` picks which axis pairs
        to emit -- drop the ones that are never visible and the frame gets
        cheaper still."""
        cx, cy, cz = center
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        z0, z1 = cz - hz, cz + hz

        if "z" in faces:
            self.poly([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                      colour, True)
            self.poly([(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)],
                      colour, True)
        if "x" in faces:
            self.poly([(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
                      colour, True)
            self.poly([(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)],
                      colour, True)
        if "y" in faces:
            self.poly([(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)],
                      colour, True)
            self.poly([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                      colour, True)

    # -- shapes that are not boxes -------------------------------------------

    def frustum(self, center, lower, upper, height, colour,
                lean: float = 0.0, caps: str = "tb") -> None:
        """A box with a different footprint at the top than at the bottom.

        This is the workhorse for anything that is wider at one end: a torso
        above a waist, a skirt, a lampshade, a jaw under a skull. Costs
        exactly what a box costs and stops everything reading as a crate.

        `lower` and `upper` are (width, depth) pairs; `lean` shifts the top
        along z, which is what tips a head forward or a chair back.
        """
        cx, cy, cz = center
        hy = height * 0.5
        y0, y1 = cy - hy, cy + hy
        lx, lz = lower[0] * 0.5, lower[1] * 0.5
        ux, uz = upper[0] * 0.5, upper[1] * 0.5
        tz = cz + lean

        # Bottom ring then top ring, both counter-clockwise seen from above.
        b = [(cx - lx, y0, cz + lz), (cx + lx, y0, cz + lz),
             (cx + lx, y0, cz - lz), (cx - lx, y0, cz - lz)]
        t = [(cx - ux, y1, tz + uz), (cx + ux, y1, tz + uz),
             (cx + ux, y1, tz - uz), (cx - ux, y1, tz - uz)]
        for i in range(4):
            j = (i + 1) % 4
            self.poly([b[i], b[j], t[j], t[i]], colour, True)
        if "t" in caps:
            self.poly(t, colour, True)
        if "b" in caps:
            self.poly(b[::-1], colour, True)

    def tube(self, p0, p1, r0, r1, colour, sides: int = 6,
             caps: str = "", roll: float = 0.0) -> None:
        """A tapered n-sided prism from p0 to p1.

        Limbs are tubes. At six sides a forearm reads as round rather than as
        a plank, for two more polygons than a box, and it can point in any
        direction -- which is the thing boxes cannot do and the reason
        everything built out of them looks like a crate stack.

        Caps are off by default: a limb's ends are inside the body and inside
        the hand, and two polygons each is not worth paying for.
        """
        ax, ay, az = p0
        bx, by, bz = p1
        dx, dy, dz = bx - ax, by - ay, bz - az
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            return
        dx, dy, dz = dx / length, dy / length, dz / length

        # Any vector not parallel to the axis will do to start the frame.
        if abs(dy) < 0.9:
            ux, uy, uz = -dz, 0.0, dx
        else:
            ux, uy, uz = 1.0, 0.0, 0.0
        un = math.sqrt(ux * ux + uy * uy + uz * uz)
        ux, uy, uz = ux / un, uy / un, uz / un
        # v = d x u, so (u, v, d) is right-handed and the rings below run
        # counter-clockwise seen from the far end -- which is what makes the
        # side quads come out wound outward.
        vx = dy * uz - dz * uy
        vy = dz * ux - dx * uz
        vz = dx * uy - dy * ux

        ring0, ring1 = [], []
        for i in range(sides):
            a = math.tau * i / sides + roll
            ca, sa = math.cos(a), math.sin(a)
            ox, oy, oz = ux * ca + vx * sa, uy * ca + vy * sa, uz * ca + vz * sa
            ring0.append((ax + ox * r0, ay + oy * r0, az + oz * r0))
            ring1.append((bx + ox * r1, by + oy * r1, bz + oz * r1))

        for i in range(sides):
            j = (i + 1) % sides
            self.poly([ring0[i], ring0[j], ring1[j], ring1[i]], colour, True)
        if "t" in caps:
            self._fan(ring1, colour)
        if "b" in caps:
            self._fan(ring0[::-1], colour)

    def _fan(self, ring, colour) -> None:
        """An n-gon as quads from one corner: three polygons for a hexagon
        instead of six triangles."""
        for k in range(1, len(ring) - 2, 2):
            quad = [ring[0], ring[k], ring[k + 1], ring[min(k + 2, len(ring) - 1)]]
            self.poly(quad, colour, True)
        if len(ring) % 2 == 0 and len(ring) > 3:
            self.poly([ring[0], ring[-2], ring[-1], ring[-1]], colour, True)

    def revolve(self, base, profile, colour, sides: int = 8,
                roll: float = 0.0, caps: str = "t",
                squash: float = 1.0, lean=(0.0, 0.0), arc=None) -> None:
        """A surface of revolution about the y axis through `base`.

        `profile` is a list of (height above base, radius) going upward. A
        skull is a dome, a shoulder is a dome, a lampshade is two points, a
        mug is three. This is the primitive that stops a round thing being a
        cylinder with a lid: a head built as one tube has a flat top and a
        hard rim, and at this resolution the rim is the first thing that
        says "box with the corners knocked off".

        `squash` scales the z radius against the x radius, so a head can be
        narrower front-to-back than it is wide. `lean` shifts the top of the
        stack in (x, z), proportionally with height, which tips a skull
        forward or sweeps a hairline back.

        `arc` is a (start, end) pair in radians, for a band that does not
        close: a horseshoe of hair around the back of a bald head, a curtain
        hanging past the ears. An open band is a surface, not a solid, so it
        is never culled -- otherwise it disappears the moment the head turns.
        """
        bx, by, bz = base
        top = max(1e-6, profile[-1][0])
        closed = arc is None
        if closed:
            angles = [math.tau * i / sides + roll for i in range(sides)]
        else:
            a0, a1 = arc
            angles = [a0 + (a1 - a0) * i / sides for i in range(sides + 1)]
        trig = [(math.cos(a), math.sin(a)) for a in angles]

        rings = []
        for hy, r in profile:
            k = hy / top
            cx, cz = bx + lean[0] * k, bz + lean[1] * k
            if r <= 1e-5:
                rings.append([(cx, by + hy, cz)] * len(trig))
                continue
            rings.append([(cx + ca * r, by + hy, cz + sa * r * squash)
                          for ca, sa in trig])

        # Wound outward. The rings run clockwise seen from above -- the
        # opposite hand to `tube`, whose frame is built from a cross product
        # -- so the quads are emitted in reverse and the caps swap. Getting
        # this backwards does not look like a hole: back-face culling then
        # keeps the far side of every dome and drops the near side, so a
        # head drew over the hair covering it and everybody went bald.
        span = len(trig) if closed else len(trig) - 1
        for lo, hi in zip(rings, rings[1:]):
            for i in range(span):
                j = (i + 1) % len(trig)
                if lo[i] == lo[j]:                  # degenerate bottom point
                    self.poly([lo[i], hi[i], hi[j]], colour, closed)
                elif hi[i] == hi[j]:                # degenerate top point
                    self.poly([lo[j], lo[i], hi[i]], colour, closed)
                else:
                    self.poly([lo[j], lo[i], hi[i], hi[j]], colour, closed)
        if closed and "t" in caps and rings[-1][0] != rings[-1][1]:
            self._fan(rings[-1][::-1], colour)
        if closed and "b" in caps and rings[0][0] != rings[0][1]:
            self._fan(rings[0], colour)

    def run_z(self, x, y, z0, z1, size, colour, faces: str = "xy",
              step: float = 1.4) -> None:
        """A length of something running away from the camera: a cornice, a
        baseboard, the lip of a stage.

        These are the cheapest depth in the whole renderer. A room reads as
        a flat backdrop until something in it converges, and three boxes
        running in z do more for the sense of a volume than fifty polygons
        of furniture standing parallel to the lens.

        Cut into lengths, because a single box spanning the whole room has
        every one of its faces crossing the near plane once the lens is
        alongside it, and the renderer drops those whole: the cornice simply
        was not there in any shot taken from inside the room.
        """
        n = max(1, int(math.ceil((z1 - z0) / step)))
        dz = (z1 - z0) / n
        for j in range(n):
            self.box((x, y, z0 + (j + 0.5) * dz), (size[0], size[1], dz),
                     colour, faces=faces)

    def plane_x(self, z0, z1, y0, y1, x, colour, step: float = 1.6) -> None:
        """A side wall, cut into lengths across depth.

        Same reason as `plane_y`: a wall that runs from the back of the room
        to behind the lens crosses the near plane and vanishes entirely,
        taking the corner of the room with it and leaving raw void down the
        edge of the frame.
        """
        n = max(1, int(math.ceil((z1 - z0) / step)))
        dz = (z1 - z0) / n
        for j in range(n):
            az = z0 + j * dz
            self.quad_x(az, az + dz, y0, y1, x, colour)

    def wedge(self, center, size, colour, point: float = 1.0) -> None:
        """A triangular prism lying along z: a nose, a shoe toe, a roof.

        `point` is how far the apex sits above the base, as a fraction of the
        height; the apex runs along the +z edge.
        """
        cx, cy, cz = center
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy * point
        z0, z1 = cz - hz, cz + hz
        apex0 = (cx, y1, z1)
        self.poly([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                  colour, True)                                   # base
        self.poly([(x0, y0, z1), (x1, y0, z1), apex0], colour, True)
        self.poly([(x1, y0, z0), (x0, y0, z0), (cx, y1, z0)], colour, True)
        self.poly([(x1, y0, z1), (x1, y0, z0), (cx, y1, z0), apex0], colour, True)
        self.poly([(x0, y0, z0), (x0, y0, z1), apex0, (cx, y1, z0)], colour, True)

    def quad_y(self, x0, x1, z0, z1, y, colour) -> None:
        """A horizontal slab face -- counter tops, light pools."""
        self.poly([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], colour)

    def plane_y(self, x0, x1, z0, z1, y, colour, step: float = 1.4) -> None:
        """A floor or ceiling, cut into bands across depth.

        Faces that cross the near plane are dropped whole rather than
        clipped, and a floor or a ceiling always extends past the camera --
        as one giant quad it vanishes completely the moment the lens is
        over it, which puts a band of raw void across the bottom or the top
        of the frame. Banding costs only the strip the lens is actually
        standing in.

        Banded in z only. Cutting in x as well doubled the polygon count to
        solve a problem that does not exist: no camera here is ever outside
        the room sideways.
        """
        n = max(1, int(math.ceil((z1 - z0) / step)))
        dz = (z1 - z0) / n
        for j in range(n):
            az = z0 + j * dz
            self.poly([(x0, y, az), (x1, y, az),
                       (x1, y, az + dz), (x0, y, az + dz)], colour)

    def quad_z(self, x0, x1, y0, y1, z, colour) -> None:
        """A wall face facing along z."""
        self.poly([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], colour)

    def quad_x(self, z0, z1, y0, y1, x, colour) -> None:
        self.poly([(x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y1, z0)], colour)

    def ring(self, center, radius, thickness, colour, segments: int = 10) -> None:
        """A flat annulus in the x/y plane -- a bicycle wheel, at this
        resolution, is a dark ring and nothing else."""
        cx, cy, cz = center
        inner = max(0.01, radius - thickness)
        for i in range(segments):
            a0 = math.tau * i / segments
            a1 = math.tau * (i + 1) / segments
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            self.poly(
                [
                    (cx + c0 * inner, cy + s0 * inner, cz),
                    (cx + c1 * inner, cy + s1 * inner, cz),
                    (cx + c1 * radius, cy + s1 * radius, cz),
                    (cx + c0 * radius, cy + s0 * radius, cz),
                ],
                colour,
            )


# Keyed from the front-left and well off vertical. Lighting almost
# straight down gives a vertical face and a side face nearly the same
# lambert term, which is why a tapered solid used to read as a box:
# the taper was there and the shading never showed it.
_LIGHT = np.array([0.62, 0.55, 0.56], dtype=np.float32)
_LIGHT /= np.linalg.norm(_LIGHT)

# A dim fill from the opposite side, low and behind. With a single key,
# every plane turned away from it lands on flat ambient -- so a back wall
# and the side wall meeting it came out the same value and the corner of
# the room disappeared. The fill is weak enough not to flatten the key and
# strong enough that no two perpendicular planes share a colour.
_FILL = np.array([-0.70, 0.18, -0.68], dtype=np.float32)
_FILL /= np.linalg.norm(_FILL)

# And a bounce off the floor, which is the only thing that lights a ceiling.
# With a key and a fill both above the horizon, every downward-facing plane
# landed on flat ambient: ceilings came out as a band of near-black across
# the top of every frame, and nothing had an underside.
_BOUNCE = np.array([0.06, -0.99, 0.12], dtype=np.float32)
_BOUNCE /= np.linalg.norm(_BOUNCE)

_KEY_K, _FILL_K, _BOUNCE_K = 0.62, 0.22, 0.16


def render(scene: Scene, surface: pygame.Surface, cam: Camera,
           ambient: float = 0.52, light=None, fog=None) -> int:
    """Draw the scene. Returns the number of polygons actually filled.

    `fog` is `(colour, near, far, strength)`: faces are mixed toward the
    colour with distance. A room built of flat-shaded polygons has no
    aerial perspective at all -- the far wall is lit exactly as brightly as
    the chair in front of it -- and without it the set reads as a painted
    backdrop however much geometry is standing in front of it. It is one
    numpy lerp, which is the cheapest depth available anywhere.
    """
    if not scene.faces:
        return 0

    V = np.asarray(scene.verts, dtype=np.float32)
    F = np.asarray(scene.faces, dtype=np.int32)
    C = np.asarray(scene.colours, dtype=np.float32)
    L = _LIGHT if light is None else np.asarray(light, dtype=np.float32)

    # World -> camera, all vertices in one go.
    Vc = (V - cam.pos) @ cam.basis
    depth = -Vc[:, 2]

    safe = np.maximum(depth, 1e-3)
    sx = cam.cx + cam.focal * Vc[:, 0] / safe
    sy = cam.cy - cam.focal * Vc[:, 1] / safe
    P = np.stack([sx, sy], axis=1)

    fd = depth[F]                       # (M, 4)
    # Anything crossing the near plane is dropped whole rather than clipped;
    # the camera never gets close enough to a wall for that to show.
    visible = (fd > cam.near).all(axis=1)
    if not visible.any():
        return 0

    face_depth = fd.mean(axis=1)

    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-6)

    # Point every normal at the camera before shading, so a polygon wound the
    # "wrong" way is lit the same as one wound the right way.
    to_cam = cam.pos - a
    flip = np.sign(np.einsum("ij,ij->i", normals, to_cam))
    flip[flip == 0] = 1.0
    normals *= flip[:, None]

    # A face of a closed solid that points away from the camera has another
    # face of the same solid in front of it, always. Dropping those halves
    # the fill cost of every box, frustum and tube in the frame, and the fill
    # loop is where the time goes.
    if scene.culls:
        cullable = np.asarray(scene.culls, dtype=bool)
        visible &= ~(cullable & (flip < 0))
        if not visible.any():
            return 0

    headroom = 1.0 - ambient
    lam = (ambient
           + headroom * _KEY_K * np.clip(normals @ L, 0.0, 1.0)
           + headroom * _FILL_K * np.clip(normals @ _FILL, 0.0, 1.0)
           + headroom * _BOUNCE_K * np.clip(normals @ _BOUNCE, 0.0, 1.0))
    lit = C * lam[:, None]

    if fog is not None:
        fog_colour, near, far, strength = fog
        k = np.clip((face_depth - near) / max(1e-3, far - near), 0.0, 1.0)
        k = (k * k * strength)[:, None]
        lit = lit * (1.0 - k) + np.asarray(fog_colour, np.float32) * k

    colours = np.clip(lit, 0, 255).astype(np.uint8)

    pts = np.round(P[F]).astype(np.int32)   # (M, 4, 2)

    # Far-to-near within each layer, layers in order.
    layers = np.asarray(scene.layers, dtype=np.int32)
    order = np.lexsort((-face_depth, layers))
    drawn = 0
    poly = pygame.draw.polygon
    for i in order:
        if not visible[i]:
            continue
        poly(surface, colours[i], pts[i])
        drawn += 1
    return drawn
