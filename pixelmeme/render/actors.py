"""The cast, as 3D geometry.

Nobody is a stack of boxes. A torso is a frustum, wide at the shoulders and
narrow at the waist; limbs are tubes running between joint positions, so they
taper and can point anywhere; a head is a revolved cranium over a revolved
jaw with a face built on the front of it. Between 180 and 334 polygons each
depending on how near the camera is, half of which the renderer culls as
back faces.

Three things carry most of the weight here:

**They turn.** Every actor is emitted under a yaw about their own vertical
axis, so they stand at an angle to the lens and face whoever is talking. Four
people square to the camera is a police lineup, and a head that only ever
shows one facet cannot read as a solid however many facets it has.

**The face is built, not drawn on.** A brow ridge, a nose and a pair of lips
that stand proud of the skull catch the key light and cast their own plane
changes. Painted flat on the front facet -- which is what a dark bar for a
mouth and two dark squares for eyes amount to -- they read as a decal on a
tube at any distance.

**Detail follows the camera.** `build_geometry` takes a detail level from
the stage, which knows how far away the lens is. The close-up gets ears,
sclera, nostrils, lids and a nine-sided skull; the far mark in a wide gets a
six-sided skull and a suggestion, for a bit over half the polygons.

There is still no skeleton and no interpolation: the pose is a handful of
joint positions computed from a phase clock, and the geometry is rebuilt
from scratch every frame. That is cheaper than it sounds and much simpler
than rigging.
"""

from __future__ import annotations

import math
import random

from . import r3d


def _tone(colour, k: float):
    """A material in its own shade. Shading by hand where a silhouette needs
    an edge survives the camera moving; relying on the lambert term does
    not."""
    return tuple(max(0, min(255, int(c * k))) for c in colour)


# What a body is doing. Pose is computed from these plus a phase clock.
IDLE, TALK, WALK, STARE, GESTURE, TWITCH = (
    "idle", "talk", "walk", "stare", "gesture", "twitch",
)

# Nobody turns further than this off the lens. Past about thirty degrees the
# face stops being readable at forty pixels tall, and a sitcom does not shoot
# the back of anybody's head anyway.
MAX_YAW = 0.55

# The four leads. Fixed, not randomised -- a recognisable recurring cast,
# for a personal, non-commercial build running on one device in one room.
#
# What makes each of them recognisable at this size is, in order: overall
# height, the shape on top of the head, and the one colour they own. Those
# three are deliberately as far apart as four people can be.
ARCHETYPES = (
    dict(
        name="JERRY", role="lead",
        height=1.80, build=0.405, girth=0.86,
        skin=(224, 182, 144), skin_warm=(236, 198, 160),
        shirt=(104, 150, 200), trousers=(62, 76, 114),
        hair=(46, 33, 27), hair_style="neat",
        shoes=(242, 240, 232), collar=(198, 210, 228),
        shirt_detail="placket", eye=(60, 70, 92),
    ),
    dict(
        name="GEORGE", role="neurotic",
        # Short and round rather than wide: a stocky build given as shoulder
        # width made his torso as broad as it was tall, which is a wardrobe.
        height=1.58, build=0.385, girth=1.14,
        skin=(218, 174, 134), skin_warm=(230, 188, 148),
        shirt=(172, 160, 104), trousers=(84, 74, 62),
        hair=(92, 70, 52), hair_style="horseshoe",
        glasses=True, jacket=(122, 106, 76),
        shoes=(68, 54, 46), collar=(192, 182, 130),
        eye=(58, 46, 36),
    ),
    dict(
        name="ELAINE", role="sharp",
        height=1.68, build=0.352, girth=0.84,
        skin=(234, 196, 160), skin_warm=(246, 210, 174),
        # A cream blouse under a plum vest, not one dark top: she was
        # reading as a silhouette with a haircut.
        shirt=(226, 214, 192), trousers=(44, 40, 50),
        jacket=(126, 72, 110), hair=(52, 34, 28), hair_style="curls",
        shoes=(54, 46, 50), collar=(238, 230, 214),
        skirt=True, eye=(64, 52, 40),
    ),
    dict(
        name="KRAMER", role="wildcard",
        height=1.97, build=0.372, girth=0.80,
        skin=(222, 180, 140), skin_warm=(234, 194, 152),
        # The loud shirt is the whole point of him, so nothing goes over it.
        shirt=(198, 116, 62), trousers=(58, 54, 62),
        hair=(40, 31, 30), hair_style="pompadour",
        shoes=(50, 44, 46), collar=(228, 158, 98),
        shirt_detail="bands", band=(96, 112, 140), eye=(52, 44, 44),
    ),
)


class Actor:
    """One character: fixed identity, mutable pose, emitted as polygons."""

    def __init__(self, seed: int = 0, **traits):
        rng = random.Random(seed)
        self.name = traits.get("name", "SOMEONE")
        self.role = traits.get("role", "")
        self.height = float(traits.get("height", 1.75))
        self.build = float(traits.get("build", 0.40))
        self.girth = float(traits.get("girth", 1.0))
        self.skin = traits.get("skin", (220, 180, 140))
        self.skin_warm = traits.get("skin_warm", _tone(self.skin, 1.06))
        # Skin in its own shadow, for a brow ridge or the underside of a
        # jaw. Shading those by hand rather than by lighting means they
        # survive the camera swinging round to the other side of the room.
        self.shade = _tone(self.skin, 0.80)
        self.deep = _tone(self.skin, 0.62)
        self.shirt = traits.get("shirt", (120, 140, 170))
        self.trousers = traits.get("trousers", (70, 70, 90))
        self.hair = traits.get("hair", (60, 44, 36))
        self.hair_style = traits.get("hair_style", "neat")
        self.shoes = traits.get("shoes", (52, 48, 50))
        self.collar = traits.get("collar")
        self.jacket = traits.get("jacket")
        self.glasses = bool(traits.get("glasses", False))
        self.skirt = bool(traits.get("skirt", False))
        self.shirt_detail = traits.get("shirt_detail")
        # Only the striped shirt has one. Giving everybody a default
        # spent palette entries on a colour three of them never wear.
        self.band = traits.get("band")
        self.eye = traits.get("eye", (52, 44, 44))
        # Eyebrows are hair, not skin in shadow. As a shadow tone they
        # vanished against the forehead from every camera but one.
        self.brow = _tone(self.hair, 0.90)

        # World position. x is across the set, z is depth (negative = further
        # from camera). Staging happens mostly in z -- see world.MARKS.
        self.x = 0.0
        self.z = 0.0
        # Which way they are turned: a yaw about their own vertical axis,
        # 0 being square to the lens and positive turning their right
        # shoulder away. This replaced a `facing` of +1/-1, which could only
        # mirror a body that was always square to the camera.
        self.yaw = 0.0
        self.yaw_target = 0.0
        self.state = IDLE
        self.phase = rng.random() * 10.0
        self.target: tuple[float, float] | None = None
        self.visible = True
        # Rises while talking, decays after, so gestures wind down rather
        # than stopping dead.
        self.energy = 0.0

    # -- behaviour -----------------------------------------------------------

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            if state == TALK:
                self.energy = 1.0

    def walk_to(self, x: float, z: float) -> None:
        self.target = (x, z)
        self.set_state(WALK)

    def look_at(self, x: float, z: float) -> None:
        """Turn toward a point on the floor, within the limit."""
        dx, dz = x - self.x, z - self.z
        if abs(dx) < 0.02 and abs(dz) < 0.02:
            return
        yaw = math.atan2(dx, dz)
        # Behind them is still in front of the camera; wrap rather than
        # letting anybody spin round to look upstage.
        if yaw > math.pi * 0.5:
            yaw = math.pi - yaw
        elif yaw < -math.pi * 0.5:
            yaw = -math.pi - yaw
        self.yaw_target = max(-MAX_YAW, min(MAX_YAW, yaw))

    def face(self, other) -> None:
        if isinstance(other, Actor):
            self.look_at(other.x, other.z)
        else:
            self.look_at(float(other), self.z)

    def update(self, dt: float) -> None:
        self.phase += dt
        self.energy = max(0.0, self.energy - dt * 0.35)

        # Heads turn over about a third of a second rather than snapping.
        d = self.yaw_target - self.yaw
        self.yaw += d * min(1.0, dt * 7.0)

        if self.state == WALK and self.target is not None:
            tx, tz = self.target
            dx, dz = tx - self.x, tz - self.z
            dist = math.hypot(dx, dz)
            if dist < 0.06:
                self.x, self.z = tx, tz
                self.target = None
                self.set_state(IDLE)
            else:
                step = min(dist, 1.5 * dt)
                self.x += dx / dist * step
                self.z += dz / dist * step
                self.look_at(tx, tz)

    # -- the head ------------------------------------------------------------

    def _head(self, sc: r3d.Scene, cx: float, chin: float, cz: float,
              hh: float, hw: float, detail: int, t: float) -> None:
        """Cranium, jaw and face, with the chin at `chin`.

        Built as two revolved solids rather than one tube with a lid. A head
        made of a single prism has a flat top and a hard rim all the way
        round, and that rim is the first thing that says "cylinder with the
        corners knocked off" -- it is the loudest single tell in the whole
        picture, because it is what the viewer is looking at.
        """
        r = hw * 0.5
        sides = (6, 8, 9)[detail]
        squash = 1.06                    # heads are deeper than they are wide
        roll = math.pi * 0.5 + math.pi / sides   # a flat facet facing front

        skin, warm, shade, deep = self.skin, self.skin_warm, self.shade, self.deep

        # Jaw: narrow at the chin, widening to the cheekbones.
        jaw = ([(0.0, r * 0.46), (hh * 0.26, r * 0.80), (hh * 0.56, r * 1.00)]
               if detail == 0 else
               [(0.0, r * 0.44), (hh * 0.11, r * 0.62), (hh * 0.27, r * 0.82),
                (hh * 0.43, r * 0.95), (hh * 0.56, r * 1.00)])
        sc.revolve((cx, chin, cz), jaw, skin, sides, roll, caps="b",
                   squash=squash)
        # Cranium: a dome, not a cap.
        dome = ([(0.0, r * 1.00), (hh * 0.26, r * 0.95), (hh * 0.44, r * 0.50)]
                if detail == 0 else
                [(0.0, r * 1.00), (hh * 0.15, r * 1.04), (hh * 0.29, r * 0.99),
                 (hh * 0.38, r * 0.84), (hh * 0.44, r * 0.46)])
        sc.revolve((cx, chin + hh * 0.56, cz), dome, skin, sides, roll,
                   squash=squash)

        # The front facet of an n-sided ring sits at cos(pi/n) of the radius.
        face_z = cz + r * squash * math.cos(math.pi / sides)
        # Eyes a little above the middle of the head, which is where they
        # are. Put at 0.60 they left a stretch of bare cheek between the
        # nose and the chin and everybody came out long in the face.
        eye_y = chin + hh * 0.575
        eye_dx = hw * 0.198

        def plate(x, y, z, w, h, colour):
            """A feature painted on the front of the face. A single quad --
            nobody turns far enough for the back of one to matter, and at
            two polygons each these would cost more than the skull."""
            sc.poly([(x - w * 0.5, y - h * 0.5, z), (x + w * 0.5, y - h * 0.5, z),
                     (x + w * 0.5, y + h * 0.5, z), (x - w * 0.5, y + h * 0.5, z)],
                    colour)

        # -- eyes ------------------------------------------------------------
        for side in (-1, 1):
            ex = cx + side * eye_dx
            if detail == 0:
                plate(ex, eye_y, face_z + 0.004, hw * 0.21, hh * 0.070,
                      self.eye)
            else:
                # Socket first: a recess in shadow is what gives an eye a
                # place to sit. Two dark squares on flat skin read as holes.
                plate(ex, eye_y, face_z + 0.003, hw * 0.29, hh * 0.105,
                      _tone(skin, 0.88))
                plate(ex, eye_y, face_z + 0.006, hw * 0.205, hh * 0.070,
                      (234, 228, 214))
                plate(ex + side * hw * 0.012, eye_y, face_z + 0.009,
                      hw * 0.090, hh * 0.062, self.eye)
                if detail == 2:
                    # A lid over the top third, which is most of what makes
                    # a pair of eyes look open rather than painted.
                    plate(ex, eye_y + hh * 0.040, face_z + 0.012,
                          hw * 0.22, hh * 0.028, shade)
            # Brow, standing proud of the forehead so it has an underside.
            # Kept thin and close to the eye: a thick bar high on the
            # forehead meets the hairline and the two read as one visor.
            sc.box((ex, eye_y + hh * 0.088, face_z + 0.010),
                   (hw * 0.27, hh * 0.034, 0.016), self.brow,
                   faces="zy" if detail else "z")

        # -- nose ------------------------------------------------------------
        nose_y = chin + hh * 0.355
        if detail == 0:
            plate(cx, (eye_y + nose_y) * 0.5, face_z + 0.004,
                  hw * 0.13, hh * 0.20, warm)
            plate(cx, nose_y - hh * 0.035, face_z + 0.005, hw * 0.20,
                  hh * 0.035, deep)
        else:
            # A bridge and a tip that actually stick out. The light is keyed
            # from the front-left, so a nose with sides catches one and
            # shades the other, and that is the whole of what tells a viewer
            # a face is a solid.
            sc.box((cx, (eye_y + nose_y) * 0.5 + hh * 0.02, face_z + 0.010),
                   (hw * 0.135, hh * 0.24, 0.022), warm, faces="xz")
            sc.box((cx, nose_y + hh * 0.015, face_z + 0.018),
                   (hw * 0.21, hh * 0.105, 0.038), warm, faces="xyz")
            plate(cx, nose_y - hh * 0.042, face_z + 0.022, hw * 0.23,
                  hh * 0.030, deep)
            if detail == 2:
                for side in (-1, 1):
                    plate(cx + side * hw * 0.068, nose_y - hh * 0.022,
                          face_z + 0.024, hw * 0.045, hh * 0.028, deep)

        # -- mouth -----------------------------------------------------------
        mouth_y = chin + hh * 0.190
        gap = (0.030 + 0.105 * (0.5 + 0.5 * math.sin(t * 11.0))
               if self.state == TALK else 0.030)
        plate(cx, mouth_y + hh * 0.048, face_z + 0.005, hw * 0.27,
              hh * 0.038, shade)                          # upper lip
        plate(cx, mouth_y, face_z + 0.006, hw * 0.28, hh * gap, (74, 40, 42))
        if detail:
            plate(cx, mouth_y - hh * (gap * 0.5 + 0.026), face_z + 0.006,
                  hw * 0.25, hh * 0.040, _tone(skin, 0.94))   # lower lip

        # -- ears ------------------------------------------------------------
        if detail == 2:
            for side in (-1, 1):
                sc.tube((cx + side * r * 0.84, eye_y + hh * 0.01, cz - r * 0.12),
                        (cx + side * r * 1.08, eye_y - hh * 0.01, cz - r * 0.14),
                        hh * 0.085, hh * 0.058, skin, 5)

        # -- glasses ---------------------------------------------------------
        if self.glasses:
            # Round wire frames. As one black bar across the face he was
            # wearing a censor strip, not spectacles.
            for side in (-1, 1):
                sc.ring((cx + side * eye_dx, eye_y, face_z + 0.026),
                        hw * 0.175, hw * 0.032, (58, 52, 50),
                        segments=8 if detail else 6)
            plate(cx, eye_y + hh * 0.010, face_z + 0.024, hw * 0.075,
                  hh * 0.018, (58, 52, 50))
            if detail:
                for side in (-1, 1):
                    sc.box((cx + side * (eye_dx + hw * 0.17), eye_y,
                            face_z - r * 0.26),
                           (0.012, hh * 0.022, r * 0.52), (58, 52, 50),
                           faces="xy")

        self._hair(sc, cx, chin, cz, hh, hw, r, sides, roll, squash, detail,
                   face_z, eye_y)

    def _hair(self, sc, cx, chin, cz, hh, hw, r, sides, roll, squash, detail,
              face_z, eye_y) -> None:
        """Whatever is on top. This is the loudest identity cue there is:
        at forty pixels tall the silhouette above the eyeline is most of
        what distinguishes four people of similar colouring."""
        hair = self.hair
        style = self.hair_style

        # Hair sits no lower than this at the FRONT: anything below it meets
        # the brow and the two read as one dark visor across the face. The
        # back and sides come lower, as hair does, via an open band.
        line = chin + hh * 0.76

        def back_band(low, rad=1.07, grow=1.09):
            sc.revolve((cx, low, cz),
                       [(0.0, r * rad), (line - low, r * grow)]
                       if not detail else
                       [(0.0, r * rad), ((line - low) * 0.5, r * grow),
                        (line - low, r * grow)],
                       hair, 5 if not detail else 7, caps="", squash=squash,
                       arc=(math.pi * 0.70, math.pi * 2.30))

        # A cap that closes over the crown. It has to stay wider than the
        # skull at every height: narrower anywhere and the skull's own top
        # disc shows through as a bald patch with a rim of hair round it.
        cap = ([(0.0, r * 1.09), (hh * 0.21, r * 0.96), (hh * 0.32, r * 0.14)]
               if not detail else
               [(0.0, r * 1.09), (hh * 0.12, r * 1.08), (hh * 0.21, r * 0.96),
                (hh * 0.27, r * 0.62), (hh * 0.32, r * 0.14)])

        if style == "neat":
            sc.revolve((cx, line, cz), cap, hair, sides, roll, squash=squash,
                       lean=(0.0, -r * 0.08))
            back_band(chin + hh * 0.56)
            # The quiff: a lift at the front, which is the one thing that
            # separates this head from a swimming cap.
            if detail:
                sc.revolve((cx - hw * 0.03, chin + hh * 0.955, cz + r * 0.46),
                           [(0.0, r * 0.50), (hh * 0.10, r * 0.44),
                            (hh * 0.16, r * 0.18)],
                           hair, 6, squash=0.8)
            if detail:
                for side in (-1, 1):       # sideburns past the ear
                    sc.box((cx + side * r * 0.90, eye_y + hh * 0.055,
                            cz - r * 0.10),
                           (0.02, hh * 0.20, r * 0.44), hair, faces="xy")

        elif style == "horseshoe":
            # Bald on top, with a band of hair running round the back from
            # one temple to the other. A dome of hair sat behind the skull
            # read as a full head of it from every camera below eye level.
            sc.revolve((cx, chin + hh * 0.54, cz),
                       [(0.0, r * 1.05), (hh * 0.16, r * 1.05),
                        (hh * 0.32, r * 0.92)],
                       hair, 7, caps="", squash=squash,
                       arc=(math.pi * 0.66, math.pi * 2.34))

        elif style == "curls":
            # Volume, and most of it above and at the sides. A curtain of
            # hair hanging flat is a wig. Built as one closed mass it also
            # swallowed her face, so the part that falls past the jaw is an
            # open band with the front left out.
            sc.revolve((cx, line - hh * 0.04, cz - r * 0.05),
                       [(0.0, r * 1.26), (hh * 0.48, r * 1.46),
                        (hh * 0.82, r * 0.52)]
                       if not detail else
                       [(0.0, r * 1.26), (hh * 0.24, r * 1.44),
                        (hh * 0.48, r * 1.46), (hh * 0.68, r * 1.24),
                        (hh * 0.82, r * 0.52)],
                       hair, sides + 1, roll, squash=0.97)
            sc.revolve((cx, chin + hh * 0.16, cz - r * 0.16),
                       [(0.0, r * 1.00), (hh * 0.56, r * 1.40)]
                       if not detail else
                       [(0.0, r * 1.00), (hh * 0.26, r * 1.28),
                        (hh * 0.56, r * 1.40)],
                       hair, 6 if not detail else 9, caps="", squash=0.97,
                       arc=(math.pi * 0.86, math.pi * 2.14))
            if detail:
                for side in (-1, 1):       # bumps, so it is not one egg
                    sc.revolve((cx + side * r * 0.80, line + hh * 0.56,
                                cz - r * 0.18),
                               [(0.0, r * 0.60), (hh * 0.18, r * 0.36)],
                               hair, 5)

        elif style == "pompadour":
            sc.revolve((cx, line - hh * 0.02, cz + r * 0.04), cap, hair,
                       sides, roll, squash=squash)
            back_band(chin + hh * 0.52)
            # Swept up and back off the forehead. Stacked straight up it was
            # a cone -- a wizard's hat, not hair; the mass stays about as
            # wide as the head and leans back as it rises.
            sc.revolve((cx, chin + hh * 0.96, cz + r * 0.28),
                       [(0.0, r * 0.84), (hh * 0.31, r * 0.78),
                        (hh * 0.41, r * 0.40)]
                       if not detail else
                       [(0.0, r * 0.84), (hh * 0.16, r * 0.90),
                        (hh * 0.31, r * 0.78), (hh * 0.41, r * 0.40)],
                       hair, sides, roll, squash=0.92,
                       lean=(0.0, -r * 0.46))
            if detail:
                for side in (-1, 1):
                    sc.box((cx + side * r * 0.90, eye_y + hh * 0.07,
                            cz - r * 0.10),
                           (0.02, hh * 0.24, r * 0.42), hair, faces="xy")

    # -- geometry ------------------------------------------------------------

    def build_geometry(self, sc: r3d.Scene, detail: int = 2) -> None:
        """Emit this actor. `detail` is 0 (far), 1 (medium) or 2 (close)."""
        if not self.visible:
            return

        t = self.phase
        h = self.height
        bw = self.build                   # shoulder width
        depth = bw * 0.58 * self.girth    # body thickness, front to back

        leg_h = h * 0.475
        torso_h = h * 0.325
        head_h = h * 0.129
        neck_h = h * 0.040

        bob = lean = arm_swing = arm_lift = step = 0.0
        head_tilt = 0.0

        if self.state == IDLE:
            bob = math.sin(t * 1.6) * 0.008
        elif self.state == TALK:
            bob = math.sin(t * 6.5) * 0.012
            arm_lift = (0.5 + 0.5 * math.sin(t * 4.5)) * 0.14 * (0.35 + self.energy)
            arm_swing = math.sin(t * 3.1) * 0.05
            head_tilt = math.sin(t * 2.3) * 0.012
        elif self.state == WALK:
            step = math.sin(t * 7.5) * 0.14
            bob = abs(step) * 0.10
            arm_swing = -step * 0.6
        elif self.state == GESTURE:
            arm_lift = 0.22 + math.sin(t * 2.6) * 0.09
            bob = math.sin(t * 1.8) * 0.01
            lean = 0.02
        elif self.state == TWITCH:
            bob = (random.random() - 0.5) * 0.03
            lean = (random.random() - 0.5) * 0.05
            arm_lift = random.random() * 0.2

        # Everything below is built square to the world and then turned as a
        # whole, which is what lets anybody stand at an angle to the lens.
        sc.set_yaw(self.yaw, (self.x, 0.0, self.z))
        try:
            self._body(sc, t, h, bw, depth, leg_h, torso_h, head_h, neck_h,
                       bob, lean, arm_swing, arm_lift, step, head_tilt, detail)
        finally:
            sc.set_yaw()

    def _body(self, sc, t, h, bw, depth, leg_h, torso_h, head_h, neck_h,
              bob, lean, arm_swing, arm_lift, step, head_tilt, detail) -> None:
        x, z = self.x, self.z
        y0 = bob

        hip_y = y0 + leg_h
        shoulder_y = hip_y + torso_h
        waist_w = bw * 0.80 * self.girth
        shoulder_w = bw * 1.04
        leg_x = bw * 0.24
        limb = bw * 0.21                  # thigh radius; everything scales off it
        sides = 4 if detail == 0 else 6

        # -- legs ------------------------------------------------------------
        # Joint positions first, then tubes between them. The stride runs
        # forward along the actor's own axis, which the yaw then turns; swung
        # in world z alone it was under a pixel from half the cameras this
        # show uses, so walking looked like gliding.
        for side in (-1, 1):
            sx = x + side * leg_x
            phase = step * (1.0 if side < 0 else -1.0)
            lift = max(0.0, -phase) * 0.55
            hip = (sx, hip_y - torso_h * 0.04, z)
            knee = (sx + side * bw * 0.02, y0 + leg_h * 0.48 + lift * 0.35,
                    z + phase * 0.55)
            ankle = (sx + side * bw * 0.01, y0 + 0.075 + lift, z + phase)
            sc.tube(hip, knee, limb * 1.02, limb * 0.80, self.trousers, sides)
            sc.tube(knee, ankle, limb * 0.80, limb * 0.56, self.trousers, sides)
            foot_z = ankle[2] + depth * 0.10
            sc.frustum((ankle[0], ankle[1] - 0.04, foot_z),
                       (bw * 0.34, depth * 0.78), (bw * 0.30, depth * 0.62),
                       0.07, self.shoes)
            sc.wedge((ankle[0], ankle[1] - 0.045, foot_z + depth * 0.5),
                     (bw * 0.30, 0.06, depth * 0.34), self.shoes, 0.7)

        # -- pelvis and torso -------------------------------------------------
        sc.frustum((x + lean * 0.4, hip_y - torso_h * 0.06, z),
                   (bw * 0.86, depth * 0.84), (waist_w, depth * 0.82),
                   torso_h * 0.26, self.trousers, caps="")
        if self.shirt_detail == "bands":
            # A striped shirt is the material changing, not stickers laid on
            # a plain one: bands pasted on the front stood out past the body
            # at the waist and poked through both arms. Stacked frusta
            # follow the taper because they are the taper.
            n = 7 if detail else 3
            for i in range(n):
                u0, u1 = i / n, (i + 1) / n
                w0 = waist_w + (shoulder_w - waist_w) * u0
                w1 = waist_w + (shoulder_w - waist_w) * u1
                d0 = depth * (0.82 + 0.18 * u0)
                d1 = depth * (0.82 + 0.18 * u1)
                sc.frustum((x + lean,
                            hip_y + torso_h * 0.92 * (u0 + u1) * 0.5, z),
                           (w0, d0), (w1, d1), torso_h * 0.92 / n,
                           self.band if i % 3 == 1 else self.shirt,
                           lean=lean * 0.5, caps="")
        else:
            sc.frustum((x + lean, hip_y + torso_h * 0.46, z),
                       (waist_w, depth * 0.82), (shoulder_w, depth * 1.0),
                       torso_h * 0.92, self.shirt, lean=lean * 0.5, caps="")

        front_z = z + depth * 0.52
        if self.shirt_detail == "placket" and detail:
            # A button band down the front, and the buttons on it. Flat
            # colour from collar to belt is a sweatshirt; this is a shirt.
            sc.box((x + lean, hip_y + torso_h * 0.52, front_z),
                   (bw * 0.11, torso_h * 0.80, 0.02),
                   _tone(self.shirt, 0.88), faces="zx")
            for i in range(3):
                sc.box((x + lean, hip_y + torso_h * (0.28 + i * 0.24),
                        front_z + 0.012),
                       (bw * 0.045, bw * 0.045, 0.012),
                       _tone(self.shirt, 1.18), faces="z")
        if self.jacket is not None:
            # Open at the front, with a lapel either side, rather than a
            # closed shell -- which is how a jacket turned him into a slab
            # with sleeves and hid the shirt entirely.
            for side in (-1, 1):
                # Each panel tapers with the body under it. As a pair of
                # axis-aligned boxes the jacket was a slab of cardboard with
                # sleeves, which is the one shape this renderer has to stay
                # away from.
                sc.frustum((x + lean + side * shoulder_w * 0.345,
                            hip_y + torso_h * 0.50, z),
                           (shoulder_w * 0.34, depth * 0.94),
                           (shoulder_w * 0.40, depth * 1.06),
                           torso_h * 0.94, self.jacket,
                           lean=lean * 0.5, caps="")
                sc.box((x + lean + side * shoulder_w * 0.20,
                        hip_y + torso_h * 0.80, front_z + 0.015),
                       (shoulder_w * 0.20, torso_h * 0.34, 0.02),
                       _tone(self.jacket, 1.12), faces="zx")

        # Belt.
        sc.frustum((x + lean * 0.6, hip_y + torso_h * 0.05, z),
                   (waist_w * 1.03, depth * 0.86), (waist_w * 1.03, depth * 0.86),
                   torso_h * 0.08, (48, 42, 44), caps="")

        # A sloped yoke rather than a flat top with a lump laid across it.
        # Square shoulders with a notch under them are what made the
        # silhouette read as a cabinet.
        yoke = self.jacket if self.jacket is not None else self.shirt
        sc.frustum((x + lean, shoulder_y - torso_h * 0.06, z),
                   (shoulder_w * (1.02 if self.jacket else 1.0), depth * 1.02),
                   (bw * 0.56, depth * 0.72),
                   torso_h * 0.16, yoke, caps="t")

        if self.collar is not None:
            # Around the neck, not across the chest -- at full shoulder width
            # it read as a bib.
            sc.frustum((x + lean, shoulder_y + neck_h * 0.20, z),
                       (bw * 0.30, depth * 0.50), (bw * 0.255, depth * 0.42),
                       neck_h * 0.95, self.collar, caps="")

        if self.skirt:
            sc.frustum((x, hip_y - leg_h * 0.16, z),
                       (bw * 1.20, depth * 1.18), (waist_w * 1.02, depth * 0.88),
                       leg_h * 0.44, self.trousers, caps="")

        # -- arms --------------------------------------------------------------
        sleeve = self.jacket if self.jacket is not None else _tone(self.shirt, 0.88)
        arm_r = limb * 0.76
        shell = 1.05 if self.jacket is not None else 1.0
        for side in (-1, 1):
            # Overlapping the torso, not hanging beside it with daylight in
            # between.
            ax = x + side * (shoulder_w * 0.5 * shell - arm_r * 0.15) + lean
            shoulder = (ax, shoulder_y - torso_h * 0.12, z)
            # The deltoid bridges torso to arm so the joint is not a gap.
            sc.tube((x + side * shoulder_w * 0.10 + lean,
                     shoulder_y - torso_h * 0.10, z),
                    shoulder, arm_r * 1.30, arm_r * 1.06, yoke, sides)
            elbow = (ax + side * bw * 0.01,
                     shoulder_y - torso_h * 0.60 + arm_lift * 0.45,
                     z + arm_swing * side + depth * (0.1 + arm_lift * 0.6))
            wrist = (elbow[0] - side * bw * 0.02,
                     elbow[1] - torso_h * 0.46 + arm_lift * 0.9,
                     elbow[2] + depth * (0.30 + arm_lift * 1.4))
            # A short sleeve stops at the bicep; a jacket runs to the elbow.
            hem = 1.0 if self.jacket is not None else 0.55
            cuff = tuple(s + (e - s) * hem for s, e in zip(shoulder, elbow))
            sc.tube(shoulder, cuff, arm_r * 1.06, arm_r * 0.92, sleeve, sides)
            if hem < 1.0:
                sc.tube(cuff, elbow, arm_r * 0.90, arm_r * 0.80, self.skin, sides)
            sc.tube(elbow, wrist, arm_r * 0.80, arm_r * 0.64, self.skin, sides)
            # The hand carries on down the forearm axis. As an axis-aligned
            # box it detached from the wrist the moment the arm moved.
            dx = [w - e for w, e in zip(wrist, elbow)]
            length = math.sqrt(sum(c * c for c in dx)) or 1.0
            hand_end = tuple(w + c / length * arm_r * 1.7
                             for w, c in zip(wrist, dx))
            sc.tube(wrist, hand_end, arm_r * 0.86, arm_r * 0.64,
                    self.skin, 5, caps="t")

        # -- neck and head -----------------------------------------------------
        neck_top = shoulder_y + neck_h
        sc.tube((x + lean, shoulder_y - torso_h * 0.04, z),
                (x + lean, neck_top, z - depth * 0.02),
                bw * 0.21, bw * 0.17, self.skin, sides)

        # Head width off height, not off build: tying it to shoulder width
        # gave the stocky one a head half again as wide as it was tall.
        head_w = h * 0.101
        self._head(sc, x + lean + head_tilt, neck_top - head_h * 0.06,
                   z - depth * 0.02 + head_tilt * 0.4, head_h, head_w,
                   detail, t)


def make_sitcom_cast(seed: int = 0) -> list[Actor]:
    """The four leads, same identities every run. The seed only perturbs
    each one's idle phase so they are not all bobbing in lockstep."""
    rng = random.Random(seed)
    return [Actor(rng.randrange(1 << 30), **a) for a in ARCHETYPES]


def by_role(cast: list[Actor], role: str) -> Actor:
    for actor in cast:
        if actor.role == role:
            return actor
    return cast[0]
