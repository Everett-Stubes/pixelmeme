"""The continuous shot.

Beats arrive from the director; the stage builds a 3D scene for each frame
and renders it. When a beat ends the next one starts on the same frame.
Nothing cuts to black, nothing resets, there is no episode boundary.

The camera cuts rather than moves -- a wide that establishes the room and a
medium on whoever is talking, swapped on beat boundaries. That is both
cheaper than animating a camera and closer to how the reference shoots.

Layout is fixed: the render fills everything above world-space and the
subtitle bar owns the strip below it, like a badly encoded broadcast.
"""

from __future__ import annotations

import math
import random

import pygame

from . import actors as actors_mod, effects, laughtrack, pixelfont as pf, r3d, world
from ..brain import director as director_mod

# The stage occupies everything above the subtitle bar.
STAGE_FRACTION = 0.775
LINE_LEAD = 2
MAX_SUB_LINES = 3

# Characters revealed per second. Fast enough not to annoy, slow enough that
# the line lands as speech rather than as a title card.
TYPE_SPEED = 26.0

# Sky/void behind any gap in the geometry.
_VOID = (26, 24, 30)


class Stage:
    def __init__(self, cfg, size: tuple[int, int], seed: int | None = None):
        self.w, self.h = size
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.director = director_mod.Director(cfg, seed)
        self.cast = self.director.cast

        self.stage_h = int(self.h * STAGE_FRACTION)
        self.bar_top = self.stage_h
        self.name_y = self.bar_top + 3
        self.text_y = self.bar_top + 11

        self.t = 0.0
        self.beat: director_mod.Beat | None = None
        self.beat_t = 0.0
        # The audience. Laughs ride over the set rather than taking a beat of
        # their own, so one can still be going when the next line starts.
        self.laugh = laughtrack.LaughTrack(self.rng, width=self.w)
        self._laughed = False
        self.set_name = self.director.set_name
        self.set_seed = self.rng.randrange(1 << 30)
        self.transition = 0.0
        # How many scenes have played since anyone last reset it. A scene is
        # one continuous stretch on one set; main.py counts them to decide
        # when to cut to the climb.
        self.scenes_played = 0

        self._scratch = pygame.Surface(size)
        self._stage_rect = pygame.Rect(0, 0, self.w, self.stage_h)
        self._cam = world.wide_camera(self.set_name, (self.w, self.stage_h))
        self.poly_count = 0

        self._place_cast(initial=True)
        self._advance()

    # -- staging -------------------------------------------------------------

    def _place_cast(self, initial: bool = False) -> None:
        marks = world.marks_for(self.set_name)
        for i, actor in enumerate(self.cast):
            x, z = marks[i % len(marks)]
            actor.x, actor.z = float(x), float(z)
            actor.visible = True
            if initial:
                actor.set_state(actors_mod.IDLE)
        centre = sum(a.x for a in self.cast) / max(1, len(self.cast))
        for actor in self.cast:
            actor.face(centre)

    def _blocked(self, speaker, side: float) -> bool:
        """Is somebody else standing in the way of this medium shot?

        The camera sits off to one side of the speaker and looks at them.
        Anyone close to that line of sight and nearer the lens ends up as a
        head in the middle of frame while somebody off-screen talks. With
        the cast staged in depth that was about a quarter of all beats.
        """
        cam_x, cam_z = speaker.x + 0.72 * side, speaker.z + 2.25
        dx, dz = speaker.x - cam_x, speaker.z - cam_z
        length = math.hypot(dx, dz) or 1.0
        dx, dz = dx / length, dz / length
        for other in self.cast:
            if other is speaker or not other.visible:
                continue
            ox, oz = other.x - cam_x, other.z - cam_z
            along = ox * dx + oz * dz
            if not 0.2 < along < length:
                continue
            if abs(ox * dz - oz * dx) < 0.40:
                return True
        return False

    def _pick_camera(self, beat) -> None:
        """A cut, chosen fresh for this beat."""
        size = (self.w, self.stage_h)
        if beat.speaker is not None and self.rng.random() < 0.70:
            speaker = self.cast[beat.speaker]
            side = 1.0 if speaker.x < 0 else -1.0
            if self._blocked(speaker, side):
                side = -side
            if not self._blocked(speaker, side):
                # Roughly one cut to the speaker in three is a close. A
                # medium puts a head sixteen pixels tall, which is a
                # silhouette; the face only gets to act in the close.
                if self.rng.random() < 0.34:
                    self._cam = world.close_camera(
                        self.set_name, size, (speaker.x, speaker.z),
                        speaker.height, side,
                    )
                else:
                    self._cam = world.medium_camera(
                        self.set_name, size, (speaker.x, speaker.z), side
                    )
                return
        self._cam = world.wide_camera(self.set_name, size)

    def _advance(self) -> None:
        beat = self.director.next_beat()
        self.beat = beat
        self.beat_t = 0.0
        self._laughed = False

        if beat.set_name and beat.set_name != self.set_name:
            self.set_name = beat.set_name
            self.set_seed = self.rng.randrange(1 << 30)
            self.transition = 1.0
            self.scenes_played += 1
            self._place_cast()

        for i, actor in enumerate(self.cast):
            if beat.speaker is not None and i == beat.speaker:
                actor.set_state(beat.action)
            else:
                actor.set_state(beat.others)

        if beat.speaker is not None:
            speaker = self.cast[beat.speaker]
            for i, actor in enumerate(self.cast):
                if i != beat.speaker:
                    actor.face(speaker)
            nearest = min(
                (a for i, a in enumerate(self.cast) if i != beat.speaker),
                key=lambda a: abs(a.x - speaker.x),
                default=None,
            )
            if nearest is not None:
                speaker.face(nearest)

        if beat.walk_in is not None:
            actor = self.cast[beat.walk_in]
            marks = world.marks_for(self.set_name)
            x, z = marks[beat.walk_in % len(marks)]
            # Comes in through the door at the back right.
            actor.x, actor.z = 1.6, -3.1
            actor.walk_to(float(x), float(z))

        self._pick_camera(beat)

    # -- per frame -----------------------------------------------------------

    def update(self, dt: float) -> None:
        self.t += dt
        self.beat_t += dt
        self.transition = max(0.0, self.transition - dt * 1.8)

        for actor in self.cast:
            actor.update(dt)

        self.director.pump()
        self.laugh.update(dt)
        self._maybe_laugh()

        if self.beat is None or self.beat_t >= self.beat.duration:
            self._advance()

    def _maybe_laugh(self) -> None:
        """Laughs land on the end of the line, not the start of it.

        A laugh track that fires while the joke is still typing out is just
        noise; the audience has to have heard the whole thing.
        """
        beat = self.beat
        if beat is None or self._laughed or beat.laugh <= 0.0:
            return
        if beat.kind == director_mod.LINE and beat.text:
            cue = len(beat.text) / TYPE_SPEED + 0.15
        else:
            cue = beat.duration * 0.55
        if self.beat_t >= min(cue, beat.duration):
            self._laughed = True
            self.laugh.fire(beat.laugh)

    def _visible_text(self) -> str:
        if not self.beat or not self.beat.text:
            return ""
        return self.beat.text[: int(self.beat_t * TYPE_SPEED)]

    # -- drawing -------------------------------------------------------------

    def _draw_bar(self, s: pygame.Surface) -> None:
        s.fill((10, 8, 14), pygame.Rect(0, self.bar_top, self.w,
                                        self.h - self.bar_top))
        s.fill((28, 22, 34), pygame.Rect(0, self.bar_top, self.w, 1))

        beat = self.beat
        if beat is None:
            return
        text = self._visible_text()
        if not text:
            return

        if beat.kind == director_mod.CAPTION:
            pf.draw_block(
                s, text, self.w // 2, self.text_y + 3, self.w - 6,
                scale=1, colour=(120, 158, 152), outline=None,
                leading=LINE_LEAD, max_lines=MAX_SUB_LINES,
            )
            return

        if beat.speaker is not None:
            speaker = self.cast[beat.speaker]
            pf.draw_text(s, speaker.name + ":", 3, self.name_y, 1,
                         speaker.shirt, None)
        pf.draw_block(
            s, text, self.w // 2, self.text_y, self.w - 6,
            scale=1, colour=(248, 238, 210), outline=None,
            leading=LINE_LEAD, max_lines=MAX_SUB_LINES,
        )

    def draw(self, target: pygame.Surface) -> None:
        s = self._scratch
        view = s.subsurface(self._stage_rect)
        view.fill(_VOID)

        scene = r3d.Scene()
        world.build_set(scene, self.set_name, self.t, self.set_seed)
        # Detail follows the lens. A head four metres away is nine pixels
        # tall; sclera, nostrils and ears on it are polygons spent on
        # something nobody can resolve, and they are polygons the close-up
        # needs. The far mark in a wide costs about a third of the near one.
        #
        # Only one of them gets the top level, however close the rest
        # happen to stand: in a close-up the whole cast can fall inside two
        # metres, and four full-detail bodies made those frames half as
        # expensive again as every other frame, which is exactly the spike
        # that shows up as a stutter rather than as a lower frame rate.
        cx, _cy, cz = self._cam.pos
        order = sorted(
            self.cast, key=lambda a: math.hypot(a.x - cx, a.z - cz)
        )
        for rank, actor in enumerate(order):
            d = math.hypot(actor.x - cx, actor.z - cz)
            if d < 2.0 and rank == 0:
                detail = 2
            elif d < 3.4:
                detail = 1
            else:
                detail = 0
            actor.build_geometry(scene, detail)
        # Ambient 0.60 left only a 1.67x range between an unlit and a lit
        # face -- not enough for flat shading to separate one plane of a
        # solid from the next. The fog is the set's own haze: without it the
        # far wall is lit exactly as brightly as the chair in front of it,
        # and the room reads as a painted backdrop.
        self.poly_count = r3d.render(scene, view, self._cam, ambient=0.46,
                                     fog=world.fog_for(self.set_name,
                                                       self.set_seed))

        glitch = max(self.beat.glitch if self.beat else 0.0, self.transition)
        if glitch > 0.25 and self.rng.random() < 0.3 + glitch * 0.5:
            effects.glitch_rows(view, glitch, self.rng, bands=2 + int(glitch * 4))
        if glitch > 0.5:
            effects.chromatic_shift(view, 1)
        if self.transition > 0.6:
            effects.flash(view, (248, 238, 210), int(self.transition * 90))

        self._draw_bar(s)
        # Two pixels clear of the subtitle bar, so the laugh reads as an
        # overlay on the picture rather than as another row of the bar.
        self.laugh.draw(s, self.w, self.bar_top - 2)

        # A faint scanline crawling down the whole frame, so the picture
        # reads as transmitted rather than drawn.
        crawl = int((self.t * 14) % self.h)
        s.fill((14, 16, 20), pygame.Rect(0, crawl, self.w, 2),
               special_flags=pygame.BLEND_ADD)

        target.blit(s, (0, 0))

    # -- status --------------------------------------------------------------

    @property
    def status(self) -> str:
        beat = self.beat
        who = (
            self.cast[beat.speaker].name
            if beat and beat.speaker is not None
            else beat.kind.upper() if beat else "-"
        )
        return f"{self.set_name[:5]} {who} p{self.poly_count}"
