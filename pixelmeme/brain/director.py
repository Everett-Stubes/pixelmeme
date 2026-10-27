"""Decides what happens next, forever.

There are no episodes and nothing ever finishes. The director keeps a queue of
beats; when it runs low it opens another scene and appends it. Nothing resets,
so the stream has no seams.

A scene is: draw a subject -- a thing, a culprit, a place, a grievance -- then
hand it to the generator in dialogue.py, which walks a conversational-move
machine and writes the turns. The director's job is casting and staging: who
says which turn, where they are standing, when the audience laughs, and when
the broadcast glitches.

Nothing outside feeds this. There is no microphone and no transcript; the
subject comes out of the lexicon, and the broadcast generates itself forever
whether anyone is in the room or not.
"""

from __future__ import annotations

import collections
import itertools
import random
import time
from dataclasses import dataclass

from . import dialogue
from ..render import actors as actors_mod
from ..render import world

# Beat kinds.
LINE, CAPTION, SILENCE = "line", "caption", "silence"


@dataclass
class Beat:
    kind: str = LINE
    speaker: int | None = None       # index into the cast
    # A skeleton with its slots still in, for beats whose text is realised at
    # the moment they are spoken (captions, the glitch cards, the lines that
    # quote the room). Generated dialogue arrives with its text already
    # written, because the turns refer back to each other and have to be
    # realised in order.
    template: str = ""
    text: str = ""
    duration: float = 3.0
    action: str = actors_mod.TALK    # what the speaker does
    others: str = actors_mod.IDLE    # what everyone else does
    set_name: str | None = None      # None = stay where we are
    glitch: float = 0.0              # extra distortion, 0..1
    walk_in: int | None = None       # cast index that enters on this beat
    walk_out: int | None = None
    # Which scene this belongs to. Beats sharing an id share a subject.
    scene: int = 0
    # How hard the audience goes after this line. 0 = no laugh.
    laugh: float = 0.0
    # The slip: one line that does not belong, which nobody acknowledges.
    # It is flagged rather than inferred because the laugh track has to know
    # to skip it -- the silence after it is the joke, and an audience that
    # laughs is the machine explaining itself. This used to be guaranteed by
    # inserting the slip *after* the laughs were assigned; now that the
    # model chooses it and writes the turns around it, the beat arrives
    # before `_dress` runs and has to say so.
    odd: bool = False


def _read_time(text: str) -> float:
    """Long enough to read, short enough that it does not drag."""
    return max(2.0, min(7.0, 1.5 + len(text) / 13.0))


class Director:
    def __init__(self, cfg, seed: int | None = None):
        self.rng = random.Random(seed)
        self.cast = actors_mod.make_sitcom_cast(self.rng.randrange(1 << 30))
        self.set_name = "APARTMENT"
        self.queue: list[Beat] = []
        # Exchanges written by a model, waiting to be used. They are a supply
        # the director draws on, NOT beats spliced into the queue.
        #
        # Splicing was the obvious way and it was wrong. A model fast enough
        # to keep the queue full meant `_next_bit` never ran, and everything
        # that makes this a show rather than a chat window lives in there:
        # drawing a new subject, counting toward a set change, the silent
        # beats, the stand-up, the glitch. Measured: 100 seconds of model
        # dialogue, `_next_bit` called zero times, one subject and one room
        # for the whole stretch. The director stays in charge; the model
        # supplies the lines inside an exchange.
        self.written: collections.deque[dict] = collections.deque(maxlen=4)
        # Routines are kept apart from exchanges. They are not
        # interchangeable: a set is one voice and lands on the brick wall,
        # so using one as a kitchen conversation would put three people in
        # a room agreeing with themselves.
        self.written_solo: collections.deque[dict] = collections.deque(maxlen=2)
        # Whether anything written has ever arrived. Until it has, and for
        # a few seconds only, the show keeps its mouth shut rather than
        # letting the lexicon write the cold open -- see `_warming`.
        self._ever_supplied = False
        self._opened_at = time.monotonic()
        self._warm_bits = 0
        # Whether a writer is *configured*, which is not the same as one
        # having loaded. `model_writing` cannot answer this: Stage.__init__
        # advances a beat before main has finished picking a writer, and
        # that first beat is the cold open. Being wrong here costs at most
        # WARMUP_SECONDS of staged business, which is a legitimate opening
        # for a sitcom in its own right.
        self.expect_writer = bool(cfg.get("llm.enabled")) or bool(
            cfg.get("cloud.enabled"))
        # Set by main when a model is writing. It does not change who is in
        # charge -- the director still draws the subjects, moves the show
        # between sets and decides when nobody speaks -- it only thins out
        # the places the lexicon puts words in somebody's mouth.
        self.model_writing = False
        self.bits_since_set_change = 0
        self.beats_emitted = 0
        # Skeletons used in the last few scenes. The generator takes this as
        # a hint and reaches for something else, which is what keeps the
        # short replies ("SO?", "IN PASSING.") from surfacing every minute.
        self._recent: collections.deque[str] = collections.deque(maxlen=64)

        # What has actually been said out loud, most recent last. The
        # procedural generator does not need this -- it keeps a structured
        # Exchange instead -- but a language model has no state of its own,
        # and handing it the last two turns is the difference between
        # continuing a conversation and starting the same one again.
        self.spoken: collections.deque[tuple[str, str]] = (
            collections.deque(maxlen=6))

        # Scenes, keyed by id. Only the last few are kept: once a scene has
        # finished playing nothing refers to it again.
        self._scene_ids = itertools.count(1)
        self._scenes: collections.OrderedDict[int, dialogue.Scene] = \
            collections.OrderedDict()
        self.scene = dialogue.new_scene(self.rng)
        self.scene_key = 0

        # The runner joke: generated and realised once, then reused verbatim
        # for the rest of the stream.
        self.callback_text: str | None = None

        scene = self._open_scene()
        self.queue.append(
            Beat(
                kind=CAPTION,
                template=self.rng.choice(dialogue.COLD_OPEN),
                duration=3.0,
                others=actors_mod.IDLE,
                scene=scene,
            )
        )

    # -- subjects ------------------------------------------------------------

    def _open_scene(self) -> int:
        """Draw a subject and open a scene around it. Returns its id.

        The new scene inherits from the outgoing one where it can -- the same
        culprit, or the same place, for a few scenes at a time. Drawing every
        noun fresh every twenty seconds is what made an hour of this feel
        like six hundred unrelated clips rather than one evening.
        """
        scene = dialogue.new_scene(self.rng, previous=self.scene)
        key = next(self._scene_ids)
        self._scenes[key] = scene
        self.scene = scene
        self.scene_key = key
        while len(self._scenes) > 8:
            self._scenes.popitem(last=False)
        return key

    # -- realising -----------------------------------------------------------

    def _realise(self, skeleton: str, speaker: int | None,
                 scene: int = 0) -> str:
        """Fill a skeleton that was not produced by the dialogue generator:
        a caption, a glitch card, a line that quotes the room."""
        sc = self._scenes.get(scene, self.scene)
        if speaker is not None:
            name = self.cast[speaker].name
        elif "{NAME}" in skeleton:
            # Captions have no speaker to attribute to, but "SOMEONE" reads
            # limp next to four characters with actual names -- pick one.
            # This only fills the slot; it does not change who the beat poses
            # or who stage.py treats as talking.
            name = self.rng.choice(self.cast).name
        else:
            name = "SOMEONE"
        others = [a.name for i, a in enumerate(self.cast) if i != speaker]
        other = self.rng.choice(others) if others else "SOMEONE"
        text = dialogue.realise(skeleton, sc, self.rng, other=other)
        return " ".join(text.replace("{NAME}", name).split())

    def _role(self, role: str) -> int:
        return self.cast.index(actors_mod.by_role(self.cast, role))

    def _casting(self, mode: str) -> list[int]:
        """Who fills the generator's speaker slots, for this kind of scene.

        The generator does not know the cast exists; it emits slot 0, slot 1
        and occasionally slot 2. Casting is where the four of them stop being
        interchangeable: the schemes are the wildcard's, the spirals are the
        neurotic's, and nobody else gets the microphone.
        """
        lead, neurotic = self._role("lead"), self._role("neurotic")
        sharp, wildcard = self._role("sharp"), self._role("wildcard")
        if mode == "scheme":
            first, second = wildcard, self.rng.choice((lead, neurotic, sharp))
        elif mode == "spiral":
            first, second = neurotic, self.rng.choice((lead, sharp))
        elif mode == "verdict":
            first, second = self.rng.sample((sharp, neurotic, lead), 2)
        else:
            first, second = self.rng.sample(range(len(self.cast)), 2)
        third = self.rng.choice(
            [i for i in range(len(self.cast)) if i not in (first, second)]
        )
        return [first, second, third]

    def _maybe_establish_callback(self) -> None:
        if self.callback_text is not None:
            return
        # Needs some history behind it, or the catchphrase is fixed before
        # the stream has any character to call back to.
        if self.beats_emitted < 24:
            return
        skeleton = self.rng.choice(dialogue.CALLBACK_SETUP)
        # Realised once, right now, and never again -- that is what makes it
        # a catchphrase instead of just another line.
        self.callback_text = self._realise(skeleton, None, 0)

    # -- scenes --------------------------------------------------------------

    def _dress(self, beats: list[Beat], scene: int, laughs: bool = True,
               slipped: bool = False) -> list[Beat]:
        """Everything that happens to a generated exchange after it is
        written: the audience, the slip, the occasional stage direction."""
        if laughs and beats:
            # The audience is generous and not very discriminating, which is
            # the correct behaviour for a laugh track. The last line is the
            # button and nearly always gets one; anything before it gets one
            # often enough that the room never feels dead.
            for beat in beats[:-1]:
                if beat.kind == LINE and not beat.odd                         and self.rng.random() < 0.34:
                    beat.laugh = self.rng.uniform(0.35, 0.65)
            if beats[-1].kind == LINE and not beats[-1].odd                     and self.rng.random() < 0.88:
                beats[-1].laugh = self.rng.uniform(0.75, 1.0)

        # A slip: one line that does not belong, which nobody acknowledges.
        # It never gets a laugh -- the silence after it is the joke. Never
        # from whoever just spoke: two lines from one mouth read as one
        # thought rather than as an interruption.
        #
        # Rarer when a model is writing: the slip comes from the lexicon by
        # design, and at one exchange in five it was most of what kept the
        # model under its share of the lines. One in eight still lands often
        # enough to be unsettling, which is the only job it has.
        # Nothing here when the exchange already has one. The model is
        # offered the same pool and writes the turns around its choice, so a
        # second slip on top would be two non-sequiturs in four lines, which
        # is not unsettling, it is just noise.
        slip_chance = 0.0 if (self.model_writing or slipped) else 0.22
        if len(beats) > 2 and self.rng.random() < slip_chance:
            at = self.rng.randrange(1, len(beats))
            neighbours = {beats[at - 1].speaker, beats[at].speaker}
            choices = [i for i in range(len(self.cast)) if i not in neighbours]
            speaker = self.rng.choice(choices or range(len(self.cast)))
            fresh = [s for s in dialogue.SLIPS if s not in self._recent]
            slip = self.rng.choice(fresh or dialogue.SLIPS)
            self._recent.append(slip)
            beats.insert(at, Beat(
                kind=LINE, speaker=speaker,
                text=self._realise(slip, speaker, scene),
                action=actors_mod.TWITCH, others=actors_mod.STARE,
                glitch=0.25, scene=scene, odd=True,
            ))

        if self.rng.random() < 0.18:
            beats.append(self._caption(scene))
        return beats

    def _exchange(self, mode: str, set_name: str | None = None,
                  **beat_kw) -> list[Beat]:
        """Open a scene, get the conversation, cast it, stage it.

        The conversation comes from a model if one has written something and
        from the lexicon otherwise. Everything after that -- the audience,
        the slip, the stage directions, the set -- is the same either way,
        which is the point: the writer changed, the direction did not.
        """
        if mode != "standup":
            # Stand-up is the lead alone at a mic, and the lexicon has a
            # purpose-built generator for it. A four-way exchange remapped
            # onto one mouth is not a routine.
            written = self._written_exchange(set_name, **beat_kw)
            if written:
                return written

        key = self._open_scene()
        scene = self._scenes[key]
        cast_slots = self._casting(mode)

        avoid = set(self._recent)
        if mode == "standup":
            turns = dialogue.standup(self.rng, scene, avoid=avoid)
            cast_slots = [self._role("lead")] * 3
        else:
            turns = dialogue.generate(self.rng, scene, mode, avoid=avoid)
        self._recent.extend(avoid - set(self._recent))

        beats = [
            Beat(
                kind=LINE, speaker=cast_slots[slot % len(cast_slots)],
                text=line, duration=_read_time(line),
                action=actors_mod.TALK, others=actors_mod.STARE,
                set_name=set_name if i == 0 else None,
                scene=key,
                **beat_kw,
            )
            for i, (slot, line) in enumerate(turns)
        ]
        return self._dress(beats, key)

    def supply(self, payload: dict) -> bool:
        """Hand the director an exchange somebody else wrote.

        The only way text from outside the project becomes dialogue, so it
        is the one place that has to be careful. Returns False if there is
        nothing usable in it, in which case the lexicon covers as usual.
        """
        if not isinstance(payload, dict):
            return False
        usable = False
        for item in (payload.get("lines") or []):
            text = item.get("text") if isinstance(item, dict) else item
            if str(text or "").strip():
                usable = True
                break
        if not usable:
            return False
        self._ever_supplied = True
        if payload.get("mode") == "standup":
            self.written_solo.append(payload)
        else:
            self.written.append(payload)
        return True

    # How long the show will wait, at startup, for the first written line
    # rather than let the lexicon open the broadcast.
    #
    # It is bounded because the alternative is a stream that is silent for
    # ever if the model never answers.
    #
    # Twelve seconds, not six. The *first* generation is the slow one -- the
    # prompt's KV cache is cold, measured 6.4s against 2.7s once it is warm
    # -- so a six-second window expired just before the first exchange
    # landed, and the cost of being half a second short is not half a line,
    # it is eight: the director asks for an exchange, and if nothing has
    # been written the lexicon writes the whole thing. The opening cutaway
    # is longer than twelve seconds anyway, so in practice nobody waits.
    WARMUP_SECONDS = 12.0

    # And a bound in bits, because the clock is the wrong unit on its own.
    # The director is driven by virtual time -- a test or a fast-forward runs
    # thousands of beats in under a real second -- so a wall-clock-only
    # warmup never expired there and the show staged business for ever: one
    # subject, one room, nobody speaking. Six bits of business is eighteen
    # to forty seconds of airtime, so on a real broadcast the clock still
    # runs out first and this bound never fires.
    WARMUP_BITS = 6

    def _warming(self) -> bool:
        """True while it is worth staying quiet and waiting for the model."""
        return (self.expect_writer
                and not self._ever_supplied
                and self._warm_bits < self.WARMUP_BITS
                and time.monotonic() - self._opened_at < self.WARMUP_SECONDS)

    def _written_exchange(self, set_name: str | None = None,
                          **beat_kw) -> list[Beat]:
        """The next model-written exchange as beats, or [] if there is none."""
        while self.written:
            payload = self.written.popleft()
            # The subject it was written about, not whatever is current now:
            # the request went out several seconds ago, and a slip or a
            # caption dressed onto these beats has to refer to the same
            # thing they do.
            key = payload.get("scene_key") or self.scene_key
            if key not in self._scenes:
                key = self.scene_key
            beats: list[Beat] = []
            odd = False
            for item in (payload.get("lines") or [])[:6]:
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                    who = item.get("speaker")
                else:
                    text, who = str(item).strip(), None
                if not text:
                    continue
                text = " ".join(text.upper().split())[:72]
                if text.startswith("[") or who is None:
                    beats.append(Beat(kind=CAPTION, text=text, scene=key,
                                      duration=_read_time(text)))
                elif isinstance(item, dict) and item.get("odd"):
                    # The slip, chosen by the model out of the curated pool
                    # and written around rather than inserted afterwards.
                    # Same staging as the one the director used to splice
                    # in; the difference is that the turn after it was
                    # written knowing it was there, so the room ignores it
                    # on purpose instead of by accident.
                    odd = True
                    # `beat_kw` can already carry a glitch -- the dispute and
                    # argument bits both set one for the whole exchange -- so
                    # it is merged rather than passed alongside. Handing Beat
                    # two values for the same keyword is a TypeError, and it
                    # only happened when a slip payload met one of those two
                    # bits: one frame in several thousand, which is exactly
                    # the kind of fault an installation meant to run for
                    # weeks hits at four in the morning with nobody watching.
                    kw = dict(beat_kw)
                    glitch = max(0.25, float(kw.pop("glitch", 0.0) or 0.0))
                    beats.append(Beat(
                        kind=LINE, speaker=int(who) % len(self.cast),
                        text=text, duration=_read_time(text),
                        action=actors_mod.TWITCH, others=actors_mod.STARE,
                        glitch=glitch, scene=key, odd=True, **kw,
                    ))
                else:
                    beats.append(Beat(
                        kind=LINE, speaker=int(who) % len(self.cast),
                        text=text, duration=_read_time(text),
                        action=actors_mod.TALK, others=actors_mod.STARE,
                        scene=key, **beat_kw,
                    ))
            if not beats:
                continue
            if set_name:
                beats[0].set_name = set_name
            drift = payload.get("mode") == "existential"
            beats = self._dress(beats, key, laughs=not drift, slipped=odd)
            if drift:
                # No laugh over it. The audience not knowing what to do with
                # it is the whole bit, and a laugh track would be the
                # machine explaining the joke to itself.
                beats.append(Beat(kind=SILENCE, duration=2.4, glitch=0.18,
                                  others=actors_mod.STARE, scene=key))
            # Draw the next subject now, so the exchange after this one is
            # about something else. The lexicon path gets that from
            # _open_scene; the model path has to do it too, or the show sits
            # on one subject for as long as the model keeps writing.
            self._open_scene()
            return beats
        return []

    # -- bits ----------------------------------------------------------------

    def _bit_small_talk(self, ) -> list[Beat]:
        return self._exchange("talk")

    def _bit_verdict(self, ) -> list[Beat]:
        return self._exchange("verdict")

    def _bit_plan(self, ) -> list[Beat]:
        return self._exchange("plan")

    def _bit_dispute(self, ) -> list[Beat]:
        return self._exchange("dispute", glitch=0.12)

    def _bit_argument(self, ) -> list[Beat]:
        return self._exchange("dispute", glitch=0.3)

    def _bit_monologue(self, ) -> list[Beat]:
        return self._exchange("spiral")

    def _bit_neurosis(self, ) -> list[Beat]:
        return self._exchange("spiral")

    def _bit_standup(self, ) -> list[Beat]:
        """The lead alone at the mic, on the brick wall, and then back to
        wherever everyone was.

        It used to be one-way: the show cut to the club and stayed there,
        because nothing ever said to come back. The brick wall ended up
        being 45% of the running time and the apartment -- the home set --
        17%, which is the wrong way round for this show and is most of why
        a stretch of it looked like one background.
        """
        home = self.set_name
        beats = self._solo_exchange() or self._exchange(
            "standup", set_name="STANDUP")
        self.set_name = "STANDUP"
        if beats:
            beats.extend(self._move_to(home, quiet=True))
        return beats

    def _solo_exchange(self) -> list[Beat]:
        """A model-written routine, staged on the brick wall, or [].

        Stand-up was the last bit the lexicon still wrote, on the reasoning
        that a four-way exchange remapped onto one mouth is not a routine.
        That was true of remapping an exchange and false of the conclusion:
        the writer is told it is writing a set and given a grammar with one
        name in it, so there is no second speaker to answer and the form is
        unreachable rather than discouraged.
        """
        lead = self._role("lead")
        while self.written_solo:
            payload = self.written_solo.popleft()
            key = payload.get("scene_key") or self.scene_key
            if key not in self._scenes:
                key = self.scene_key
            beats: list[Beat] = []
            for item in (payload.get("lines") or [])[:4]:
                text = str((item.get("text") if isinstance(item, dict)
                            else item) or "").strip()
                if not text:
                    continue
                text = " ".join(text.upper().split())[:72]
                beats.append(Beat(
                    kind=LINE, speaker=lead, text=text,
                    duration=_read_time(text), action=actors_mod.TALK,
                    others=actors_mod.IDLE, scene=key,
                ))
            if not beats:
                continue
            beats[0].set_name = "STANDUP"
            # Dressed without a slip: a non-sequitur in the middle of a set
            # reads as the comedian losing his place, not as the room
            # declining to notice. Nobody is in the room.
            beats = self._dress(beats, key, slipped=True)
            self._open_scene()
            return beats
        return []

    def _move_to(self, dest: str, quiet: bool = False) -> list[Beat]:
        """Change location. Quiet by default: a beat of nobody saying
        anything on the new set, which is how a sitcom changes scene.

        Every location change used to be a transmission fault with a glitch
        card over it. Once in a while that is a joke; every time, it is the
        only way the show knows how to move, and it makes the place it
        arrives at feel like an error rather than a scene.
        """
        if dest == self.set_name:
            return []
        self.set_name = dest
        self.bits_since_set_change = 0
        beats = [Beat(
            kind=SILENCE, duration=self.rng.uniform(1.0, 1.8),
            set_name=dest, others=actors_mod.IDLE, scene=self.scene_key,
            glitch=0.0 if quiet else 0.5,
        )]
        if not quiet and self.rng.random() < 0.5:
            beats.append(self._caption(self.scene_key, duration=1.8))
        return beats

    def _bit_move(self, ) -> list[Beat]:
        """Go somewhere else and start a scene there."""
        dest = self.rng.choice(
            [s for s in world.SET_NAMES
             if s != self.set_name and s != "STANDUP"]
        )
        return self._move_to(dest) + self._exchange(
            self.rng.choice(("talk", "verdict", "plan", "dispute"))
        )

    def _bit_scheme(self, ) -> list[Beat]:
        beats = self._exchange("scheme")
        if self.rng.random() < 0.4:
            beats.insert(0, Beat(
                kind=SILENCE, duration=1.3, scene=beats[0].scene,
                walk_in=self._role("wildcard"), others=actors_mod.IDLE,
            ))
        return beats

    def _bit_entrance(self, ) -> list[Beat]:
        """The wildcard's entrance, specifically -- it is his bit."""
        beats = self._exchange("scheme")
        beats.insert(0, Beat(
            kind=SILENCE, duration=1.8, walk_in=self._role("wildcard"),
            others=actors_mod.IDLE, scene=beats[0].scene,
            # The entrance gets the laugh, before he has said anything.
            laugh=0.9,
        ))
        return beats

    def _bit_callback(self, ) -> list[Beat]:
        """Deliver the established catchphrase. Its text was realised once
        and is never rewritten."""
        speaker = self.rng.randrange(len(self.cast))
        return [
            Beat(
                kind=LINE, speaker=speaker, text=self.callback_text,
                duration=_read_time(self.callback_text),
                others=actors_mod.STARE, laugh=1.0,
            )
        ]

    def _bit_ritual(self) -> list[Beat]:
        # Not a new subject: the ritual is the room chanting the one that is
        # already on the table.
        key = self.scene_key
        beats = []
        order = list(range(len(self.cast)))
        self.rng.shuffle(order)
        # Short. It is a seasoning, not a scene.
        for i, speaker in enumerate(order[:3]):
            beats.append(
                Beat(
                    kind=LINE, speaker=speaker,
                    template=self.rng.choice(dialogue.RITUAL),
                    duration=self.rng.uniform(0.9, 1.5),
                    action=actors_mod.TALK, others=actors_mod.STARE,
                    glitch=min(0.5, i * 0.08), scene=key,
                )
            )
        beats.append(Beat(kind=SILENCE, duration=2.0, others=actors_mod.STARE,
                          scene=key))
        return beats

    def _bit_malfunction(self, ) -> list[Beat]:
        new_set = self.rng.choice(
            [s for s in world.SET_NAMES
             if s != self.set_name and s != "STANDUP"]
        )
        # A glitch card is not a subject either.
        key = self.scene_key
        beats = [
            Beat(
                kind=CAPTION,
                template=self.rng.choice(dialogue.GLITCH),
                duration=self.rng.uniform(1.0, 2.0),
                glitch=1.0,
                others=actors_mod.TWITCH,
                scene=key,
            ),
            Beat(
                kind=SILENCE, duration=1.2, glitch=0.8,
                set_name=new_set, others=actors_mod.STARE, scene=key,
            ),
        ]
        self.set_name = new_set
        self.bits_since_set_change = 0
        return beats

    def _caption(self, scene: int, duration: float = 2.0) -> Beat:
        return Beat(
            kind=CAPTION,
            template=self.rng.choice(dialogue.STAGE_DIRECTIONS),
            duration=duration,
            others=actors_mod.IDLE,
            scene=scene,
        )

    def _bit_staring(self, ) -> list[Beat]:
        # Pacing. Without these the stream is exhausting.
        beats = [
            Beat(kind=SILENCE, duration=self.rng.uniform(2.0, 4.5),
                 others=actors_mod.STARE, scene=self.scene_key),
        ]
        if self.rng.random() < 0.4:
            beats.append(self._caption(self.scene_key, duration=2.0))
        return beats

    def _bit_business(self, ) -> list[Beat]:
        """Nobody says anything. One person drifts, fidgets, or just stands
        there for a while; everyone else does their own thing.

        A room with people in it is mostly not dialogue, and a beat with no
        text is genuinely silent on screen -- nothing renders at all, not
        even dimmed.
        """
        mover = self.rng.randrange(len(self.cast))
        action = self.rng.choice(
            (actors_mod.IDLE, actors_mod.GESTURE, actors_mod.TWITCH)
        )
        others = self.rng.choice((actors_mod.STARE, actors_mod.IDLE))
        return [
            Beat(
                kind=SILENCE, speaker=mover, action=action, others=others,
                duration=self.rng.uniform(3.0, 6.5), scene=self.scene_key,
            )
        ]

    # -- scheduling ----------------------------------------------------------

    # Talk outweighs silence roughly three to one by running time: the silent
    # bits are one beat each, the exchanges are five or six. "business" and
    # "staring" exist to stop the room narrating itself without a pause, not
    # to be the main event -- that was the old failure mode, where a quiet
    # mic meant forty minutes of people muttering "HUH." at each other.
    _WEIGHTED = (
        ("small_talk", 7),
        ("verdict", 5),
        ("plan", 4),
        ("dispute", 4),
        ("business", 6),
        ("staring", 4),
        ("argument", 2),
        ("standup", 3),
        ("scheme", 3),
        ("neurosis", 3),
        ("monologue", 2),
        ("entrance", 2),
        ("ritual", 1),
        ("malfunction", 1),
        ("move", 2),
    )

    def _next_bit(self) -> list[Beat]:
        if self._warming():
            # Nothing is written yet and the broadcast has only just come
            # up. Stage business rather than let the lexicon open the show:
            # the cold open is the one stretch where waiting costs nothing,
            # because a room with people quietly doing nothing in it is a
            # perfectly good first ten seconds of a sitcom.
            self._warm_bits += 1
            return self._bit_business()

        self._maybe_establish_callback()

        # The catchphrase, rarely -- it is only funny because it is not
        # expected. At this rate it lands every few minutes, not every bit.
        if self.callback_text and self.rng.random() < 0.015:
            return self._bit_callback()

        self.bits_since_set_change += 1
        names, weights = zip(*self._WEIGHTED)
        # The longer we have been in one place, the more likely we move.
        weights = list(weights)
        if self.model_writing:
            # The ritual is the room chanting the subject back at itself,
            # three turns of it, out of a four-phrase lexicon list. There is
            # nothing in it for a model to write -- the content is the
            # subject, which the director already owns -- so with a writer
            # attached it is three lexicon lines at a time and nothing else.
            # It is a seasoning at weight 1; the show does not miss it.
            weights[names.index("ritual")] = 0
        # The longer we have been in one place the more likely we move, and
        # the move is the ordinary kind rather than the glitch.
        #
        # Counted in bits, but the pressure is meant to be felt in minutes,
        # and a bit is not a fixed length. With a model writing, an exchange
        # is six lines instead of four and the show runs about one bit every
        # seventeen seconds against the lexicon's seven, so the old
        # thresholds came up two and a half times slower in wall-clock
        # terms: measured, one room change in four minutes. Same intent,
        # scaled to the writer.
        patience, glitch_after = (3, 5) if self.model_writing else (5, 8)
        if self.bits_since_set_change > patience:
            weights[names.index("move")] += self.bits_since_set_change
        if self.bits_since_set_change > glitch_after:
            weights[names.index("malfunction")] += self.bits_since_set_change
        choice = self.rng.choices(names, weights=weights, k=1)[0]
        return getattr(self, f"_bit_{choice}")()

    def pump(self, minimum: int = 2) -> None:
        """Keep the queue stocked. Called every frame; cheap when full.

        This fills from `_next_bit`, which is also what draws subjects and
        moves the show between sets, so it has to keep running whoever is
        writing the dialogue. A model's exchanges arrive through `supply`
        and get picked up inside a bit, never spliced in ahead of one.
        """
        guard = 0
        while len(self.queue) < minimum and guard < 8:
            guard += 1
            self.queue.extend(self._next_bit())

    def next_beat(self) -> Beat:
        self.pump()
        self.beats_emitted += 1
        beat = self.queue.pop(0)
        if beat.template:
            # Skeletons that were held back are realised now, at the moment
            # they are spoken, so a line that quotes the room quotes the
            # freshest thing it has said.
            beat.text = self._realise(beat.template, beat.speaker, beat.scene)
            if beat.kind == LINE and beat.duration == 3.0:
                beat.duration = _read_time(beat.text)
        if beat.kind == LINE and beat.speaker is not None and beat.text:
            self.spoken.append((self.cast[beat.speaker].name, beat.text))
        return beat

    def recent_lines(self) -> list[tuple[str, str]]:
        """The last few turns as (name, text). For whoever writes the next."""
        return list(self.spoken)

    # -- cloud ---------------------------------------------------------------

    def lead_name(self) -> str:
        """Who holds the microphone. The writer needs the name, not the slot:
        it is the only name its stand-up grammar will accept."""
        return self.cast[self._role("lead")].name

    def cast_names(self) -> list[str]:
        return [a.name for a in self.cast]
