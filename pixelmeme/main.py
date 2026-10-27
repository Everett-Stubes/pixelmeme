"""pixelmeme -- a continuous pixel broadcast of a room that does not exist.

A small fixed cast acts out half-coherent conversations forever, generated on
the device from a lexicon and a conversational-move machine. Every so often --
a station break, not a fallback -- it cuts away to a looping climb for a few
minutes, then cuts back.

Nothing here is episodic and nothing feeds it from outside. The stage never
resets and the stream never cuts except for that one scheduled break.

Run with --help for the flags.
"""

from __future__ import annotations

import argparse
import random
import signal
import sys
import traceback

import pygame

from . import config as config_module
from .brain.cloud import CloudCaptioner
from .brain.llama import LlamaScripter
from .render import pixelfont as pf
from .render.display import Display
from .render.idle import IdleScene
from .render.stage import Stage

LIVE, IDLE = "LIVE", "IDLE"
TRANSITION_SECONDS = 1.6


class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.running = True
        self.state = IDLE
        self.show_debug = bool(cfg.get("debug.overlay", False))
        self.frame_errors = 0

        # The climb is a scheduled segment, not a reaction to anything. It
        # fires after a few scenes rather than on a clock -- a "scene" being
        # one continuous stretch on one set, the same unit a call sheet uses.
        self.cutaway_every = float(cfg.get("cutaway.every_scenes", 3.0))
        self.cutaway_duration = float(cfg.get("cutaway.duration_minutes", 5.0)) * 60.0
        # Floor and ceiling in seconds, so a run of very short scenes cannot
        # cut away every twenty seconds and a very long one still breaks.
        self.cutaway_min_live = float(cfg.get("cutaway.min_live_seconds", 150.0))
        self.cutaway_max_live = float(cfg.get("cutaway.max_live_seconds", 1500.0))
        self._live_elapsed = 0.0
        # Opens on the climb so that is the first thing on screen -- but only
        # briefly. Sitting on a five-minute cutaway before the sitcom has
        # been seen once reads as the thing being broken.
        self.opening_seconds = float(cfg.get("cutaway.opening_seconds", 25.0))
        self._cutaway_elapsed = 0.0
        self._first_cutaway = True
        self._scenes_target = self._roll_scene_target()
        self._transition_text: str | None = None
        self._transition_t = 0.0

        self.display = Display(cfg)
        self.base_size = (self.display.base_w, self.display.base_h)
        self.live_palette = cfg.get("display.palette")
        self.idle_palette = cfg.get("display.idle_palette")

        self.stage = Stage(cfg, self.base_size)
        print(f"[stage] cast: {', '.join(self.stage.director.cast_names())}")

        # Who writes the dialogue. Two optional writers in front of the
        # lexicon generator, and the lexicon generator underneath both of
        # them covering every frame neither one has produced a line for.
        # The API wins if both are on: it is the more deliberate switch.
        self.cloud = CloudCaptioner(cfg)
        self.llama = LlamaScripter(cfg)
        self.writer: CloudCaptioner | LlamaScripter | None = None
        for name, candidate in (("claude", self.cloud), ("llama", self.llama)):
            if self.writer is not None or not candidate.enabled:
                continue
            if candidate.start():
                self.writer = candidate
                print(f"[writer] {name}: {candidate.status}")
            else:
                print(f"[writer] {name} unavailable: {candidate.status}")
        if self.writer is None:
            print("[writer] the lexicon generator")
        else:
            self.stage.director.model_writing = True

        self.idle_scene = IdleScene(cfg, self.base_size)
        print(f"[idle] backend: {self.idle_scene.backend_name}")

        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, _signum, _frame) -> None:
        self.running = False

    # -- state ---------------------------------------------------------------

    def _roll_scene_target(self) -> int:
        """Average `cutaway_every` scenes, give or take one."""
        lo = max(1, int(round(self.cutaway_every)) - 1)
        hi = int(round(self.cutaway_every)) + 1
        return random.randint(lo, hi)

    def _go_live(self) -> None:
        self.state = LIVE
        self.display.use_palette(self.live_palette)
        self._live_elapsed = 0.0
        self.stage.scenes_played = 0
        self._scenes_target = self._roll_scene_target()
        self._transition_text = "AND NOW, BACK TO OUR REGULARLY SCHEDULED NOTHING"
        self._transition_t = 0.0

    def _go_idle(self) -> None:
        self.state = IDLE
        self.display.use_palette(self.idle_palette)
        self._cutaway_elapsed = 0.0
        self._transition_text = "MEANWHILE..."
        self._transition_t = 0.0

    # -- main loop -----------------------------------------------------------

    def run(self) -> int:
        self.display.use_palette(self.idle_palette)
        while self.running:
            dt = self.display.tick()
            self._handle_events()
            try:
                self._step(dt)
            except Exception:
                # A broken frame must never take the installation down; it is
                # supposed to run unattended for weeks.
                self.frame_errors += 1
                if self.frame_errors <= 5:
                    traceback.print_exc()
                elif self.frame_errors == 6:
                    print("[main] further frame errors suppressed")
                self.display.base.fill((0, 0, 0))
            self.display.present()
        self.shutdown()
        return 0

    def _schedule(self, dt: float) -> None:
        """Advance the clocks and switch segment if one is due.

        This runs before any of the work a frame actually does, and it is
        the only thing that decides which segment we are in. The run loop
        swallows a frame that raises so the installation cannot be taken
        down by one bad beat -- but that means anything after the throw is
        skipped, and if the clocks lived down there a permanently broken
        frame would pin the broadcast inside whichever segment it was in.
        A stuck sitcom is a black screen; a stuck cutaway is five minutes
        of climbing that never ends. Neither can happen from up here.
        """
        if self.state == LIVE:
            self._live_elapsed += dt
            enough_scenes = self.stage.scenes_played >= self._scenes_target
            if self._live_elapsed >= self.cutaway_max_live or (
                enough_scenes and self._live_elapsed >= self.cutaway_min_live
            ):
                self._go_idle()
        else:
            self._cutaway_elapsed += dt
            if self._cutaway_elapsed >= self._cutaway_limit:
                self._first_cutaway = False
                self._go_live()

    @property
    def _cutaway_limit(self) -> float:
        return (
            self.opening_seconds if self._first_cutaway
            else self.cutaway_duration
        )

    def _step(self, dt: float) -> None:
        if self._transition_text is not None:
            self._transition_t += dt
            if self._transition_t >= TRANSITION_SECONDS:
                self._transition_text = None

        self._schedule(dt)

        if self.state == LIVE:
            self.stage.update(dt)
            self._maybe_write()
        else:
            self.idle_scene.update(dt)
            # Keep the script stocked so coming back is instant.
            self.stage.director.pump()

        self._draw()

    def _maybe_write(self) -> None:
        """Collect a finished exchange and start the next one.

        Both writers are asynchronous and neither is waited for. A generation
        that is slower than the queue drains is not a stall, it is a stretch
        of lexicon dialogue -- which is why the floor exists.
        """
        writer = self.writer
        if writer is None:
            return
        director = self.stage.director
        # Drain everything that is ready, not one per frame: the writer can
        # finish two exchanges between two of these calls, and one left in
        # its buffer is an exchange the lexicon writes instead.
        while True:
            result = writer.take()
            if result is None:
                break
            director.supply(result)
        # And keep it working whenever it has somewhere to put the answer.
        # Waiting for the queue to run down guaranteed the lexicon got there
        # first; the writer's own buffer and its minimum interval pace it now.
        writer.request(director.scene, director.cast_names(),
                       director.recent_lines(), director.scene_key,
                       director.lead_name())

    def _draw(self) -> None:
        base = self.display.base
        if self._transition_text is not None:
            # A quick cut-to-black card between segments, like a station
            # break -- cheaper and clearer than crossfading the scenes.
            base.fill((10, 8, 14))
            pf.draw_block(
                base, self._transition_text, self.base_size[0] // 2,
                self.base_size[1] // 2 - 6, self.base_size[0] - 10,
                scale=1, colour=(236, 238, 242), outline=None, max_lines=3,
            )
        elif self.state == IDLE:
            self.idle_scene.draw(base)
        else:
            self.stage.draw(base)

        if self.show_debug:
            self._draw_debug(base)

    def _draw_debug(self, base) -> None:
        if self.state == LIVE:
            timing = (f"SC{self.stage.scenes_played}/{self._scenes_target} "
                      f"{self._live_elapsed:.0f}S")
        else:
            timing = f"BACK {self._cutaway_limit - self._cutaway_elapsed:.0f}S"
        lines = [
            f"{self.state} {self.display.clock.get_fps():.0f}F",
            timing,
            self.stage.status,
        ]
        if self.writer is self.llama:
            lines.append(f"LLM {self.llama.status[:12]}")
            lines.append(f"{self.llama.calls}OK {self.llama.failures}X "
                         f"{self.llama.drifts}E")
            # The two registers the lexicon used to own, so a glance at the
            # overlay says whether the writer is actually covering them.
            lines.append(f"{self.llama.standups}S {self.llama.slips}P")
        elif self.writer is not None:
            lines.append(f"CLD {self.cloud.status[:12]} {self.cloud.calls}")
        for i, line in enumerate(lines):
            pf.draw_text(base, line, 1, 1 + i * 7, 1, (0, 230, 230), (0, 0, 0))

    # -- input ---------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_f:
                    self.display.toggle_fullscreen()
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_SPACE:
                    # Skip the current beat.
                    if self.stage.beat is not None:
                        self.stage.beat_t = self.stage.beat.duration
                elif event.key == pygame.K_i:
                    if self.state == IDLE:
                        self._go_live()
                    else:
                        self._go_idle()

    def shutdown(self) -> None:
        self.display.shutdown()
        pygame.quit()


# -- entry point -------------------------------------------------------------


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="pixelmeme", description=__doc__)
    p.add_argument("--config", help="path to a YAML config file")
    p.add_argument("--windowed", action="store_true", help="do not go fullscreen")
    p.add_argument("--fps", type=int, help="frame rate cap")
    p.add_argument("--debug", action="store_true", help="show the debug overlay")
    p.add_argument(
        "--cloud",
        action="store_true",
        help="write the script with the Claude API (needs ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="skip the on-device model and write from the lexicon",
    )
    p.add_argument("--llm-model", help="path to a GGUF model file")
    p.add_argument("--palette", help="sitcom, vhs, fried, gameboy, stone")
    p.add_argument("--rotate", type=int, choices=(0, 90, 180, 270))
    p.add_argument(
        "--fit",
        choices=("auto", "fill", "stretch"),
        help="auto = crisp integer scale, fill = use the whole screen height",
    )
    p.add_argument(
        "--base",
        help="internal render size, WxH, e.g. 90x160 (must be portrait)",
    )
    return p.parse_args(argv)


def _apply_overrides(cfg, args) -> None:
    data = cfg._data
    if args.windowed:
        data["display"]["fullscreen"] = False
    if args.fps:
        data["display"]["fps"] = args.fps
    if args.debug:
        data["debug"]["overlay"] = True
    if args.cloud:
        data["cloud"]["enabled"] = True
    if args.no_llm:
        data["llm"]["enabled"] = False
    if args.llm_model:
        data["llm"]["model_path"] = args.llm_model
        data["llm"]["enabled"] = True
    if args.palette:
        data["display"]["palette"] = args.palette
    if args.rotate is not None:
        data["display"]["rotate"] = args.rotate
    if args.fit:
        data["display"]["fit"] = args.fit
    if args.base:
        w, _, h = args.base.lower().partition("x")
        data["display"]["base_width"] = int(w)
        data["display"]["base_height"] = int(h)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    cfg = config_module.load(args.config)
    _apply_overrides(cfg, args)

    if cfg.get("display.base_width") >= cfg.get("display.base_height"):
        print("[display] warning: base size is not portrait; this is meant to be tall")

    # display only: nothing here makes a sound, so the mixer is never
    # initialised and the app does not care whether the box has one.
    pygame.display.init()
    app = App(cfg)
    try:
        return app.run()
    except KeyboardInterrupt:
        app.shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
