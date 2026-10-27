"""Check that the broadcast always comes back.

The stream has exactly two segments -- the sitcom and the climbing cutaway --
and the only thing that moves between them is App._schedule. There is nobody
watching the installation, so a segment that fails to end is a failure nobody
reports: it just looks like the show stopped. That happened once already, and
not as a crash -- the cutaway was five minutes against a two-and-a-half-minute
sitcom floor, so the wall had 62% of the running time and the first real break
was twelve times longer than the twenty-five-second opener everyone had
already been shown. Nothing was stuck. It only looked that way, which is the
same thing to someone in the room.

So: drive the real scheduler with fake time and fake scene changes, and
assert the shape of the result rather than any one number. Scenes are advanced
on a fixed cadence instead of by running the stage, which keeps this instant;
what is under test is the scheduler, not the director.

Run after touching anything under `cutaway` in the config. Nothing reads this
at runtime.
"""
import os
import statistics
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

pygame.display.init()

from pixelmeme import config as C
from pixelmeme import main as main_mod

# How long one scene lasts, measured off the director: a location change every
# fifty seconds or so. The scheduler only ever sees the count.
SCENE_SECONDS = 50.0
DT = 0.5                      # coarse: the scheduler has no per-frame state
HOURS = 3.0

cfg = C.load(None)
cfg._data["display"]["fullscreen"] = False
app = main_mod.App(cfg)

opener = float(cfg.get("cutaway.opening_seconds"))
break_len = float(cfg.get("cutaway.duration_minutes")) * 60.0
floor = float(cfg.get("cutaway.min_live_seconds"))
ceiling = float(cfg.get("cutaway.max_live_seconds"))

t = 0.0
since_scene = 0.0
segments = [[app.state, 0.0, 0.0]]       # state, start, end
while t < HOURS * 3600.0:
    t += DT
    if app.state == main_mod.LIVE:
        since_scene += DT
        if since_scene >= SCENE_SECONDS:
            since_scene -= SCENE_SECONDS
            app.stage.scenes_played += 1
    was = app.state
    app._schedule(DT)
    segments[-1][2] = t
    if app.state != was:
        since_scene = 0.0
        segments.append([app.state, t, t])

# The last segment was cut off by the clock, not by the scheduler.
segments.pop()

spans = {main_mod.LIVE: [], main_mod.IDLE: []}
for state, start, end in segments:
    spans[state].append(end - start)

fails = []


def check(ok: bool, label: str, detail: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {label}: {detail}")
    if not ok:
        fails.append(label)


live, idle = spans[main_mod.LIVE], spans[main_mod.IDLE]
print(f"{HOURS:.0f}h: {len(live)} sitcom stretches, {len(idle)} cutaways")

check(bool(live) and bool(idle), "both segments play",
      f"{len(live)} live, {len(idle)} idle")
if not live or not idle:
    sys.exit(1)

check(len(live) >= 10, "the sitcom returns, repeatedly",
      f"{len(live)} times in {HOURS:.0f}h")

# Every segment ended. Any that did not would have been the last one, which
# is popped above -- so this is really a check that the clock outlived many
# of them rather than one.
longest = max(live)
check(longest <= ceiling + DT, "no sitcom stretch outruns max_live_seconds",
      f"longest {longest:.0f}s, ceiling {ceiling:.0f}s")
check(min(live) >= floor - DT, "no sitcom stretch undercuts min_live_seconds",
      f"shortest {min(live):.0f}s, floor {floor:.0f}s")

# The opener is the first cutaway and is meant to be short; every one after it
# is the real break and they are all the same length.
check(abs(idle[0] - opener) <= DT * 2, "the opener is the short one",
      f"{idle[0]:.0f}s, configured {opener:.0f}s")
rest = idle[1:]
check(bool(rest) and max(rest) - min(rest) <= DT * 2,
      "every real break is the configured length",
      f"{min(rest):.0f}-{max(rest):.0f}s, configured {break_len:.0f}s")

# The two that actually matter. A break longer than the programme is not a
# break, and a first break many times longer than the opener trains the
# viewer to expect twenty-five seconds and then hands them minutes.
mean_live = statistics.mean(live)
check(break_len < mean_live, "the break is shorter than the programme",
      f"break {break_len:.0f}s vs sitcom mean {mean_live:.0f}s")
check(break_len <= opener * 4, "the first real break is near the opener",
      f"{break_len:.0f}s vs opener {opener:.0f}s "
      f"({break_len / opener:.1f}x)")

total = sum(live) + sum(idle)
share = sum(live) / total * 100
check(share >= 70.0, "the sitcom is the main event",
      f"{share:.0f}% of the running time")
print(f"sitcom stretch min/mean/max {min(live):.0f}/{mean_live:.0f}/"
      f"{max(live):.0f}s, break {break_len:.0f}s")

print()
print(f"==== {'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
sys.exit(1 if fails else 0)
