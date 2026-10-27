"""Check the on-device writer: does it produce usable dialogue, and fast enough.

Two halves, and the first one runs anywhere:

**The filter.** Fixed samples of the things a small model actually does wrong
-- prose instead of dialogue, a speaker who is not in the cast, curly quotes,
a paragraph where a line should be, markdown, the few-shot examples recited
back -- pushed through the real parser, asserting what survives. No model
needed, so this part runs in CI and on a Pi with nothing installed.

**The model.** If the GGUF is actually present: generate a batch of exchanges
against real scenes and report the yield, the latency, the length
distribution, and every line it wrote, so the register can be read rather
than inferred. Skips cleanly when the file is missing -- the app's whole
design is that an absent model is not an error.

    python tools/check_llm.py            # filter checks, then the model if present
    python tools/check_llm.py --n 12     # more samples
    python tools/check_llm.py --filter-only
"""
from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pixelmeme import config as C                        # noqa: E402
from pixelmeme.brain import dialogue, llama              # noqa: E402

NAMES = ["JERRY", "GEORGE", "ELAINE", "KRAMER"]

# Whether a line is one of the curated large ones. Exact, not vocabulary:
# the first version of this check scored the register by looking for words
# like REMEMBER and ALONE, and it reported 6 of 8 on a run where the real
# answer was 8 of 8 -- it simply had no entry for "WILL STILL BE HERE WHEN
# I'M NOT". A metric that is wrong in the reassuring direction is worse than
# no metric, so this matches the pool's own templates instead, with the
# slots turned into patterns.
_POOL = tuple(
    re.compile(
        "^" + re.escape(line.text)
        .replace(r"\{N\}", r"\d+")
        .replace(r"\{IT!\}", ".+")
        .replace(r"\{IT\}", ".+")
        .replace(r"\{GUY\}", ".+")
        .replace(r"\{PLACE\}", ".+") + "$"
    )
    for line in dialogue.MOVES["existential"]
)


def is_large(text: str) -> bool:
    return any(p.match(text) for p in _POOL)


fails: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {label}{': ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


# -- the filter ---------------------------------------------------------------

def filter_checks() -> None:
    print("-- filter " + "-" * 60)

    # Clean output survives intact.
    got = llama.parse(
        "JERRY: HE TOOK THE SECOND ONE.\nELAINE: THERE WAS ONLY ONE.\n",
        NAMES,
    )
    check(got == [("JERRY", "HE TOOK THE SECOND ONE."),
                  ("ELAINE", "THERE WAS ONLY ONE.")],
          "clean dialogue passes through", str(got))

    # Narration, markdown and preamble are not dialogue.
    noise = (
        "Sure! Here is a short exchange:\n"
        "## The Bagel\n"
        "*They are standing in the kitchen.*\n"
        "[LAUGHTER]\n"
        "JERRY: SO WHERE IS IT.\n"
        "The scene continues as they argue about the bagel for some time.\n"
        "ELAINE: IN THE DRAWER, WHERE YOU LEFT IT.\n"
    )
    got = llama.parse(noise, NAMES)
    check(got == [("JERRY", "SO WHERE IS IT."),
                  ("ELAINE", "IN THE DRAWER, WHERE YOU LEFT IT.")],
          "narration and markdown are dropped", str(got))

    # A speaker who is not in the room never reaches the screen: the subtitle
    # bar colours the name from the actor, so an unknown one is not a
    # cosmetic problem, it is a lookup that has to not happen.
    got = llama.parse("NEWMAN: HELLO, JERRY.\nJERRY: HELLO, NEWMAN.\n", NAMES)
    check(got == [("JERRY", "HELLO, NEWMAN.")],
          "a speaker outside the cast is dropped", str(got))

    # Characters the font cannot draw would render as solid boxes.
    got = llama.parse("JERRY: “IT’S” MINE — ALWAYS WAS… \U0001f600\n"
                      "GEORGE: SINCE WHEN.\n", NAMES)
    check(got and got[0][1] == "IT'S MINE - ALWAYS WAS...",
          "smart quotes, dashes and emoji are folded or dropped",
          repr(got[0][1] if got else None))
    for _, text in got:
        check(set(text) <= set(llama.ALPHABET), "output stays in the alphabet",
              repr(text))

    # Long lines are cut on a word boundary: the bar is three lines of about
    # twenty-six characters and the renderer will not scroll.
    long_line = "JERRY: " + "THE SECOND BAGEL WAS NEVER MINE AND I HAVE SAID SO REPEATEDLY FOR YEARS NOW.\n"
    got = llama.parse(long_line + "GEORGE: SO?\n", NAMES)
    check(bool(got) and len(got[0][1]) <= llama.MAX_LINE,
          "a long line is cut to the subtitle width",
          f"{len(got[0][1]) if got else 0} chars: {got[0][1] if got else ''}")
    check(bool(got) and " " not in got[0][1][-1:] and not got[0][1].endswith("-"),
          "the cut lands on a word boundary", got[0][1] if got else "")

    # A finished line is left alone. The trim that removes a dangling last
    # word took the "BEFORE" off a curated existential line and would have
    # taken the "FOR" off "SO WHAT IS THE GUEST TOWEL FOR." -- it is only
    # supposed to fix a line the word bound cut mid-clause, not edit writing
    # that was already finished.
    for intact in (
        "I'M NOT AFRAID OF THE END. I'M AFRAID OF THE PART BEFORE.",
        "SO WHAT IS THE GUEST TOWEL FOR.",
        "WHERE IS IT FROM.",
        "SO?",
    ):
        check(llama.clean(intact) == intact,
              "a finished line is not trimmed", repr(llama.clean(intact)))

    # ...but a cut that landed mid-clause is repaired.
    for broken, want in (
        ("WE HAD TOLD HIM TO COME IN BY NOW, BUT?",
         "WE HAD TOLD HIM TO COME IN BY NOW."),
        ("IT WAS IN THE", "IT WAS IN."),
    ):
        check(llama.clean(broken) == want,
              "a mechanical cut is repaired", repr(llama.clean(broken)))

    # The few-shot examples recited back are not writing.
    got = llama.parse("JERRY: THERE WAS A SECOND BAGEL?\n"
                      "GEORGE: THERE IS ALWAYS A SECOND BAGEL.\n", NAMES)
    check(got == [], "lines copied from the examples are dropped", str(got))

    # A line already spoken is not a new line.
    got = llama.parse("JERRY: SO WHERE IS IT.\nELAINE: I NEVER TOUCHED IT.\n",
                      NAMES, recent=[("GEORGE", "SO WHERE IS IT.")])
    check(got == [("ELAINE", "I NEVER TOUCHED IT.")],
          "a line just spoken is not repeated", str(got))

    # Two turns from one mouth read as one thought, so the attribution moves.
    got = llama.parse("JERRY: IT WAS IN THE DRAWER.\n"
                      "JERRY: AND NOW IT IS NOT.\n", NAMES)
    check(len(got) == 2 and got[0][0] != got[1][0],
          "consecutive turns are reattributed", str(got))

    # Nobody addresses themselves by name. Small models do this constantly
    # and it is the clearest possible sign that nobody is home.
    got = llama.parse("ELAINE: THERE WAS A LAST HAIRCUT, ELAINE.\n"
                      "JERRY: I SAW HIM WITH HIS PANTS DOWN, KRAMER.\n", NAMES)
    check(got and got[0][1] == "THERE WAS A LAST HAIRCUT.",
          "a self-addressed vocative is dropped", str(got[:1]))
    check(len(got) > 1 and got[1][1].endswith(", KRAMER."),
          "a vocative to somebody else is kept", str(got[1:2]))

    # The same thought twice running is the show stuttering, and a
    # contraction is not a new line.
    got = llama.parse("GEORGE: WE ARE THE SAME BOSS!\n"
                      "ELAINE: WE'RE THE SAME BOSS!\n"
                      "KRAMER: SO WHAT NOW.\n", NAMES)
    check(len(got) == 2, "a near-repeat is dropped, contraction or not",
          str([t for _, t in got]))

    # Nothing usable means nothing is injected -- the lexicon covers it.
    check(llama.parse("I cannot help with that request.", NAMES) == [],
          "a refusal yields no lines")
    check(llama.parse("", NAMES) == [], "empty output yields no lines")

    # The two rules must not collide. Lines copied from the few-shot examples
    # are dropped, and the existential lines are forced by the grammar -- so
    # if an example ever quoted a pool line verbatim, every forced line would
    # be silently thrown away and the register would vanish with no error
    # anywhere. They are deliberately paraphrases of each other, which is
    # exactly the kind of thing a later edit breaks without noticing.
    rng = random.Random(0)
    for label, pool in (("existential", llama.large_lines),
                        ("slip", llama.slip_lines)):
        clash = [t for t in pool(dialogue.new_scene(rng), rng, 99)
                 if t in llama._SHOT_TEXT]
        check(not clash,
              f"no forced {label} line is droppable as an example copy",
              str(clash[:2]))

    # Stand-up is a set, not a conversation, and the grammar is the only
    # thing that guarantees it. One name reachable means there is nobody to
    # answer; with four, the model wrote a four-way exchange and called it
    # a routine. The prompt asking for one voice did not do this.
    solo = llama.grammar_for(["JERRY"], 4, min_lines=3)
    name_rule = [r for r in solo.splitlines() if r.startswith("name ::=")][0]
    check(name_rule.strip() == 'name ::= "JERRY"',
          "the stand-up grammar reaches exactly one speaker", name_rule)
    check(solo.startswith("root ::= line line line line?"),
          "the stand-up grammar is three or four lines",
          solo.splitlines()[0])

    # `parse` reattributes two consecutive turns from one mouth, which is
    # right for an exchange and catastrophic for a routine: it handed every
    # tag after the premise to somebody else in the room.
    # Not a line from the few-shot examples: the filter drops those as
    # recitations, which would make this look like a parser failure.
    routine = ("JERRY: THE LIST HAS A SECOND PAGE.\n"
               "JERRY: NOBODY HAS EVER TURNED IT OVER.\n")
    check([w for w, _ in llama.parse(routine, NAMES, solo=True)]
          == ["JERRY", "JERRY"],
          "a routine stays in one mouth when parsed solo")
    check(len({w for w, _ in llama.parse(routine, NAMES)}) == 2,
          "an exchange still gets its second turn reattributed")

    # The word cap. Ten was met by welding two words into one rather than by
    # stopping -- "MORE THANYOURSELF" -- and every such line was exactly ten
    # words long. The number is load-bearing, so it is asserted here.
    check("{1,11} end" in llama.grammar_for(NAMES, 4),
          "a line is bounded at twelve words, not ten",
          [r for r in llama.grammar_for(NAMES, 4).splitlines()
           if r.startswith("text")][0])

    # Every shape keeps an ordinary turn after the forced one. A forced line
    # in last position is an ending, not a register: the deflection after it
    # is what makes it bearable.
    for shape in llama.FORCED_SHAPES:
        check(shape.split()[-1] == "line" and "big" in shape,
              f"forced shape {shape!r} keeps a turn after the odd line")

    # The grammar. "It compiled" is not a check -- that is what was here
    # before, and it passed happily while llama.cpp was rejecting the whole
    # grammar over one escaped hyphen and returning something that
    # constrained nothing. from_string does not raise on a bad grammar.
    #
    # So: scan the text we emit for the thing that actually goes wrong. The
    # only escape llama.cpp's parser accepts in what we generate is a
    # backslash-n; any other backslash kills the whole grammar silently.
    allowed = {chr(92) + "n"}
    for label, text in (
        ("plain", llama.grammar_for(NAMES, 4)),
        ("drift", llama.grammar_for(NAMES, 4, ["NOTHING LASTS HERE."])),
        ("standup", llama.grammar_for(["JERRY"], 4, min_lines=3)),
        ("slip", llama.grammar_for(NAMES, 4, ["THE CEILING KNOWS."],
                                   shape="line line big line")),
    ):
        escapes = {text[i:i + 2] for i in range(len(text) - 1)
                   if text[i] == chr(92)}
        check(escapes <= allowed,
              f"the {label} grammar uses no escape llama.cpp lacks",
              str(sorted(escapes)))

    # Whether it is *enforced* cannot be checked without the model, and that
    # is what the forced-line count in the model half is for.
    try:
        from llama_cpp import LlamaGrammar
        LlamaGrammar.from_string(llama.grammar_for(NAMES, 4), verbose=False)
        check(True, "the GBNF grammar is accepted (weak: see above)")
    except ImportError:
        print("skip  the GBNF grammar (llama-cpp-python not installed)")
    except Exception as exc:
        check(False, "the GBNF grammar is accepted",
              f"{type(exc).__name__}: {exc}")


# -- the model ----------------------------------------------------------------

def model_checks(n: int, drift_rate: float | None = None,
                 model: str | None = None, threads: int | None = None) -> None:
    print()
    print("-- model " + "-" * 61)
    cfg = C.load(None)
    if drift_rate is not None:
        cfg._data["llm"]["existential_chance"] = drift_rate
    if model:
        # Pinning one file is how the boards get compared: the Pi runs a
        # different model from the desktop, so "it reads fine" has to be
        # checked against the one that will actually be running there.
        cfg._data["llm"]["model_path"] = [model]
        cfg._data["llm"]["ram_headroom_mb"] = 0.0
    if threads:
        cfg._data["llm"]["threads"] = threads
    scripter = llama.LlamaScripter(cfg, rng=random.Random(7))
    if not scripter.start():
        print(f"skip  {scripter.status}")
        print("      python tools/get_model.py")
        return

    print(f"{Path(scripter.model_path).name}, {scripter.threads} threads, "
          f"n_ctx {scripter.n_ctx}")

    # Does the worst-case prompt fit? The drift variant is the long one, and
    # an overflow is not a wrong line, it is llama.cpp raising and the whole
    # exchange being dropped -- which looks like the model being bad at its
    # job rather than like a number being too small.
    scripter._load()
    worst = (llama.SYSTEM + chr(10) * 2 + llama.DRIFT
             + "".join(a + b for a, b in llama.SHOTS[:2])
             + llama._brief(dialogue.new_scene(random.Random(2)),
                            [("KRAMER", "X" * 70), ("JERRY", "Y" * 70)],
                            NAMES))
    used = len(scripter._llm.tokenize(worst.encode("utf-8")))
    need = used + scripter.max_tokens
    check(need <= scripter.n_ctx,
          "the worst-case prompt fits the context window",
          f"{used} prompt + {scripter.max_tokens} output = {need} "
          f"of {scripter.n_ctx}")

    rng = random.Random(11)
    scene = dialogue.new_scene(rng)
    recent: list[tuple[str, str]] = []
    ok, empty, lat, lengths, drifts = 0, 0, [], [], 0
    drift_hits = 0

    for i in range(n):
        # Alternate: half the calls get the previous exchange as context, so
        # the continuity path is exercised and not just the cold one.
        if i % 2 == 0:
            scene = dialogue.new_scene(rng, previous=scene)
            recent = []
        t0 = time.monotonic()
        payload = scripter._generate(
            llama._brief(scene, recent, NAMES), NAMES, recent, scene
        )
        lat.append((time.monotonic() - t0) * 1000.0)
        if payload is None:
            empty += 1
            print(f"  {i:2d} ---- {scripter.status}")
            continue
        ok += 1
        drift = payload["mode"] == "existential"
        if drift:
            drifts += 1
            if any(is_large(l["text"]) for l in payload["lines"]):
                drift_hits += 1
        mark = "~" if drift else " "
        print(f"  {i:2d} {mark} [{scene.thing}]")
        recent = []
        for line in payload["lines"]:
            who = NAMES[line["speaker"]]
            lengths.append(len(line["text"]))
            recent.append((who, line["text"]))
            big = "  <--" if drift and is_large(line["text"]) else ""
            print(f"       {who:>7}: {line['text']}{big}")

    print()
    check(ok > 0, "the model produces usable exchanges",
          f"{ok}/{n} ({empty} dropped by the filter)")
    if ok:
        check(ok / n >= 0.6, "most calls yield something usable",
              f"{ok / n * 100:.0f}%")
        check(max(lengths) <= llama.MAX_LINE, "every line fits the bar",
              f"longest {max(lengths)} of {llama.MAX_LINE}")
        print(f"      line length min/mean/max "
              f"{min(lengths)}/{statistics.mean(lengths):.0f}/{max(lengths)}")
        print(f"      latency    min/mean/max "
              f"{min(lat):.0f}/{statistics.mean(lat):.0f}/{max(lat):.0f}ms")
        print(f"      existential {drifts}/{ok}, of which "
              f"{drift_hits} actually said something large")
        if drifts:
            # The register is the thing that was asked for, so it is the
            # thing that has to be checked rather than assumed. Not every
            # drift exchange has to land -- it is a 1B model -- but if the
            # prompt is doing nothing at all, this is where that shows.
            check(drift_hits / drifts >= 0.5,
                  "the existential prompt actually produces the register",
                  f"{drift_hits}/{drifts}")
        # The queue holds roughly twenty seconds of dialogue. Slower than
        # that is not a failure -- the lexicon covers the gap -- but it is
        # the number that decides how much of the show the model writes.
        if statistics.mean(lat) > 20000:
            print("      note: slower than the queue drains; expect the "
                  "lexicon to carry most of the show on this box")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="model samples")
    ap.add_argument("--filter-only", action="store_true")
    ap.add_argument("--drift", type=float, default=None,
                    help="force the existential rate, e.g. 1.0, to sample it")
    ap.add_argument("--model", help="a specific GGUF, instead of the configured one")
    ap.add_argument("--threads", type=int, help="override the thread count")
    args = ap.parse_args(argv)

    filter_checks()
    if not args.filter_only:
        model_checks(args.n, args.drift, args.model, args.threads)

    print()
    print(f"==== {'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
