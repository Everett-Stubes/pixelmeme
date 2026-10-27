"""Local text generation: a small instruct model, running on the device.

Llama 3.2 1B Q4 where there is memory for it, Qwen2.5-0.5B Q4 on a 1GB Pi 3.
`llm.model_path` is a list and the first entry that is present and small
enough to fit wins, so one config covers every board -- see `start`.

The procedural generator in dialogue.py is still here and still correct. This
is the writer that sits in front of it: the model is loaded from a GGUF file
through llama-cpp-python and asked for one short exchange at a time, on a
worker thread. Nothing about the broadcast waits for it. If the file is
missing, the package is not installed, the box is too small, or the model
writes something unusable, the stream carries on out of the lexicon and the
only trace is a line in the debug overlay.

A 1B model is small enough to be wrong about everything. Three things make it
usable here:

**The grammar.** llama.cpp will constrain sampling to a GBNF grammar, so
`NAME: LINE`, from this cast, two to ten words, ending on a full stop or a
question mark, in the alphabet the pixel font can draw, is not a request --
it is the only thing the sampler can emit. That removes, before generation
rather than after, every failure that actually hurts: prose paragraphs,
markdown, stage directions, emoji, speakers who are not in the cast, lines
that run to the token limit, and characters that render as a solid box.

Everything the prompt merely asks for, it gets ignored on. "Short. Under
twelve words." was a prompt line first and did nothing; in the grammar it is
simply true. Treat the prompt as steering and the grammar as the contract.

**The examples.** At this size, instructions are weaker than demonstrations.
The small register is carried by few-shot pairs rather than by adjectives.
They are the *same* pairs every call by default (`llm.stable_prompt`), so the
prompt prefix never changes and llama.cpp can reuse the KV cache for it
instead of re-evaluating 500 tokens -- which on a Pi is most of the work.
They used to be rotated, to stop the model reciting one back; at temperature
0.75 it does not, and the filter drops copies anyway.

**The filter.** Everything that comes back is still treated as untrusted
input: re-cased, stripped to the drawable alphabet, trimmed of a trailing
word that cannot end a sentence, cut to the subtitle width, checked against
the last few lines actually spoken, and dropped entirely unless enough usable
turns survive. The show is 20 fps on a fixed-size subtitle bar; a line that
does not fit is not a line.

Mostly it writes the small stuff -- who moved the chair, whether that counts
as a second bagel. Once in a while the exchange is the other thing instead:
one turn far too large for the room, in the same flat voice, which nobody
picks up. That one is **forced by the grammar from a curated pool**, not
asked for; see DRIFT below for why, and for what asking produced instead.
"""

from __future__ import annotations

import importlib.util
import os
import queue
import random
import re
import threading
import time
from dataclasses import replace
from pathlib import Path

from . import dialogue
from ..render import pixelfont as pf

# The subtitle bar is 108px wide at 1px scale: about 26 characters a line,
# three lines. 70 characters is a comfortable two and a bit.
MAX_LINE = 70
MIN_LINE = 6
MIN_LINES = 2

# What a line may contain. Deliberately narrower than what the font can draw:
# brackets are how the show marks a stage direction, and a colon is how the
# output format marks a speaker, so neither may appear inside speech.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.'?!-"

# A guard, not a style choice: if the font cannot draw it, it reaches a frame
# as a solid box.
assert set(ALPHABET) <= pf.SUPPORTED, set(ALPHABET) - pf.SUPPORTED

# Curly quotes, dashes and ellipses are the characters a model reaches for
# first and the ones this font has never heard of.
_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": "", "\u201d": "", "\u201e": "", "\u2033": "",
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...", "\u00a0": " ", "`": "'", "\u00b4": "'",
    ";": ",", ":": ",", "\u2022": " ",
}

_SPEECH = re.compile(r"^\s*([A-Z][A-Z \-']*?)\s*:\s*(.+?)\s*$")

# Words that cannot end a sentence. A 1B model rambles, so lines do get cut,
# and a cut on the nearest word boundary leaves things like "I HAVE SAID SO
# REPEATEDLY FOR." -- which reads as a fault rather than as clipped speech.
# Dropping the dangling word is the difference between a line and a glitch.
_DANGLING = frozenset("""
AND OR BUT SO FOR NOR YET IF AS THAN THAT WHICH WHO WHOSE WHILE BECAUSE
TO OF IN ON AT BY FROM WITH WITHOUT INTO ONTO OVER UNDER ABOUT AFTER BEFORE
THE A AN THIS THESE THOSE MY YOUR HIS HER ITS OUR THEIR
""".split())


# -- the prompt ---------------------------------------------------------------

SYSTEM = """
You write dialogue for a pixel-art sitcom that never ends. Four friends in a
New York apartment argue about nothing in particular, forever.

FORMAT. Dialogue only, one turn per line, exactly like this:
NAME: WHAT THEY SAY
Four to six turns. No stage directions, no narration, no titles, no
explanations, no quotation marks, no asterisks.

STYLE.
- Every turn is about the one thing you are given. Never change the subject.
- Short. Under twelve words. Ordinary spoken English, present tense.
- The argument is never about whether it matters. It is about the detail:
  which one, how many, what time, who said so.
- Treat something trivial as a matter of principle. That is the whole joke.
- Nobody wins, nobody apologises, nobody sums it up at the end.
- Nothing cruel, nothing about anybody's body, health or private business.
""".strip()

# The thing that is occasionally asked for instead. Deliberately one turn --
# the register only works if the room does not notice it.
#
# This prompt is no longer what produces the register; the grammar is.
# Two rounds of prompting for it scored **zero out of ten** on whether
# the exchange contained anything large at all -- the model cheerfully
# marked an exchange existential and then wrote "MY BOSS SAID THAT
# WOULD BE A DISASTER". Dedicated few-shot examples did not move it
# either. A 1B model will not hold two registers at once and no amount
# of adjectives changes that, so the large line is drawn from the
# curated pool in dialogue.py and forced by the grammar. The model then
# does what it is actually good at: choosing which one fits here, and
# writing the small lines either side of it including the deflection
# that follows. The prompt stays because it still steers those.
DRIFT = """
One of the turns, and only one, is different. Somebody says something far too
large for the room -- about time, or being the only person who is really here,
or how long this has all been going on -- in exactly the same flat voice they
would use about a sandwich. They are not upset and they do not explain it.
Nobody answers it. The next turn carries on with the small thing.
""".strip()

# Demonstrations, not instructions: at this size the register comes from
# these, not from the adjectives in SYSTEM. The first two are used, in order,
# on every call -- see `stable_prompt` for why that matters more than variety
# does.
SHOTS = (
    (
        "JERRY, GEORGE, ELAINE, KRAMER\n"
        "The thing: THE SECOND BAGEL\n"
        "Who is to blame: MY DENTIST\n"
        "Where: THE DELI",
        "GEORGE: HE TOOK THE SECOND BAGEL.\n"
        "JERRY: THERE WAS A SECOND BAGEL?\n"
        "GEORGE: THERE IS ALWAYS A SECOND BAGEL.\n"
        "ELAINE: NOT AT FOUR IN THE AFTERNOON.\n",
    ),
    (
        "JERRY, GEORGE, ELAINE, KRAMER\n"
        "The thing: THE PARKING SPOT\n"
        "Who is to blame: THE NIGHT MANAGER\n"
        "Where: THE PARKING GARAGE",
        "ELAINE: HE SIGNED FOR IT.\n"
        "KRAMER: YOU CANNOT SIGN FOR A SPOT.\n"
        "ELAINE: HE HAD A CLIPBOARD, KRAMER.\n",
    ),
    (
        "JERRY, GEORGE, ELAINE, KRAMER\n"
        "The thing: THE GUEST TOWEL\n"
        "Who is to blame: MY COUSIN\n"
        "Where: THE LAUNDRY ROOM",
        "KRAMER: SHE USED THE GUEST TOWEL.\n"
        "JERRY: SHE IS A GUEST.\n"
        "KRAMER: SHE LIVES HERE NOW. THAT IS DIFFERENT.\n"
        "GEORGE: SO WHAT IS THE GUEST TOWEL FOR.\n",
    ),
)

# The drift needs its own demonstrations, not a paragraph of instruction
# appended to the small-register ones. Asking for "one turn that is far too
# large for the room" while still showing four examples of people arguing
# about a towel produced exchanges marked existential that contained nothing
# of the kind -- "MY BOSS SAID THAT WOULD BE A DISASTER" -- because at 1B the
# demonstrations beat the instruction every time. That is the whole reason
# the register is carried by examples here; writing prose for the one case
# that most needed an example was exactly the wrong way round.
#
# What these show and the words cannot: small, enormous, small. The large
# line is flat and unexplained, and the next turn carries on with the towel.
DRIFT_SHOTS = (
    (
        "JERRY, GEORGE, ELAINE, KRAMER\n"
        "The thing: THE SECOND BAGEL\n"
        "Who is to blame: MY DENTIST\n"
        "Where: THE DELI",
        "GEORGE: HE TOOK THE SECOND BAGEL.\n"
        "JERRY: THERE WAS A SECOND BAGEL?\n"
        "GEORGE: NONE OF THIS GETS REMEMBERED BY ANYBODY.\n"
        "JERRY: SO THAT IS A NO ON THE BAGEL.\n",
    ),
    (
        "JERRY, GEORGE, ELAINE, KRAMER\n"
        "The thing: THE PARKING SPOT\n"
        "Who is to blame: THE NIGHT MANAGER\n"
        "Where: THE PARKING GARAGE",
        "ELAINE: HE SIGNED FOR IT.\n"
        "KRAMER: WE ARE ALL JUST WAITING IN DIFFERENT ROOMS.\n"
        "ELAINE: HE HAD A CLIPBOARD, KRAMER.\n",
    ),
    (
        "JERRY, GEORGE, ELAINE, KRAMER\n"
        "The thing: THE GUEST TOWEL\n"
        "Who is to blame: MY COUSIN\n"
        "Where: THE LAUNDRY ROOM",
        "KRAMER: SHE USED THE GUEST TOWEL.\n"
        "GEORGE: THE BUILDING WILL OUTLIVE ALL OF US.\n"
        "KRAMER: SHE LIVES HERE NOW. THAT IS DIFFERENT.\n",
    ),
)

# Stand-up used to be the one bit the lexicon kept, on the grounds that a
# four-way exchange remapped onto one mouth is not a routine. That was true
# and it was the wrong conclusion: the fix is to tell the model it is writing
# a routine, not to take the pen off it. One name in the grammar is all it
# takes to make the form unreachable -- there is no second speaker to answer,
# so the model cannot write a conversation even by accident.
STANDUP = """
You write stand-up for a pixel-art sitcom that never ends. One comedian, on
stage, in front of a brick wall.

FORMAT. Dialogue only, one sentence per line, exactly like this:
NAME: WHAT THEY SAY
Three or four lines. No stage directions, no narration, no titles, no
explanations, no quotation marks, no asterisks.

STYLE.
- It is a set, not a conversation. Nobody answers. There is no audience in
  the writing -- no "folks", no "am I right", no "you ever notice".
- First line sets up the one thing you are given. The rest are tags on it:
  the same observation pushed one step further each time.
- Short. Ordinary spoken English, present tense.
- The comedian is annoyed by something too small to be annoyed by, and
  completely sincere about it.
- No punchline that explains the joke. Stop on the smallest detail.
- Nothing cruel, nothing about anybody's body, health or private business.
""".strip()

STANDUP_SHOTS = (
    (
        "JERRY\n"
        "The thing: THE SECOND BAGEL\n"
        "Who is to blame: MY DENTIST\n"
        "Where: THE DELI",
        "JERRY: THERE IS ALWAYS A SECOND BAGEL.\n"
        "JERRY: NOBODY ORDERS IT. IT ARRIVES.\n"
        "JERRY: AND NOW IT IS YOUR BAGEL.\n",
    ),
    (
        "ELAINE\n"
        "The thing: THE PARKING SPOT\n"
        "Who is to blame: THE NIGHT MANAGER\n"
        "Where: THE PARKING GARAGE",
        "ELAINE: THE SPOT WAS MINE AT SIX.\n"
        "ELAINE: AT SEVEN IT BELONGS TO A MAN WITH A CLIPBOARD.\n"
        "ELAINE: THE CLIPBOARD IS THE WHOLE SYSTEM.\n",
    ),
    (
        "GEORGE\n"
        "The thing: THE GUEST TOWEL\n"
        "Who is to blame: MY COUSIN\n"
        "Where: THE LAUNDRY ROOM",
        "GEORGE: WE HAVE A TOWEL NOBODY IS ALLOWED TO DRY ANYTHING ON.\n"
        "GEORGE: IT HANGS THERE FOR YEARS.\n"
        "GEORGE: WAITING FOR A BETTER CLASS OF HAND.\n",
    ),
)

_SHOT_TEXT = " ".join(
    reply for _, reply in SHOTS + DRIFT_SHOTS + STANDUP_SHOTS
).upper()


def grammar_for(names, max_lines=4, large=None, min_lines=MIN_LINES,
                shape="line big line"):
    """GBNF: turns of `NAME: SPEECH`, and nothing else.

    This is the whole reason a 1B model is viable. Everything the show cannot
    render -- a markdown heading, a paragraph of narration, a speaker who is
    not in the room, a curly quote -- is unreachable rather than filtered.

    With `large`, the shape is fixed at small / large / small and the middle
    turn can only be one of those lines. The model still chooses which one
    and still writes the two around it; what it can no longer do is quietly
    decline to change register, which is what it did every time it was asked.
    """
    speakers = " | ".join('"%s"' % n for n in names)
    # The alphabet as a GBNF character class. The hyphen goes LAST and
    # unescaped: llama.cpp's grammar parser has no \- escape and rejects the
    # whole grammar if it sees one -- and `LlamaGrammar.from_string` prints
    # "error parsing grammar" to stderr and then hands back an object that
    # constrains nothing at all, rather than raising. So a grammar with one
    # bad character in it is not a crash, it is every guarantee in this file
    # silently not applying while the output still looks fine because the
    # few-shot examples are carrying the format on their own.
    #
    # The line is bounded in words and has to end on a full stop, question
    # mark or exclamation. "Short. Under twelve words." was in the prompt and
    # was simply not obeyed: once the grammar let a line run to the token
    # limit, a turn after a forced one degenerated into "MY BOSS HAD TO GO BY
    # THE CLOCK. KRAMER. ELAINE. JERRY. ELAINE. KRAMER." A bound in the
    # grammar cannot be ignored the way a sentence in a prompt can, and it
    # ends the line on a cadence instead of on a word count.
    #
    # The word bound is twelve, not ten. At ten, a model that had planned a
    # longer sentence met the cap by welding the last two words into one --
    # "MORE THANYOURSELF", "A BAGEL OF MYOWN", "DROPPING BY FOR THEACK" --
    # and every single one of those lines was exactly ten words long. A
    # non-word on screen reads as a rendering fault rather than as a joke,
    # which is worse than a long line. Twelve words is about seventy
    # characters, which is where `_shorten` cuts anyway, so the guillotine
    # now belongs to the filter: it trims at a word boundary and `_undangle`
    # tidies the result, instead of the model improvising a compound noun.
    rules = [
        'line ::= name ": " text "\\n"\n',
        "name ::= %s\n" % speakers,
        'text ::= word (" " word){1,11} end\n',
        "word ::= [A-Z0-9] [A-Z0-9'-]* \",\"?\n",
        'end ::= "." | "?" | "!"\n',
    ]
    if large:
        # `shape` is where the forced turn sits. It has to have at least one
        # ordinary turn after it: the room not picking the line up, out
        # loud, and carrying on about the soup is the entire device, and a
        # forced line in last position is just an ending.
        rules.insert(0, "root ::= %s\n" % shape)
        rules.append('big ::= name ": " huge "\\n"\n')
        rules.append("huge ::= %s\n" % " | ".join('"%s"' % t for t in large))
    else:
        least = max(1, min(min_lines, max_lines))
        required = " ".join("line" for _ in range(least))
        optional = " ".join("line?" for _ in range(max(0, max_lines - least)))
        rules.insert(0, "root ::= %s %s\n" % (required, optional))
    return "".join(rules)


def compile_grammar(text: str):
    """Compile a GBNF grammar, raising if llama.cpp could not parse it.

    `LlamaGrammar.from_string` does not raise on a bad grammar. It writes
    "error parsing grammar" to the C library's stderr -- not Python's -- and
    returns an object that constrains nothing, so the sampler runs free while
    every guarantee in this module silently stops applying. That is a much
    worse failure than an exception, because the output still looks right:
    the few-shot examples hold the format on their own most of the time.

    One escaped hyphen cost two rounds of measuring the wrong thing, so the
    stderr of the C call is captured at the file-descriptor level and a parse
    error is turned into the exception it should have been.
    """
    from llama_cpp import LlamaGrammar

    saved = os.dup(2)
    reader, writer = os.pipe()
    try:
        os.dup2(writer, 2)
        os.close(writer)
        try:
            grammar = LlamaGrammar.from_string(text, verbose=False)
        finally:
            os.dup2(saved, 2)
        noise = os.read(reader, 1 << 16).decode("utf-8", "replace")
    finally:
        os.close(reader)
        os.close(saved)

    if "error" in noise.lower():
        raise ValueError("llama.cpp rejected the grammar: %s"
                         % " ".join(noise.split())[:200])
    return grammar


# Where the forced turn can sit. Every one of them keeps at least one
# ordinary turn afterwards, for the deflection.
FORCED_SHAPES = (
    "line big line",
    "line big line line",
    "line line big line",
    "line line big line line",
)


def forced_lines(pool, scene, rng, limit=12):
    """Candidate lines from a curated pool, realised against this scene.

    Used for the two registers a 1B model will not produce on request: the
    existential turn and the slip. In both cases the model still chooses
    which line belongs here and still writes the turns either side of it --
    including the deflection that has to follow -- but it cannot quietly
    decline to change register, which is what it did every single time it
    was merely asked to.
    """
    local = replace(scene, said=set(getattr(scene, "said", set())))
    out = []
    for line in pool:
        raw = getattr(line, "text", line)
        text = clean(dialogue.realise(raw, local, rng))
        if MIN_LINE <= len(text) <= MAX_LINE and '"' not in text:
            out.append(text)
    rng.shuffle(out)
    return out[:limit]


def large_lines(scene, rng, limit=12):
    """Candidate large lines, realised against this scene.

    They come from `dialogue.MOVES["existential"]` rather than from a second
    copy here: that pool is the register, it is already written, and it
    already knows it gets exactly one deflection and then the scene goes back
    to being about a parking spot.
    """
    # On a copy: realising a line marks the subject as named, and the lexicon
    # generator reads that flag to decide whether to say "IT". Burning it here
    # would leave a later line referring to something never mentioned.
    return forced_lines(dialogue.MOVES["existential"], scene, rng, limit)


def slip_lines(scene, rng, limit=12):
    """Candidate slips: one line that does not belong, from `dialogue.SLIPS`.

    The director used to insert these itself, after the model had finished,
    which made every slip a lexicon line in the middle of a written exchange.
    Offering the pool to the model instead costs nothing -- it is the same
    curated line either way -- and buys something the insertion could not:
    the turn after it is written knowing the slip is there, so the room
    ignores it on purpose rather than by accident.
    """
    return forced_lines(dialogue.SLIPS, scene, rng, limit)


def _brief(scene, recent: list[tuple[str, str]], names: list[str]) -> str:
    """The scene's own facts, plus what was actually just said."""
    lines = [
        ", ".join(names),
        "The thing: %s" % (getattr(scene, "thing", "")
                           or "NOTHING IN PARTICULAR"),
        "Who is to blame: %s" % (getattr(scene, "guy", "") or "NOBODY"),
        "Where: %s" % (getattr(scene, "place", "") or "HERE"),
    ]
    if recent:
        # Continuity. Without this every exchange opens cold on the same
        # subject, which reads as the conversation restarting every ten
        # seconds -- the exact thing the procedural generator was rewritten
        # to stop doing.
        said = "\n".join("%s: %s" % (who, text) for who, text in recent[-2:])
        lines.append("Already said just now:\n%s" % said)
    return "\n".join(lines)


# -- the model ----------------------------------------------------------------


def _win_mem() -> tuple[float, float] | None:
    """(total, available) MB on Windows, or None."""
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem = _Mem()
        mem.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        return mem.ullTotalPhys / 1e6, mem.ullAvailPhys / 1e6
    except Exception:
        return None


def total_ram_mb() -> float | None:
    """Physical RAM, or None if we cannot tell."""
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e6
    except (AttributeError, ValueError, OSError):
        pass
    win = _win_mem()
    return win[0] if win else None


def available_ram_mb() -> float | None:
    """RAM actually free for a new allocation, or None if we cannot tell.

    Total is the wrong number to gate on. A Pi 3 has 1GB on paper, and what
    is left after the GPU split, the OS and -- if there is one -- a desktop
    session is what decides whether a model loads or the board spends the
    next week swapping to an SD card. On Linux that is MemAvailable, which
    is the kernel's own estimate and accounts for reclaimable cache.
    """
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024
    except OSError:
        pass
    win = _win_mem()
    return win[1] if win else None


class LlamaScripter:
    """Generates exchanges on a worker thread. Never blocks a frame."""

    def __init__(self, cfg, rng: random.Random | None = None):
        c = cfg["llm"]
        self.enabled = bool(c["enabled"])
        # One config for every board. `model_path` may be a list, and the
        # first entry that is both on disk and small enough to fit wins --
        # so the Pi, which only ever has the 0.5B file, and the desktop,
        # which has the 1B, read the same config and need no profiles.
        raw = c["model_path"]
        paths = [raw] if isinstance(raw, str) else list(raw)
        self.candidates = [cfg.resolve_path(p) for p in paths]
        self.model_path = None
        self.headroom_mb = float(c["ram_headroom_mb"])
        self.n_ctx = int(c["n_ctx"])
        # One core stays free for the render loop. llama.cpp drops the GIL
        # during inference, so this is not a Python problem -- it is four
        # threads of matrix multiply against a 20 fps budget on the same four
        # cores, and the frame loses. Generation being slower is invisible;
        # the frame rate dropping is the only thing anybody can see.
        # One core left for the renderer, capped at eight because a model
        # this small stops scaling there.
        #
        # Thread count turned out not to be the frame-rate lever. Cutting to
        # four changed nothing (14.4 fps against 14.9) and cost a tenth of
        # the lines: what the broadcast loses is the per-token Python work
        # in llama-cpp-python's sampling loop, which holds the GIL and costs
        # the same whatever the matrix multiply is doing. More threads
        # finish sooner, so the window the render loop has to share is
        # shorter. The real lever is generating less often -- see
        # max_lines_per_call.
        self.threads = int(c["threads"]) or min(
            8, max(1, (os.cpu_count() or 2) - 1)
        )
        self.max_tokens = int(c["max_tokens"])
        self.temperature = float(c["temperature"])
        self.top_p = float(c["top_p"])
        self.repeat_penalty = float(c["repeat_penalty"])
        self.max_lines = int(c["max_lines_per_call"])
        self.drift_chance = float(c["existential_chance"])
        self.standup_chance = float(c["standup_chance"])
        self.slip_chance = float(c["slip_chance"])
        self.min_interval = float(c["min_interval_seconds"])
        # Rotating the few-shot examples changes the prompt prefix on every
        # call, and llama.cpp can only reuse a prefix it has already
        # evaluated. Measured on the 1B: stable prefix 6.4s then 3.2s then
        # 2.7s as the cache warms, against a flat 4.3-4.9s rotating. On a Pi
        # the prompt is most of the work, so this is the single biggest knob
        # there. Rotation existed to stop the model reciting an example back,
        # and at 0.75 it does not -- and the filter drops copies anyway.
        self.stable_prompt = bool(c["stable_prompt"])

        self.rng = rng or random.Random()
        self.status = "disabled"
        self.calls = 0
        self.failures = 0
        self.drifts = 0
        self.standups = 0
        self.slips = 0
        self.last_ms = 0.0

        self._llm = None
        self._grammar = None
        self._grammar_for: list[str] = []
        # Stand-up gets its own cached grammar: one name in it, so the model
        # has nobody to write an answer for.
        self._solo_grammar = None
        self._solo_for = ""
        # Four deep: the model is faster than the broadcast on a
        # desktop, and a full buffer is what lets it stay ahead of
        # the queue instead of the lexicon covering the gap.
        self._results: queue.Queue[dict] = queue.Queue(maxsize=4)
        self._inflight = threading.Event()
        self._lock = threading.Lock()
        self._dead = False
        self._last_request = 0.0

    # -- startup -------------------------------------------------------------

    def start(self) -> bool:
        """Check what we can check without paying for a model load.

        The load itself happens on the first request, on the worker thread:
        a 1B model takes seconds to map and the broadcast is already on
        screen by then.
        """
        if not self.enabled:
            return False
        # find_spec rather than an import: importing llama_cpp loads the
        # native library, and at this point we are still inside startup with
        # a frame due.
        if importlib.util.find_spec("llama_cpp") is None:
            return self._disable("llama-cpp-python not installed")

        present = [p for p in self.candidates if p and Path(p).exists()]
        if not present:
            return self._disable("no model at %s" % (self.candidates[0]
                                                     if self.candidates else "?"))

        # The gate is the model's own size against the memory actually free,
        # not a fixed number. A fixed 1800MB floor was both too strict and
        # too lax: it refused a 400MB model on a 1GB board that fits it
        # perfectly well, and it would have waved through a 2GB model on a
        # 2GB board that cannot hold it. Measured here: the app is 52MB and
        # the 1B at Q4 brings the process to 958MB resident, which is the
        # file plus about 150MB of context and overhead.
        free = available_ram_mb()
        too_big = []
        for path in present:
            need = Path(path).stat().st_size / 1e6 + self.headroom_mb
            if free is None or free >= need:
                self.model_path = path
                self.status = "ready"
                return True
            too_big.append((Path(path).name, need))

        # Report the *smallest* one that still did not fit: naming the 1B
        # when a 0.5B was also rejected tells you to find a smaller model you
        # have already got. Not a crash if one were allowed through -- worse.
        # The board would swap to the SD card and the frame rate would go.
        name, need = min(too_big, key=lambda t: t[1])
        return self._disable("%s needs %.0fMB, %.0fMB free"
                             % (name, need, free))

    def _disable(self, why: str) -> bool:
        self.status = why
        self.enabled = False
        return False

    def _load(self) -> bool:
        if self._llm is not None:
            return True
        if self._dead:
            return False
        from llama_cpp import Llama

        try:
            t0 = time.monotonic()
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.threads,
                # Small batch: this is a 4-core SBC with no GPU, and a big
                # batch buys throughput we do not need at one exchange a
                # minute while costing memory we do not have.
                n_batch=64,
                # -1 reseeds every call, which is what keeps two identical
                # prompts from producing the identical exchange.
                seed=-1,
                verbose=False,
            )
            self.status = "loaded in %.1fs" % (time.monotonic() - t0)
            return True
        except Exception as exc:
            self._dead = True
            self.status = "load failed: %s" % type(exc).__name__
            return False

    # -- request side --------------------------------------------------------

    def ready_to_request(self) -> bool:
        if not self.enabled or self._dead:
            return False
        if self._inflight.is_set():
            return False
        if self._results.full():
            return False
        return (time.monotonic() - self._last_request) >= self.min_interval

    def request(self, scene, cast_names: list[str],
                recent: list[tuple[str, str]] | None = None,
                scene_key: int = 0, lead: str | None = None) -> bool:
        """Start one generation on a worker thread. False if skipped."""
        if not self.ready_to_request():
            return False
        self._inflight.set()
        self._last_request = time.monotonic()
        # The scene is frozen here rather than read on the worker: the
        # director moves on to the next one while this is in flight, and an
        # exchange written about one subject with the large line realised
        # against a different one is worse than either.
        snapshot = replace(scene, said=set(getattr(scene, "said", set())))
        threading.Thread(
            target=self._worker,
            args=(_brief(snapshot, list(recent or []), cast_names),
                  list(cast_names), list(recent or []), snapshot, scene_key,
                  lead or (cast_names[0] if cast_names else "JERRY")),
            name="llama-script",
            daemon=True,
        ).start()
        return True

    def _worker(self, brief: str, names: list[str],
                recent: list[tuple[str, str]], scene,
                scene_key: int = 0, lead: str = "JERRY") -> None:
        try:
            payload = self._generate(brief, names, recent, scene, lead)
            if payload is not None:
                # Which subject this was written about. By the time it is
                # used the director has moved on, and a slip dressed onto
                # these beats has to refer to the same thing they do.
                payload["scene_key"] = scene_key
                try:
                    self._results.put_nowait(payload)
                except queue.Full:
                    pass
        except Exception as exc:          # a frame must never see this
            self.failures += 1
            self.status = "error: %s" % type(exc).__name__
        finally:
            self._inflight.clear()

    def _generate(self, brief: str, names: list[str],
                  recent: list[tuple[str, str]], scene,
                  lead: str = "JERRY") -> dict | None:
        with self._lock:                  # one llama.cpp context, one caller
            if not self._load():
                return None
            # One register per call, picked here. They are exclusive on
            # purpose: a 1B model given two things to do at once does
            # neither, which is the whole lesson of the existential turn.
            roll = self.rng.random()
            standup = roll < self.standup_chance
            drift = (not standup
                     and roll < self.standup_chance + self.drift_chance)
            slip = (not standup and not drift
                    and roll < (self.standup_chance + self.drift_chance
                                + self.slip_chance))

            if standup:
                # A routine, not a conversation, and not a continuation of
                # one: `recent` is deliberately dropped so the prompt prefix
                # stays identical call to call and the comedian is not
                # answering something somebody said in the kitchen.
                brief = _brief(scene, [], [lead])
                system, pool = STANDUP, list(STANDUP_SHOTS)
            elif drift:
                system, pool = SYSTEM + "\n\n" + DRIFT, list(DRIFT_SHOTS)
            else:
                system, pool = SYSTEM, list(SHOTS)
            messages = [{"role": "system", "content": system}]
            # Same two examples, same order, every call: that keeps the
            # prompt prefix byte-identical so llama.cpp reuses the KV cache
            # it already built instead of evaluating 500 tokens again.
            shots = pool[:2] if self.stable_prompt else self.rng.sample(pool, 2)
            for shot, reply in shots:
                messages.append({"role": "user", "content": shot})
                messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": brief})

            forced: list[str] = []
            if standup:
                # One name in the grammar. The form is unreachable rather
                # than discouraged: there is no second speaker to answer,
                # so the model cannot drift back into an exchange.
                if self._solo_for != lead:
                    # Three or four, and the grammar says so. The prompt
                    # already asked for three or four and got six every
                    # time: a premise with five tags on it is not a tighter
                    # routine, it is the same sentence restarting, and the
                    # last two lines were always where it came apart.
                    self._solo_grammar = compile_grammar(
                        grammar_for([lead], 4, min_lines=3)
                    )
                    self._solo_for = lead
                grammar = self._solo_grammar
            elif drift or slip:
                # Rebuilt every time: the candidate lines are realised
                # against the scene that is on now, so this grammar is only
                # correct for this one call.
                forced = (large_lines(scene, self.rng) if drift
                          else slip_lines(scene, self.rng))
                grammar = compile_grammar(
                    grammar_for(names, self.max_lines, forced,
                                shape=self.rng.choice(FORCED_SHAPES))
                )
            else:
                if self._grammar_for != names:
                    self._grammar = compile_grammar(
                        grammar_for(names, self.max_lines)
                    )
                    self._grammar_for = list(names)
                grammar = self._grammar

            t0 = time.monotonic()
            try:
                out = self._llm.create_chat_completion(
                    messages=messages,
                    grammar=grammar,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repeat_penalty=self.repeat_penalty,
                )
            except Exception as exc:
                self.failures += 1
                self.status = "generate failed: %s" % type(exc).__name__
                return None
            self.last_ms = (time.monotonic() - t0) * 1000.0

        raw = out["choices"][0]["message"]["content"] or ""
        # Stand-up is parsed against the whole cast so the speaker index
        # still means something, even though only one name was reachable.
        lines = parse(raw, names, [] if standup else recent, solo=standup)
        # A drift exchange needs its deflection. The grammar guarantees three
        # turns -- small, large, small -- but the filter can still drop the
        # last one, and a large line with nothing after it is the register
        # without the thing that makes it bearable: the room is supposed to
        # not pick it up, out loud, and then carry on about the soup. Better
        # to drop the exchange and let the lexicon write this one.
        # A forced line needs the turn after it: the room is supposed to
        # not pick it up, out loud, and then carry on about the soup. A
        # large line with nothing after it is the register without the
        # thing that makes it bearable. Stand-up needs three -- a premise
        # and one tag is a sentence, not a set.
        need = MIN_LINES + 1 if (drift or slip or standup) else MIN_LINES
        if len(lines) < need:
            self.failures += 1
            self.status = "unusable (%d/%d lines, %.0fms)" % (
                len(lines), need, self.last_ms)
            return None

        self.calls += 1
        if drift:
            self.drifts += 1
        if standup:
            self.standups += 1
        self.status = "ok %.0fms" % self.last_ms
        odd = set(forced) if slip else set()
        if slip and not odd & {t for _, t in lines}:
            # The grammar guarantees a forced line, but the filter can drop
            # it -- as a near-repeat, usually. What is left is an ordinary
            # exchange, so let it be one rather than dressing a normal turn
            # up as the slip.
            slip = False
        if slip:
            self.slips += 1
        return {
            "lines": [{"speaker": names.index(who), "text": text,
                       "odd": text in odd}
                      for who, text in lines],
            "mode": ("standup" if standup
                     else "existential" if drift
                     else "slip" if slip else "bit"),
        }

    # -- result side ---------------------------------------------------------

    def take(self) -> dict | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None


# -- filtering ----------------------------------------------------------------


def clean(text: str) -> str:
    """Model output -> something the subtitle bar can actually draw."""
    for bad, good in _FOLD.items():
        text = text.replace(bad, good)
    text = "".join(ch for ch in text.upper() if ch in ALPHABET)
    text = " ".join(text.split())
    text = text.strip(" ,-'")
    if len(text) > MAX_LINE:
        text = _shorten(text)
    return _undangle(text)


def _undangle(text: str) -> str:
    """Drop a trailing word that cannot end a sentence -- carefully.

    The grammar bounds a turn at ten words and a bound has to fall somewhere,
    which produced "WE HAD TOLD HIM TO COME IN BY NOW, BUT?" -- a question
    mark after a conjunction, not clipped speech but a line that broke.

    The first version of this trimmed any dangling last word, and that was
    too blunt in both directions. It ate the "BEFORE" off the curated line
    "I'M NOT AFRAID OF THE END. I'M AFRAID OF THE PART BEFORE." -- mangling
    writing that was already finished -- and it would have taken the "FOR"
    off "SO WHAT IS THE GUEST TOWEL FOR.", which is the better line with it.
    Plenty of these words end a sentence perfectly well in a question.

    So a word is only dangling when the line gives a second reason to think
    the cut was mechanical: either there is no terminal punctuation at all,
    because the truncation above put it there, or a comma sits right in front
    of the word, which no finished sentence does. Three words are left alone
    whatever they say, because "SO?" is one of this show's best lines.
    """
    words = text.split()
    finished = bool(words) and words[-1][-1:] in ".?!"
    while len(words) > 3:
        bare = words[-1].strip(",.'?!-")
        if bare not in _DANGLING:
            break
        if finished and not words[-2].endswith(","):
            break                      # a real sentence, ending on a real word
        words.pop()
    out = " ".join(words).rstrip(" ,-'")
    if out and out[-1] not in ".?!":
        out += "."
    return out


def _shorten(text: str) -> str:
    """Cut an over-long line down to something that reads as speech.

    A sentence that already ends inside the limit is the best cut there is,
    so take that first. Otherwise cut on a word boundary; `_undangle` then
    deals with whatever word the cut landed on.
    """
    head = text[:MAX_LINE + 1]
    stop = max(head.rfind(". "), head.rfind("? "), head.rfind("! "))
    if stop >= MIN_LINE:
        return head[:stop + 1]
    return head.rsplit(" ", 1)[0] or text[:MAX_LINE]


def parse(raw: str, names: list[str],
          recent: list[tuple[str, str]] | None = None,
          solo: bool = False) -> list[tuple[str, str]]:
    """Pull `NAME: LINE` turns out of whatever came back.

    The grammar means this should be a formality, but it runs anyway: the
    grammar is gone the moment someone configures a different backend, and
    the cost of being wrong here is garbage on a screen in somebody's room.
    """
    seen = {_key(text) for _, text in (recent or [])}
    out: list[tuple[str, str]] = []
    for line in raw.splitlines():
        match = _SPEECH.match(line.strip())
        if not match:
            continue
        who = match.group(1).strip()
        if who not in names:
            continue
        text = _devocative(clean(match.group(2)), who)
        if len(text) < MIN_LINE or not any(c.isalpha() for c in text):
            continue
        if _key(text) in seen:              # already said, by anyone
            continue
        if text in _SHOT_TEXT:              # recited from the examples
            continue
        if out and who == out[-1][0] and not solo:
            # Two turns from one mouth read as one thought rather than as a
            # conversation. The content is fine; the attribution is not.
            #
            # Except in a routine, where one mouth is the entire form. This
            # rule would have handed every tag after the premise to somebody
            # else in the room, which is precisely the four-way exchange the
            # stand-up grammar exists to prevent.
            others = [n for n in names if n != who]
            if not others:
                continue
            who = others[hash(text) % len(others)]
        seen.add(_key(text))
        out.append((who, text))
    return out


# Expanded before a line is reduced to its dedupe key. Only consistency
# matters here, never correctness: this is applied to both sides of every
# comparison and the result is never shown to anybody, so "MY BOSS'S CAR"
# becoming "MY BOSS IS CAR" costs nothing.
_CONTRACTIONS = (
    ("N'T", " NOT"), ("'RE", " ARE"), ("'LL", " WILL"), ("'VE", " HAVE"),
    ("'M", " AM"), ("'D", " WOULD"), ("'S", " IS"),
)


def _key(text: str) -> str:
    """A loose identity for a line, for spotting near-repeats.

    Exact matching let the same thought through twice in a row as "WE ARE THE
    SAME BOSS!" and "WE'RE THE SAME BOSS!", which on screen is the show
    stuttering. Contractions are expanded and everything but letters and
    digits dropped, so punctuation cannot disguise a line as a new one.
    """
    for short, long in _CONTRACTIONS:
        text = text.replace(short, long)
    return "".join(c for c in text if c.isalnum())


def _devocative(text: str, who: str) -> str:
    """Stop somebody addressing themselves by name.

    Small models do this constantly -- Elaine saying "THERE WAS A LAST
    HAIRCUT, ELAINE." -- and it is the single most obvious tell that nobody
    is home. The name is only wrong because of who is saying it, so the
    cheapest correct fix is to drop the vocative and keep the line.
    """
    for form in (", %s." % who, ", %s?" % who, ", %s!" % who, ", %s" % who):
        if text.endswith(form):
            kept = text[: -len(form)]
            return kept + (form[-1] if form[-1] in ".?!" else ".")
    lead = "%s, " % who
    if text.startswith(lead) and len(text) - len(lead) >= MIN_LINE:
        return text[len(lead):]
    return text
