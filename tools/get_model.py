"""Download the writer: Llama 3.2 1B Instruct, GGUF, Q4_K_M.

About 770MB over the wire and about 1.3GB of RAM while running. Lands at the
path `llm.model_path` in config.yaml, which the app reads at startup; nothing
else has to be configured.

    python tools/get_model.py            # Llama 3.2 1B Q4_K_M, the default
    python tools/get_model.py pi         # Qwen2.5 0.5B Q4_K_M, for a 1GB Pi
    python tools/get_model.py --auto     # whichever fits this machine
    python tools/get_model.py --list     # what is on offer

Why Q4_K_M: this is a 1B model writing three short lines of sitcom dialogue
every twenty seconds or so. The quantisation is not what limits the writing at
this size, and Q4 is the one that fits a Pi 4 next to a framebuffer, a pygame
surface and numpy. Q8 is here because it is cheap to offer, not because it is
expected to read differently.

No API key, no account. Plain HTTPS from Hugging Face, and it verifies the
size it was given before moving the file into place.

It resumes and retries on its own, because an 800MB transfer to a Raspberry Pi
over somebody's home connection does not reliably arrive in one piece -- the
first run of this stopped at 79MB of 770MB with no error at all, just a closed
socket and a short file. A download that needs a human to notice and re-run it
is a download that fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pixelmeme import config as config_module  # noqa: E402

# (repo, filename, megabytes). The sizes are real Content-Lengths, so the
# memory arithmetic in the README can be checked against this table.
MODELS = {
    "1b": ("unsloth/Llama-3.2-1B-Instruct-GGUF",
           "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 808),
    "1b-q5": ("unsloth/Llama-3.2-1B-Instruct-GGUF",
              "Llama-3.2-1B-Instruct-Q5_K_M.gguf", 912),
    "1b-q8": ("unsloth/Llama-3.2-1B-Instruct-GGUF",
              "Llama-3.2-1B-Instruct-Q8_0.gguf", 1321),
    # For a 1GB board. A 0.5B at Q4 reads better than a 1B crushed to Q2,
    # and it leaves room for the renderer and the OS, which a 581MB Q2 does
    # not. It is weaker than the 1B -- more malformed words, more repetition
    # -- and on a Pi 3 the lexicon still writes most of the show.
    "pi": ("bartowski/Qwen2.5-0.5B-Instruct-GGUF",
           "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", 398),
    # Smaller still, if even that will not fit. Noticeably weaker again.
    "tiny": ("unsloth/SmolLM2-360M-Instruct-GGUF",
             "SmolLM2-360M-Instruct-Q4_K_M.gguf", 271),
}
URL = "https://huggingface.co/{repo}/resolve/main/{name}?download=true"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _size(part: Path) -> int:
    return part.stat().st_size if part.exists() else 0


def fetch(url: str, dest: Path, retries: int = 8) -> bool:
    """Download with resume and retry. True only once the whole file is here."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        complete, fatal = _attempt(url, dest, part)
        if complete:
            return True
        if fatal or attempt == retries:
            break
        wait = min(30, 2 ** attempt)
        print(f"  attempt {attempt}/{retries} stopped at {human(_size(part))}"
              f"; resuming in {wait}s")
        time.sleep(wait)
    print("gave up. The partial file is kept, so running this again resumes.")
    return False


def _attempt(url: str, dest: Path, part: Path) -> tuple[bool, bool]:
    """One resumable pass. Returns (got the whole file, stop trying)."""
    have = _size(part)
    req = urllib.request.Request(url, headers={"User-Agent": "pixelmeme"})
    if have:
        print(f"resuming at {human(have)}")
        req.add_header("Range", f"bytes={have}-")

    total = 0
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if have and resp.status != 206:
                # The server ignored the range. Start again rather than
                # appending a second copy of the file to the first.
                print("server will not resume; starting again")
                part.unlink(missing_ok=True)
                have = 0
            total = int(resp.headers.get("Content-Length", 0)) + have
            with part.open("ab" if have else "wb") as fh:
                done = have
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {human(done)} / {human(total)} "
                              f"({done / total * 100:.0f}%)",
                              end="", flush=True)
            print()
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:
            # The range started past the end: the file is already complete.
            os.replace(part, dest)
            return True, True
        print(f"\nHTTP {exc.code}: {exc.reason}")
        # A missing or forbidden file will not appear by trying again; a 5xx
        # or a rate limit might.
        return False, exc.code in (400, 401, 403, 404, 410)
    except (urllib.error.URLError, OSError) as exc:
        print(f"\ninterrupted: {exc}")
        return False, False

    size = _size(part)
    if total and size != total:
        print(f"short file: {human(size)} of {human(total)}")
        return False, False
    os.replace(part, dest)
    return True, False


def _configured(cfg) -> list[str]:
    raw = cfg.get("llm.model_path") or []
    return [raw] if isinstance(raw, str) else list(raw)


def _model_dir(cfg) -> Path:
    """Where the app already looks, so a download needs no config edit."""
    paths = _configured(cfg)
    if paths:
        return Path(cfg.resolve_path(paths[0])).parent
    return ROOT / "assets" / "models"


def _pick(cfg) -> str:
    """The biggest model this machine can actually hold.

    Same arithmetic the app uses to choose at startup: the file's own size
    plus headroom against the memory that is genuinely free, rather than a
    guess from the board name.
    """
    from pixelmeme.brain import llama

    free = llama.available_ram_mb()
    headroom = float(cfg.get("llm.ram_headroom_mb", 220.0))
    if free is None:
        return "1b"
    for key in ("1b", "pi", "tiny"):
        if free >= MODELS[key][2] + headroom:
            return key
    return "tiny"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="1b", choices=sorted(MODELS),
                    help="which model (default 1b)")
    ap.add_argument("--auto", action="store_true",
                    help="pick by the memory this machine actually has free")
    ap.add_argument("--list", action="store_true", help="list and exit")
    ap.add_argument("--out", help="write here instead of the configured dir")
    ap.add_argument("--retries", type=int, default=8,
                    help="resume attempts before giving up (default 8)")
    args = ap.parse_args(argv)

    if args.list:
        for key, (repo, name, mb) in sorted(MODELS.items(),
                                            key=lambda kv: -kv[1][2]):
            print(f"  {key:6s} {mb:5d} MB  {repo}/{name}")
        return 0

    cfg = config_module.load(None)
    which = args.which
    if args.auto:
        which = _pick(cfg)
        print(f"auto: {which}")
    repo, name, _mb = MODELS[which]
    dest = Path(args.out) if args.out else _model_dir(cfg) / name

    if dest.exists():
        print(f"already here: {dest} ({human(dest.stat().st_size)})")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"{repo}\n  -> {dest}")
    if not fetch(URL.format(repo=repo, name=name), dest, args.retries):
        return 1

    print(f"done: {dest} ({human(dest.stat().st_size)})")
    if dest.name in {Path(p).name for p in _configured(cfg)}:
        print("\nllm.model_path already lists it. Check it with:\n"
              "  python tools/check_llm.py")
    else:
        print(f"\nadd this to llm.model_path in config.yaml:\n  {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
