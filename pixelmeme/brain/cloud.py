"""Optional: let Claude write the script instead of the local generator.

Off by default. The broadcast is designed to run with no network at all, so
this is strictly an upgrade path: requests run on a background thread, the
local generator keeps the stream going while one is in flight, and any
failure (no key, no network, a timeout, a refusal) falls back without a
visible seam.
"""

from __future__ import annotations

import json
import queue
import threading
import time

SYSTEM_PROMPT = """
You write dialogue for a continuous, never-ending pixel-art broadcast. A small
cast stands around on a cheap set and talks. It never cuts to a new episode and
nothing is ever resolved.

You get the subject the current scene is about, and the cast. Write the next
short exchange.

The register: flat, certain, and wrong. Characters state impossible things the
way you would mention the weather. Nobody is confused, nobody explains, nobody
asks a follow-up question. Replies do not answer the thing that was just said.

Rules:
- All caps. Each line at most 70 characters. Short lines land harder.
- Everything in the exchange is about the subject you were given, and the
  subject is always something trivial treated as enormous.
- This is not a joke with a punchline. It is a conversation that almost makes
  sense and then does not.
- Never mean about the people in the room, and never about anyone's appearance,
  health, or private business. Surreal, not cruel.
- No hashtags, no emoji, no quotation marks, and never explain the bit.
- If the subject gives you nothing, write about the silence, or the building,
  or something being slightly wrong with the room.

Return 2 to 5 lines. Each has a `speaker` (the index of a cast member) and
`text`. For a stage direction nobody says out loud -- [ LAUGHTER ],
[ SILENCE ], [ NOBODY MOVES ] -- use speaker -1 and wrap the text in square
brackets. At most one of those per exchange.
""".strip()

_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


class CloudCaptioner:
    def __init__(self, cfg):
        c = cfg["cloud"]
        self.enabled = bool(c["enabled"])
        self.model = str(c["model"])
        self.effort = str(c["effort"])
        self.timeout = float(c["timeout_seconds"])
        self.min_interval = float(c["min_interval_seconds"])

        self.status = "disabled"
        self._client = None
        self._results: queue.Queue[dict] = queue.Queue(maxsize=4)
        self._inflight = threading.Event()
        self._last_request = 0.0
        self.calls = 0
        self.failures = 0

    def start(self) -> bool:
        if not self.enabled:
            return False
        try:
            import anthropic
        except ImportError:
            self.status = "anthropic package not installed"
            self.enabled = False
            return False
        try:
            # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
            # `ant auth login` profile -- whichever the Pi actually has.
            self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=1)
        except Exception as exc:
            self.status = f"client init failed: {exc}"
            self.enabled = False
            return False
        self.status = "ready"
        return True

    # -- request side --------------------------------------------------------

    def ready_to_request(self) -> bool:
        if not self.enabled or self._client is None:
            return False
        if self._inflight.is_set():
            return False
        return (time.monotonic() - self._last_request) >= self.min_interval

    def request(self, scene, cast_names: list[str],
                recent: list[tuple[str, str]] | None = None,
                scene_key: int = 0) -> bool:
        """Fire a request on a worker thread. False if skipped."""
        if not self.ready_to_request():
            return False
        # Only the scene's own facts cross the network, and they were made up
        # here a moment ago.
        subject = {
            "subject": getattr(scene, "thing", ""),
            "culprit": getattr(scene, "guy", ""),
            "place": getattr(scene, "place", ""),
            "verdict": getattr(scene, "trait", ""),
        }

        self._inflight.set()
        self._last_request = time.monotonic()
        threading.Thread(
            target=self._worker,
            args=(subject, list(cast_names), scene_key),
            name="claude-script",
            daemon=True,
        ).start()
        return True

    def _worker(self, subject: dict, cast_names: list[str],
                scene_key: int = 0) -> None:
        try:
            payload = self._call(subject, cast_names)
            if payload is not None:
                payload["scene_key"] = scene_key
                try:
                    self._results.put_nowait(payload)
                except queue.Full:
                    pass
        finally:
            self._inflight.clear()

    def _call(self, subject: dict, cast_names: list[str]) -> dict | None:
        import anthropic

        roster = "\n".join(f"{i}: {name}" for i, name in enumerate(cast_names))
        user = (
            f"Cast:\n{roster}\n\n"
            f"The scene is about: {subject.get('subject') or 'nothing in particular'}\n"
            f"Somebody absent who is to blame: {subject.get('culprit') or 'unknown'}\n"
            f"Where it happened: {subject.get('place') or 'here'}\n"
            f"The verdict on them: {subject.get('verdict') or 'undecided'}"
        )
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": _SCHEMA},
                },
            )
        except anthropic.APITimeoutError:
            self.failures += 1
            self.status = "timeout"
            return None
        except anthropic.RateLimitError:
            self.failures += 1
            self.status = "rate limited"
            # Back off hard: no point hammering a Pi's uplink.
            self._last_request = time.monotonic() + 60.0
            return None
        except anthropic.APIStatusError as exc:
            self.failures += 1
            self.status = f"api error {exc.status_code}"
            return None
        except anthropic.APIConnectionError:
            self.failures += 1
            self.status = "offline"
            return None
        except Exception as exc:
            self.failures += 1
            self.status = type(exc).__name__
            return None

        # A safety classifier can decline; that is a 200, not an exception.
        if response.stop_reason == "refusal":
            self.failures += 1
            category = getattr(response.stop_details, "category", None)
            self.status = f"declined ({category})"
            return None

        try:
            text = next(b.text for b in response.content if b.type == "text")
            payload = json.loads(text)
        except (StopIteration, json.JSONDecodeError, AttributeError):
            self.failures += 1
            self.status = "unparseable response"
            return None

        self.calls += 1
        self.status = "ok"
        return payload

    # -- result side ---------------------------------------------------------

    def take(self) -> dict | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None
