"""A 3x5 bitmap font, drawn by hand.

TrueType rendered at 8px is a grey smudge. A real bitmap font scaled by an
integer factor stays crisp at any size, which is the entire point here.
Uppercase only -- meme text is uppercase anyway.
"""

from __future__ import annotations

import pygame

GLYPH_W, GLYPH_H = 3, 5

_G: dict[str, tuple[str, ...]] = {
    "A": ("###", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": ("###", "#..", "#..", "#..", "###"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "###", "#..", "###"),
    "F": ("###", "#..", "###", "#..", "#.."),
    "G": ("###", "#..", "#.#", "#.#", "###"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", "###"),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"),
    "N": ("##.", "#.#", "#.#", "#.#", "#.#"),
    "O": ("###", "#.#", "#.#", "#.#", "###"),
    "P": ("###", "#.#", "###", "#..", "#.."),
    "Q": ("###", "#.#", "###", "..#", "..#"),
    "R": ("###", "#.#", "##.", "#.#", "#.#"),
    "S": ("###", "#..", "###", "..#", "###"),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"),
    "V": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", "###", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", "###", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", "..#", "..#", "..#"),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
    " ": ("...", "...", "...", "...", "..."),
    ".": ("...", "...", "...", "...", ".#."),
    ",": ("...", "...", "...", ".#.", "#.."),
    "!": (".#.", ".#.", ".#.", "...", ".#."),
    "?": ("###", "..#", ".##", "...", ".#."),
    "-": ("...", "...", "###", "...", "..."),
    "_": ("...", "...", "...", "...", "###"),
    ":": ("...", ".#.", "...", ".#.", "..."),
    ";": ("...", ".#.", "...", ".#.", "#.."),
    "(": ("..#", ".#.", ".#.", ".#.", "..#"),
    ")": ("#..", ".#.", ".#.", ".#.", "#.."),
    "[": ("##.", "#..", "#..", "#..", "##."),
    "]": (".##", "..#", "..#", "..#", ".##"),
    "{": ("..#", ".#.", "##.", ".#.", "..#"),
    "}": ("#..", ".#.", ".##", ".#.", "#.."),
    "/": ("..#", "..#", ".#.", "#..", "#.."),
    "&": ("##.", "##.", "###", "#.#", "###"),
    "%": ("#.#", "..#", ".#.", "#..", "#.#"),
    "$": ("###", "##.", "###", ".##", "###"),
    "*": ("#.#", ".#.", "###", ".#.", "#.#"),
    "+": ("...", ".#.", "###", ".#.", "..."),
    "=": ("...", "###", "...", "###", "..."),
    "<": ("..#", ".#.", "#..", ".#.", "..#"),
    ">": ("#..", ".#.", "..#", ".#.", "#.."),
    "@": ("###", "#.#", "###", "#..", "###"),
    "#": ("#.#", "###", "#.#", "###", "#.#"),
    "|": (".#.", ".#.", ".#.", ".#.", ".#."),
    "'": (".#.", ".#.", "...", "...", "..."),
    '"': ("#.#", "#.#", "...", "...", "..."),
    "\\": ("#..", "#..", ".#.", "..#", "..#"),
}

_FALLBACK = ("###", "#.#", "#.#", "#.#", "###")

# Every character this font can actually draw. Anything else comes out as
# _FALLBACK -- a solid box -- so text from outside the project (a language
# model, an API) has to be filtered against this before it reaches a frame.
SUPPORTED = frozenset(_G)

# Pixel offsets for a 1px outline. Diagonals included so thin strokes stay
# readable against noisy backgrounds like the granite wall.
_OUTLINE_OFFSETS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)


def advance(scale: int = 1, tracking: int = 1) -> int:
    """Horizontal distance from one glyph origin to the next."""
    return GLYPH_W * scale + tracking * scale


def text_width(text: str, scale: int = 1, tracking: int = 1) -> int:
    if not text:
        return 0
    return len(text) * advance(scale, tracking) - tracking * scale


def line_height(scale: int = 1, leading: int = 2) -> int:
    return GLYPH_H * scale + leading * scale


def max_chars(width_px: int, scale: int = 1, tracking: int = 1) -> int:
    step = advance(scale, tracking)
    return max(1, (width_px + tracking * scale) // step)


def wrap(text: str, width_chars: int) -> list[str]:
    words = str(text).upper().split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        # A word longer than the line gets hard-split rather than pushed off
        # the edge.
        while len(word) > width_chars:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[: width_chars - 1] + "-")
            word = word[width_chars - 1 :]
        candidate = (current + " " + word).strip()
        if len(candidate) <= width_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _blit_glyph(
    surface: pygame.Surface,
    rows: tuple[str, ...],
    x: int,
    y: int,
    scale: int,
    colour: tuple[int, int, int],
) -> None:
    fill = pygame.Rect(0, 0, scale, scale)
    for ry, row in enumerate(rows):
        for rx, cell in enumerate(row):
            if cell == "#":
                fill.topleft = (x + rx * scale, y + ry * scale)
                surface.fill(colour, fill)


def draw_text(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    scale: int = 1,
    colour: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] | None = (0, 0, 0),
    tracking: int = 1,
) -> int:
    """Draw one line with its top-left at (x, y). Returns the width drawn."""
    text = str(text).upper()
    glyphs = [(_G.get(ch, _FALLBACK), i) for i, ch in enumerate(text)]
    step = advance(scale, tracking)

    if outline is not None:
        for ox, oy in _OUTLINE_OFFSETS:
            dx, dy = ox * scale, oy * scale
            for rows, i in glyphs:
                _blit_glyph(surface, rows, x + i * step + dx, y + dy, scale, outline)
    for rows, i in glyphs:
        _blit_glyph(surface, rows, x + i * step, y, scale, colour)
    return text_width(text, scale, tracking)


def draw_centered(
    surface: pygame.Surface,
    text: str,
    cx: int,
    y: int,
    scale: int = 1,
    colour: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] | None = (0, 0, 0),
    tracking: int = 1,
) -> None:
    w = text_width(str(text), scale, tracking)
    draw_text(surface, text, cx - w // 2, y, scale, colour, outline, tracking)


def _fit_scale(
    text: str, width_px: int, scale: int, tracking: int, max_lines: int
) -> tuple[int, list[str]]:
    """Largest scale <= `scale` whose wrap fits in `max_lines`."""
    while scale > 1:
        if len(wrap(text, max_chars(width_px, scale, tracking))) <= max_lines:
            break
        scale -= 1
    return scale, wrap(text, max_chars(width_px, scale, tracking))[:max_lines]


def draw_block(
    surface: pygame.Surface,
    text: str,
    cx: int,
    y: int,
    width_px: int,
    scale: int = 2,
    colour: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] | None = (0, 0, 0),
    leading: int = 2,
    tracking: int = 1,
    max_lines: int = 4,
) -> int:
    """Word-wrapped centred paragraph. Returns total height drawn.

    Shrinks the scale before it truncates, so long captions stay whole.
    """
    scale, lines = _fit_scale(str(text), width_px, scale, tracking, max_lines)
    lh = line_height(scale, leading)
    for i, line in enumerate(lines):
        draw_centered(surface, line, cx, y + i * lh, scale, colour, outline, tracking)
    return len(lines) * lh


def block_height(
    text: str,
    width_px: int,
    scale: int = 2,
    leading: int = 2,
    tracking: int = 1,
    max_lines: int = 4,
) -> int:
    """Height draw_block would use, for laying out around it."""
    scale, lines = _fit_scale(str(text), width_px, scale, tracking, max_lines)
    return len(lines) * line_height(scale, leading)
