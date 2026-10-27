"""The climbing scene: a sandstone wall and the person on it.

Both are generated, not loaded. The wall tiles vertically so it can scroll
forever without a seam, and the climber is drawn from joint positions rather
than sprite frames so the motion is continuous.
"""

from __future__ import annotations

import dataclasses
import math
import random

import numpy as np
import pygame

def _rect(s: pygame.Surface, c, x: int, y: int, w: int, h: int) -> None:
    if w > 0 and h > 0:
        s.fill(c, pygame.Rect(int(x), int(y), int(w), int(h)))


def _line(s: pygame.Surface, c, x1, y1, x2, y2, thick: int = 1) -> None:
    """Integer Bresenham line drawn as little squares, so it stays blocky."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        _rect(s, c, x1, y1, thick, thick)
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def _disc(s: pygame.Surface, c, cx: int, cy: int, r: int) -> None:
    """Filled circle, scanline by scanline, no anti-aliasing."""
    for dy in range(-r, r + 1):
        span = int(math.sqrt(max(0.0, r * r - dy * dy)))
        _rect(s, c, cx - span, cy + dy, span * 2 + 1, 1)


# --- The climbing scene -----------------------------------------------------


def _periodic_noise(
    w: int, h: int, cells_x: int, cells_y: int, rng: np.random.Generator
) -> np.ndarray:
    """Bilinear value noise that wraps vertically, so the wall can scroll
    forever without a visible seam."""
    grid = rng.random((cells_x + 1, cells_y + 1)).astype(np.float32)
    grid[-1, :] = grid[0, :]
    grid[:, -1] = grid[:, 0]

    xs = np.linspace(0, cells_x, w, endpoint=False, dtype=np.float32)
    ys = np.linspace(0, cells_y, h, endpoint=False, dtype=np.float32)
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    fx, fy = (xs - x0)[:, None], (ys - y0)[None, :]
    # Smoothstep: removes the diamond artefacts of raw bilinear.
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)

    g00 = grid[np.ix_(x0, y0)]
    g10 = grid[np.ix_(x0 + 1, y0)]
    g01 = grid[np.ix_(x0, y0 + 1)]
    g11 = grid[np.ix_(x0 + 1, y0 + 1)]
    top = g00 * (1 - fx) + g10 * fx
    bottom = g01 * (1 - fx) + g11 * fx
    return top * (1 - fy) + bottom * fy


def make_alps_backdrop(width: int, height: int, seed: int = 11) -> pygame.Surface:
    """A static panorama behind the climb: sky, a hazed range across the
    valley, and the valley floor a very long way down.

    Built in depth layers, each one drawn at the contrast its distance
    earns. That is the whole trick for exposure: the eye reads "far" from
    loss of contrast long before it reads it from size, so the far range is
    nearly sky-coloured, the near range carries the snow, and only the
    valley floor is allowed a saturated colour. The previous version drew
    everything at full contrast and the climb looked twenty feet off the
    ground no matter how long it scrolled.

    Rendered once and cached by the caller -- at this distance nothing in it
    needs to move for the climb itself to read as going somewhere.
    """
    s = pygame.Surface((width, height))
    rng = random.Random(seed)

    # Low horizon: more sky above it than valley below. Looking out from high
    # on a wall, the far tops sit at about eye level and the ground is a long
    # way under you.
    horizon = int(height * 0.52)

    # -- sky, darkening upward ------------------------------------------------
    # Five bands, not three. The palette has three blues, so the extra steps
    # come from how much of each band the next one covers rather than from
    # new colours -- a thin band of the next blue up, then a thick one, reads
    # as a gradient without anything to dither.
    #
    # The bands are cut at cumulative edges rather than at independently
    # rounded positions. Sizing each one separately left row 78 belonging to
    # no band at all -- a one-pixel black line straight across the sky,
    # because a fresh pygame Surface starts black and nothing had painted
    # there. It read as a wire strung between the mountains.
    sky = ((60, 96, 140), (110, 160, 210), (190, 220, 245))
    edges = (0, int(horizon * 0.34), int(horizon * 0.80), horizon)
    for i, colour in enumerate(sky):
        _rect(s, colour, 0, edges[i], width, edges[i + 1] - edges[i])

    # Cloud banks, flat-bottomed and sitting below the top band. Drawn in the
    # palest sky blue and the off-white, so they read as cloud rather than as
    # a hole in the sky.
    for _ in range(rng.randrange(2, 4)):
        cy = rng.randrange(int(horizon * 0.22), int(horizon * 0.62))
        cw = rng.randrange(width // 3, width)
        cx = rng.randrange(-cw // 3, width)
        for i in range(rng.randrange(3, 6)):
            bw = rng.randrange(cw // 4, max(5, cw // 2))
            bx = cx + rng.randrange(0, max(1, cw - bw))
            bh = rng.randrange(2, 4)
            _rect(s, (190, 220, 245), bx, cy - bh, bw, bh)
            _rect(s, (208, 204, 194), bx + 1, cy - bh, max(1, bw - 2), 1)

    # -- the far range: almost sky --------------------------------------------
    # One step off the horizon blue. At this distance a mountain is a shape,
    # not a surface, and giving it any texture at all pulls it forward.
    far = (190, 220, 245)
    x = -rng.randrange(0, 12)
    while x < width:
        pw = rng.randrange(18, 34)
        ph = rng.randrange(int(height * 0.08), int(height * 0.17))
        pygame.draw.polygon(s, far, [
            (x, horizon),
            (x + pw // 2 + rng.randrange(-pw // 3, pw // 3 + 1), horizon - ph),
            (x + pw, horizon)])
        x += pw - rng.randrange(4, 10)

    # -- the near range: the one that carries the snow ------------------------
    rock, snow, shade = (176, 170, 160), (236, 238, 242), (140, 134, 126)
    x = -rng.randrange(0, 10)
    while x < width:
        pw = rng.randrange(14, 26)
        ph = rng.randrange(int(height * 0.10), int(height * 0.22))
        # Off-centre apex: a real skyline is a row of asymmetric wedges, and
        # putting every summit exactly halfway along its base made a
        # sawtooth that the eye reads as a pattern rather than as mountains.
        apex = (x + pw // 2 + rng.randrange(-pw // 4, pw // 4 + 1),
                horizon - ph)
        pygame.draw.polygon(s, rock, [(x, horizon), apex, (x + pw, horizon)])
        # The shaded face, on the right of each peak: the sun is on the left
        # here exactly as it is on the pillar.
        pygame.draw.polygon(s, shade,
                            [apex, (x + pw, horizon),
                             (x + pw // 2 + pw // 6, horizon)])
        cap = max(2, ph // 3)
        pygame.draw.polygon(
            s, snow,
            [(apex[0] - pw // 5, apex[1] + cap), apex,
             (apex[0] + pw // 5, apex[1] + cap)],
        )
        x += pw - rng.randrange(2, 6)

    # -- the valley floor, a long way down ------------------------------------
    # Straight to the floor. A grey band along the foot of the range read as
    # a motorway, and stripes of lighter green across the valley read as a
    # deckchair -- both were detail bought at the cost of the thing the
    # layer is for, which is being quietly far away.
    _rect(s, (70, 110, 70), 0, horizon, width, height - horizon)

    # Haze where the ground meets the distance. One pale row is worth more
    # than any amount of detail further down: it is the only cue that says
    # the green is receding rather than standing up behind the pillar.
    _rect(s, (190, 220, 245), 0, horizon, width, 1)
    # The far half of the valley floor, in the lighter green, with a ragged
    # lower edge. Everything else in this picture got a near tone and a far
    # tone; without one the ground was a single flat slab filling the bottom
    # of both side strips, which is the one place the eye had nothing to
    # read distance from. The edge is stepped so it is the far side of a
    # valley rather than a horizontal rule.
    far_floor = max(2, int(height * 0.13))
    edge = horizon + 1
    for row in range(far_floor):
        wobble = rng.randrange(0, 3) if row > far_floor - 4 else 0
        _rect(s, (130, 150, 90), 0, edge + row, width, 1)
        if wobble:
            for seg in range(0, width, rng.randrange(5, 13)):
                _rect(s, (70, 110, 70), seg, edge + row,
                      rng.randrange(2, 7), 1)

    # Fields, as a few big soft patches rather than as bands. Full-width
    # bands of a lighter green turned the valley into a deckchair; what
    # farmland from a thousand feet actually looks like is irregular blocks
    # of slightly different green, most of them larger than you expect.
    for _ in range(rng.randrange(3, 6)):
        fw = rng.randrange(width // 4, width // 2)
        fh = rng.randrange(3, max(4, int(height * 0.06)))
        fx = rng.randrange(-fw // 3, width)
        fy = rng.randrange(horizon + 2, max(horizon + 3, height - fh))
        # Stepped on one side, so the edge is a field boundary and not a box.
        for row in range(fh):
            inset = rng.randrange(0, 3)
            _rect(s, (130, 150, 90), fx + inset, fy + row,
                  max(1, fw - inset * 2), 1)

    # The village: four or five marks, total. At this distance it is the
    # suggestion of a village -- the old one was a continuous terrace of
    # nine-pixel chalets with windows and a spire, which is a town you could
    # walk to rather than one you are a thousand feet above.
    vy = horizon + max(3, int(height * 0.06))
    for _ in range(rng.randrange(4, 7)):
        vx = rng.randrange(2, max(3, width - 4))
        _rect(s, (208, 204, 194), vx, vy, 2, 1)
        _rect(s, (150, 80, 58), vx, vy - 1, 2, 1)
        vy += rng.randrange(-2, 3)

    return s


# The sandstone ramp, dark to light. Eight steps rather than six: the extra
# two are what let the bedding planes, the ledge shadows and the lit face of
# the pillar all be distinguishable from each other instead of collapsing
# into the same two tones.
#
# These are palette entries exactly, not hand-picked browns that happen to
# look right. The rock is flat cel art -- every pixel of it is one of these
# eight and nothing in between -- so a colour that misses the palette has
# the quantiser choosing between the two entries it sits between, which an
# ordered dither then lays out as a grid of dots across a whole bed of rock.
# Picked by eye, all eight missed, and two of the beds checkerboarded.
_SANDSTONE = np.array(
    [
        (62, 40, 26),
        (86, 56, 38),
        (112, 76, 50),
        (140, 98, 66),
        (168, 122, 84),
        (194, 148, 106),
        (216, 174, 132),
        (234, 200, 160),
    ],
    dtype=np.uint8,
)

# The climber, likewise straight off the palette. Deriving his lit and shaded
# tones by multiplying a base colour is how the shirt highlight ended up
# between two reds with nothing to snap to.
_SKIN = (194, 148, 106)
_SKIN_LIT = (216, 174, 132)
_SKIN_DARK = (168, 122, 84)
_SHIRT = (168, 48, 42)
_SHIRT_LIT = (206, 74, 62)
_SHIRT_DARK = (116, 32, 30)
_PANTS = (70, 64, 62)
_PANTS_DARK = (44, 40, 44)
_INK = (18, 20, 26)
_CHALK = (236, 238, 242)
# Chalk seen on rock rather than on a hand: the same material a step
# dimmer, because a full-brightness smear competes with the snow caps.
_CHALK_ROCK = (208, 204, 194)
_BAG = (104, 98, 92)
_RUBBER = (150, 80, 58)


def _h(seed: int, salt: float) -> float:
    """Deterministic 0..1 from a seed and a number, without touching either
    of the generators in `make_wall` -- both of their draw orders are relied
    on by features further down."""
    n = math.sin(seed * 12.9898 + salt * 78.233) * 43758.5453
    return n - math.floor(n)


def make_wall(width: int, height: int, seed: int = 7) -> pygame.Surface:
    """A tall slab of sandstone that tiles vertically.

    Built as an index field rather than as colours, so every feature -- the
    bedding planes, the ledges, the lit and shaded sides of the pillar --
    is a shift along one ramp. Nothing can land off-palette that way, and
    the rock keeps reading as one rock under all of it.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((width, height), dtype=np.float32)
    amplitude = 1.0
    cells = 2
    for _ in range(5):          # one more octave than before: finer grain
        field += _periodic_noise(width, height, cells, cells * 2, rng) * amplitude
        amplitude *= 0.52
        cells *= 2
    field -= field.min()
    field /= max(1e-6, field.max())

    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)

    # Bedding planes: sandstone is laid down in horizontal beds and they are
    # the single most recognisable thing about it from a distance. Every
    # harmonic is a whole number of cycles over the height, so the tiling
    # still wraps exactly.
    #
    # Three harmonics, not one. A single clean sine at nine cycles put nine
    # evenly spaced bands down the tile, and evenly spaced bands on a warm
    # brown ramp is not rock, it is a plank -- which is what this looked
    # like. Beating three of them against each other gives beds of visibly
    # different thickness and strength, which is what stops the eye finding
    # the period.
    beds = np.zeros(height, dtype=np.float32)
    for cycles, weight in ((5.0, 1.0), (9.0, 0.62), (14.0, 0.34)):
        beds += np.sin(ys / height * math.tau * cycles
                       + _h(seed, cycles) * math.tau) * weight
    beds /= 1.96
    field += (beds[None, :] * 0.5 + 0.5) * 0.18

    # Desert varnish: dark mineral streaks weeping down the face. This is the
    # single most recognisable thing about desert sandstone at a distance and
    # it runs vertically, which is the axis the old wall had nothing in at
    # all -- so it breaks the banding and says "rock" in the same stroke.
    # Vertical features wrap up the tile for free.
    varnish = np.zeros((width, height), dtype=np.float32)
    for _ in range(max(3, width // 9)):
        cx = rng.integers(0, width)
        half = float(rng.integers(2, max(3, width // 7)))
        # Across: a soft bell, so the streak has no hard edge.
        d = np.abs(((xs - cx + width / 2) % width) - width / 2) / half
        across_w = np.clip(1.0 - d * d, 0.0, 1.0)
        # Down: fades in and out, a whole number of cycles so it wraps.
        cycles = float(rng.integers(1, 4))
        down = (np.sin(ys / height * math.tau * cycles
                       + float(rng.random()) * math.tau) * 0.5 + 0.5) ** 1.4
        varnish += np.outer(across_w, down) * float(rng.uniform(0.18, 0.42))
    field -= np.clip(varnish, 0.0, 0.75)

    # The pillar is round, not flat: light from the left, shadow curling
    # away to the right edge.
    across = xs / max(1.0, width - 1.0)
    curve = np.cos((across - 0.34) * 2.1) ** 2
    field = field * 0.78 + curve[:, None] * 0.34

    steps = len(_SANDSTONE)
    field -= field.min()
    field /= max(1e-6, field.max())
    idx = np.clip((field * steps).astype(np.int32), 0, steps - 1)

    # Ledges: a dark line with a lit lip above it. Two rows of index shift
    # read as a roof and the shelf under it, which is most of what gives the
    # scrolling any sense of scale.
    prng = random.Random(seed)
    for _ in range(max(2, height // 40)):
        y = prng.randrange(6, max(7, height - 6))
        x0 = prng.randrange(0, width)
        run = prng.randrange(width // 3, width)
        for i in range(run):
            xx = (x0 + i) % width
            yy = y + int(math.sin(i * 0.14 + y) * 2.0)
            if 2 <= yy < height - 2:
                idx[xx, yy] = max(0, idx[xx, yy] - 4)       # shadow under
                idx[xx, yy + 1] = max(0, idx[xx, yy + 1] - 3)
                idx[xx, yy - 1] = min(steps - 1, idx[xx, yy - 1] + 2)  # lit lip

    # Pockets and huecos. Sandstone erodes into holes and they are the only
    # features on the wall a climber could actually be using -- so some of
    # them get chalked, which is the cheapest way to say the route has been
    # climbed before and that the man is following something rather than
    # crawling up a blank face. Chalk is the brightest thing on the rock and
    # a two-pixel smear of it reads from across a room.
    chalked = []
    for _ in range(max(3, height // 20)):
        cx = prng.randrange(1, width - 1)
        cy = prng.randrange(3, height - 3)
        r = prng.randrange(1, 3)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                xx, yy = (cx + dx) % width, cy + dy
                if 0 <= yy < height:
                    idx[xx, yy] = max(0, idx[xx, yy] - (3 if dy <= 0 else 1))
        # Only near the middle, and only sometimes. A route is a line up
        # the rock, so chalk clustered into a corridor reads as the way
        # somebody went; chalk spread evenly over the whole face reads as
        # snow. The climber sways about the centre, so the centre is where
        # his hands have been.
        if abs(cx - width / 2) < width * 0.22 and prng.random() < 0.3:
            chalked.append((cx, cy, r))

    # Cracks. Mostly VERTICAL now: a crack system on a sandstone pillar runs
    # up it, which is both what the rock does and what a climber would be
    # following. The old generator ran them horizontally with a small
    # vertical wobble, and a warm brown field with wavy dark horizontal lines
    # in it is wood grain -- that one loop was most of the plank.
    #
    # A vertical crack also wraps for free: it runs the full height, so there
    # is no end for the tile seam to cut through.
    # Three of them, not ten, and none of them running the whole way. Ten
    # full-height lines crossing the horizontal ledges made a grid: the
    # pillar read as scaffolding, which is a worse failure than the plank
    # it replaced. A crack has ends, and it is a seam in the rock rather
    # than a line drawn on it -- so it darkens by a few steps instead of
    # slamming to black, and it tapers out at both ends.
    for _ in range(max(2, width // 28)):
        x = prng.randrange(0, width)
        y0 = prng.randrange(0, height)
        run = prng.randrange(height // 6, height // 2)
        drift = prng.uniform(0.012, 0.05) * prng.choice((-1, 1))
        wobble = prng.uniform(1.0, 2.4)
        for i in range(run):
            yy = (y0 + i) % height
            xx = int(x + i * drift + math.sin(i * 0.09 + x) * wobble) % width
            # Taper: full depth in the middle, nothing at the two ends, so
            # it stops without a cut.
            edge = min(i, run - 1 - i) / max(1.0, run * 0.22)
            depth = round(4 * min(1.0, edge))
            if depth:
                idx[xx, yy] = max(0, idx[xx, yy] - depth)

    # A few short horizontal ones left in, for the beds to part along. Rare
    # enough that they read as detail rather than as grain.
    for _ in range(max(1, height // 48)):
        y = prng.randrange(4, max(5, height - 4))
        x = prng.randrange(0, width)
        length = prng.randrange(width // 8, width // 3)
        for i in range(length):
            yy = y + int(math.sin(i * 0.25) * 2)
            if 0 <= yy < height:
                idx[(x + i) % width, yy] = 0

    # -- the silhouette ------------------------------------------------------
    # A rectangle of rock is the one thing that gives this away as a texture
    # on a quad. The real edge of a pillar wanders: it bulges, it pinches,
    # and it has flakes missing. Both edges are cut by a profile built from
    # whole-number harmonics over the tile height, so the cut wraps exactly
    # when the tile does.
    def profile(seed_offset: int, scale: float) -> np.ndarray:
        prof = np.zeros(height, dtype=np.float32)
        ph = random.Random(seed + seed_offset)
        for harmonic, weight in ((1, 1.0), (2, 0.55), (3, 0.3), (5, 0.22),
                                 (8, 0.14), (13, 0.09), (21, 0.06)):
            phase = ph.random() * math.tau
            prof += np.sin(ys / height * math.tau * harmonic + phase) * weight
        prof -= prof.min()
        prof /= max(1e-6, prof.max())
        return prof * scale

    # Up to about a fifth of the width off each side, so the pillar visibly
    # narrows and widens as it goes past.
    cut = width * 0.12
    left = profile(31, cut)
    right = width - 1 - profile(97, cut)

    # Flakes: short stretches where a slab has come away, square-edged, which
    # is how sandstone actually breaks.
    fl = random.Random(seed + 5)
    for _ in range(max(2, height // 30)):
        y0 = fl.randrange(0, height)
        run = fl.randrange(3, 14)
        bite = fl.uniform(2.0, width * 0.12)
        side = fl.random() < 0.5
        for i in range(run):
            yy = (y0 + i) % height
            if side:
                left[yy] += bite
            else:
                right[yy] -= bite

    columns = np.arange(width, dtype=np.float32)[:, None]
    outside = (columns < left[None, :]) | (columns > right[None, :])

    rgb = _SANDSTONE[idx]

    # Chalk: a smear above the lip of a chalked pocket, where a hand would
    # have pulled down on it. Off the ramp entirely -- it is magnesium
    # carbonate on sandstone, not a lighter sandstone -- so it is applied to
    # the resolved colours rather than to the index field.
    for cx, cy, r in chalked:
        for dx in range(-r, r + 1):
            xx, yy = (cx + dx) % width, cy - r
            if 0 <= yy < height and prng.random() < 0.75:
                rgb[xx, yy] = _CHALK_ROCK
    # The arete: a hard dark line just inside the cut on both sides, so the
    # rock separates from the sky instead of bleeding into it.
    shade = (~outside) & (columns > right[None, :] - 3.5)
    rgb[shade] = _SANDSTONE[2]
    rim = (~outside) & (
        (columns < left[None, :] + 1.0) | (columns > right[None, :] - 1.0)
    )
    rgb[rim] = _SANDSTONE[0]

    key = (255, 0, 255)
    rgb[outside] = key

    surface = pygame.Surface((width, height))
    pygame.surfarray.blit_array(surface, rgb)
    surface.set_colorkey(key)
    return surface


# Offsets for the outline pass: the whole figure is drawn once per offset in
# a flat outline colour, then once more in real colours at (0, 0). Cheaper
# than per-shape outlining and it is the only thing that reliably separates
# a warm-toned figure from warm-toned rock.
_HALO_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _tone(colour: tuple[int, int, int], k: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * k))) for c in colour)


# -- what the body is doing --------------------------------------------------


@dataclasses.dataclass
class Pose:
    """One frame of climbing, as four numbers per side and a rise.

    Keeping the pose separate from the drawing is what lets him do more than
    one thing: the renderer knows how to draw any pose, and the sequence
    below decides which poses to walk through.
    """

    # Per side, keyed by -1 (camera-left) and +1 (camera-right).
    reach: dict = dataclasses.field(default_factory=lambda: {-1: 0.0, 1: 0.0})
    hang: dict = dataclasses.field(default_factory=lambda: {-1: 0.0, 1: 0.0})
    foot: dict = dataclasses.field(default_factory=lambda: {-1: 0.0, 1: 0.0})
    facing: int = 1

    # There is deliberately no `rise`. Lifting the figure up the screen
    # during a press and dropping it back at the end of the move was a
    # seven-pixel snap every time, and it is also the wrong place for it:
    # the camera follows him, so gaining height moves the WALL, not him.
    # The press reads from the legs straightening and the holds coming down
    # past his hands, which is what actually happens.


def _ease(t: float) -> float:
    """Smoothstep. Climbing is slow at the ends of a move and quick through
    the middle, which is most of what separates it from a metronome."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _tri(t: float) -> float:
    """Out and back: 0 at both ends, 1 in the middle."""
    return _ease(t * 2.0) if t < 0.5 else _ease((1.0 - t) * 2.0)


# Each move is (name, seconds, height gained in body units). A height, not a
# speed: the wall is scrolled by exactly the distance the move says he
# climbed, eased on the same curve the limbs move on, so the rock cannot
# drift out of step with him. Given as a speed, the wall slid past at a
# constant rate through a press while his body accelerated and decelerated
# under it, which is the thing that read as him swimming up a texture.
#
# He is about twelve units tall, so a press gaining four and a half units is
# a little under half his height -- about what standing up on a high foot is
# actually worth. A shake-out gains nothing at all, because it is a rest.
MOVES = (
    ("reach", 2.2, 4.1),
    ("step", 2.4, 1.7),
    ("press", 2.0, 7.8),
    ("shake", 3.0, 0.0),
    ("match", 2.2, 1.7),
    ("settle", 1.8, 0.7),
)

# What tends to follow what. A high step wants a press on it; a press leaves
# him stretched out and wanting a hand; a shake-out is a rest and comes after
# something strenuous.
_AFTER = {
    "reach": (("reach", 3), ("step", 4), ("match", 2), ("shake", 2), ("settle", 1)),
    "step": (("press", 6), ("reach", 2)),
    "press": (("reach", 5), ("shake", 2), ("settle", 2)),
    "shake": (("reach", 5), ("step", 2)),
    "match": (("step", 3), ("shake", 2), ("reach", 3)),
    "settle": (("reach", 4), ("step", 3)),
}


class ClimbSequence:
    """Strings moves together, forever, and reports the pose for right now."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.move = "reach"
        self.side = 1
        self.t = 0.0
        self.duration = 2.2
        self.gain = 1.1
        self.facing = 1
        # Total height climbed so far, in body units. The caller scrolls the
        # wall by the change in this and nothing else, which is what keeps
        # the rock and the climber in step: there is only one number.
        self.climbed = 0.0
        # Carried between moves so a limb that was high stays high until
        # something moves it, instead of snapping back between moves.
        self.hold = {"reach": {-1: 0.1, 1: 0.1}, "foot": {-1: 0.0, 1: 0.0}}
        self._begin("reach")

    def _begin(self, name: str) -> None:
        self.move = name
        duration, gain = next((d, g) for n, d, g in MOVES if n == name)
        # Stretching a move stretches the time, not the height: a slow press
        # covers the same ground more slowly.
        self.duration = duration * self.rng.uniform(0.85, 1.25)
        self.gain = gain
        self.t = 0.0
        if name in ("reach", "step", "shake"):
            # Alternate sides rather than choosing freely: climbing with one
            # arm twice in a row is a thing that happens, four times in a row
            # is a bug.
            self.side = -self.side if self.rng.random() < 0.8 else self.side
            self.facing = self.side

    def update(self, dt: float) -> None:
        """Advance, accumulating height on the same curve the limbs move on.

        Stepped move by move rather than by frame, so a long frame that
        crosses a boundary still credits the rest of the move it is leaving
        and only the part of the next one it has entered. Crediting the
        whole frame to whichever move happened to be current at the end of
        it is how the wall jumped on a slow frame.
        """
        remaining = max(0.0, dt)
        while remaining > 0.0:
            step = min(remaining, self.duration - self.t)
            f0 = self.t / self.duration
            self.t += step
            f1 = min(1.0, self.t / self.duration)
            self.climbed += self.gain * (_ease(f1) - _ease(f0))
            remaining -= step
            if self.t < self.duration:
                break
            options = _AFTER[self.move]
            names = [n for n, _ in options]
            weights = [w for _, w in options]
            self._begin(self.rng.choices(names, weights=weights, k=1)[0])

    def pose(self) -> Pose:
        f = self.t / max(1e-6, self.duration)
        p = Pose(facing=self.facing)
        near, far = self.side, -self.side
        p.reach = dict(self.hold["reach"])
        p.foot = dict(self.hold["foot"])

        if self.move == "reach":
            p.reach[near] = _tri(f) * 0.6 + _ease(f) * 0.4
            p.reach[far] = 0.15 + 0.1 * (1.0 - f)
            if f > 0.92:
                self.hold["reach"] = {near: 0.25, far: 0.15}
        elif self.move == "step":
            p.foot[near] = _ease(f)
            p.reach[near] = 0.2
            p.reach[far] = 0.3
            if f > 0.92:
                self.hold["foot"] = {near: 1.0, far: 0.0}
        elif self.move == "press":
            # Standing up on the high foot: the legs straighten and the arms
            # come down relative to the body, because the holds stayed where
            # they were and he did not.
            p.foot = {s: self.hold["foot"][s] * (1.0 - _ease(f) * 0.9)
                      for s in (-1, 1)}
            p.reach = {s: max(0.1, self.hold["reach"][s] - _ease(f) * 0.5)
                       for s in (-1, 1)}
            if f > 0.92:
                self.hold["foot"] = {-1: 0.0, 1: 0.0}
                self.hold["reach"] = {-1: 0.12, 1: 0.12}
        elif self.move == "shake":
            # The rest. One arm off, hanging straight down, shaking the blood
            # back into it -- the most recognisable thing a soloist does.
            p.hang[near] = _tri(f)
            p.reach[near] = 0.0
            p.reach[far] = 0.75
        elif self.move == "match":
            both = 0.78 + 0.2 * _tri(f)
            p.reach = {-1: both, 1: both}
        else:   # settle
            p.reach = {s: max(0.08, self.hold["reach"][s] * (1.0 - f * 0.5))
                       for s in (-1, 1)}
        return p


def draw_climber(
    surface: pygame.Surface,
    x: int,
    y: int,
    scale: int,
    phase: float,
    facing: int = 1,
    pose: Pose | None = None,
    skin: tuple[int, int, int] = _SKIN,
    shirt: tuple[int, int, int] = _SHIRT,
    hair: tuple[int, int, int] = _INK,
    pants: tuple[int, int, int] = _PANTS,
    bag: tuple[int, int, int] = _BAG,
    outline: tuple[int, int, int] = _INK,
    chalk: tuple[int, int, int] = _CHALK,
) -> None:
    """Honnold, free soloing, in a three-quarter turn.

    He is a specific-looking person and the likeness is all in three things:
    a jaw as wide as the skull above it with a chin hung off the front of it,
    ears that stand off the sides, and a swimmer's V of shoulders over a
    narrow waist. Everything here is built to put those three on screen at
    about sixty pixels tall. Red tee, black hair, chalk bag, no rope.

    The turn is faked by layering rather than by perspective: the far-side
    limbs are drawn first, thinner and tucked in; the near-side limbs last,
    full thickness and reaching further out. A camera-facing figure is
    symmetric about x and this one deliberately is not. `facing` is +1
    (turned right) or -1 (left).

    Shapes are collected before anything is drawn so the whole figure can be
    stamped out in one flat colour for the outline pass and then again in
    real colours, without building it twice.
    """
    u = float(scale)
    beat = phase % 1.0
    if pose is None:
        # No sequence driving it: fall back to the plain alternating reach,
        # which is what this did before it had a repertoire.
        swing = math.sin(beat * math.tau)
        pose = Pose(reach={1: max(0.0, swing), -1: max(0.0, -swing)},
                    facing=facing)
    facing = pose.facing
    reach, hang, foot = pose.reach, pose.hang, pose.foot
    # Breathing, and nothing else. Height gained goes into the wall.
    bob = math.sin(beat * math.tau) * u * 0.25
    far, near = -facing, facing

    # Only the defaults are guaranteed to be on the palette, so a caller
    # passing its own colours still gets multiplied tones.
    lit = (_SKIN_LIT, _SKIN_DARK) if skin == _SKIN else (_tone(skin, 1.15),
                                                         _tone(skin, 0.74))
    skin_lit, skin_dark = lit
    worn = (_SHIRT_LIT, _SHIRT_DARK) if shirt == _SHIRT else (
        _tone(shirt, 1.22), _tone(shirt, 0.68))
    shirt_lit, shirt_dark = worn
    pants_dark = _PANTS_DARK if pants == _PANTS else _tone(pants, 0.72)
    rubber = _RUBBER

    ops: list[tuple] = []

    def R(ux: float, uy: float, uw: float, uh: float, colour) -> None:
        """Rect in body units, centred on (ux, uy) horizontally."""
        ops.append((
            "r", round(ux * u - uw * u / 2), round(uy * u + bob),
            max(1, round(uw * u)), max(1, round(uh * u)), colour,
        ))

    def L(p0, p1, thick: float, colour) -> None:
        ops.append((
            "l", round(p0[0] * u), round(p0[1] * u + bob),
            round(p1[0] * u), round(p1[1] * u + bob),
            max(1, round(thick * u)), colour,
        ))

    # -- head ----------------------------------------------------------------
    # Drawn as skull + jaw + chin rather than as one block: the jaw being
    # nearly as wide as the skull, and the chin sitting proud of it, is the
    # whole likeness.
    hx = 0.45 * facing
    R(hx, -3.0, 1.6, 1.15, skin)                         # skull
    R(hx - 0.05 * facing, -1.9, 1.56, 0.9, skin)         # jaw, barely narrower
    R(hx + 0.26 * facing, -1.2, 0.82, 0.3, skin)         # chin, hung forward
    R(hx + far * 0.56, -2.95, 0.4, 2.0, skin_dark)       # far cheek in shadow
    R(hx + far * 0.9, -2.3, 0.38, 0.66, skin)            # ear, standing off
    R(hx + far * 0.98, -2.18, 0.18, 0.36, skin_dark)
    R(hx + facing * 0.85, -2.05, 0.36, 0.42, skin)       # nose in profile
    R(hx, -2.46, 1.5, 0.2, skin_dark)                    # brow
    R(hx + facing * 0.28, -2.26, 0.28, 0.24, _INK)        # eye
    R(hx, -3.32, 1.7, 0.52, hair)                        # hair, cropped short
    R(hx + far * 0.72, -3.0, 0.5, 1.0, hair)             # back of the head
    R(hx + facing * 0.4, -2.74, 0.9, 0.26, hair)         # fringe
    R(hx + far * 0.74, -2.5, 0.26, 0.4, hair)            # sideburn

    # Neck: long, and in the shade of his own jaw.
    R(hx * 0.5, -0.5, 0.78, 1.3, skin_dark)

    # -- the limbs on the far side, first ------------------------------------

    def arm(side: int, thick: float, sleeve_thick: float) -> None:
        r = reach.get(side, 0.0)
        h = hang.get(side, 0.0)
        # At rest the arm is bent and the hand is on a hold at chest height;
        # only the reaching side goes overhead. Both arms up at once is a
        # jumping jack, not a climber.
        shoulder = (side * 1.3, 0.62)
        elbow = (side * (1.75 + 0.2 * r), 2.4 - 2.1 * r)
        hand = (side * (1.62 + 0.33 * r), 1.5 - 3.7 * r)
        if h > 0.0:
            # Shaking out: the arm comes off the wall and hangs, and the hand
            # wanders because he is flicking the blood back into it.
            wobble = math.sin(phase * 17.0) * 0.22 * h
            elbow = (elbow[0] * (1 - h) + side * 1.5 * h,
                     elbow[1] * (1 - h) + 3.4 * h)
            hand = (hand[0] * (1 - h) + (side * 1.35 + wobble) * h,
                    hand[1] * (1 - h) + 7.3 * h)
        mid = ((shoulder[0] + elbow[0]) / 2, (shoulder[1] + elbow[1]) / 2)
        L(shoulder, elbow, thick, skin)
        L(shoulder, mid, sleeve_thick, shirt)            # short sleeve
        if r > 0.25 and h < 0.4:
            R(mid[0], mid[1] - 0.25, 0.5, 0.5, skin_lit)  # bicep, under load
        L(elbow, hand, thick * 0.85, skin)
        R(hand[0], hand[1] - 0.3, 0.8, 0.6, skin_lit)     # hand on the hold
        if h < 0.5:
            for f in (-1, 0, 1):                          # fingers, crimping
                R(hand[0] + f * 0.26, hand[1] - 0.62, 0.18, 0.34, skin_lit)
        if r > 0.85 or h > 0.8:
            R(hand[0] - side * 0.6, hand[1] - 0.9, 0.5, 0.5, chalk)

    def leg(side: int, thick: float, colour) -> None:
        r = reach.get(side, 0.0)
        hi = foot.get(side, 0.0)
        hip = (side * 0.58, 4.5)
        # A high step: the knee comes up and out and the foot lands level
        # with his own hip, which is the pose that reads as climbing even
        # when the body is still.
        knee = (side * (1.0 + 0.25 * r + 1.05 * hi), 6.9 - 2.5 * hi)
        ft = (side * (1.35 + 0.5 * r + 0.35 * hi), 9.3 - 4.3 * hi)
        L(hip, knee, thick, colour)
        L(knee, ft, thick * 0.8, colour)
        R(ft[0], ft[1], 1.0, 0.44, _INK)                  # shoe
        R(ft[0] + facing * 0.4, ft[1] + 0.1, 0.4, 0.26, rubber)  # toe rand

    arm(far, 0.5, 0.62)
    leg(far, 0.62, pants_dark)

    # -- torso ---------------------------------------------------------------
    # Rows of decreasing width: shoulders to waist, which is the silhouette
    # doing the work. One row is one pixel at the scale this runs at.
    rows = 22
    for i in range(rows):
        f = i / (rows - 1)
        w = 3.0 - 1.05 * f
        ty = 0.25 + f * 4.25
        R(0.0, ty, w, 0.22, shirt)
        R(far * (w / 2 - 0.22), ty, 0.45, 0.22, shirt_dark)
        R(near * (w / 2 - 0.12), ty, 0.24, 0.22, shirt_lit)
    R(0.0, 0.2, 3.0, 0.5, shirt)                          # trapezius

    # Chalk bag at the back of the hip, and the waist strap holding it on --
    # the only two pieces of kit a soloist has.
    R(0.0, 4.26, 2.2, 0.17, (44, 40, 44))
    R(far * 0.95, 4.5, 0.86, 1.1, bag)
    R(far * 0.95, 4.54, 0.86, 0.16, (176, 170, 160))

    leg(near, 0.74, pants)
    arm(near, 0.6, 0.74)

    # -- stamp it out --------------------------------------------------------

    def emit(dx: int, dy: int, mono) -> None:
        for op in ops:
            if op[0] == "r":
                _rect(surface, mono or op[5], x + op[1] + dx, y + op[2] + dy,
                      op[3], op[4])
            else:
                _line(surface, mono or op[6], x + op[1] + dx, y + op[2] + dy,
                      x + op[3] + dx, y + op[4] + dy, op[5])

    # Cast shadow first, down and to the right -- the light on the wall comes
    # from the left. Nothing else in the frame tells you he is standing off
    # the rock rather than painted onto it.
    #
    # Offset by his own size, not by a fixed two pixels. At two it fell
    # entirely underneath the outline halo, which is drawn one pixel out on
    # all four sides, so the only part of it that ever showed was a smudge
    # under his shoes. Proportional to scale it stays correct if he is ever
    # drawn larger, and it clears the halo at every size.
    shadow_dx, shadow_dy = max(2, round(u * 0.55)), max(3, round(u * 0.8))
    emit(shadow_dx, shadow_dy, (86, 56, 38))
    for dx, dy in _HALO_OFFSETS:
        emit(dx, dy, outline)
    emit(0, 0, None)
