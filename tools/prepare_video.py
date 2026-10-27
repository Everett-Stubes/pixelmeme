#!/usr/bin/env python3
"""Bake a video clip into the pixel format the idle loop plays.

A Pi 3 cannot decode H.264 and palette-quantise it 20 times a second on top
of everything else it is doing. So we do the expensive part once, here, on a
real computer, and ship the result as an array of palette indices the Pi can
blit straight to the screen.

    python tools/prepare_video.py honnold.mp4 --seconds 45

Writes assets/idle/honnold.pxv.npz by default, which pixelmeme picks up
automatically on the next start.

Supply your own clip. Nothing is downloaded and no footage is bundled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pixelmeme.render.palette import PALETTES, Quantizer  # noqa: E402


def center_crop_to_aspect(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Crop (h, w, 3) to the target aspect ratio without squashing anybody."""
    h, w = frame.shape[:2]
    want = target_w / target_h
    have = w / h
    if have > want:
        new_w = int(round(h * want))
        x0 = (w - new_w) // 2
        return frame[:, x0 : x0 + new_w]
    new_h = int(round(w / want))
    y0 = (h - new_h) // 2
    return frame[y0 : y0 + new_h, :]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="source clip (anything OpenCV can open)")
    p.add_argument("-o", "--out", help="output .npz (default assets/idle/<name>.pxv.npz)")
    p.add_argument("--width", type=int, default=90)
    p.add_argument("--height", type=int, default=160)
    p.add_argument("--fps", type=float, default=12.0, help="playback rate to bake")
    p.add_argument("--start", type=float, default=0.0, help="skip this many seconds")
    p.add_argument("--seconds", type=float, default=60.0, help="how much to keep")
    p.add_argument("--palette", default="stone", choices=sorted(PALETTES))
    p.add_argument("--dither", type=float, default=0.55)
    p.add_argument(
        "--name",
        default="honnold",
        help="output stem; the default is what config.yaml expects",
    )
    args = p.parse_args(argv)

    try:
        import cv2
    except ImportError:
        print("opencv-python is needed for this tool:  pip install opencv-python")
        return 1

    source = Path(args.video)
    if not source.exists():
        print(f"no such file: {source}")
        return 1

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"OpenCV could not open {source}")
        return 1

    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"source: {src_fps:.2f} fps, {total} frames")

    # Sample every Nth source frame to land on the requested playback rate.
    step = max(1, int(round(src_fps / args.fps)))
    start_frame = int(args.start * src_fps)
    wanted = int(args.seconds * args.fps)
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    quantizer = Quantizer(args.palette, args.dither)
    palette = quantizer.palette
    frames: list[np.ndarray] = []

    index = 0
    while len(frames) < wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step:
            index += 1
            continue
        index += 1

        frame = center_crop_to_aspect(frame, args.width, args.height)
        frame = cv2.resize(
            frame, (args.width, args.height), interpolation=cv2.INTER_AREA
        )
        rgb = frame[:, :, ::-1]  # BGR -> RGB
        # pygame.surfarray works in (w, h, 3); OpenCV gives (h, w, 3).
        rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2))

        # Store indices, not RGB: a third of the memory, and playback is then
        # one fancy-index per frame.
        frames.append(quantizer.indices(rgb).astype(np.uint8))

        if len(frames) % 25 == 0:
            print(f"  {len(frames)}/{wanted} frames", end="\r", flush=True)

    capture.release()
    if not frames:
        print("no frames were read -- wrong --start, or an unreadable codec")
        return 1

    stack = np.stack(frames)
    out_path = Path(args.out) if args.out else ROOT / "assets" / "idle" / f"{args.name}.pxv.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, frames=stack, palette=palette, fps=np.float32(args.fps))

    on_disk = out_path.stat().st_size / 1e6
    in_ram = stack.nbytes / 1e6
    print(
        f"\nwrote {out_path}\n"
        f"  {len(frames)} frames at {args.fps:g} fps "
        f"({len(frames) / args.fps:.1f}s loop)\n"
        f"  {on_disk:.1f} MB on disk, {in_ram:.1f} MB in RAM when playing"
    )
    if in_ram > 120:
        print("  that is a lot for a Pi 3 -- consider a shorter --seconds or lower --fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
