#!/usr/bin/env python3
"""
Smooth wrapping carousel of the four visual abstraction levels.

Renders an animated GIF (works headless on compute nodes).
If a display is available, also opens a Tkinter playback window.

Usage:
    python visualize_abstraction.py
    python visualize_abstraction.py --image_id alexander_mcqueen_fall_1996_000000
    python visualize_abstraction.py --output carousel.gif --fps 30

Dependencies: Pillow, opencv-python, numpy, tkinter (optional).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent  # ~/copalette
CLOTHING_DIR = REPO / "clothing"
SILHOUETTE_DIR = REPO / "silhouettes"
EDGE_DIR = REPO / "edges"

BG_RGB = (10, 10, 10)
TEXT_RGB = (255, 255, 255)
SUBTITLE_RGB = (136, 136, 136)
ACCENT_RGB = (74, 144, 217)
DIM_RGB = (42, 42, 42)
COUNTER_RGB = (85, 85, 85)

IMG_SIZE = 500
CANVAS_W = IMG_SIZE + 160
CANVAS_H = IMG_SIZE + 200

DWELL_S = 2.5
TRANSITION_S = 0.6
FPS_DEFAULT = 30

LEVELS = [
    {"key": "fullcolor",   "label": "Full Color",  "subtitle": "hue, texture, shape retained"},
    {"key": "grayscale",   "label": "Grayscale",   "subtitle": "luminance and texture retained, hue removed"},
    {"key": "silhouette",  "label": "Silhouette",  "subtitle": "shape retained, surface detail removed"},
    {"key": "edge",        "label": "Edge Map",    "subtitle": "contour and seam geometry only"},
]

# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

_FONT_SEARCH = [
    "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    "/usr/share/fonts/google-droid/DroidSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_FONT_SEARCH_REGULAR = [p.replace("-Bold", "-Regular").replace("Sans-Bold", "Sans") for p in _FONT_SEARCH]


def _load_font(size, bold=True):
    candidates = _FONT_SEARCH if bold else _FONT_SEARCH_REGULAR
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    # Also try bold paths for regular if regular not found
    if not bold:
        for p in _FONT_SEARCH:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    try:
        return ImageFont.truetype("LiberationSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT_LABEL = _load_font(28, bold=True)
FONT_SUBTITLE = _load_font(15, bold=False)
FONT_SMALL = _load_font(11, bold=False)

# ---------------------------------------------------------------------------
# Image loading / generation
# ---------------------------------------------------------------------------


def _find_default_image_id():
    """Pick the first clothing crop that has matching silhouette and edge."""
    for path in sorted(CLOTHING_DIR.glob("*_clothing.jpg")):
        base = path.stem.replace("_clothing", "")
        if (SILHOUETTE_DIR / f"{base}_silhouette.jpg").exists() and \
           (EDGE_DIR / f"{base}_edge.jpg").exists():
            return base
    first = next(CLOTHING_DIR.glob("*_clothing.jpg"), None)
    if first:
        return first.stem.replace("_clothing", "")
    return None


def _load_or_generate(image_id):
    """Return dict of {level_key: PIL.Image} for the four abstraction levels."""
    clothing_path = CLOTHING_DIR / f"{image_id}_clothing.jpg"
    if not clothing_path.exists():
        raise FileNotFoundError(f"Clothing crop not found: {clothing_path}")

    clothing = Image.open(clothing_path).convert("RGB")
    imgs = {}

    # Full color
    imgs["fullcolor"] = clothing.copy()

    # Grayscale
    gray_np = np.array(clothing.convert("L"))
    imgs["grayscale"] = Image.fromarray(np.stack([gray_np] * 3, axis=-1))

    # Silhouette
    sil_path = SILHOUETTE_DIR / f"{image_id}_silhouette.jpg"
    if sil_path.exists():
        imgs["silhouette"] = Image.open(sil_path).convert("RGB")
    else:
        _, thresh = cv2.threshold(gray_np, 127, 255, cv2.THRESH_BINARY)
        if thresh.mean() < 128:
            sil = (np.array(clothing).sum(axis=-1) > 30).astype(np.uint8) * 255
        else:
            sil = thresh
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, kernel)
        imgs["silhouette"] = Image.fromarray(np.stack([sil] * 3, axis=-1))

    # Edge map — always generate from full-res grayscale so it stays sharp;
    # the pre-generated edges are only 224x224 (model input size).
    edge_path = EDGE_DIR / f"{image_id}_edge.jpg"
    use_pregenerated = False
    if edge_path.exists():
        pre = Image.open(edge_path)
        if min(pre.size) >= min(clothing.size):
            imgs["edge"] = pre.convert("RGB")
            use_pregenerated = True
    if not use_pregenerated:
        blurred = cv2.GaussianBlur(gray_np, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dk = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges_dilated = cv2.dilate(edges, dk, iterations=1)
        edges_final = 255 - edges_dilated  # white bg, dark edges
        imgs["edge"] = Image.fromarray(np.stack([edges_final] * 3, axis=-1))

    return imgs


def _prepare_display_images(raw_imgs):
    """Resize all images to IMG_SIZE, centered on dark background."""
    result = {}
    for key, img in raw_imgs.items():
        img = img.copy()
        img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_RGB)
        x = (IMG_SIZE - img.width) // 2
        y = (IMG_SIZE - img.height) // 2
        canvas.paste(img, (x, y))
        result[key] = canvas
    return result


# ---------------------------------------------------------------------------
# Frame renderer (pure Pillow — no display needed)
# ---------------------------------------------------------------------------


def _ease_in_out(t):
    """Cubic ease-in-out: 0→0, 0.5→0.5, 1→1."""
    return t * t * (3.0 - 2.0 * t)


def _text_centered(draw, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (CANVAS_W - tw) // 2
    draw.text((x, y), text, font=font, fill=fill)


def render_frame(images, t_global):
    """Render a single animation frame at time t_global (seconds).

    Returns a PIL Image of size (CANVAS_W, CANVAS_H).
    """
    n = len(LEVELS)
    cycle_s = DWELL_S + TRANSITION_S
    total_cycle = n * cycle_s

    t = t_global % total_cycle
    level_idx = int(t // cycle_s) % n
    phase_t = t - level_idx * cycle_s  # time within this level's phase

    cur = LEVELS[level_idx]
    nxt = LEVELS[(level_idx + 1) % n]

    if phase_t < DWELL_S:
        # Dwelling — current image centered
        slide = 0.0
    else:
        # Transitioning
        raw_t = (phase_t - DWELL_S) / TRANSITION_S
        slide = _ease_in_out(min(raw_t, 1.0))

    frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_RGB)

    # Image region: centered horizontally, top-padded
    img_cx = CANVAS_W // 2
    img_cy = 30 + IMG_SIZE // 2
    slide_span = IMG_SIZE + 80

    offset_px = int(slide * slide_span)

    # Paste current image sliding left
    cur_left = img_cx - IMG_SIZE // 2 - offset_px
    frame.paste(images[cur["key"]], (cur_left, img_cy - IMG_SIZE // 2))

    # Paste next image sliding in from right
    if slide > 0:
        nxt_left = img_cx - IMG_SIZE // 2 + slide_span - offset_px
        frame.paste(images[nxt["key"]], (nxt_left, img_cy - IMG_SIZE // 2))

    # Clip the image region (mask away overflow with background bars)
    draw = ImageDraw.Draw(frame)
    pad_left = img_cx - IMG_SIZE // 2
    pad_right = img_cx + IMG_SIZE // 2
    draw.rectangle([0, 0, pad_left - 1, img_cy + IMG_SIZE // 2], fill=BG_RGB)
    draw.rectangle([pad_right + 1, 0, CANVAS_W, img_cy + IMG_SIZE // 2], fill=BG_RGB)

    # Text: label and subtitle
    active = cur if slide < 0.5 else nxt
    text_y = img_cy + IMG_SIZE // 2 + 24
    _text_centered(draw, text_y, active["label"], FONT_LABEL, TEXT_RGB)
    _text_centered(draw, text_y + 36, active["subtitle"], FONT_SUBTITLE, SUBTITLE_RGB)

    # Progress bar
    bar_y = CANVAS_H - 28
    bar_h = 6
    seg_total = CANVAS_W - 120
    seg_w = seg_total // n
    bar_x0 = (CANVAS_W - seg_w * n) // 2

    display_idx = level_idx if slide < 0.5 else (level_idx + 1) % n
    for i in range(n):
        x0 = bar_x0 + i * seg_w
        x1 = x0 + seg_w - 4
        fill = ACCENT_RGB if i == display_idx else DIM_RGB
        draw.rectangle([x0, bar_y, x1, bar_y + bar_h], fill=fill)

    # Counter
    counter_text = f"{display_idx + 1} / {n}"
    _text_centered(draw, bar_y + bar_h + 8, counter_text, FONT_SMALL, COUNTER_RGB)

    return frame


# ---------------------------------------------------------------------------
# Static row export (for README / paper figures)
# ---------------------------------------------------------------------------


def export_row(raw_imgs, output_path, cell_size=400, padding=24):
    """Render all four abstraction levels side by side as a single PNG.

    Each image is sized to cell_size, with labels underneath,
    on a dark background.
    """
    n = len(LEVELS)
    row_w = n * cell_size + (n + 1) * padding
    row_h = cell_size + 80 + padding * 2  # image + text + top/bottom pad

    canvas = Image.new("RGB", (row_w, row_h), BG_RGB)
    draw = ImageDraw.Draw(canvas)

    for i, level in enumerate(LEVELS):
        img = raw_imgs[level["key"]].copy()
        img.thumbnail((cell_size, cell_size), Image.LANCZOS)

        # Center within cell
        x0 = padding + i * (cell_size + padding)
        ix = x0 + (cell_size - img.width) // 2
        iy = padding + (cell_size - img.height) // 2
        canvas.paste(img, (ix, iy))

        # Label
        label = level["label"]
        bbox = draw.textbbox((0, 0), label, font=FONT_LABEL)
        lw = bbox[2] - bbox[0]
        lx = x0 + (cell_size - lw) // 2
        ly = padding + cell_size + 8
        draw.text((lx, ly), label, font=FONT_LABEL, fill=TEXT_RGB)

        # Subtitle
        sub = level["subtitle"]
        bbox_s = draw.textbbox((0, 0), sub, font=FONT_SMALL)
        sw = bbox_s[2] - bbox_s[0]
        sx = x0 + (cell_size - sw) // 2
        sy = ly + 32
        draw.text((sx, sy), sub, font=FONT_SMALL, fill=SUBTITLE_RGB)

    canvas.save(output_path, quality=95)
    print(f"Saved static row to {output_path} "
          f"({canvas.size[0]}x{canvas.size[1]}, {os.path.getsize(output_path) / 1e3:.0f} KB)")


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------


def export_gif(images, output_path, fps=30):
    """Render one full carousel cycle and save as animated GIF.

    Dwell phases use a single frame with a long delay.
    Transition phases are rendered at full fps for smooth animation.
    """
    n = len(LEVELS)
    cycle_s = DWELL_S + TRANSITION_S
    dt = 1.0 / fps

    frames = []
    durations = []  # per-frame duration in ms

    print(f"Rendering {n * cycle_s:.1f}s animation at {fps} fps (transitions only)...")

    for level_idx in range(n):
        base_t = level_idx * cycle_s

        # Dwell: single frame, held for DWELL_S
        frame = render_frame(images, base_t)
        frames.append(frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
        durations.append(int(DWELL_S * 1000))

        # Transition: render every frame at fps
        t = 0.0
        while t < TRANSITION_S:
            frame = render_frame(images, base_t + DWELL_S + t)
            frames.append(frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
            durations.append(int(1000 / fps))
            t += dt

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    total_dur = sum(durations) / 1000
    print(f"Saved {len(frames)} frames ({total_dur:.1f}s) to {output_path} "
          f"({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Tkinter playback (optional — only when display is available)
# ---------------------------------------------------------------------------


def _has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform == "darwin")


def play_tkinter(images):
    """Open a Tkinter window and play the carousel live."""
    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("FASH-iCNN  \u2014  Visual Abstraction Ladder")
    root.configure(bg="#0a0a0a")
    root.geometry(f"{CANVAS_W}x{CANVAS_H}")
    root.resizable(False, False)

    label = tk.Label(root, bg="#0a0a0a", bd=0, highlightthickness=0)
    label.pack(fill="both", expand=True)

    start_time = time.monotonic()
    photo_ref = [None]  # prevent GC

    def tick():
        t = time.monotonic() - start_time
        frame = render_frame(images, t)
        photo_ref[0] = ImageTk.PhotoImage(frame)
        label.configure(image=photo_ref[0])
        root.after(16, tick)  # ~60 fps

    tick()
    root.mainloop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Animated carousel of visual abstraction levels.")
    parser.add_argument(
        "--image_id", type=str, default=None,
        help="Base image ID, e.g. 'alexander_mcqueen_fall_1996_000000'. "
             "Auto-selects if omitted.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output GIF path. Defaults to abstraction_carousel.gif next to this script.",
    )
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT, help=f"GIF frame rate (default {FPS_DEFAULT}).")
    parser.add_argument("--no-window", action="store_true", help="Skip Tkinter playback even if display is available.")
    args = parser.parse_args()

    image_id = args.image_id or _find_default_image_id()
    if image_id is None:
        print(f"ERROR: No clothing crops found in {CLOTHING_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading abstraction images for: {image_id}")
    raw = _load_or_generate(image_id)
    images = _prepare_display_images(raw)

    sil_src = "pre-generated" if (SILHOUETTE_DIR / f"{image_id}_silhouette.jpg").exists() else "generated on-the-fly"
    # Edge is regenerated from full-res clothing when pre-generated is smaller
    edge_pre = EDGE_DIR / f"{image_id}_edge.jpg"
    if edge_pre.exists() and min(Image.open(edge_pre).size) >= min(raw["fullcolor"].size):
        edge_src = "pre-generated"
    else:
        edge_src = "generated from full-res crop" if edge_pre.exists() else "generated on-the-fly"
    print(f"  fullcolor:  {raw['fullcolor'].size}")
    print(f"  grayscale:  generated from clothing crop")
    print(f"  silhouette: {sil_src}")
    print(f"  edge:       {edge_src}")

    out_dir = Path(__file__).resolve().parent

    # Static row PNG (for README)
    row_path = str(out_dir / "abstraction_ladder.png")
    export_row(raw, row_path)

    # Animated GIF carousel
    output = args.output or str(out_dir / "abstraction_carousel.gif")
    export_gif(images, output, fps=args.fps)

    # Optionally open Tkinter window
    if not args.no_window and _has_display():
        print("Opening live preview window (close to exit)...")
        play_tkinter(images)
    elif not args.no_window:
        print("No display detected — skipping Tkinter window. Use --no-window to suppress this message.")


if __name__ == "__main__":
    main()
