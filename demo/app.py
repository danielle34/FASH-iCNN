#!/usr/bin/env python3
"""
FASH-iCNN: Editorial Fashion Identity Inspector

Gradio demo that takes a garment photo and returns color, house, and era
predictions grounded in Vogue runway editorial data.
Runs full analysis independently at each of four abstraction levels.
"""

import logging
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# Self-contained imports from this package
from colors import (
    CSS_COLORS_RGB, CSS_TO_BK, BK_HEX, CHROMATIC_BK_NAMES,
    dominant_color_from_image, lab_to_hex, rgb_to_hex,
)
from preprocessing import segment_clothing, generate_abstraction_ladder, detect_and_crop_face
from inference import (
    load_all_models, ABSTRACTION_LEVELS,
    predict_designer, predict_decade, predict_bk_color, predict_css_color,
    DESIGNER_DISPLAY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOUSES = [
    "Alexander McQueen", "Balenciaga", "Calvin Klein Collection", "Chanel",
    "Christian Dior", "Fendi", "Gucci", "Hermes", "Louis Vuitton", "Prada",
    "Ralph Lauren", "Saint Laurent", "Valentino", "Versace",
]
YEARS = [str(y) for y in range(1991, 2025)]

LEVEL_DISPLAY = {
    "fullcolor": "Full Color",
    "grayscale": "Grayscale",
    "silhouette": "Silhouette",
    "edge": "Edge Map",
}

# ---------------------------------------------------------------------------
# HTML builders (compact versions for per-column display)
# ---------------------------------------------------------------------------

def _swatch(hex_color, size=28):
    return (
        f'<div style="display:inline-block;width:{size}px;height:{size}px;'
        f'background-color:{hex_color};border:2px solid #333;border-radius:4px;'
        f'vertical-align:middle;margin-right:6px;"></div>'
    )


def _build_designer_html(designer_result):
    if designer_result is None:
        return '<p style="font-family:sans-serif;color:#999;"><i>Model not available</i></p>'

    parts = ['<div style="font-family:sans-serif;font-size:0.9em;">']
    parts.append('<p style="margin-bottom:4px;"><b>Designer</b></p>')

    for rank, (name, conf) in enumerate(designer_result, 1):
        bar_width = int(conf * 200)
        color = "#4a90d9" if rank == 1 else "#7ab3ef" if rank == 2 else "#a8ccf0"
        weight = "bold" if rank == 1 else "normal"
        parts.append(
            f'<div style="margin:3px 0;">'
            f'<div style="font-weight:{weight};margin-bottom:1px;">{rank}. {name}</div>'
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="background:{color};height:16px;width:{bar_width}px;'
            f'border-radius:3px;flex-shrink:0;"></div>'
            f'<span style="color:#666;font-size:0.85em;">{conf:.1%}</span>'
            f'</div></div>'
        )

    parts.append('</div>')
    return "\n".join(parts)


def _build_decade_html(decade_result):
    if decade_result is None:
        return '<p style="font-family:sans-serif;color:#999;"><i>Model not available</i></p>'

    label, conf = decade_result
    bar_width = int(conf * 200)
    return (
        '<div style="font-family:sans-serif;font-size:0.9em;">'
        f'<p style="margin-bottom:4px;"><b>Decade:</b> '
        f'<span style="font-size:1.1em;font-weight:bold;">{label}</span> '
        f'<span style="color:#666;">({conf:.1%})</span></p>'
        f'<div style="background:#4a90d9;height:16px;width:{bar_width}px;'
        f'border-radius:3px;"></div>'
        '</div>'
    )


def _build_color_html(color_info, bk_result, css_result):
    parts = ['<div style="font-family:sans-serif;font-size:0.9em;line-height:1.6;">']

    # Berlin-Kay family
    if bk_result:
        bk_name, bk_conf = bk_result
        bk_hex = BK_HEX.get(bk_name, "#808080")
        parts.append(
            f'<p style="margin:2px 0;"><b>BK:</b> {_swatch(bk_hex, 22)} '
            f'<b>{bk_name.title()}</b> '
            f'<span style="color:#666;">({bk_conf:.1%})</span></p>'
        )
    else:
        parts.append('<p style="margin:2px 0;"><b>BK:</b> <i>Model not available</i></p>')

    # CSS named color
    if css_result:
        css_name, css_conf = css_result
        css_rgb = CSS_COLORS_RGB.get(css_name, (128, 128, 128))
        css_hex = rgb_to_hex(*css_rgb)
        parts.append(
            f'<p style="margin:2px 0;"><b>CSS:</b> {_swatch(css_hex, 22)} '
            f'<b>{css_name}</b> '
            f'<span style="color:#666;">({css_conf:.1%})</span></p>'
        )
    else:
        parts.append('<p style="margin:2px 0;"><b>CSS:</b> <i>Model not available</i></p>')

    # CIELAB coordinate + swatch
    lab = color_info["lab"]
    parts.append(
        f'<p style="margin:2px 0;"><b>LAB:</b> {_swatch(color_info["hex"], 22)} '
        f'L*={lab[0]:.1f} a*={lab[1]:.1f} b*={lab[2]:.1f}</p>'
    )

    parts.append('</div>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-level analysis
# ---------------------------------------------------------------------------

def _analyze_one_level(level, abstraction_img, year_int):
    """Run designer, decade, and color predictions on a single abstraction image."""
    designer_result = predict_designer(abstraction_img, level=level)
    decade_result = predict_decade(abstraction_img)
    bk_result = predict_bk_color(abstraction_img, year=year_int)
    css_result = predict_css_color(abstraction_img)
    color_info = dominant_color_from_image(
        np.array(abstraction_img.convert("RGB").resize((224, 224)))
    )
    return designer_result, decade_result, bk_result, css_result, color_info


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(garment_img, face_img, year, house):
    """Run the full prediction pipeline at all four abstraction levels.

    Returns 16 values: for each of 4 levels ->
        (pil_image, designer_html, decade_html, color_html).
    """
    if garment_img is None:
        placeholder = '<p style="font-family:sans-serif;">Please upload a garment image.</p>'
        return [None, placeholder, "", ""] + [None, "", "", ""] * 3

    pil_garment = Image.fromarray(garment_img) if isinstance(garment_img, np.ndarray) else garment_img

    # --- Segment clothing ---
    clothing_crop, mask = segment_clothing(pil_garment)

    # --- Abstraction ladder ---
    ladder = generate_abstraction_ladder(clothing_crop)

    # --- Face detection (optional) ---
    face_crop = None
    if face_img is not None:
        pil_face = Image.fromarray(face_img) if isinstance(face_img, np.ndarray) else face_img
        face_crop = detect_and_crop_face(pil_face)

    # --- Parse year ---
    year_int = int(year) if year and year.strip() else None

    # --- Run analysis at each abstraction level ---
    results = []
    for level in ABSTRACTION_LEVELS:
        abs_img = ladder[level]
        designer_result, decade_result, bk_result, css_result, color_info = (
            _analyze_one_level(level, abs_img, year_int)
        )

        results.append(abs_img)
        results.append(_build_designer_html(designer_result))
        results.append(_build_decade_html(decade_result))
        results.append(_build_color_html(color_info, bk_result, css_result))

    return results


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="FASH-iCNN: Editorial Fashion Identity Inspector",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# FASH-iCNN: Editorial Fashion Identity Inspector\n"
            "Upload a garment photo to analyze its editorial identity: which fashion "
            "house it most resembles, which era it belongs to, and how that signal "
            "changes as visual information is stripped away. Color prediction is "
            "available but experimental."
        )

        with gr.Row():
            # --- Left column: inputs ---
            with gr.Column(scale=1, min_width=260):
                garment_input = gr.Image(
                    label="Garment Photo (required)",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                face_input = gr.Image(
                    label="Face Photo (optional)",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                year_dropdown = gr.Dropdown(
                    choices=[""] + YEARS,
                    value="",
                    label="Year (optional)",
                    info="Select the garment year to improve color prediction accuracy",
                )
                house_dropdown = gr.Dropdown(
                    choices=[""] + HOUSES,
                    value="",
                    label="Fashion House (optional)",
                    info="For reference only; does not affect predictions",
                )
                submit_btn = gr.Button("Analyze Garment", variant="primary", size="lg")

            # --- Right area: four abstraction-level columns ---
            with gr.Column(scale=4):
                gr.Markdown("### Per-Abstraction-Level Analysis")
                with gr.Row(equal_height=False):
                    level_imgs = []
                    level_designer_htmls = []
                    level_decade_htmls = []
                    level_color_htmls = []
                    for level in ABSTRACTION_LEVELS:
                        with gr.Column(min_width=200):
                            gr.Markdown(f"**{LEVEL_DISPLAY[level]}**")
                            img = gr.Image(
                                label=LEVEL_DISPLAY[level],
                                type="pil",
                                height=280,
                                show_label=False,
                            )
                            designer_html = gr.HTML()
                            decade_html = gr.HTML()
                            with gr.Accordion(
                                "Weak Signals (Color Prediction)",
                                open=False,
                            ):
                                gr.Markdown(
                                    "<small>Color prediction is less reliable on "
                                    "achromatic or dark garments.</small>"
                                )
                                color_html = gr.HTML()
                            level_imgs.append(img)
                            level_designer_htmls.append(designer_html)
                            level_decade_htmls.append(decade_html)
                            level_color_htmls.append(color_html)

        # Flatten: img0, designer0, decade0, color0, img1, ...
        outputs = []
        for i in range(4):
            outputs.append(level_imgs[i])
            outputs.append(level_designer_htmls[i])
            outputs.append(level_decade_htmls[i])
            outputs.append(level_color_htmls[i])

        submit_btn.click(
            fn=predict,
            inputs=[garment_input, face_input, year_dropdown, house_dropdown],
            outputs=outputs,
        )

    return demo


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Loading models...")
    load_all_models()
    log.info("Building UI...")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
