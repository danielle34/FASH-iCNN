# Demo

Interactive Gradio web application for FASH-iCNN. Upload a garment image and optionally a face image to receive a dominant-color recommendation at three specificity levels (Berlin-Kay family, CSS named color, CIELAB coordinate) alongside retrievable editorial runway precedents.

**Paper sections:** Section 4.1 (Architecture), Section 5.1 (Interaction Implications)

## Contents

- `app.py` — Gradio web interface
- `inference.py` — Model inference pipeline (loads trained checkpoints, runs hierarchical prediction)
- `preprocessing.py` — Image preprocessing (SegFormer clothing extraction, MediaPipe face cropping)
- `colors.py` — Color mappings (Berlin-Kay, CSS named colors, CIELAB centroids)

## Usage

```bash
pip install -r ../requirements.txt
python app.py
```

This launches a local Gradio server. Requires trained model checkpoints (not included in this repository).
