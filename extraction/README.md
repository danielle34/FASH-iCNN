# Extraction

Clothing crop extraction from Vogue runway images using SegFormer (ADE20K label 3) and face crop extraction via MediaPipe.

**Paper section:** Section 3 (Dataset and Corpus)

## Contents

- `run_extract_clothing.py` — Extracts clothing regions from runway images using SegFormer semantic segmentation, and face crops using MediaPipe Face Mesh.

## Usage

```bash
python run_extract_clothing.py --data_dir /path/to/vogue-runway --output_dir /path/to/crops
```

Produces 65,541 garment crops and 77,269 face crops from the 84,596 quality-filtered images.
