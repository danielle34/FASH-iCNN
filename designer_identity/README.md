# Designer Identity

Fashion house identification from clothing crops at multiple levels of visual abstraction. Full-color garment appearance identifies the house at 78.2% top-1 across 14 houses (vs. 9.3% majority baseline).

**Paper section:** Section 4.2 (Garment Appearance Encodes Editorial Identity), Table 3

## Subfolders

### `abstraction/`
The core abstraction experiment: trains independent EfficientNet-B0 models on four representations (full color, grayscale, silhouette, edge map) for 14-way designer classification. Produces the Designer Top-1 column in Table 3.

```bash
cd abstraction && python run_abstraction_designer.py
```

### `full_designer/`
Full-crop designer classification with optional face input and designer embedding experiments (Section 4.4, face-to-designer implicit encoding).

```bash
cd full_designer && python run_full_designer.py
```

### `silhouette_designer/`
Silhouette-only designer classification, isolating shape signal from color and texture.

```bash
cd silhouette_designer && python run_silhouette_designer.py
```

## Shared modules

Each subfolder contains its own `dataset.py`, `model.py`, `train.py`, and `evaluate.py`.
