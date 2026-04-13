# Color Prediction

Hierarchical BK → CSS → constrained-LAB color prediction pipeline. Reduces perceptual error from ΔE₀₀ = 15.0 (unconstrained regression) to 9.10 (full pipeline).

**Paper sections:** Section 4.1 (Architecture), Section 4.2 (Per-designer constrained models), Section 4.3 (Visual Abstraction Analysis), Section 4.5 (Palette Prediction)

## Subfolders

### `hierarchical_lab/`
Constrained LAB regression conditioned on predicted BK and CSS classes. This is the final stage of the three-stage pipeline.

```bash
cd hierarchical_lab && python run_hierarchical_lab.py
```

### `hierarchical_color/`
Combined BK → CSS hierarchical classification pipeline.

```bash
cd hierarchical_color && python run_hierarchical.py
```

### `css_clothing/`
CSS named-color classification from clothing crops. Predicts fine-grained color (54--69 classes).

```bash
cd css_clothing && python run_css_clothing.py
```

### `clothing_constrained/`
Per-designer constrained BK9 color models. Each model is trained and evaluated within a single house's chromatic subset (Table 2).

```bash
cd clothing_constrained && python run_clothing_constrained.py
```

## Shared modules

Each subfolder contains its own `colors.py` (color mappings), `dataset.py` (data loading), `model.py` (architecture), `train.py` (training loop), and `evaluate.py` (evaluation metrics).
