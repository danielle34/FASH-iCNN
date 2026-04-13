# Temporal Identity

Temporal editorial identity prediction from clothing crops alone. Garment appearance encodes both which decade and which specific year a runway image belongs to.

**Paper section:** Section 4.2 (Garment Appearance Encodes Editorial Identity)

## Subfolders

### `decade/`
Decade classification (4 classes: 1991--2000, 2001--2010, 2011--2020, 2021--2024). Reaches 88.6% top-1 against a 45.2% majority baseline.

```bash
cd decade && python run_clothing_decade.py
```

### `year/`
Fine-grained year prediction (34 classes, 1991--2024). Reaches 58.3% top-1 against a 2.9% random baseline, with mean absolute error of 2.2 years and 73.2% of predictions landing within 2 years of the correct answer.

```bash
cd year && python run_clothing_year.py
```

## Shared modules

Each subfolder contains its own `dataset.py`, `model.py`, `train.py`, and `evaluate.py`.
