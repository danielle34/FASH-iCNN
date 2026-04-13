"""
Model loading and inference for the FASH-iCNN demo.
Loads four EfficientNet-B0 classifiers from checkpoints.
All run on CPU. Missing checkpoints -> dummy mode.
Zero imports from other copalette modules.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve checkpoint paths relative to *this* file's parent
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent  # ~/copalette

_DESIGNER_CKPT_DIR = _REPO / "copalette_clothing_designer_abstraction" / "outputs" / "checkpoints"
CKPT_DESIGNER_BY_LEVEL = {
    "fullcolor":   _DESIGNER_CKPT_DIR / "ckpt_fullcolor.pth",
    "grayscale":   _DESIGNER_CKPT_DIR / "ckpt_grayscale.pth",
    "silhouette":  _DESIGNER_CKPT_DIR / "ckpt_silhouette.pth",
    "edge":        _DESIGNER_CKPT_DIR / "ckpt_edge.pth",
}
CKPT_DECADE = _REPO / "copalette_clothing_decade" / "outputs" / "checkpoints" / "ckpt_clothing_only.pth"
CKPT_CSS = _REPO / "copalette_css_clothing" / "outputs" / "checkpoints" / "ckpt_A_clothing_only.pth"

# BK color checkpoints are per-decade; map decade label -> checkpoint
_BK_CKPT_DIR = _REPO / "copalette_clothing_constrained" / "outputs" / "checkpoints"
CKPT_BK_BY_DECADE = {
    "1991-2000": _BK_CKPT_DIR / "ckpt_bk_1991-2000.pth",
    "2001-2010": _BK_CKPT_DIR / "ckpt_bk_2001-2010.pth",
    "2011-2020": _BK_CKPT_DIR / "ckpt_bk_2011-2020.pth",
    "2021-2024": _BK_CKPT_DIR / "ckpt_bk_2021-2024.pth",
}
CKPT_BK_DEFAULT = _BK_CKPT_DIR / "ckpt_bk_2011-2020.pth"

# ---------------------------------------------------------------------------
# Label definitions (must match training order exactly)
# ---------------------------------------------------------------------------

DESIGNER_LABELS = [
    "alexander mcqueen", "balenciaga", "calvin klein collection", "chanel",
    "christian dior", "fendi", "gucci", "hermes", "louis vuitton", "prada",
    "ralph lauren", "saint laurent", "valentino", "versace",
]

DECADE_LABELS = ["1991-2000", "2001-2010", "2011-2020", "2021-2024"]

# Chromatic BK (sorted alphabetically, excludes black/gray)
BK_LABELS = ["blue", "brown", "green", "orange", "pink", "purple", "red", "white", "yellow"]

CSS_LABELS = [
    "aliceblue", "antiquewhite", "beige", "bisque", "brown",
    "burlywood", "chocolate", "coral", "crimson", "darkgoldenrod",
    "darkkhaki", "darkolivegreen", "darkred", "darksalmon", "darkseagreen",
    "darkslateblue", "deeppink", "firebrick", "floralwhite", "ghostwhite",
    "gold", "goldenrod", "indianred", "khaki", "lavender",
    "lavenderblush", "lightcoral", "lightpink", "lightsteelblue", "linen",
    "maroon", "mediumvioletred", "midnightblue", "mistyrose", "oldlace",
    "orangered", "palegoldenrod", "palevioletred", "peachpuff", "peru",
    "pink", "plum", "red", "rosybrown", "saddlebrown",
    "sandybrown", "seashell", "sienna", "steelblue", "tan",
    "thistle", "tomato", "wheat", "whitesmoke", "yellowgreen",
]

DESIGNER_DISPLAY = {
    "alexander mcqueen": "Alexander McQueen",
    "balenciaga": "Balenciaga",
    "calvin klein collection": "Calvin Klein Collection",
    "chanel": "Chanel",
    "christian dior": "Christian Dior",
    "fendi": "Fendi",
    "gucci": "Gucci",
    "hermes": "Hermes",
    "louis vuitton": "Louis Vuitton",
    "prada": "Prada",
    "ralph lauren": "Ralph Lauren",
    "saint laurent": "Saint Laurent",
    "valentino": "Valentino",
    "versace": "Versace",
}


# ---------------------------------------------------------------------------
# Model architectures (self-contained, matching training code)
# ---------------------------------------------------------------------------

class SingleStreamClassifier(nn.Module):
    """EfficientNet-B0 with custom classification head."""

    def __init__(self, num_classes, head_hidden=512, dropout=0.3):
        super().__init__()
        bb = models.efficientnet_b0(weights=None)
        bb.classifier = nn.Identity()
        self.backbone = bb
        self.head = nn.Sequential(
            nn.Linear(1280, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


class DualStreamClassifier(nn.Module):
    """Dual-stream EfficientNet-B0 with optional face+clothing fusion."""

    def __init__(self, num_classes, use_face=True, use_clothing=True):
        super().__init__()
        self.use_face = use_face
        self.use_clothing = use_clothing

        if use_face:
            face_bb = models.efficientnet_b0(weights=None)
            face_bb.classifier = nn.Identity()
            self.face_backbone = face_bb
            self.face_dim = 1280
        else:
            self.face_backbone = None
            self.face_dim = 0

        if use_clothing:
            clothing_bb = models.efficientnet_b0(weights=None)
            clothing_bb.classifier = nn.Identity()
            self.clothing_backbone = clothing_bb
            self.clothing_dim = 1280
        else:
            self.clothing_backbone = None
            self.clothing_dim = 0

        fusion_dim = self.face_dim + self.clothing_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, face_img, clothing_img):
        feats = []
        if self.face_backbone is not None:
            feats.append(self.face_backbone(face_img))
        if self.clothing_backbone is not None:
            feats.append(self.clothing_backbone(clothing_img))
        fused = torch.cat(feats, dim=1)
        return self.head(fused)


# ---------------------------------------------------------------------------
# Image transforms (eval-time)
# ---------------------------------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model registry -- loaded once at startup
# ---------------------------------------------------------------------------

_models = {}


def _load_single_stream(name, ckpt_path, num_classes, head_hidden=512, dropout=0.3,
                         state_key="model_state_dict"):
    """Load a SingleStreamClassifier checkpoint. Returns None on failure."""
    if not ckpt_path.exists():
        log.warning(f"[{name}] Checkpoint not found: {ckpt_path}")
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = SingleStreamClassifier(num_classes, head_hidden=head_hidden, dropout=dropout)
        model.load_state_dict(ckpt[state_key])
        model.eval()
        log.info(f"[{name}] Loaded from {ckpt_path}")
        return model
    except Exception as e:
        log.warning(f"[{name}] Failed to load: {e}")
        return None


def _load_dual_stream(name, ckpt_path, num_classes, use_face, use_clothing,
                       state_key="model_state_dict"):
    """Load a DualStreamClassifier checkpoint. Returns None on failure."""
    if not ckpt_path.exists():
        log.warning(f"[{name}] Checkpoint not found: {ckpt_path}")
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DualStreamClassifier(num_classes, use_face=use_face, use_clothing=use_clothing)
        model.load_state_dict(ckpt[state_key])
        model.eval()
        log.info(f"[{name}] Loaded from {ckpt_path}")
        return model
    except Exception as e:
        log.warning(f"[{name}] Failed to load: {e}")
        return None


ABSTRACTION_LEVELS = ["fullcolor", "grayscale", "silhouette", "edge"]


def load_all_models():
    """Load all model families at startup."""
    # Designer classifiers (14-way, hidden=512, dropout=0.3) -- one per abstraction level
    _models["designer"] = {}
    for level, path in CKPT_DESIGNER_BY_LEVEL.items():
        m = _load_single_stream(
            f"designer_{level}", path, num_classes=14,
            head_hidden=512, dropout=0.3, state_key="model_state_dict",
        )
        if m is not None:
            _models["designer"][level] = m

    # Decade classifier (4-way, hidden=256, dropout=0.3)
    _models["decade"] = _load_single_stream(
        "decade", CKPT_DECADE, num_classes=4,
        head_hidden=256, dropout=0.3, state_key="model_state",
    )

    # BK color classifiers (9-way, hidden=256, dropout=0.5) -- one per decade
    _models["bk"] = {}
    for decade, path in CKPT_BK_BY_DECADE.items():
        m = _load_single_stream(
            f"bk_{decade}", path, num_classes=9,
            head_hidden=256, dropout=0.5, state_key="model_state",
        )
        if m is not None:
            _models["bk"][decade] = m

    # CSS color classifier (55-way, clothing-only variant)
    _models["css"] = _load_dual_stream(
        "css", CKPT_CSS, num_classes=55,
        use_face=False, use_clothing=True, state_key="model_state_dict",
    )

    avail = []
    designer_models = _models.get("designer", {})
    if designer_models:
        avail.append(f"designer({','.join(designer_models.keys())})")
    if _models.get("decade"):
        avail.append("decade")
    if _models.get("bk"):
        avail.append(f"bk({len(_models['bk'])} decades)")
    if _models.get("css"):
        avail.append("css")
    log.info(f"Models loaded: {', '.join(avail) if avail else 'NONE'}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(pil_img):
    """Convert PIL image to a normalized (1, 3, 224, 224) tensor."""
    return eval_transform(pil_img.convert("RGB")).unsqueeze(0)


def _year_to_decade(year):
    """Map a year to its decade bucket."""
    if year is None:
        return None
    y = int(year)
    if y <= 2000:
        return "1991-2000"
    elif y <= 2010:
        return "2001-2010"
    elif y <= 2020:
        return "2011-2020"
    else:
        return "2021-2024"


@torch.no_grad()
def predict_designer(clothing_pil, level="fullcolor"):
    """Predict fashion house using the checkpoint for the given abstraction level.

    Args:
        clothing_pil: PIL image (already at the desired abstraction level).
        level: One of "fullcolor", "grayscale", "silhouette", "edge".

    Returns list of (name, confidence) top-3, or None if model unavailable.
    """
    designer_models = _models.get("designer", {})
    model = designer_models.get(level)
    if model is None:
        return None
    x = _pil_to_tensor(clothing_pil)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze()
    top3 = torch.topk(probs, k=min(3, len(probs)))
    results = []
    for idx, prob in zip(top3.indices.tolist(), top3.values.tolist()):
        label = DESIGNER_LABELS[idx]
        results.append((DESIGNER_DISPLAY.get(label, label), prob))
    return results


@torch.no_grad()
def predict_decade(clothing_pil):
    """Predict decade. Returns (label, confidence)."""
    model = _models.get("decade")
    if model is None:
        return None
    x = _pil_to_tensor(clothing_pil)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze()
    idx = probs.argmax().item()
    return (DECADE_LABELS[idx], probs[idx].item())


@torch.no_grad()
def predict_bk_color(clothing_pil, year=None):
    """Predict Berlin-Kay color family. Returns (label, confidence)."""
    bk_models = _models.get("bk", {})
    if not bk_models:
        return None
    decade = _year_to_decade(year)
    if decade and decade in bk_models:
        model = bk_models[decade]
    else:
        # Fall back to 2011-2020 or first available
        model = bk_models.get("2011-2020") or next(iter(bk_models.values()))
    x = _pil_to_tensor(clothing_pil)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze()
    idx = probs.argmax().item()
    return (BK_LABELS[idx], probs[idx].item())


@torch.no_grad()
def predict_css_color(clothing_pil):
    """Predict CSS named color. Returns (label, confidence)."""
    model = _models.get("css")
    if model is None:
        return None
    x = _pil_to_tensor(clothing_pil)
    # DualStreamClassifier with use_face=False only uses clothing backbone
    # but forward() signature requires both args
    logits = model(x, x)
    probs = torch.softmax(logits, dim=1).squeeze()
    idx = probs.argmax().item()
    return (CSS_LABELS[idx], probs[idx].item())
