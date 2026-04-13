"""
Self-contained color utilities for copalette_hierarchical_lab.

Provides:
  - Berlin-Kay 11 centroids (LAB) and the chromatic-9 ordering used by
    the upstream multimodal F checkpoint
  - CSS 140 named colors (RGB + LAB centroids)
  - CSS -> Berlin-Kay family mapping
  - LAB conversions, BK lookup, CSS lookup, ΔE76, full CIEDE2000
  - black/gray palette weight filter (matches copalette_noblack /
    copalette_multimodal_color)

NOTE: CHROMATIC_BK_NAMES is in the exact ORDER used to train the
upstream multimodal F checkpoint:
    ["white", "red", "orange", "yellow", "green",
     "blue", "purple", "pink", "brown"]
This order MUST NOT be changed — Stage 1 inference depends on the head
weights matching these indices.
"""

from collections import OrderedDict

import numpy as np
from skimage.color import rgb2lab


# ---------------------------------------------------------------------------
# Berlin-Kay 11 centroids in LAB
# ---------------------------------------------------------------------------
BERLIN_KAY_CENTROIDS_LAB = OrderedDict([
    ("white",  np.array([100.0,   0.0,   0.0])),
    ("black",  np.array([  0.0,   0.0,   0.0])),
    ("red",    np.array([ 53.2,  80.1,  67.2])),
    ("green",  np.array([ 46.2, -51.7,  49.9])),
    ("yellow", np.array([ 97.1, -21.6,  94.5])),
    ("blue",   np.array([ 32.3,  79.2, -107.9])),
    ("brown",  np.array([ 37.0,  15.0,  30.0])),
    ("purple", np.array([ 29.8,  58.9, -36.5])),
    ("pink",   np.array([ 76.0,  40.0,   7.0])),
    ("orange", np.array([ 74.9,  23.9,  78.9])),
    ("gray",   np.array([ 50.0,   0.0,   0.0])),
])
BK_NAMES = list(BERLIN_KAY_CENTROIDS_LAB.keys())
BK_LAB_ARRAY = np.stack(list(BERLIN_KAY_CENTROIDS_LAB.values()))

# Order MUST match the upstream multimodal F checkpoint head.
CHROMATIC_BK_NAMES = ["white", "red", "orange", "yellow", "green",
                     "blue", "purple", "pink", "brown"]
CHROMATIC_BK_TO_IDX = {n: i for i, n in enumerate(CHROMATIC_BK_NAMES)}
CHROMATIC_BK_IDX_TO_NAME = {i: n for n, i in CHROMATIC_BK_TO_IDX.items()}
NUM_CHROMATIC_BK = len(CHROMATIC_BK_NAMES)


# ---------------------------------------------------------------------------
# CSS 140 named colors
# ---------------------------------------------------------------------------
CSS_COLORS_RGB = OrderedDict([
    ("aliceblue", (240, 248, 255)), ("antiquewhite", (250, 235, 215)),
    ("aqua", (0, 255, 255)), ("aquamarine", (127, 255, 212)),
    ("azure", (240, 255, 255)), ("beige", (245, 245, 220)),
    ("bisque", (255, 228, 196)), ("black", (0, 0, 0)),
    ("blanchedalmond", (255, 235, 205)), ("blue", (0, 0, 255)),
    ("blueviolet", (138, 43, 226)), ("brown", (165, 42, 42)),
    ("burlywood", (222, 184, 135)), ("cadetblue", (95, 158, 160)),
    ("chartreuse", (127, 255, 0)), ("chocolate", (210, 105, 30)),
    ("coral", (255, 127, 80)), ("cornflowerblue", (100, 149, 237)),
    ("cornsilk", (255, 248, 220)), ("crimson", (220, 20, 60)),
    ("cyan", (0, 255, 255)), ("darkblue", (0, 0, 139)),
    ("darkcyan", (0, 139, 139)), ("darkgoldenrod", (184, 134, 11)),
    ("darkgray", (169, 169, 169)), ("darkgreen", (0, 100, 0)),
    ("darkkhaki", (189, 183, 107)), ("darkmagenta", (139, 0, 139)),
    ("darkolivegreen", (85, 107, 47)), ("darkorange", (255, 140, 0)),
    ("darkorchid", (153, 50, 204)), ("darkred", (139, 0, 0)),
    ("darksalmon", (233, 150, 122)), ("darkseagreen", (143, 188, 143)),
    ("darkslateblue", (72, 61, 139)), ("darkslategray", (47, 79, 79)),
    ("darkturquoise", (0, 206, 209)), ("darkviolet", (148, 0, 211)),
    ("deeppink", (255, 20, 147)), ("deepskyblue", (0, 191, 255)),
    ("dimgray", (105, 105, 105)), ("dodgerblue", (30, 144, 255)),
    ("firebrick", (178, 34, 34)), ("floralwhite", (255, 250, 240)),
    ("forestgreen", (34, 139, 34)), ("fuchsia", (255, 0, 255)),
    ("gainsboro", (220, 220, 220)), ("ghostwhite", (248, 248, 255)),
    ("gold", (255, 215, 0)), ("goldenrod", (218, 165, 32)),
    ("gray", (128, 128, 128)), ("green", (0, 128, 0)),
    ("greenyellow", (173, 255, 47)), ("honeydew", (240, 255, 240)),
    ("hotpink", (255, 105, 180)), ("indianred", (205, 92, 92)),
    ("indigo", (75, 0, 130)), ("ivory", (255, 255, 240)),
    ("khaki", (240, 230, 140)), ("lavender", (230, 230, 250)),
    ("lavenderblush", (255, 240, 245)), ("lawngreen", (124, 252, 0)),
    ("lemonchiffon", (255, 250, 205)), ("lightblue", (173, 216, 230)),
    ("lightcoral", (240, 128, 128)), ("lightcyan", (224, 255, 255)),
    ("lightgoldenrodyellow", (250, 250, 210)), ("lightgray", (211, 211, 211)),
    ("lightgreen", (144, 238, 144)), ("lightpink", (255, 182, 193)),
    ("lightsalmon", (255, 160, 122)), ("lightseagreen", (32, 178, 170)),
    ("lightskyblue", (135, 206, 250)), ("lightslategray", (119, 136, 153)),
    ("lightsteelblue", (176, 196, 222)), ("lightyellow", (255, 255, 224)),
    ("lime", (0, 255, 0)), ("limegreen", (50, 205, 50)),
    ("linen", (250, 240, 230)), ("magenta", (255, 0, 255)),
    ("maroon", (128, 0, 0)), ("mediumaquamarine", (102, 205, 170)),
    ("mediumblue", (0, 0, 205)), ("mediumorchid", (186, 85, 211)),
    ("mediumpurple", (147, 112, 219)), ("mediumseagreen", (60, 179, 113)),
    ("mediumslateblue", (123, 104, 238)), ("mediumspringgreen", (0, 250, 154)),
    ("mediumturquoise", (72, 209, 204)), ("mediumvioletred", (199, 21, 133)),
    ("midnightblue", (25, 25, 112)), ("mintcream", (245, 255, 250)),
    ("mistyrose", (255, 228, 225)), ("moccasin", (255, 228, 181)),
    ("navajowhite", (255, 222, 173)), ("navy", (0, 0, 128)),
    ("oldlace", (253, 245, 230)), ("olive", (128, 128, 0)),
    ("olivedrab", (107, 142, 35)), ("orange", (255, 165, 0)),
    ("orangered", (255, 69, 0)), ("orchid", (218, 112, 214)),
    ("palegoldenrod", (238, 232, 170)), ("palegreen", (152, 251, 152)),
    ("paleturquoise", (175, 238, 238)), ("palevioletred", (219, 112, 147)),
    ("papayawhip", (255, 239, 213)), ("peachpuff", (255, 218, 185)),
    ("peru", (205, 133, 63)), ("pink", (255, 192, 203)),
    ("plum", (221, 160, 221)), ("powderblue", (176, 224, 230)),
    ("purple", (128, 0, 128)), ("rebeccapurple", (102, 51, 153)),
    ("red", (255, 0, 0)), ("rosybrown", (188, 143, 143)),
    ("royalblue", (65, 105, 225)), ("saddlebrown", (139, 69, 19)),
    ("salmon", (250, 128, 114)), ("sandybrown", (244, 164, 96)),
    ("seagreen", (46, 139, 87)), ("seashell", (255, 245, 238)),
    ("sienna", (160, 82, 45)), ("silver", (192, 192, 192)),
    ("skyblue", (135, 206, 235)), ("slateblue", (106, 90, 205)),
    ("slategray", (112, 128, 144)), ("snow", (255, 250, 250)),
    ("springgreen", (0, 255, 127)), ("steelblue", (70, 130, 180)),
    ("tan", (210, 180, 140)), ("teal", (0, 128, 128)),
    ("thistle", (216, 191, 216)), ("tomato", (255, 99, 71)),
    ("turquoise", (64, 224, 208)), ("violet", (238, 130, 238)),
    ("wheat", (245, 222, 179)), ("white", (255, 255, 255)),
    ("whitesmoke", (245, 245, 245)), ("yellow", (255, 255, 0)),
    ("yellowgreen", (154, 205, 50)),
])

CSS_TO_BK = {
    "aliceblue": "white", "antiquewhite": "white", "aqua": "blue",
    "aquamarine": "green", "azure": "white", "beige": "white",
    "bisque": "orange", "black": "black", "blanchedalmond": "white",
    "blue": "blue", "blueviolet": "purple", "brown": "red",
    "burlywood": "brown", "cadetblue": "blue", "chartreuse": "green",
    "chocolate": "brown", "coral": "orange", "cornflowerblue": "blue",
    "cornsilk": "white", "crimson": "red", "cyan": "blue",
    "darkblue": "blue", "darkcyan": "blue", "darkgoldenrod": "brown",
    "darkgray": "gray", "darkgreen": "green", "darkkhaki": "yellow",
    "darkmagenta": "purple", "darkolivegreen": "green",
    "darkorange": "orange", "darkorchid": "purple", "darkred": "red",
    "darksalmon": "orange", "darkseagreen": "green",
    "darkslateblue": "purple", "darkslategray": "gray",
    "darkturquoise": "blue", "darkviolet": "purple", "deeppink": "pink",
    "deepskyblue": "blue", "dimgray": "gray", "dodgerblue": "blue",
    "firebrick": "red", "floralwhite": "white", "forestgreen": "green",
    "fuchsia": "pink", "gainsboro": "gray", "ghostwhite": "white",
    "gold": "yellow", "goldenrod": "yellow", "gray": "gray",
    "green": "green", "greenyellow": "green", "honeydew": "white",
    "hotpink": "pink", "indianred": "red", "indigo": "purple",
    "ivory": "white", "khaki": "yellow", "lavender": "white",
    "lavenderblush": "white", "lawngreen": "green",
    "lemonchiffon": "white", "lightblue": "blue", "lightcoral": "pink",
    "lightcyan": "white", "lightgoldenrodyellow": "white",
    "lightgray": "gray", "lightgreen": "green", "lightpink": "pink",
    "lightsalmon": "orange", "lightseagreen": "blue",
    "lightskyblue": "blue", "lightslategray": "gray",
    "lightsteelblue": "blue", "lightyellow": "white", "lime": "green",
    "limegreen": "green", "linen": "white", "magenta": "pink",
    "maroon": "red", "mediumaquamarine": "green", "mediumblue": "blue",
    "mediumorchid": "purple", "mediumpurple": "purple",
    "mediumseagreen": "green", "mediumslateblue": "purple",
    "mediumspringgreen": "green", "mediumturquoise": "blue",
    "mediumvioletred": "pink", "midnightblue": "blue",
    "mintcream": "white", "mistyrose": "white", "moccasin": "yellow",
    "navajowhite": "orange", "navy": "blue", "oldlace": "white",
    "olive": "green", "olivedrab": "green", "orange": "orange",
    "orangered": "red", "orchid": "purple", "palegoldenrod": "yellow",
    "palegreen": "green", "paleturquoise": "blue",
    "palevioletred": "pink", "papayawhip": "white",
    "peachpuff": "orange", "peru": "brown", "pink": "pink",
    "plum": "purple", "powderblue": "blue", "purple": "purple",
    "rebeccapurple": "purple", "red": "red", "rosybrown": "brown",
    "royalblue": "blue", "saddlebrown": "brown", "salmon": "orange",
    "sandybrown": "orange", "seagreen": "green", "seashell": "white",
    "sienna": "brown", "silver": "gray", "skyblue": "blue",
    "slateblue": "purple", "slategray": "gray", "snow": "white",
    "springgreen": "green", "steelblue": "blue", "tan": "brown",
    "teal": "blue", "thistle": "purple", "tomato": "red",
    "turquoise": "blue", "violet": "purple", "wheat": "white",
    "white": "white", "whitesmoke": "white", "yellow": "yellow",
    "yellowgreen": "green",
}

CSS_NAMES = list(CSS_COLORS_RGB.keys())
_css_rgb = np.array([CSS_COLORS_RGB[n] for n in CSS_NAMES], dtype=np.float64)
CSS_LAB_ARRAY = rgb2lab(_css_rgb.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
CSS_NAME_TO_LAB = {n: CSS_LAB_ARRAY[i] for i, n in enumerate(CSS_NAMES)}
BLACKGRAY_CSS = frozenset(n for n, bk in CSS_TO_BK.items() if bk in ("black", "gray"))


def css_to_lab(css_name):
    """Lookup the LAB centroid of a single CSS color name."""
    return CSS_NAME_TO_LAB.get(css_name, np.array([50.0, 0.0, 0.0]))


def css_to_lab_batch(css_names):
    """Vectorized: list of CSS names -> (N, 3) LAB array."""
    return np.stack([CSS_NAME_TO_LAB.get(n, np.array([50.0, 0.0, 0.0]))
                     for n in css_names], axis=0)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
def rgb_to_lab_array(rgb_array):
    rgb = np.asarray(rgb_array, dtype=np.float64).reshape(1, -1, 3) / 255.0
    return rgb2lab(rgb).reshape(-1, 3)


def rgb_to_lab_single(r, g, b):
    rgb = np.array([[[r, g, b]]], dtype=np.float64) / 255.0
    return rgb2lab(rgb)[0, 0]


def lab_to_berlin_kay(lab):
    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim == 1:
        lab = lab.reshape(1, 3)
    dists = np.linalg.norm(lab[:, None, :] - BK_LAB_ARRAY[None, :, :], axis=2)
    idx = np.argmin(dists, axis=1)
    return [BK_NAMES[i] for i in idx]


def lab_to_berlin_kay_single(lab):
    return lab_to_berlin_kay(np.array(lab).reshape(1, 3))[0]


def lab_to_css_name(lab):
    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim == 1:
        lab = lab.reshape(1, 3)
    dists = np.linalg.norm(lab[:, None, :] - CSS_LAB_ARRAY[None, :, :], axis=2)
    idx = np.argmin(dists, axis=1)
    return [CSS_NAMES[i] for i in idx]


def css_name_to_berlin_kay(name):
    return CSS_TO_BK.get(name, "white")


# ---------------------------------------------------------------------------
# Black/gray palette weight (matches copalette_noblack / multimodal_color)
# ---------------------------------------------------------------------------
def compute_blackgray_weight(row):
    total_pct = 0.0
    bg_pct = 0.0
    for i in range(1, 7):
        pct = row.get(f"c{i}_pct", 0)
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            continue
        pct = float(pct)
        total_pct += pct
        r = row.get(f"c{i}_r", 0)
        g = row.get(f"c{i}_g", 0)
        b = row.get(f"c{i}_b", 0)
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [r, g, b]):
            continue
        lab = rgb_to_lab_single(float(r), float(g), float(b))
        bk = lab_to_berlin_kay_single(lab)
        if bk in ("black", "gray"):
            bg_pct += pct
    if total_pct == 0:
        return 0.0
    return bg_pct / total_pct


# ---------------------------------------------------------------------------
# Delta-E
# ---------------------------------------------------------------------------
def delta_e_cie76(pred_lab, true_lab):
    pred = np.asarray(pred_lab, dtype=np.float64)
    true = np.asarray(true_lab, dtype=np.float64)
    return np.sqrt(np.sum((pred - true) ** 2, axis=-1))


def delta_e_ciede2000(lab1, lab2):
    """Full CIEDE2000 (Sharma, Wu, Dalal 2005). lab1, lab2: (N, 3)."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg ** 7
    G = 0.5 * (1.0 - np.sqrt(C_avg7 / (C_avg7 + 25.0 ** 7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = np.zeros_like(h1p)
    mask1 = (C1p * C2p) != 0
    diff = h2p - h1p
    cond_a = mask1 & (np.abs(diff) <= 180)
    cond_b = mask1 & (diff > 180)
    cond_c = mask1 & (diff < -180)
    dhp[cond_a] = diff[cond_a]
    dhp[cond_b] = diff[cond_b] - 360.0
    dhp[cond_c] = diff[cond_c] + 360.0
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0
    hp_avg = np.zeros_like(h1p)
    sum_h = h1p + h2p
    abs_diff_h = np.abs(h1p - h2p)
    m_le180 = mask1 & (abs_diff_h <= 180)
    m_gt180_lt = mask1 & (abs_diff_h > 180) & (sum_h < 360)
    m_gt180_ge = mask1 & (abs_diff_h > 180) & (sum_h >= 360)
    m_one_zero = ~mask1
    hp_avg[m_le180] = sum_h[m_le180] / 2.0
    hp_avg[m_gt180_lt] = (sum_h[m_gt180_lt] + 360.0) / 2.0
    hp_avg[m_gt180_ge] = (sum_h[m_gt180_ge] - 360.0) / 2.0
    hp_avg[m_one_zero] = sum_h[m_one_zero]

    T = (1.0
         - 0.17 * np.cos(np.radians(hp_avg - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp_avg))
         + 0.32 * np.cos(np.radians(3.0 * hp_avg + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp_avg - 63.0)))
    SL = 1.0 + 0.015 * (Lp_avg - 50.0) ** 2 / np.sqrt(20.0 + (Lp_avg - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T
    Cp_avg7 = Cp_avg ** 7
    RC = 2.0 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25.0 ** 7))
    d_theta = 30.0 * np.exp(-((hp_avg - 275.0) / 25.0) ** 2)
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC

    kL = kC = kH = 1.0
    dE = np.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return dE
