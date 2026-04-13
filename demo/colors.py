"""
Self-contained color utilities for the FASH-iCNN demo.
Berlin-Kay 11-term centroids, CSS 140 named colors, LAB conversions.
Zero imports from other copalette modules.
"""

from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# Berlin-Kay 11 centroids in CIELAB
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

# Chromatic subset (excluding black and gray) — sorted alphabetically
# These are the 9 classes used by the BK color classifier
CHROMATIC_BK_NAMES = sorted([n for n in BK_NAMES if n not in ("black", "gray")])
# => ['blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

# Representative hex for each BK color (for display swatches)
BK_HEX = {
    "white": "#ffffff", "black": "#000000", "red": "#e60000",
    "green": "#228b22", "yellow": "#ffd700", "blue": "#0000ff",
    "brown": "#8b4513", "purple": "#800080", "pink": "#ff69b4",
    "orange": "#ff8c00", "gray": "#808080",
}

# ---------------------------------------------------------------------------
# CSS 140 named colors  (name -> (R, G, B))
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

CSS_NAMES = list(CSS_COLORS_RGB.keys())

# CSS name -> Berlin-Kay family
CSS_TO_BK = {
    "aliceblue": "white", "antiquewhite": "white", "aqua": "blue",
    "aquamarine": "green", "azure": "white", "beige": "white",
    "bisque": "orange", "black": "black", "blanchedalmond": "white",
    "blue": "blue", "blueviolet": "purple", "brown": "brown",
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


# ---------------------------------------------------------------------------
# Conversion helpers (no skimage dependency — pure numpy)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c):
    """sRGB [0,1] -> linear RGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    """Linear RGB -> sRGB [0,1]."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def rgb_to_lab(r, g, b):
    """Convert a single (R,G,B) 0-255 to CIELAB (L*,a*,b*)."""
    rgb = np.array([r, g, b], dtype=np.float64) / 255.0
    lin = _srgb_to_linear(rgb)
    # Linear RGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = M @ lin
    # Normalize by D65 white point
    xyz /= np.array([0.95047, 1.0, 1.08883])
    # f(t)
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[1] - 16.0
    a = 500.0 * (f[0] - f[1])
    b_val = 200.0 * (f[1] - f[2])
    return np.array([L, a, b_val])


def lab_to_rgb_uint8(L, a, b_val):
    """Convert CIELAB to (R,G,B) 0-255."""
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_val / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    x = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.array([x, y, z]) * np.array([0.95047, 1.0, 1.08883])
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    lin = M_inv @ xyz
    srgb = _linear_to_srgb(np.clip(lin, 0, None))
    return tuple(int(max(0, min(255, round(v * 255)))) for v in srgb)


def lab_to_hex(L, a, b_val):
    """CIELAB -> '#rrggbb' hex string."""
    r, g, b = lab_to_rgb_uint8(L, a, b_val)
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_hex(r, g, b):
    """(R,G,B) 0-255 -> '#rrggbb'."""
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def dominant_color_from_image(img_rgb):
    """Compute the mean color of non-black pixels in an RGB numpy image.
    Returns (L, a, b) in CIELAB, the CSS name, and the BK family name.
    """
    pixels = img_rgb.reshape(-1, 3).astype(np.float64)
    # Exclude near-black (background) pixels
    mask = pixels.sum(axis=1) > 30
    if mask.sum() < 10:
        mask = np.ones(len(pixels), dtype=bool)
    mean_rgb = pixels[mask].mean(axis=0)
    lab = rgb_to_lab(mean_rgb[0], mean_rgb[1], mean_rgb[2])

    # Find nearest CSS color
    css_lab_list = []
    for name in CSS_NAMES:
        r, g, b = CSS_COLORS_RGB[name]
        css_lab_list.append(rgb_to_lab(r, g, b))
    css_lab_arr = np.stack(css_lab_list)
    dists = np.linalg.norm(css_lab_arr - lab[None, :], axis=1)
    css_idx = int(np.argmin(dists))
    css_name = CSS_NAMES[css_idx]
    bk_name = CSS_TO_BK.get(css_name, "gray")

    return {
        "lab": lab.tolist(),
        "hex": lab_to_hex(lab[0], lab[1], lab[2]),
        "css_name": css_name,
        "css_rgb": CSS_COLORS_RGB[css_name],
        "css_hex": rgb_to_hex(*CSS_COLORS_RGB[css_name]),
        "bk_name": bk_name,
        "bk_hex": BK_HEX.get(bk_name, "#808080"),
    }
