"""
Preprocessing pipeline for the FASH-iCNN demo.
- SegFormer clothing segmentation (HuggingFace)
- Abstraction ladder generation (grayscale, silhouette, edge map)
- Face detection via MediaPipe
Zero imports from other copalette modules.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SegFormer clothing segmentation
# ---------------------------------------------------------------------------

_segformer_model = None
_segformer_processor = None


def _load_segformer():
    """Lazy-load SegFormer model for clothing segmentation."""
    global _segformer_model, _segformer_processor
    if _segformer_model is not None:
        return _segformer_model, _segformer_processor
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        import torch
        model_name = "isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing"
        _segformer_processor = SegformerImageProcessor.from_pretrained(model_name)
        _segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        _segformer_model.eval()
        log.info("SegFormer loaded successfully")
    except Exception as e:
        log.warning(f"Failed to load SegFormer: {e}")
        _segformer_model = None
        _segformer_processor = None
    return _segformer_model, _segformer_processor


def segment_clothing(pil_img):
    """Extract the clothing region from an image using SegFormer.

    Returns:
        clothing_crop: PIL Image of the clothing region (background zeroed),
                       or the original image if segmentation fails.
        mask: Binary numpy mask of clothing pixels, or None.
    """
    model, processor = _load_segformer()
    if model is None or processor is None:
        log.warning("SegFormer not available — using full image")
        return pil_img, None

    try:
        import torch
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # (1, num_classes, H', W')
        # Upsample to original size
        upsampled = torch.nn.functional.interpolate(
            logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        # Label 3 = clothing
        mask = (seg_map == 3).astype(np.uint8)
        if mask.sum() < 100:
            log.warning("Clothing mask too small — using full image")
            return pil_img, None

        img_np = np.array(pil_img)
        masked = img_np.copy()
        masked[mask == 0] = 0
        return Image.fromarray(masked), mask

    except Exception as e:
        log.warning(f"Segmentation failed: {e}")
        return pil_img, None


# ---------------------------------------------------------------------------
# Abstraction ladder
# ---------------------------------------------------------------------------

def generate_abstraction_ladder(pil_img):
    """Generate four abstraction levels from a (segmented) garment image.

    Returns dict with keys: fullcolor, grayscale, silhouette, edge
    Each value is a PIL Image (RGB, 224x224).
    """
    img = pil_img.convert("RGB").resize((224, 224), Image.LANCZOS)
    img_np = np.array(img)

    # 1. Full color — as-is
    fullcolor = img.copy()

    # 2. Grayscale — convert then replicate to 3 channels
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    grayscale = Image.fromarray(np.stack([gray, gray, gray], axis=-1))

    # 3. Silhouette — threshold, flood fill, binary mask
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # If most pixels are dark (segmented background), invert
    if thresh.mean() < 128:
        # Use non-black pixels as the silhouette
        nonblack = (img_np.sum(axis=-1) > 30).astype(np.uint8) * 255
        silhouette_gray = nonblack
    else:
        silhouette_gray = thresh
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    silhouette_gray = cv2.morphologyEx(silhouette_gray, cv2.MORPH_CLOSE, kernel)
    silhouette = Image.fromarray(
        np.stack([silhouette_gray, silhouette_gray, silhouette_gray], axis=-1)
    )

    # 4. Edge map — Gaussian blur, Canny, invert, dilate
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = 255 - edges  # Invert: white background, dark edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(255 - edges, dilate_kernel, iterations=1)
    edges = 255 - edges  # Re-invert to get white bg + dark lines
    edge_img = Image.fromarray(np.stack([edges, edges, edges], axis=-1))

    return {
        "fullcolor": fullcolor,
        "grayscale": grayscale,
        "silhouette": silhouette,
        "edge": edge_img,
    }


# ---------------------------------------------------------------------------
# Face detection via MediaPipe
# ---------------------------------------------------------------------------

def detect_and_crop_face(pil_img):
    """Detect a face using MediaPipe and return a 224x224 crop.

    Returns:
        face_crop: PIL Image (224x224 RGB) or None if no face found.
    """
    try:
        import mediapipe as mp
        img_np = np.array(pil_img.convert("RGB"))
        h, w = img_np.shape[:2]

        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as detector:
            results = detector.process(img_np)

        if not results.detections:
            log.info("No face detected")
            return None

        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        face_crop = pil_img.crop((x1, y1, x2, y2)).resize((224, 224), Image.LANCZOS)
        return face_crop

    except Exception as e:
        log.warning(f"Face detection failed: {e}")
        return None
