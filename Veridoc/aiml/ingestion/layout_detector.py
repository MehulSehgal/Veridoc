"""
Document layout region detection — classical computer vision, no downloaded
model weights. Finds figure/table-shaped regions on a rendered page image
using OpenCV: adaptive thresholding + morphological closing to merge nearby
ink into blocks, then contour filtering by size/aspect-ratio/fill-ratio to
keep blocks that look like a figure or table rather than a paragraph of text.

This trades the accuracy of a trained document-layout model (e.g.
DocLayout-YOLO) for zero external dependencies: no model download, no
internet access needed at runtime, works fully offline out of the box.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

MIN_REGION_FRACTION = 0.02   # region must cover at least 2% of the page area
MAX_REGION_FRACTION = 0.85   # ...and no more than 85% (avoid flagging the whole page)


@dataclass
class LayoutRegion:
    label: str  # "figure" (this detector doesn't distinguish figure vs table)
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel coords
    confidence: float
    source: str  # "opencv_heuristic"


def detect_regions(image: Image.Image) -> list[LayoutRegion]:
    arr = np.array(image.convert("L"))
    h, w = arr.shape
    page_area = h * w

    # Body text is thin, high-frequency ink; figures/tables/plots tend to have
    # larger connected blocks of ink once dilated. Adaptive threshold copes
    # with scan/render brightness variation better than a single global cutoff.
    thresh = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # Dilate horizontally+vertically to fuse nearby strokes/lines/bars into
    # solid blobs -- this is what separates "a paragraph" (thin, sparse after
    # dilation) from "a chart" (dense blob after dilation).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[LayoutRegion] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        area_frac = area / page_area
        if area_frac < MIN_REGION_FRACTION or area_frac > MAX_REGION_FRACTION:
            continue

        aspect = cw / max(ch, 1)
        if aspect > 12 or aspect < 0.08:
            continue  # skip long thin strips (rules, headers/footers)

        # fill ratio inside the *original* threshold mask (before dilation):
        # a photo/plot region is fairly densely inked; a dilated paragraph
        # block, ironically, is also dense -- so we mainly use this to drop
        # near-empty boxes (whitespace regions dilation occasionally catches).
        roi = thresh[y : y + ch, x : x + cw]
        fill_ratio = float((roi > 0).mean())
        if fill_ratio < 0.03:
            continue

        confidence = min(0.5 + fill_ratio, 0.95)
        regions.append(
            LayoutRegion(
                label="figure",
                bbox=(x, y, x + cw, y + ch),
                confidence=confidence,
                source="opencv_heuristic",
            )
        )

    return _dedupe_nested(regions)


def _dedupe_nested(regions: list[LayoutRegion], overlap_thresh: float = 0.8) -> list[LayoutRegion]:
    """Drop a region if another region already contains most of it (contour
    nesting from the dilation step can produce boxes-within-boxes)."""
    kept: list[LayoutRegion] = []
    regions_sorted = sorted(regions, key=lambda r: -(r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
    for r in regions_sorted:
        x0, y0, x1, y1 = r.bbox
        area = (x1 - x0) * (y1 - y0)
        nested = False
        for kept_r in kept:
            kx0, ky0, kx1, ky1 = kept_r.bbox
            ix0, iy0 = max(x0, kx0), max(y0, ky0)
            ix1, iy1 = min(x1, kx1), min(y1, ky1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if area > 0 and inter / area > overlap_thresh:
                nested = True
                break
        if not nested:
            kept.append(r)
    return kept


def crop_region(image: Image.Image, region: LayoutRegion, pad: int = 4) -> Image.Image:
    x0, y0, x1, y1 = region.bbox
    w, h = image.size
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    return image.crop((x0, y0, x1, y1))
