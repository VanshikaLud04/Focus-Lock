"""
geometry.py — Bounding-box math engine
=======================================
IoU, overlap-ratio, and phone-in-hand heuristics used by
the detection pipeline to reduce false positives.

Key insight
-----------
A phone sitting on the desk is NOT a distraction.
A phone that overlaps the person bounding box IS likely in hand.

We use 'contained_ratio' (fraction of phone box inside person box)
rather than classic IoU, because the person box is ~20× larger than
the phone box and classic IoU would always be near zero even when
the phone is fully inside the person.
"""

from __future__ import annotations


def iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Intersection-over-Union for two [x1, y1, x2, y2] boxes.

    Returns a float in [0, 1].  1 = perfect overlap, 0 = no overlap.
    """
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter   = inter_w * inter_h

    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def contained_ratio(small_box: list[float], large_box: list[float]) -> float:
    """
    Fraction of 'small_box' that is inside 'large_box'.

    This is the right metric for "is the phone inside the person region?"
    because person boxes are typically 10-30× larger than phone boxes,
    which makes classic IoU useless.

    Returns a float in [0, 1].
    """
    xi1 = max(small_box[0], large_box[0])
    yi1 = max(small_box[1], large_box[1])
    xi2 = min(small_box[2], large_box[2])
    yi2 = min(small_box[3], large_box[3])

    inter_w    = max(0.0, xi2 - xi1)
    inter_h    = max(0.0, yi2 - yi1)
    inter      = inter_w * inter_h
    small_area = max(1.0, (small_box[2] - small_box[0]) * (small_box[3] - small_box[1]))

    return inter / small_area


def is_phone_held(detections, overlap_thresh: float = 0.25) -> bool:
    """
    Return True if a detected phone overlaps a person bounding box
    by at least *overlap_thresh* of the phone's own area.

    Parameters
    ----------
    detections : DetectionResult
    overlap_thresh : float
        Fraction of phone box that must sit inside a person box.
        0.25 = "at least 25% of the phone is within the person region".
        Lower = more sensitive (catches held phones easier).
        Higher = stricter (only flags when phone is clearly in hand).

    Why 0.25?
    ---------
    When you hold a phone, the phone box typically overlaps 30–80 %
    with your torso/arm region.  0.25 is a safe lower bound that
    catches holding without triggering on a phone placed on a desk
    next to you.
    """
    phones  = [d for d in detections.detections if d.is_phone]
    persons = [d for d in detections.detections if d.is_person]

    if not phones or not persons:
        return False

    for phone in phones:
        for person in persons:
            ratio = contained_ratio(phone.bbox_xyxy, person.bbox_xyxy)
            if ratio >= overlap_thresh:
                return True

    return False


def phone_iou_scores(detections) -> list[tuple[float, float]]:
    """
    Debug helper — returns (iou, contained_ratio) for every
    phone × person pair.  Useful for tuning thresholds.
    """
    phones  = [d for d in detections.detections if d.is_phone]
    persons = [d for d in detections.detections if d.is_person]
    scores  = []
    for ph in phones:
        for pe in persons:
            scores.append((
                iou(ph.bbox_xyxy, pe.bbox_xyxy),
                contained_ratio(ph.bbox_xyxy, pe.bbox_xyxy),
            ))
    return scores
