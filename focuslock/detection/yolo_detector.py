"""
yolo_detector.py -- YOLOv8 inference wrapper

Wraps ultralytics YOLO so the rest of the codebase doesn't
touch the ultralytics API directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    """Single bounding-box detection from YOLO."""
    class_name: str
    confidence: float
    bbox_xyxy: list[float]   # [x1, y1, x2, y2] in pixels

    @property
    def is_phone(self) -> bool:
        return self.class_name.lower() in {"cell phone", "phone", "mobile phone"}

    @property
    def is_person(self) -> bool:
        return self.class_name.lower() == "person"


# YOLO COCO class sets for distraction detection
_EATING_CLASSES   = {
    'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
}
_DRINKING_CLASSES = {'bottle', 'wine glass', 'cup'}


@dataclass
class DetectionResult:
    """Collection of detections for a single frame."""
    detections:   list[Detection] = field(default_factory=list)
    inference_ms: float           = 0.0

    @property
    def has_phone(self) -> bool:
        """Any phone visible in frame (includes phone on desk)."""
        return any(d.is_phone for d in self.detections)

    @property
    def has_phone_in_hand(self) -> bool:
        """
        Phone overlaps the person bounding box — likely being held/used.

        Uses contained_ratio from geometry.py (fraction of phone box
        that sits inside the person box).  A phone on the desk next to
        you will NOT trigger this; a phone you're holding will.

        This is the correct signal to use for distraction detection.
        `has_phone` is kept for backwards-compatibility and debug use.
        """
        from focuslock.detection.geometry import is_phone_held
        return is_phone_held(self, overlap_thresh=0.25)

    @property
    def person_count(self) -> int:
        return sum(1 for d in self.detections if d.is_person)

    @property
    def is_talking(self) -> bool:
        """Two or more people in frame -- probably talking."""
        return self.person_count >= 2

    @property
    def is_eating(self) -> bool:
        """Eating utensil or food visible."""
        names = {d.class_name.lower() for d in self.detections}
        return bool(names & _EATING_CLASSES)

    @property
    def is_drinking_only(self) -> bool:
        """Only a drink container visible, no food. Doesn't trigger distraction alert."""
        names = {d.class_name.lower() for d in self.detections}
        has_drink = bool(names & _DRINKING_CLASSES)
        has_food  = bool(names & _EATING_CLASSES)
        return has_drink and not has_food

    @property
    def distraction_objects(self) -> list[str]:
        """Human-readable list of distraction triggers in this frame."""
        out = []
        if self.has_phone:        out.append('phone')
        if self.is_talking:       out.append('talking')
        if self.is_eating:        out.append('eating')
        if self.is_drinking_only: out.append('drinking')
        return out


class YOLODetector:
    """
    Thin wrapper around ultralytics YOLOv8.

    Usage
    -----
    detector = YOLODetector(cfg["model"])
    result   = detector.detect(frame)
    if result.has_phone:
        ...
    """

    def __init__(self, cfg: dict) -> None:
        from ultralytics import YOLO
        self._conf   = cfg.get("confidence", 0.45)
        self._device = cfg.get("device", "")
        model_path   = cfg.get("path", "yolov8n.pt")
        print(f"Loading {model_path} | conf={self._conf}")
        try:
            self._model  = YOLO(model_path)
            print("Model ready.")
        except Exception as e:
            print(f"[YOLODetector] Failed to load model: {e}")
            self._model = None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.

        Returns
        -------
        DetectionResult
            All detections + inference latency.
        """
        if self._model is None:
            return DetectionResult()

        import time
        t0      = time.perf_counter()
        results = self._model.predict(
            frame, conf=self._conf, device=self._device, verbose=False
        )
        ms = (time.perf_counter() - t0) * 1000

        detections = []
        for box in results[0].boxes:
            detections.append(Detection(
                class_name = results[0].names[int(box.cls)],
                confidence = float(box.conf),
                bbox_xyxy  = box.xyxy[0].tolist(),
            ))
        return DetectionResult(detections=detections, inference_ms=ms)

    def warmup(self) -> None:
        """Run one dummy inference to pre-load weights."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        print("Warmup complete.")
