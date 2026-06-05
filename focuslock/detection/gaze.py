"""
gaze.py -- head-pose estimation and focus decision

Uses MediaPipe Face Mesh to extract 3D facial landmarks,
computes yaw and pitch, then decides whether the user is focused.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np


@dataclass
class GazeResult:
    """Output of a single gaze estimation pass."""
    yaw:          float = 0.0      # degrees, positive = turned right
    pitch:        float = 0.0      # degrees, positive = looking up
    roll:         float = 0.0      # degrees
    confidence:   float = 0.0      # landmark detection confidence
    face_found:   bool  = False    # True if a face was detected
    high_motion:  bool  = False    # True if scene motion was too high for reliable gaze
    clearly_away: bool  = False    # True if head is far out of range or no face found


class GazeEstimator:
    """
    Estimates head-pose from a live BGR frame using MediaPipe.

    Usage
    -----
    gaze    = GazeEstimator(cfg["gaze"])
    result  = gaze.estimate(frame)
    focused = gaze.is_focused(result, app_context)
    """

    # 6 landmarks used for solvePnP
    _LM_INDICES = [1, 152, 33, 263, 61, 291]

    # Seconds a face can be absent before marking DISTRACTED
    _ABSENCE_GRACE: float = 3.0

    # Corresponding 3D model points (generic face, mm)
    _MODEL_POINTS = np.array([
        (0.0,    0.0,    0.0),      # nose tip       -- lm 1
        (0.0,   -63.6, -12.5),      # chin           -- lm 152
        (-43.3,  32.7, -26.0),      # left eye       -- lm 33
        (43.3,   32.7, -26.0),      # right eye      -- lm 263
        (-28.9, -28.9, -24.1),      # left mouth     -- lm 61
        (28.9,  -28.9, -24.1),      # right mouth    -- lm 291
    ], dtype=np.float64)

    def __init__(self, cfg: dict) -> None:
        self._yaw_thresh   = cfg.get("yaw_threshold_deg",   20)
        self._pitch_thresh = cfg.get("pitch_threshold_deg", 15)
        self._absent_since: Optional[float] = None   # monotonic timestamp of first face-absent frame

        self._mp_face = None
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode        = False,
                max_num_faces            = 1,
                refine_landmarks         = True,
                min_detection_confidence = 0.5,
                min_tracking_confidence  = 0.5,
            )
            print("[GazeEstimator] MediaPipe Face Mesh ready.")
        except (ImportError, AttributeError):
            # mediapipe not installed, OR mediapipe >= 0.10.14 which removed
            # mp.solutions on Apple Silicon.  Pin: pip install mediapipe==0.10.9
            print("[GazeEstimator] Warning: mediapipe.solutions not available.")
            print("  Fix: pip install 'mediapipe==0.10.9'")
            print("  Falling back to YOLO-only focus detection.")

    def estimate(self, frame: np.ndarray) -> GazeResult:
        """
        Run head-pose estimation on a BGR frame.

        1. Convert BGR to RGB and run Face Mesh.
        2. Extract 6 key landmarks in image coordinates.
        3. Use cv2.solvePnP to recover rotation vector.
        4. Convert rotation vector to Euler angles (yaw, pitch, roll).
        """
        if self._mp_face is None:
            return GazeResult(face_found=False, clearly_away=True)

        import cv2

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = self._mp_face.process(rgb)

        if not mesh.multi_face_landmarks:
            return GazeResult(face_found=False, clearly_away=True)

        lms = mesh.multi_face_landmarks[0].landmark

        img_points = np.array([
            (lms[i].x * w, lms[i].y * h) for i in self._LM_INDICES
        ], dtype=np.float64)

        focal   = w
        cam_mat = np.array([
            [focal, 0,      w / 2],
            [0,     focal,  h / 2],
            [0,     0,      1   ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS, img_points,
            cam_mat, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return GazeResult(face_found=True, confidence=0.0, clearly_away=True)

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles

        conf = float(np.mean([lms[i].visibility for i in self._LM_INDICES]))

        # clearly_away = head is very far beyond normal thresholds
        far_out = abs(float(yaw)) > self._yaw_thresh * 2.5 or abs(float(pitch)) > self._pitch_thresh * 2.5

        return GazeResult(
            yaw          = float(yaw),
            pitch        = float(pitch),
            roll         = float(roll),
            confidence   = conf,
            face_found   = True,
            clearly_away = far_out,
        )

    def is_focused(self, result: GazeResult, app_context=None, detections=None) -> bool:
        """
        Decide if the user is focused.

        Priority order
        --------------
        1. High motion (dancing / leaving desk)          → DISTRACTED
        2. Phone in hand (IoU overlap with person box)   → DISTRACTED
        3. Talking to someone (2+ people in frame)       → DISTRACTED
        4. Eating food items                             → DISTRACTED
        5. Clearly looking away (large angle)            → DISTRACTED
        6. Face absent for > 3 s                        → DISTRACTED
        7. YOLO-only fallback (no MediaPipe)             → check detections only
        8. Gaze within thresholds                        → FOCUSED
        """
        # ── 1. Dancing / large body movement ─────────────────────
        if result.high_motion:
            return False

        # ── 2–4. YOLO-based distractions ─────────────────────────
        if detections:
            # IoU check: phone on desk = OK, phone overlapping body = distracted
            if detections.has_phone_in_hand:
                return False
            if detections.is_talking:
                return False
            if detections.is_eating and not detections.is_drinking_only:
                return False

        # ── YOLO-only fallback (MediaPipe not available) ──────────
        if self._mp_face is None:
            return True   # no distraction objects → assume focused

        # ── 5. Clearly looking in wrong direction ─────────────────
        if result.clearly_away:
            return False

        # ── 6. Sustained face absence ─────────────────────────────
        if not result.face_found:
            if self._absent_since is not None:
                if time.monotonic() - self._absent_since > self._ABSENCE_GRACE:
                    return False
            return True   # brief absence → grace period

        # ── 7. Normal gaze threshold check ───────────────────────
        yaw_thresh   = self._yaw_thresh
        pitch_thresh = self._pitch_thresh

        if app_context and getattr(app_context, "yaw_override", None) is not None:
            yaw_thresh = app_context.yaw_override

        return (abs(result.yaw) <= yaw_thresh and
                abs(result.pitch) <= pitch_thresh)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._mp_face:
            self._mp_face.close()
