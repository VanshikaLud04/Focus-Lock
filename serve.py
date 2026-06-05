"""
serve.py — Focus Lock Web Server
==================================
Starts the detection loop + Flask server, then opens the
dashboard automatically in your browser.

    source venv/bin/activate
    python serve.py

Dashboard → http://localhost:5050

How it works
------------
  1. Detection thread: camera → YOLO + gaze → FSM → writes to _state
  2. Flask thread: serves dashboard.html + MJPEG stream + /state JSON
  3. Browser: polls /state every 2 s, shows live camera via /video_feed
"""

import threading
import time
import webbrowser
from pathlib import Path

import cv2
import yaml
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS
from focuslock.alerts.lock_screen import LockScreenOverlay

# ── Shared state ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "fsm_state":             "IDLE",
    "focus_pct":             0.0,
    "session_secs":          0,
    "yaw":                   0.0,
    "pitch":                 0.0,
    "has_phone":             False,
    "is_talking":            False,
    "is_eating":             False,
    "is_drinking":           False,
    "high_motion":           False,
    "clearly_away":          False,
    "distraction_reason":    "",    # phone | talking | eating | motion | away | gaze
    "frame_jpg":             None,  # latest annotated JPEG bytes
    "active":                False, # whether detection is active
}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
PROJECT = Path(__file__).parent


@app.route("/")
def index():
    return send_from_directory(str(PROJECT), "dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _gen_frames():
    """MJPEG generator — yields latest annotated frame at ~30 fps."""
    while True:
        with _lock:
            jpg = _state["frame_jpg"]
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(1 / 30)


@app.route("/state")
def get_state():
    with _lock:
        return jsonify({k: v for k, v in _state.items() if k != "frame_jpg"})


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


@app.route("/set_active", methods=["POST"])
def set_active():
    from flask import request
    data = request.get_json()
    with _lock:
        _state["active"] = bool(data.get("active", False))
        if not _state["active"]:
            _state["frame_jpg"] = None
    return jsonify({"ok": True})


# ── Detection thread ──────────────────────────────────────────────────────────
def _run_detection(cfg: dict) -> None:
    """
    Runs in a background daemon thread.

    Frame-skipping loop
    -------------------
    The camera captures at full FPS (for live streaming), but heavy
    ML inference (YOLO + MediaPipe) only runs every
    cfg['adaptive_sampler']['min_interval_sec'] seconds.
    This keeps CPU usage low while the video feed stays smooth.
    """
    from focuslock.capture.camera          import CameraCapture
    from focuslock.detection.yolo_detector import YOLODetector
    from focuslock.detection.gaze          import GazeEstimator
    from focuslock.detection.sampler       import AdaptiveSampler
    from focuslock.fsm.focus_fsm           import FocusFSM
    from focuslock.macos.accessibility     import AccessibilityMonitor
    from focuslock.data.database           import SessionDB
    from focuslock.hud.overlay             import HUDOverlay

    lock_overlay = LockScreenOverlay()

    detector = YOLODetector(cfg["model"])
    gaze     = GazeEstimator(cfg["gaze"])
    sampler  = AdaptiveSampler(cfg["adaptive_sampler"])
    fsm      = FocusFSM(cfg["fsm"])
    a11y     = AccessibilityMonitor(cfg["accessibility"])
    db       = SessionDB(cfg["database"])
    hud      = HUDOverlay(cfg)

    print("[Focus Lock] Detection thread ready. Waiting for frontend to start timer.")

    cam = None
    session_id = None
    prev_frame = None

    try:
        while True:
            with _lock:
                active = _state.get("active", False)

            if not active:
                if cam is not None:
                    if session_id:
                        db.end_session(session_id, fsm.session_stats())
                        session_id = None
                    cam.release()
                    cam = None
                time.sleep(0.2)
                continue

            if cam is None:
                cam = CameraCapture(cfg["camera"])
                session_id = db.start_session()
                prev_frame = None

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            should_infer, interval_ms = sampler.tick(frame, prev_frame)
            prev_frame = frame.copy()

            if not should_infer:
                # Still stream the frame — just skip heavy inference
                annotated = hud.draw(frame, fsm.state, fsm.session_stats())
                _push(annotated)
                continue

            # ── Run all detectors ─────────────────────────────────
            detections  = detector.detect(frame)
            gaze_result = gaze.estimate(frame)
            app_context = a11y.get_context()

            # ── Focus decision (priority-ordered in gaze.py) ──────
            focused = gaze.is_focused(gaze_result, app_context, detections)

            # ── Distraction reason for dashboard ──────────────────
            reason = _distraction_reason(focused, gaze_result, detections)

            # ── FSM state ─────────────────────────────────────────
            state_label = fsm.update(focused)
            stats       = fsm.session_stats()

            # ── Lock screen: phone in hand → full-screen block ────
            # Uses IoU geometry: phone on desk is ignored;
            # phone overlapping person bounding box triggers lock.
            if detections.has_phone_in_hand and state_label != "BREAK":
                lock_overlay.show("📱 Phone in hand detected")
            else:
                lock_overlay.hide()

            if session_id is not None:
                db.log_event(
                    session_id  = session_id,
                    state       = state_label,
                    confidence  = gaze_result.confidence,
                    yaw_deg     = gaze_result.yaw,
                    pitch_deg   = gaze_result.pitch,
                    app_bundle  = app_context.bundle_id,
                    interval_ms = interval_ms,
                )

            # ── Annotate + stream ─────────────────────────────────
            annotated = hud.draw(frame, state_label, stats, gaze_result, detections)
            _push(annotated)

            # ── Update shared state for dashboard ─────────────────
            with _lock:
                _state["fsm_state"]          = state_label
                _state["focus_pct"]          = round(stats.focus_pct, 1)
                _state["session_secs"]       = int(stats.total_sec)
                _state["yaw"]                = round(gaze_result.yaw, 1)
                _state["pitch"]              = round(gaze_result.pitch, 1)
                _state["has_phone"]          = detections.has_phone
                _state["is_talking"]         = detections.is_talking
                _state["is_eating"]          = detections.is_eating
                _state["is_drinking"]        = detections.is_drinking_only
                _state["high_motion"]        = gaze_result.high_motion
                _state["clearly_away"]       = gaze_result.clearly_away
                _state["distraction_reason"] = reason

    finally:
        if session_id and cam:
            db.end_session(session_id, fsm.session_stats())
        if cam:
            cam.release()
        print("[Focus Lock] Thread stopped.")


def _distraction_reason(focused: bool, gaze_result, detections) -> str:
    """Map detection flags to a short reason string for the dashboard."""
    if focused:
        return ""
    if gaze_result.high_motion:
        return "motion"
    if detections.has_phone_in_hand:   # IoU-confirmed, not just phone visible
        return "phone"
    if detections.is_talking:
        return "talking"
    if detections.is_eating:
        return "eating"
    if gaze_result.clearly_away:
        return "away"
    if not gaze_result.face_found:
        return "away"
    return "gaze"


def _push(frame) -> None:
    """Encode frame to JPEG and write to shared state."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    with _lock:
        _state["frame_jpg"] = buf.tobytes()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Focus Lock web server")
    parser.add_argument("--port",       type=int, default=5050)
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser automatically")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("[Focus Lock] Starting detection thread…")
    t = threading.Thread(target=_run_detection, args=(cfg,), daemon=True)
    t.start()

    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
        print(f"[Focus Lock] Browser will open at {url}")

    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
