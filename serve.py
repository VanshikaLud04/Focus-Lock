"""
serve.py — Focus Lock Web Server
==================================
Starts the detection loop + Flask server, then opens the
dashboard automatically in your browser.

    source venv/bin/activate
    python serve.py

Dashboard → http://localhost:5050
"""

import threading
import time
import webbrowser
from pathlib import Path
import queue
import schedule

import cv2
import yaml
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO
from focuslock.alerts.lock_screen import LockScreenOverlay

from focuslock.analytics.events import EventBus
from focuslock.analytics.resources import ResourceMonitor
from focuslock.data.sqlite_writer import SQLiteWriter
from focuslock.analytics.analytics_engine import AnalyticsEngine
from focuslock.data.database import SessionDB

event_bus = EventBus()
resource_monitor = ResourceMonitor()
# Note: sqlite_writer and analytics_engine are initialized in __main__

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

# ── Flask app & SocketIO ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
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
    data = request.get_json()
    with _lock:
        _state["active"] = bool(data.get("active", False))
        if not _state["active"]:
            _state["frame_jpg"] = None
    
    # Emit state change via WebSocket
    socketio.emit("state_update", {k: v for k, v in _state.items() if k != "frame_jpg"})
    return jsonify({"ok": True})

@app.route("/api/analytics")
def api_analytics():
    session_id = request.args.get("session_id", type=int)
    # If no session_id is provided, you might want to default to the latest or something,
    # but for timeline we need a specific session or we just get today's stats.
    # For simplicity, if not provided we just return streaks, heatmap and trends.
    timeline = []
    if session_id:
        timeline = analytics_engine.build_timeline(session_id)
        
    heatmap = analytics_engine.build_heatmap(days=30)
    streak = analytics_engine.get_longest_streak(session_id)
    trends = analytics_engine.get_trends()
    
    return jsonify({
        "timeline": timeline,
        "heatmap": heatmap,
        "streak": streak,
        "trends": trends
    })

@app.route("/api/resources")
def api_resources():
    return jsonify(resource_monitor.get_metrics())


# ── Background Task: Daily Report ─────────────────────────────────────────────
def _run_scheduler(cfg: dict):
    from focuslock.data.report import generate_daily_csv
    
    # Schedule the report to run daily at midnight
    schedule.every().day.at("00:00").do(generate_daily_csv, cfg["database"])
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# ── Detection & Camera Threads ────────────────────────────────────────────────

# ── Benchmark accumulators (populated when --benchmark or --no-adaptive is set)
_BENCH = {
    "enabled":      False,
    "no_adaptive":  False,
    "latencies_ms": [],
    "infer_count":  0,
    "skip_count":   0,
}


def _print_bench_summary() -> None:
    """Print a full benchmark summary on exit."""
    lats = _BENCH["latencies_ms"]
    if not lats:
        print("[Benchmark] No inference frames recorded.")
        return
    lats_sorted = sorted(lats)
    avg = sum(lats) / len(lats)
    p50 = lats_sorted[int(len(lats) * 0.50)]
    p95 = lats_sorted[int(len(lats) * 0.95)]
    p99 = lats_sorted[int(len(lats) * 0.99)]
    total    = _BENCH["infer_count"] + _BENCH["skip_count"]
    skip_pct = (_BENCH["skip_count"] / total * 100) if total else 0
    print("\n" + "=" * 52)
    print("  FOCUS LOCK BENCHMARK REPORT")
    print("=" * 52)
    print(f"  Adaptive sampling : {'DISABLED (baseline)' if _BENCH['no_adaptive'] else 'ENABLED'}")
    print(f"  Total frames seen : {total}")
    print(f"  Inference runs    : {_BENCH['infer_count']}")
    print(f"  Frames skipped    : {_BENCH['skip_count']}  ({skip_pct:.1f}% saved by sampler)")
    print(f"  Avg latency       : {avg:.1f} ms")
    print(f"  P50 latency       : {p50:.1f} ms")
    print(f"  P95 latency       : {p95:.1f} ms")
    print(f"  P99 latency       : {p99:.1f} ms")
    print("=" * 52 + "\n")


def _run_worker(cfg: dict, frame_queue: queue.Queue):
    """Worker thread that pulls frames and runs heavy ML inference."""
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
    fsm      = FocusFSM(cfg["fsm"])
    a11y     = AccessibilityMonitor(cfg["accessibility"])
    db       = SessionDB(cfg["database"])
    hud      = HUDOverlay(cfg)

    from focuslock.analytics.attention import AttentionEngine
    attention_engine = AttentionEngine(event_bus, resource_monitor)

    if _BENCH["no_adaptive"]:
        sampler = None
        print("[Benchmark] Adaptive sampling DISABLED — inferring every frame (baseline).")
    else:
        sampler = AdaptiveSampler(cfg["adaptive_sampler"])

    print("[Focus Lock] Worker thread ready.")

    session_id = None
    prev_frame = None

    try:
        while True:
            # Block until we get a frame to process
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                with _lock:
                    if session_id and not _state.get("active", False):
                        db.end_session(session_id, fsm.session_stats())
                        session_id = None
                continue

            with _lock:
                active = _state.get("active", False)

            if not active:
                if session_id:
                    db.end_session(session_id, fsm.session_stats())
                    session_id = None
                continue

            if session_id is None:
                session_id = db.start_session()
                prev_frame = None

            if sampler is not None:
                should_infer, interval_ms = sampler.tick(frame, prev_frame)
            else:
                should_infer, interval_ms = True, 0.0

            prev_frame = frame.copy()

            if not should_infer:
                if _BENCH["enabled"]:
                    _BENCH["skip_count"] += 1
                annotated = hud.draw(frame, fsm.state, fsm.session_stats())
                _push(annotated)
                continue

            # ── Run all detectors (timed for --benchmark) ─────────
            _t0 = time.perf_counter()
            detections  = detector.detect(frame)
            gaze_result = gaze.estimate(frame)
            app_context = a11y.get_context()
            _t1 = time.perf_counter()

            if _BENCH["enabled"]:
                elapsed_ms = (_t1 - _t0) * 1000
                _BENCH["latencies_ms"].append(elapsed_ms)
                _BENCH["infer_count"] += 1
                if _BENCH["infer_count"] % 20 == 0:
                    avg = sum(_BENCH["latencies_ms"]) / len(_BENCH["latencies_ms"])
                    print(f"[Benchmark] frames={_BENCH['infer_count']} | avg={avg:.1f}ms | skipped={_BENCH['skip_count']}")

            focused = gaze.is_focused(gaze_result, app_context, detections)
            reason = _distraction_reason(focused, gaze_result, detections)
            
            # Notify resource monitor that a frame was processed for FPS tracking
            resource_monitor.tick_frame()

            state_label = fsm.update(focused)
            stats       = fsm.session_stats()

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
                
                attention_engine.process_frame(
                    session_id=session_id,
                    state=state_label,
                    confidence=gaze_result.confidence,
                    trigger=reason,
                    interval_ms=interval_ms,
                    yaw=gaze_result.yaw,
                    pitch=gaze_result.pitch,
                    active_app=app_context.bundle_id,
                    face_visible=gaze_result.face_found,
                    phone_detected=detections.has_phone_in_hand
                )

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

            # Emit state change via WebSocket
            socketio.emit("state_update", {k: v for k, v in _state.items() if k != "frame_jpg"})

    finally:
        if session_id:
            db.end_session(session_id, fsm.session_stats())
        print("[Focus Lock] Worker thread stopped.")


def _run_camera(cfg: dict, frame_queue: queue.Queue) -> None:
    """Camera thread that captures frames and pushes them to the worker."""
    from focuslock.capture.camera import CameraCapture
    
    print("[Focus Lock] Camera thread ready.")
    cam = None

    try:
        while True:
            with _lock:
                active = _state.get("active", False)

            if not active:
                if cam is not None:
                    cam.release()
                    cam = None
                time.sleep(0.2)
                continue

            if cam is None:
                cam = CameraCapture(cfg["camera"])

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Push to worker queue (drop if full)
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    finally:
        if cam:
            cam.release()
        print("[Focus Lock] Camera thread stopped.")


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
    import atexit
    parser = argparse.ArgumentParser(description="Focus Lock web server")
    parser.add_argument("--port",        type=int, default=5050)
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--no-browser",  action="store_true",
                        help="Don't open browser automatically")
    parser.add_argument("--benchmark",   action="store_true",
                        help="Print per-frame latency stats to the console")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive sampling (use for baseline CPU measurement)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _BENCH["enabled"]     = args.benchmark or args.no_adaptive
    _BENCH["no_adaptive"] = args.no_adaptive

    if _BENCH["enabled"]:
        atexit.register(_print_bench_summary)

    if args.no_adaptive:
        print("[Benchmark] STEP 1 — Adaptive sampling OFF. Open Activity Monitor, filter 'python3'.")
        print("[Benchmark]          Sit in frame for 30s, record CPU %. Then Ctrl+C.")
    elif args.benchmark:
        print("[Benchmark] STEP 2 — Adaptive sampling ON. Record latency + CPU with sampler enabled.")
        print("[Benchmark]          Step OUT of frame to watch CPU drop as sampler backs off.")

    frame_queue = queue.Queue(maxsize=1)

    print("[Focus Lock] Starting threads…")
    
    t_cam = threading.Thread(target=_run_camera, args=(cfg, frame_queue), daemon=True)
    t_cam.start()
    
    t_worker = threading.Thread(target=_run_worker, args=(cfg, frame_queue), daemon=True)
    t_worker.start()
    
    t_sched = threading.Thread(target=_run_scheduler, args=(cfg,), daemon=True)
    t_sched.start()
    
    # Initialize globals used by Flask routes
    global sqlite_writer, analytics_engine
    global_db = SessionDB(cfg["database"])
    sqlite_writer = SQLiteWriter(global_db)
    sqlite_writer.start()
    
    event_bus.subscribe("attention_events", sqlite_writer.handle_attention_event)
    analytics_engine = AnalyticsEngine(global_db)
    
    resource_monitor.start()

    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
        print(f"[Focus Lock] Browser will open at {url}")

    socketio.run(app, host="0.0.0.0", port=args.port, use_reloader=False)
