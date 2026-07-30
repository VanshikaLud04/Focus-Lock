"""
Focus Lock
==========
Run with:
    python main.py              # start live session
    python main.py --menubar    # start as macOS menu bar app
    python main.py report       # print today's session report
    python main.py report --week
"""

import argparse
import yaml
from AdaptiveFrameSampler import AdaptiveFrameSampler

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_live_session(cfg: dict) -> None:
    from focuslock.capture.camera import CameraCapture
    from focuslock.detection.yolo_detector import YOLODetector
    from focuslock.detection.gaze import GazeEstimator
    from focuslock.fsm.focus_fsm import FocusFSM
    from focuslock.macos.accessibility import AccessibilityMonitor
    from focuslock.data.database import SessionDB
    from focuslock.hud.overlay import HUDOverlay
    from focuslock.analytics.events import EventBus
    from focuslock.analytics.resources import ResourceMonitor
    from focuslock.data.sqlite_writer import SQLiteWriter
    from focuslock.analytics.attention import AttentionEngine
    import cv2
    import time
    import threading
    import queue

    print("Starting session -- press Q to quit, F to flag a false positive.")

    cam       = CameraCapture(cfg["camera"])
    db        = SessionDB(cfg["database"])
    session_id = db.start_session()
    
    frame_queue = queue.Queue(maxsize=1)
    
    # Shared state for main thread display
    state_lock = threading.Lock()
    display_state = {
        "annotated_frame": None,
        "running": True,
        "flag_fp": False
    }

    def worker_thread():
        detector  = YOLODetector(cfg["model"])
        gaze      = GazeEstimator(cfg["gaze"])
        fsm       = FocusFSM(cfg["fsm"])
        a11y      = AccessibilityMonitor(cfg["accessibility"])
        hud       = HUDOverlay(cfg)
        
        event_bus = EventBus()
        resource_monitor = ResourceMonitor()
        resource_monitor.start()
        sqlite_writer = SQLiteWriter(db)
        sqlite_writer.start()
        event_bus.subscribe("attention_events", sqlite_writer.handle_attention_event)
        
        attention_engine = AttentionEngine(event_bus, resource_monitor)
        
        # Use AdaptiveSampler from sampler.py like in serve.py, or AdaptiveFrameSampler
        try:
            from focuslock.detection.sampler import AdaptiveSampler
            sampler = AdaptiveSampler(cfg["adaptive_sampler"])
            use_adaptive_sampler = True
        except ImportError:
            sampler = AdaptiveFrameSampler()
            use_adaptive_sampler = False
            
        last_confidence = 1.0
        prev_frame = None

        while display_state["running"]:
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with state_lock:
                if display_state["flag_fp"]:
                    db.flag_last_event_as_fp(session_id)
                    display_state["flag_fp"] = False
                    print("Flagged last event as false positive.")

            if use_adaptive_sampler:
                should_infer, interval_ms = sampler.tick(frame, prev_frame)
            else:
                current_time = time.monotonic()
                movement_score = 0.0
                if prev_frame is not None:
                    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray1, gray2)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    total_pixels = frame.shape[0] * frame.shape[1]
                    movement_score = cv2.countNonZero(thresh) / total_pixels
                sampler_result = sampler.update(movement_score, last_confidence, current_time)
                should_infer = sampler_result["trigger_yolo"]
                interval_ms = max(1, int(1000 / sampler_result["fps"]))

            prev_frame = frame.copy()

            if not should_infer:
                annotated = hud.draw(frame, fsm.state, fsm.session_stats())
                with state_lock:
                    display_state["annotated_frame"] = annotated
                continue

            detections   = detector.detect(frame)
            gaze_result  = gaze.estimate(frame)
            app_context  = a11y.get_context()

            if gaze_result:
                last_confidence = gaze_result.confidence
                focused = gaze.is_focused(gaze_result, app_context, detections)
            else:
                last_confidence = 1.0
                # If gaze.py in main expects just gaze_result and app_context:
                try:
                    focused = gaze.is_focused(gaze_result, app_context)
                except TypeError:
                    focused = gaze.is_focused(gaze_result, app_context, detections)

            reason = ""
            if not focused:
                if gaze_result and gaze_result.high_motion:
                    reason = "motion"
                elif detections.has_phone_in_hand:
                    reason = "phone"
                elif detections.is_talking:
                    reason = "talking"
                elif detections.is_eating:
                    reason = "eating"
                elif gaze_result and gaze_result.clearly_away:
                    reason = "away"
                elif gaze_result and not gaze_result.face_found:
                    reason = "away"
                else:
                    reason = "gaze"
                    
            resource_monitor.tick_frame()

            state   = fsm.update(focused)

            db.log_event(
                session_id  = session_id,
                state       = state,
                confidence  = last_confidence,
                yaw_deg     = gaze_result.yaw if gaze_result else 0.0,
                pitch_deg   = gaze_result.pitch if gaze_result else 0.0,
                app_bundle  = app_context.bundle_id,
                interval_ms = interval_ms,
            )
            
            attention_engine.process_frame(
                session_id=session_id,
                state=state,
                confidence=last_confidence,
                trigger=reason,
                interval_ms=interval_ms,
                yaw=gaze_result.yaw if gaze_result else 0.0,
                pitch=gaze_result.pitch if gaze_result else 0.0,
                active_app=app_context.bundle_id,
                face_visible=gaze_result.face_found if gaze_result else False,
                phone_detected=detections.has_phone_in_hand
            )

            annotated = hud.draw(frame, state, fsm.session_stats(), gaze_result, detections)
            with state_lock:
                display_state["annotated_frame"] = annotated

        db.end_session(session_id, fsm.session_stats())
        resource_monitor.stop()
        sqlite_writer.stop()
        print("Session saved.")

    t = threading.Thread(target=worker_thread, daemon=True)
    t.start()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
                
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
                
            with state_lock:
                annotated = display_state["annotated_frame"]
                
            if annotated is not None:
                cv2.imshow("Focus Lock", annotated)
                
            key = cv2.waitKey(33) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                with state_lock:
                    display_state["flag_fp"] = True

    finally:
        display_state["running"] = False
        cam.release()
        cv2.destroyAllWindows()


def run_menubar(cfg: dict) -> None:
    from focuslock.macos.menubar import FocusLockMenuBarApp
    FocusLockMenuBarApp(cfg).run()


def run_report(args: argparse.Namespace, cfg: dict) -> None:
    from focuslock.data.report import print_report
    period = "week" if args.week else "today"
    print_report(cfg["database"], period=period)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="focuslock",
        description="Focus Lock -- real-time productivity tracker"
    )
    subparsers = parser.add_subparsers(dest="command")

    rp = subparsers.add_parser("report", help="Print session summary")
    rp.add_argument("--week", action="store_true", help="Show this week instead of today")

    parser.add_argument("--menubar", action="store_true", help="Run as macOS menu bar app")
    parser.add_argument("--config",  default="config.yaml",  help="Path to config file")

    args = parser.parse_args()
    cfg  = load_config(args.config)

    if args.command == "report":
        run_report(args, cfg)
    elif args.menubar:
        run_menubar(cfg)
    else:
        run_live_session(cfg)


if __name__ == "__main__":
    main()
