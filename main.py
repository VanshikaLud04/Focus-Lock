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


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_live_session(cfg: dict) -> None:
    from focuslock.capture.camera import CameraCapture
    from focuslock.detection.yolo_detector import YOLODetector
    from focuslock.detection.gaze import GazeEstimator
    from focuslock.detection.sampler import AdaptiveSampler
    from focuslock.fsm.focus_fsm import FocusFSM
    from focuslock.macos.accessibility import AccessibilityMonitor
    from focuslock.data.database import SessionDB
    from focuslock.hud.overlay import HUDOverlay
    import cv2

    print("Starting session -- press Q to quit, F to flag a false positive.")

    cam       = CameraCapture(cfg["camera"])
    detector  = YOLODetector(cfg["model"])
    gaze      = GazeEstimator(cfg["gaze"])
    sampler   = AdaptiveSampler(cfg["adaptive_sampler"])
    fsm       = FocusFSM(cfg["fsm"])
    a11y      = AccessibilityMonitor(cfg["accessibility"])
    db        = SessionDB(cfg["database"])
    hud       = HUDOverlay(cfg)

    session_id = db.start_session()
    prev_frame = None

    try:
        for frame in cam.stream():
            should_infer, interval_ms = sampler.tick(frame, prev_frame)
            prev_frame = frame.copy()

            if not should_infer:
                annotated = hud.draw(frame, fsm.state, fsm.session_stats())
                cv2.imshow("Focus Lock", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            detections   = detector.detect(frame)
            gaze_result  = gaze.estimate(frame)
            app_context  = a11y.get_context()

            focused = gaze.is_focused(gaze_result, app_context)
            state   = fsm.update(focused)

            db.log_event(
                session_id  = session_id,
                state       = state,
                confidence  = gaze_result.confidence,
                yaw_deg     = gaze_result.yaw,
                pitch_deg   = gaze_result.pitch,
                app_bundle  = app_context.bundle_id,
                interval_ms = interval_ms,
            )

            annotated = hud.draw(frame, state, fsm.session_stats(), gaze_result, detections)
            cv2.imshow("Focus Lock", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                db.flag_last_event_as_fp(session_id)
                print("Flagged last event as false positive.")

    finally:
        db.end_session(session_id, fsm.session_stats())
        cam.release()
        cv2.destroyAllWindows()
        print("Session saved.")


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
