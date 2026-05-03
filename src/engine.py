from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.combo_tracker import ComboTracker
from src.config import EngineConfig
from src.detectors.face_detector import FaceDetector, FaceObservation
from src.detectors.hand_detector import HandDetector, HandObservation
from src.effects.renderer import EffectRenderer
from src.state import AppState

try:
    import pyvirtualcam
except ImportError:  # pragma: no cover
    pyvirtualcam = None


class ARFilterEngine:
    def __init__(self, config: EngineConfig, state: AppState) -> None:
        self.config = config
        self.state = state
        self.renderer = EffectRenderer()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._virtual_camera = None
        self._face_detector: FaceDetector | None = None
        self._hand_detector: HandDetector | None = None
        self._last_face_observation: FaceObservation | None = None
        self._last_hand_observations: list[HandObservation] = []
        self._opencv_backend = "CPU"
        self._last_detection_size = f"{self.config.frame_width}x{self.config.frame_height}"
        self._detector_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="detector")
        self._combo_tracker = ComboTracker()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._close_virtual_camera()
        self._detector_executor.shutdown(wait=True, cancel_futures=False)

    def _initialize_detectors(self) -> bool:
        try:
            self._opencv_backend = self._configure_opencv()
            self._face_detector = FaceDetector(
                model_path=self.config.face_model_path, max_faces=self.config.max_faces
            )
            self._hand_detector = HandDetector(
                model_path=self.config.hand_model_path, max_hands=self.config.max_hands
            )
            self.state.set_error(None)
            return True
        except Exception as exc:
            self.state.set_error(str(exc))
            self._face_detector = None
            self._hand_detector = None
            return False

    def _configure_opencv(self) -> str:
        cv2.setUseOptimized(True)
        if self.config.opencv_threads > 0:
            cv2.setNumThreads(self.config.opencv_threads)
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(False)
        return f"CPU ({cv2.getNumThreads()} threads)"

    def _run_loop(self) -> None:
        self.state.set_status("starting")
        detectors_ready = self._initialize_detectors()

        capture = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(self.config.camera_index)

        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            capture.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        fps_window_start = time.perf_counter()
        frame_counter = 0
        smoothed_fps = 0.0
        loop_index = 0
        self.state.set_status("running" if capture.isOpened() and detectors_ready else "degraded")

        while not self._stop_event.is_set():
            if not capture.isOpened():
                frame = self._status_frame("Webcam unavailable", "Check camera permissions and device index.")
                ok, jpeg = cv2.imencode(".jpg", frame)
                if ok:
                    self.state.update_frame(
                        jpeg.tobytes(),
                        self._metrics(False, [], None, 0.0, 0, 0, "locked", "67_seesaw", False),
                    )
                self.state.set_error("Unable to open webcam.")
                time.sleep(0.5)
                continue

            ok, frame = capture.read()
            if not ok:
                self.state.set_error("Failed to read frame from webcam.")
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            tick = time.perf_counter()
            loop_index += 1

            face_observation: FaceObservation | None = None
            hand_observations: list[HandObservation] = []
            if detectors_ready and self._face_detector and self._hand_detector:
                should_detect = (
                    loop_index == 1
                    or self.config.detection_interval <= 1
                    or loop_index % self.config.detection_interval == 0
                )
                if should_detect:
                    timestamp_ms = int(tick * 1000)
                    detection_frame = self._downscale_for_detection(frame)
                    self._last_detection_size = (
                        f"{detection_frame.shape[1]}x{detection_frame.shape[0]}"
                    )
                    detection_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detection_rgb)
                    face_future = self._detector_executor.submit(
                        self._face_detector.detect,
                        mp_image,
                        timestamp_ms,
                    )
                    hand_future = self._detector_executor.submit(
                        self._hand_detector.detect,
                        mp_image,
                        timestamp_ms,
                    )
                    face_observation = face_future.result()
                    hand_observations = hand_future.result()
                    self._last_face_observation = face_observation
                    self._last_hand_observations = hand_observations
                else:
                    face_observation = self._last_face_observation
                    hand_observations = self._last_hand_observations
                self.state.set_error(None)
            else:
                frame = self._status_frame(
                    "Models missing",
                    "Run scripts/download_models.py, then restart the server.",
                    base_frame=frame,
                )

            combo_snapshot = self._combo_tracker.update(hand_observations, tick)
            processed = self.renderer.render(
                frame=frame,
                face=face_observation,
                hands=hand_observations,
                settings=self.state.get_settings(),
                tick=tick,
                combo=combo_snapshot,
            )
            if self.renderer.consume_session_reset():
                self._combo_tracker.reset()
                combo_snapshot = self._combo_tracker.update([], tick)

            frame_counter += 1
            elapsed = tick - fps_window_start
            if elapsed >= 1.0:
                smoothed_fps = frame_counter / elapsed
                fps_window_start = tick
                frame_counter = 0

            self._sync_virtual_camera(processed)

            preview_frame = self._downscale_for_preview(processed)
            encoded, jpeg = cv2.imencode(
                ".jpg",
                preview_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
            )
            if encoded:
                self.state.update_frame(
                    jpeg.tobytes(),
                    self._metrics(
                        face_observation is not None,
                        hand_observations,
                        face_observation,
                        smoothed_fps,
                        combo_snapshot.score,
                        combo_snapshot.streak,
                        combo_snapshot.phase,
                        combo_snapshot.active_gesture,
                        combo_snapshot.activated,
                    ),
                )

        capture.release()

    def _metrics(
        self,
        face_detected: bool,
        hands: list[HandObservation],
        face: FaceObservation | None,
        fps: float,
        score: int,
        streak: int,
        combo_phase: str,
        combo_gesture: str,
        combo_activated: bool,
    ) -> dict[str, Any]:
        expressions = []
        if face is not None:
            expressions = [name for name, active in face.expressions.items() if active]
        gestures = [hand.gesture for hand in hands if hand.gesture != "unknown"]
        return {
            "fps": round(fps, 1),
            "face_detected": face_detected,
            "hand_count": len(hands),
            "expressions": expressions,
            "gestures": gestures,
            "backend": f"{self._opencv_backend}, parallel detectors",
            "detection_size": self._last_detection_size,
            "score": score,
            "streak": streak,
            "combo_phase": combo_phase,
            "combo_gesture": combo_gesture,
            "combo_activated": combo_activated,
        }

    def _downscale_for_detection(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.config.detection_max_width:
            return frame
        scale = self.config.detection_max_width / width
        return cv2.resize(
            frame,
            (self.config.detection_max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )

    def _downscale_for_preview(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.config.preview_max_width:
            return frame
        scale = self.config.preview_max_width / width
        return cv2.resize(
            frame,
            (self.config.preview_max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _sync_virtual_camera(self, frame_bgr: np.ndarray) -> None:
        requested = self.state.virtual_camera_requested()
        if requested and self._virtual_camera is None:
            self._open_virtual_camera(frame_bgr.shape[1], frame_bgr.shape[0])
        if not requested and self._virtual_camera is not None:
            self._close_virtual_camera()
        if self._virtual_camera is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._virtual_camera.send(frame_rgb)
            self._virtual_camera.sleep_until_next_frame()

    def _open_virtual_camera(self, width: int, height: int) -> None:
        if pyvirtualcam is None:
            self.state.set_error("pyvirtualcam is not installed. Install requirements first.")
            self.state.set_virtual_camera_active(False)
            self.state.request_virtual_camera(False)
            return

        try:
            self._virtual_camera = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=self.config.target_fps,
            )
            self.state.set_virtual_camera_active(True)
            self.state.set_error(None)
        except Exception as exc:
            self._virtual_camera = None
            self.state.set_virtual_camera_active(False)
            self.state.set_error(f"Virtual camera failed to start: {exc}")
            self.state.request_virtual_camera(False)

    def _close_virtual_camera(self) -> None:
        if self._virtual_camera is not None:
            self._virtual_camera.close()
            self._virtual_camera = None
        self.state.set_virtual_camera_active(False)

    def _status_frame(
        self,
        title: str,
        subtitle: str,
        base_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        if base_frame is None:
            frame = np.zeros(
                (self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8
            )
        else:
            frame = base_frame.copy()

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (8, 16, 32), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)
        cv2.putText(
            frame,
            title,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (245, 245, 245),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            subtitle,
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (170, 220, 255),
            2,
            cv2.LINE_AA,
        )
        return frame
