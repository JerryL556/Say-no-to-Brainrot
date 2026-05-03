from __future__ import annotations

import threading
from typing import Any

from src.config import EffectSettings


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings = EffectSettings()
        self._frame_id = 0
        self._frame_jpeg = b""
        self._status = "starting"
        self._last_error: str | None = None
        self._virtual_camera_requested = True
        self._virtual_camera_active = False
        self._metrics: dict[str, Any] = {
            "fps": 0.0,
            "face_detected": False,
            "hand_count": 0,
            "expressions": [],
            "gestures": [],
            "backend": "initializing",
            "detection_size": "",
            "score": 0,
            "streak": 0,
            "combo_phase": "idle",
            "combo_gesture": "67_seesaw",
            "combo_activated": False,
        }

    def update_frame(self, jpeg_bytes: bytes, metrics: dict[str, Any]) -> None:
        with self._lock:
            self._frame_id += 1
            self._frame_jpeg = jpeg_bytes
            self._metrics = metrics

    def get_frame(self) -> tuple[int, bytes]:
        with self._lock:
            return self._frame_id, self._frame_jpeg

    def update_settings(self, patch: dict[str, bool]) -> None:
        with self._lock:
            for key, value in patch.items():
                if hasattr(self._settings, key):
                    setattr(self._settings, key, value)

    def get_settings(self) -> EffectSettings:
        with self._lock:
            return EffectSettings(**self._settings.to_dict())

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def set_error(self, error: str | None) -> None:
        with self._lock:
            self._last_error = error

    def request_virtual_camera(self, enabled: bool) -> None:
        with self._lock:
            self._virtual_camera_requested = enabled

    def virtual_camera_requested(self) -> bool:
        with self._lock:
            return self._virtual_camera_requested

    def set_virtual_camera_active(self, enabled: bool) -> None:
        with self._lock:
            self._virtual_camera_active = enabled

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "last_error": self._last_error,
                "virtual_camera_requested": self._virtual_camera_requested,
                "virtual_camera_active": self._virtual_camera_active,
                "settings": self._settings.to_dict(),
                "metrics": dict(self._metrics),
            }
