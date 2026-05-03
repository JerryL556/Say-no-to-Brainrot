from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np


@dataclass
class HandObservation:
    landmarks: list[tuple[float, float, float]]
    handedness: str
    gesture: str
    confidence: float


class HandDetector:
    def __init__(self, model_path: Path, max_hands: int = 2) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing hand model at '{model_path}'. Run scripts/download_models.py first."
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, mp_image: mp.Image, timestamp_ms: int) -> list[HandObservation]:
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        observations: list[HandObservation] = []
        for index, hand_landmarks in enumerate(result.hand_landmarks):
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            handedness = "Unknown"
            if result.handedness and len(result.handedness) > index and result.handedness[index]:
                handedness = result.handedness[index][0].category_name

            gesture, confidence = self._classify_gesture(landmarks)
            observations.append(
                HandObservation(
                    landmarks=landmarks,
                    handedness=handedness,
                    gesture=gesture,
                    confidence=confidence,
                )
            )
        return observations

    @staticmethod
    def _distance(
        landmarks: list[tuple[float, float, float]], left_index: int, right_index: int
    ) -> float:
        left = np.array(landmarks[left_index][:2], dtype=np.float32)
        right = np.array(landmarks[right_index][:2], dtype=np.float32)
        return float(np.linalg.norm(left - right))

    def _is_finger_extended(
        self,
        landmarks: list[tuple[float, float, float]],
        tip_index: int,
        pip_index: int,
        palm_center: np.ndarray,
    ) -> bool:
        tip = np.array(landmarks[tip_index][:2], dtype=np.float32)
        pip = np.array(landmarks[pip_index][:2], dtype=np.float32)
        return (
            np.linalg.norm(tip - palm_center) > np.linalg.norm(pip - palm_center) * 1.12
            and tip[1] < pip[1]
        )

    def _is_thumb_extended(
        self, landmarks: list[tuple[float, float, float]], palm_center: np.ndarray
    ) -> bool:
        tip = np.array(landmarks[4][:2], dtype=np.float32)
        ip = np.array(landmarks[3][:2], dtype=np.float32)
        return np.linalg.norm(tip - palm_center) > np.linalg.norm(ip - palm_center) * 1.18

    def _classify_gesture(
        self, landmarks: list[tuple[float, float, float]]
    ) -> tuple[str, float]:
        palm_center = np.mean(
            np.array([landmarks[0][:2], landmarks[5][:2], landmarks[17][:2]], dtype=np.float32),
            axis=0,
        )
        thumb = self._is_thumb_extended(landmarks, palm_center)
        index = self._is_finger_extended(landmarks, 8, 6, palm_center)
        middle = self._is_finger_extended(landmarks, 12, 10, palm_center)
        ring = self._is_finger_extended(landmarks, 16, 14, palm_center)
        pinky = self._is_finger_extended(landmarks, 20, 18, palm_center)
        palm_width = self._distance(landmarks, 5, 17) + 1e-6

        if thumb and pinky and not index and not middle and not ring:
            return "number_six", 0.92

        thumb_index = self._distance(landmarks, 4, 8)
        thumb_middle = self._distance(landmarks, 4, 12)
        thumb_ring = self._distance(landmarks, 4, 16)
        clustered = (
            thumb_index < palm_width * 0.42
            and thumb_middle < palm_width * 0.48
            and thumb_ring < palm_width * 0.54
            and not pinky
        )
        if clustered:
            return "number_seven", 0.88

        if thumb and index and middle and ring and pinky:
            return "open_palm", 0.96

        if index and middle and not ring and not pinky:
            tip_spread = self._distance(landmarks, 8, 12)
            return "peace_sign", float(np.clip(0.72 + (tip_spread / palm_width) * 0.18, 0.0, 0.98))

        other_folded = not index and not middle and not ring and not pinky
        thumb_tip_y = landmarks[4][1]
        wrist_y = landmarks[0][1]
        if thumb and other_folded and thumb_tip_y < wrist_y:
            return "thumbs_up", 0.9

        return "unknown", 0.0
