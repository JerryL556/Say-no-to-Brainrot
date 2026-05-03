from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np


@dataclass
class FaceObservation:
    landmarks: list[tuple[float, float, float]]
    expressions: dict[str, bool]
    scores: dict[str, float]
    blendshapes: dict[str, float]


class FaceDetector:
    def __init__(self, model_path: Path, max_faces: int = 1) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing face model at '{model_path}'. Run scripts/download_models.py first."
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, mp_image: mp.Image, timestamp_ms: int) -> FaceObservation | None:
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks]

        blendshape_map: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                blendshape_map[category.category_name] = category.score

        smile_score = max(
            self._smile_from_blendshapes(blendshape_map),
            self._smile_from_geometry(landmarks),
        )
        mouth_open_score = max(
            blendshape_map.get("jawOpen", 0.0),
            self._mouth_open_from_geometry(landmarks),
        )
        brow_raise_score = max(
            blendshape_map.get("browInnerUp", 0.0),
            self._brow_raise_from_geometry(landmarks),
        )

        scores = {
            "smile": smile_score,
            "mouth_open": mouth_open_score,
            "brows_raised": brow_raise_score,
        }
        expressions = {
            "smile": smile_score > 0.36,
            "mouth_open": mouth_open_score > 0.32,
            "brows_raised": brow_raise_score > 0.28,
        }

        return FaceObservation(
            landmarks=landmarks,
            expressions=expressions,
            scores=scores,
            blendshapes=blendshape_map,
        )

    @staticmethod
    def _distance(
        landmarks: list[tuple[float, float, float]], left_index: int, right_index: int
    ) -> float:
        left = np.array(landmarks[left_index][:2], dtype=np.float32)
        right = np.array(landmarks[right_index][:2], dtype=np.float32)
        return float(np.linalg.norm(left - right))

    def _smile_from_blendshapes(self, blendshape_map: dict[str, float]) -> float:
        left = blendshape_map.get("mouthSmileLeft", 0.0)
        right = blendshape_map.get("mouthSmileRight", 0.0)
        return (left + right) / 2.0

    def _smile_from_geometry(self, landmarks: list[tuple[float, float, float]]) -> float:
        mouth_width = self._distance(landmarks, 61, 291)
        eye_span = self._distance(landmarks, 33, 263) + 1e-6
        ratio = mouth_width / eye_span
        return float(np.clip((ratio - 0.42) * 2.8, 0.0, 1.0))

    def _mouth_open_from_geometry(self, landmarks: list[tuple[float, float, float]]) -> float:
        mouth_gap = self._distance(landmarks, 13, 14)
        mouth_width = self._distance(landmarks, 61, 291) + 1e-6
        ratio = mouth_gap / mouth_width
        return float(np.clip((ratio - 0.08) * 5.8, 0.0, 1.0))

    def _brow_raise_from_geometry(self, landmarks: list[tuple[float, float, float]]) -> float:
        left_gap = self._distance(landmarks, 105, 159)
        right_gap = self._distance(landmarks, 334, 386)
        eye_span = self._distance(landmarks, 33, 263) + 1e-6
        ratio = ((left_gap + right_gap) / 2.0) / eye_span
        return float(np.clip((ratio - 0.04) * 8.5, 0.0, 1.0))
