from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np

from src.combo_tracker import ComboSnapshot, TrackedHandVisual
from src.config import EffectSettings, ROOT_DIR
from src.detectors.face_detector import FaceObservation
from src.detectors.hand_detector import HandObservation


class EffectRenderer:
    def __init__(self) -> None:
        self._smoothed_heat = 0.0
        self._final_triggered = False
        self._final_overlay = self._load_final_overlay()
        self._nod_phase = "idle"
        self._nod_baseline: float | None = None
        self._nod_smoothed_pitch: float | None = None
        self._nod_peak_pitch: float | None = None
        self._pending_session_reset = False

    def render(
        self,
        frame: np.ndarray,
        face: FaceObservation | None,
        hands: list[HandObservation],
        settings: EffectSettings,
        tick: float,
        combo: ComboSnapshot | None = None,
    ) -> np.ndarray:
        if (
            combo is not None
            and combo.activated
            and self._should_trigger_final(combo)
        ):
            self._final_triggered = True
            if self._nod_phase == "idle":
                self._nod_phase = "waiting_down"

        if self._final_triggered:
            self._update_nod_state(face)
            self._render_final_overlay(frame)
            return frame

        if combo is not None and combo.activated:
            self._update_heat(combo.heat)
            self._render_heat_tint(frame, self._smoothed_heat)
            if face is not None:
                self._render_eye_lasers(frame, face, tick)
            self._render_combo_effects(frame, combo, tick)
        else:
            self._update_heat(0.0)
        if combo is not None and combo.activated:
            self._render_score_ui(frame, combo)
        return frame

    def _load_final_overlay(self) -> np.ndarray | None:
        image_path = ROOT_DIR / "a55e5d90-e3e3-423c-806c-6714129c4b7f.png"
        if not image_path.exists():
            return None
        return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    def _should_trigger_final(self, combo: ComboSnapshot) -> bool:
        return combo.score >= 15

    def _update_nod_state(self, face: FaceObservation | None) -> None:
        if face is None:
            return

        pitch = self._estimate_nod_pitch(face)
        if self._nod_smoothed_pitch is None:
            self._nod_smoothed_pitch = pitch
        else:
            self._nod_smoothed_pitch += (pitch - self._nod_smoothed_pitch) * 0.2

        if self._nod_baseline is None:
            self._nod_baseline = self._nod_smoothed_pitch
        elif self._nod_phase == "waiting_down":
            self._nod_baseline += (self._nod_smoothed_pitch - self._nod_baseline) * 0.05

        baseline = self._nod_baseline
        current_pitch = self._nod_smoothed_pitch
        down_threshold = baseline + 0.032
        up_threshold = baseline + 0.014

        if self._nod_phase == "waiting_down":
            if current_pitch >= down_threshold:
                self._nod_phase = "waiting_up"
                self._nod_peak_pitch = current_pitch
            return

        if self._nod_phase != "waiting_up":
            return

        if self._nod_peak_pitch is None or current_pitch > self._nod_peak_pitch:
            self._nod_peak_pitch = current_pitch

        if self._nod_peak_pitch >= down_threshold and current_pitch <= up_threshold:
            self._dismiss_final_overlay()

    def _dismiss_final_overlay(self) -> None:
        self._final_triggered = False
        self._pending_session_reset = True
        self._nod_phase = "idle"
        self._nod_baseline = None
        self._nod_smoothed_pitch = None
        self._nod_peak_pitch = None
        self._smoothed_heat = 0.0

    def consume_session_reset(self) -> bool:
        if not self._pending_session_reset:
            return False
        self._pending_session_reset = False
        return True

    def _estimate_nod_pitch(self, face: FaceObservation) -> float:
        landmarks = face.landmarks
        forehead = np.array(landmarks[10][:2], dtype=np.float32)
        chin = np.array(landmarks[152][:2], dtype=np.float32)
        nose = np.array(landmarks[1][:2], dtype=np.float32)
        left_eye = np.array(landmarks[33][:2], dtype=np.float32)
        right_eye = np.array(landmarks[263][:2], dtype=np.float32)
        eye_mid = (left_eye + right_eye) / 2.0
        face_height = max(1e-6, float(chin[1] - forehead[1]))
        return float((nose[1] - eye_mid[1]) / face_height)

    def _render_final_overlay(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        image = self._final_overlay
        if image is not None:
            composed = self._fit_overlay_to_frame(image, frame.shape[1], frame.shape[0])
            if composed.shape[2] == 4:
                alpha = composed[:, :, 3:4].astype(np.float32) / 255.0
                rgb = composed[:, :, :3].astype(np.float32)
                base = overlay.astype(np.float32)
                overlay[:, :, :] = (rgb * alpha + base * (1.0 - alpha)).astype(np.uint8)
            else:
                overlay[:, :, :] = composed[:, :, :3]
        else:
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 24, 40), -1)

        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0.0, frame)
        self._draw_final_overlay_prompt(frame)

    def _draw_final_overlay_prompt(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        panel_width = min(width - 40, 420)
        panel_height = 68
        left = (width - panel_width) // 2
        top = height - panel_height - 28

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (left, top),
            (left + panel_width, top + panel_height),
            (10, 12, 18),
            -1,
        )
        cv2.rectangle(
            overlay,
            (left, top),
            (left + panel_width, top + panel_height),
            (240, 245, 255),
            2,
        )
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)
        label = "NOD TO AFFIRM"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = top + 44
        cv2.putText(
            frame,
            label,
            (text_x + 2, text_y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (8, 10, 12),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (255, 250, 240),
            2,
            cv2.LINE_AA,
        )

    def _fit_overlay_to_frame(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        if image_height == 0 or image_width == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        scale = max(width / image_width, height / image_height)
        resized = cv2.resize(
            image,
            (max(1, int(image_width * scale)), max(1, int(image_height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        start_x = max(0, (resized.shape[1] - width) // 2)
        start_y = max(0, (resized.shape[0] - height) // 2)
        return resized[start_y:start_y + height, start_x:start_x + width]


    def _update_heat(self, target_heat: float) -> None:
        blend = 0.08 if target_heat > self._smoothed_heat else 0.035
        self._smoothed_heat += (target_heat - self._smoothed_heat) * blend

    def _render_heat_tint(self, frame: np.ndarray, heat: float) -> None:
        if heat <= 0.02:
            return
        overlay = frame.copy()
        tint_strength = min(0.26, 0.05 + heat * 0.18)
        red_color = (30, 40, 235)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), red_color, -1)
        cv2.addWeighted(overlay, tint_strength, frame, 1.0 - tint_strength, 0.0, frame)

    def _render_score_ui(self, frame: np.ndarray, combo: ComboSnapshot) -> None:
        overlay = frame.copy()
        width = frame.shape[1]
        bar_width = 240
        bar_height = 48
        left = (width - bar_width) // 2
        top = 18
        cv2.rectangle(
            overlay,
            (left, top),
            (left + bar_width, top + bar_height),
            (12, 18, 34),
            -1,
        )
        cv2.rectangle(
            overlay,
            (left, top),
            (left + bar_width, top + bar_height),
            (150, 220, 255) if combo.activated else (150, 150, 170),
            2,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0, frame)
        cv2.putText(
            frame,
            f"BRAINROT SCORE  {combo.score}",
            (left + 26, top + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 245, 220),
            2,
            cv2.LINE_AA,
        )

    def _render_eye_lasers(
        self,
        frame: np.ndarray,
        face: FaceObservation,
        tick: float,
    ) -> None:
        height, width = frame.shape[:2]
        left_eye = self._feature_center(face.landmarks, [33, 133, 159, 145], width, height)
        right_eye = self._feature_center(face.landmarks, [362, 263, 386, 374], width, height)
        forehead = self._point(face.landmarks, 10, width, height)
        nose_tip = self._point(face.landmarks, 1, width, height)
        chin = self._point(face.landmarks, 152, width, height)
        left_face = self._point(face.landmarks, 234, width, height)
        right_face = self._point(face.landmarks, 454, width, height)

        eye_mid = np.array(
            [(left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0],
            dtype=np.float32,
        )
        face_axis = np.array([chin[0] - eye_mid[0], chin[1] - forehead[1]], dtype=np.float32)
        face_axis = self._normalize(
            face_axis if np.linalg.norm(face_axis) > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)
        )

        face_width = max(1.0, float(np.linalg.norm(np.array(right_face) - np.array(left_face))))
        yaw = float((nose_tip[0] - eye_mid[0]) / face_width)
        pitch = float(((nose_tip[1] - eye_mid[1]) - (chin[1] - forehead[1]) * 0.18) / max(1.0, chin[1] - forehead[1]))
        left_gaze = self._estimate_eye_gaze(face.landmarks, True, width, height)
        right_gaze = self._estimate_eye_gaze(face.landmarks, False, width, height)
        avg_gaze = (left_gaze + right_gaze) / 2.0
        shared_dir = self._compose_forward_direction(face_axis, yaw, pitch, avg_gaze)
        shared_distance = self._ray_distance_to_frame(
            (int(eye_mid[0]), int(eye_mid[1])),
            shared_dir,
            width,
            height,
            1.0,
        )
        pulse = 0.74 + 0.26 * (0.5 + 0.5 * math.sin(tick * 14.0))

        self._draw_laser_beam(
            frame,
            left_eye,
            shared_dir,
            pulse,
            tick,
            shared_distance,
            (255, 70, 40),
            (255, 215, 170),
            (130, 225, 255),
        )
        self._draw_laser_beam(
            frame,
            right_eye,
            shared_dir,
            pulse,
            tick + 0.37,
            shared_distance,
            (255, 70, 40),
            (255, 215, 170),
            (130, 225, 255),
        )
        self._draw_eye_glow(frame, left_eye, pulse, (255, 110, 90))
        self._draw_eye_glow(frame, right_eye, pulse, (255, 110, 90))
        self._draw_eye_particles(frame, left_eye, pulse, tick, (255, 210, 170))
        self._draw_eye_particles(frame, right_eye, pulse, tick + 0.41, (255, 210, 170))

    def _compose_forward_direction(
        self,
        face_axis: np.ndarray,
        yaw: float,
        pitch: float,
        gaze: np.ndarray,
    ) -> np.ndarray:
        horizontal = (yaw * 2.1) + (gaze[0] * 1.5)
        vertical = (pitch * 2.3) + (gaze[1] * 1.9) - 0.04
        direction = np.array([horizontal, vertical], dtype=np.float32) + face_axis * 0.04
        if np.linalg.norm(direction) < 0.10:
            direction = np.array([0.0, -0.12], dtype=np.float32)
        return self._normalize(direction)
    def _render_combo_effects(self, frame: np.ndarray, combo: ComboSnapshot, tick: float) -> None:
        height, width = frame.shape[:2]
        for tracked_hand in combo.tracked_hands:
            self._draw_hand_trail(frame, tracked_hand, width, height)

        if len(combo.tracked_hands) >= 2:
            left = next((hand for hand in combo.tracked_hands if hand.handedness == "Left"), None)
            right = next((hand for hand in combo.tracked_hands if hand.handedness == "Right"), None)
            if left is not None and right is not None:
                left_center = (int(left.center[0] * width), int(left.center[1] * height))
                right_center = (int(right.center[0] * width), int(right.center[1] * height))
                self._draw_seesaw_beam(frame, left_center, right_center, combo.phase, tick)
                self._draw_gesture_tag(frame, left_center, "6", (255, 180, 70))
                self._draw_gesture_tag(frame, right_center, "7", (110, 255, 235))

        flash_age = tick - combo.last_scored_at
        if flash_age < 0.8:
            strength = 1.0 - (flash_age / 0.8)
            self._draw_combo_flash(frame, combo, strength)


    def _draw_hand_trail(
        self,
        frame: np.ndarray,
        tracked_hand: TrackedHandVisual,
        width: int,
        height: int,
    ) -> None:
        if len(tracked_hand.trail) < 2:
            return
        color = (
            (255, 180, 70)
            if tracked_hand.handedness == "Left"
            else (110, 255, 235)
        )
        points = [
            (int(point[0] * width), int(point[1] * height))
            for point in tracked_hand.trail
        ]
        for index in range(1, len(points)):
            alpha = index / len(points)
            thickness = max(1, int(2 + alpha * 6))
            blend = tuple(int(channel * alpha) for channel in color)
            cv2.line(frame, points[index - 1], points[index], blend, thickness, cv2.LINE_AA)

    def _draw_seesaw_beam(
        self,
        frame: np.ndarray,
        left_center: tuple[int, int],
        right_center: tuple[int, int],
        phase: str,
        tick: float,
    ) -> None:
        overlay = frame.copy()
        base_color = (160, 235, 255)
        cv2.line(overlay, left_center, right_center, base_color, 5, cv2.LINE_AA)
        pulse = 10 + int((math.sin(tick * 9.0) + 1.0) * 5)
        left_color = (255, 180, 70)
        right_color = (110, 255, 235)
        cv2.circle(overlay, left_center, pulse, left_color, 3, cv2.LINE_AA)
        cv2.circle(overlay, right_center, pulse, right_color, 3, cv2.LINE_AA)
        if phase == "left_up":
            cv2.arrowedLine(
                overlay,
                (left_center[0], left_center[1] + 40),
                (left_center[0], left_center[1] - 24),
                left_color,
                3,
                cv2.LINE_AA,
                tipLength=0.35,
            )
            cv2.arrowedLine(
                overlay,
                (right_center[0], right_center[1] - 40),
                (right_center[0], right_center[1] + 24),
                right_color,
                3,
                cv2.LINE_AA,
                tipLength=0.35,
            )
        elif phase == "right_up":
            cv2.arrowedLine(
                overlay,
                (left_center[0], left_center[1] - 40),
                (left_center[0], left_center[1] + 24),
                left_color,
                3,
                cv2.LINE_AA,
                tipLength=0.35,
            )
            cv2.arrowedLine(
                overlay,
                (right_center[0], right_center[1] + 40),
                (right_center[0], right_center[1] - 24),
                right_color,
                3,
                cv2.LINE_AA,
                tipLength=0.35,
            )
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

    def _draw_gesture_tag(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (center[0] - 24, center[1] - 86),
            (center[0] + 24, center[1] - 44),
            (20, 30, 54),
            -1,
        )
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)
        cv2.putText(
            frame,
            label,
            (center[0] - 11, center[1] - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_combo_flash(
        self,
        frame: np.ndarray,
        combo: ComboSnapshot,
        strength: float,
    ) -> None:
        overlay = frame.copy()
        # Keep the global flash subtle so the lightning remains the dominant effect.
        color = (160, 210, 255) if combo.phase == "scored" else (255, 210, 120)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
        cv2.addWeighted(overlay, 0.018 * strength, frame, 1.0 - (0.018 * strength), 0.0, frame)
        score_text = f"+1  BRAINROT SCORE {combo.score}"
        shadow_color = (24, 26, 36)
        text_color = (255, 245, 220)
        origin = (frame.shape[1] // 2 - 170, 92)
        self._draw_sky_lightning(frame, combo.score, strength)
        cv2.putText(
            frame,
            score_text,
            (origin[0] + 2, origin[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0 + (0.12 * strength),
            shadow_color,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            score_text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0 + (0.12 * strength),
            text_color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _point(
        landmarks: list[tuple[float, float, float]],
        index: int,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        x, y, _ = landmarks[index]
        return int(x * width), int(y * height)

    def _feature_center(
        self,
        landmarks: list[tuple[float, float, float]],
        indices: list[int],
        width: int,
        height: int,
    ) -> tuple[int, int]:
        points = np.array([landmarks[index][:2] for index in indices], dtype=np.float32)
        center = np.mean(points, axis=0)
        return int(center[0] * width), int(center[1] * height)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return vector / norm

    def _draw_laser_beam(
        self,
        frame: np.ndarray,
        origin: tuple[int, int],
        direction: np.ndarray,
        pulse: float,
        tick: float,
        distance: float,
        core_color: tuple[int, int, int],
        glow_color: tuple[int, int, int],
        arc_color: tuple[int, int, int],
    ) -> None:
        target = (
            int(origin[0] + direction[0] * distance),
            int(origin[1] + direction[1] * distance),
        )
        direction_perp = self._normalize(np.array([-direction[1], direction[0]], dtype=np.float32))

        overlay = frame.copy()
        glow_thickness = int(24 + 12 * pulse)
        core_thickness = int(7 + 3 * pulse)
        corona_thickness = int(12 + 5 * pulse)
        cv2.line(overlay, origin, target, (255, 255, 255), corona_thickness, cv2.LINE_AA)
        cv2.line(overlay, origin, target, glow_color, glow_thickness, cv2.LINE_AA)
        cv2.line(overlay, origin, target, core_color, core_thickness, cv2.LINE_AA)
        self._draw_beam_lightning(overlay, origin, target, direction_perp, tick, arc_color)
        self._draw_beam_particles(overlay, origin, target, direction_perp, tick, glow_color)

        impact_radius = int(16 + 8 * pulse)
        cv2.circle(overlay, target, impact_radius, glow_color, 2, cv2.LINE_AA)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            inner = (
                int(target[0] + math.cos(rad) * impact_radius * 0.35),
                int(target[1] + math.sin(rad) * impact_radius * 0.35),
            )
            outer = (
                int(target[0] + math.cos(rad) * impact_radius * 1.25),
                int(target[1] + math.sin(rad) * impact_radius * 1.25),
            )
            cv2.line(overlay, inner, outer, glow_color, 2, cv2.LINE_AA)
        self._draw_impact_sparks(overlay, target, tick, arc_color, glow_color)

        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0.0, frame)

    def _draw_eye_glow(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        pulse: float,
        color: tuple[int, int, int],
    ) -> None:
        overlay = frame.copy()
        radius = int(10 + 4 * pulse)
        cv2.circle(overlay, center, radius * 2, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, center, radius, (255, 245, 220), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0.0, frame)

    def _ray_distance_to_frame(
        self,
        origin: tuple[int, int],
        direction: np.ndarray,
        width: int,
        height: int,
        scale: float,
    ) -> float:
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = float(direction[0]), float(direction[1])
        candidates: list[float] = []

        if abs(dx) > 1e-6:
            tx = ((width - 1) - ox) / dx if dx > 0 else (0 - ox) / dx
            if tx > 0:
                candidates.append(tx)
        if abs(dy) > 1e-6:
            ty = ((height - 1) - oy) / dy if dy > 0 else (0 - oy) / dy
            if ty > 0:
                candidates.append(ty)

        if not candidates:
            return max(width, height) * 0.6 * scale

        return min(candidates) * scale

    def _estimate_eye_gaze(
        self,
        landmarks: list[tuple[float, float, float]],
        left_eye: bool,
        width: int,
        height: int,
    ) -> np.ndarray:
        if len(landmarks) < 478:
            return np.zeros(2, dtype=np.float32)

        if left_eye:
            iris_indices = [468, 469, 470, 471, 472]
            horizontal_indices = (33, 133)
            vertical_indices = (159, 145)
        else:
            iris_indices = [473, 474, 475, 476, 477]
            horizontal_indices = (362, 263)
            vertical_indices = (386, 374)

        iris_center = np.mean(
            np.array([landmarks[index][:2] for index in iris_indices], dtype=np.float32),
            axis=0,
        )
        horizontal_left = np.array(landmarks[horizontal_indices[0]][:2], dtype=np.float32)
        horizontal_right = np.array(landmarks[horizontal_indices[1]][:2], dtype=np.float32)
        vertical_top = np.array(landmarks[vertical_indices[0]][:2], dtype=np.float32)
        vertical_bottom = np.array(landmarks[vertical_indices[1]][:2], dtype=np.float32)

        eye_center = np.mean(
            np.array([horizontal_left, horizontal_right, vertical_top, vertical_bottom], dtype=np.float32),
            axis=0,
        )
        horizontal_span = max(1e-6, float(np.linalg.norm(horizontal_right - horizontal_left)))
        vertical_span = max(1e-6, float(np.linalg.norm(vertical_bottom - vertical_top)))
        offset = iris_center - eye_center
        gaze = np.array(
            [
                float(np.clip(offset[0] / horizontal_span, -0.45, 0.45)),
                float(np.clip(offset[1] / vertical_span, -0.45, 0.45)),
            ],
            dtype=np.float32,
        )
        return gaze

    def _draw_beam_lightning(
        self,
        overlay: np.ndarray,
        origin: tuple[int, int],
        target: tuple[int, int],
        perp: np.ndarray,
        tick: float,
        color: tuple[int, int, int],
    ) -> None:
        points = [origin]
        for step in range(1, 5):
            t = step / 5.0
            base_x = origin[0] + (target[0] - origin[0]) * t
            base_y = origin[1] + (target[1] - origin[1]) * t
            swing = math.sin(tick * 11.0 + step * 1.8) * (9 + step * 2)
            points.append(
                (
                    int(base_x + perp[0] * swing),
                    int(base_y + perp[1] * swing),
                )
            )
        points.append(target)
        for index in range(1, len(points)):
            cv2.line(overlay, points[index - 1], points[index], color, 2, cv2.LINE_AA)

    def _draw_beam_particles(
        self,
        overlay: np.ndarray,
        origin: tuple[int, int],
        target: tuple[int, int],
        perp: np.ndarray,
        tick: float,
        color: tuple[int, int, int],
    ) -> None:
        for step in range(1, 7):
            t = (step * 0.13 + tick * 0.22) % 1.0
            base_x = origin[0] + (target[0] - origin[0]) * t
            base_y = origin[1] + (target[1] - origin[1]) * t
            spread = math.sin(tick * 8.0 + step * 2.1) * (6 + step)
            center = (
                int(base_x + perp[0] * spread),
                int(base_y + perp[1] * spread),
            )
            cv2.circle(overlay, center, 3, color, -1, cv2.LINE_AA)

    def _draw_impact_sparks(
        self,
        overlay: np.ndarray,
        target: tuple[int, int],
        tick: float,
        arc_color: tuple[int, int, int],
        spark_color: tuple[int, int, int],
    ) -> None:
        for index in range(6):
            angle = tick * 3.2 + index * (math.pi / 3.0)
            inner = (
                int(target[0] + math.cos(angle) * 8),
                int(target[1] + math.sin(angle) * 8),
            )
            outer = (
                int(target[0] + math.cos(angle) * 26),
                int(target[1] + math.sin(angle) * 26),
            )
            cv2.line(overlay, inner, outer, spark_color, 2, cv2.LINE_AA)
        jag_points = []
        for step in range(5):
            angle = tick * 5.0 + step * 0.9
            radius = 10 + step * 5
            jag_points.append(
                (
                    int(target[0] + math.cos(angle) * radius),
                    int(target[1] + math.sin(angle) * radius),
                )
            )
        for index in range(1, len(jag_points)):
            cv2.line(overlay, jag_points[index - 1], jag_points[index], arc_color, 2, cv2.LINE_AA)

    def _draw_eye_particles(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        pulse: float,
        tick: float,
        color: tuple[int, int, int],
    ) -> None:
        overlay = frame.copy()
        for index in range(5):
            angle = tick * 5.5 + index * 1.25
            radius = 16 + index * 3
            particle = (
                int(center[0] + math.cos(angle) * radius),
                int(center[1] + math.sin(angle) * radius * 0.55),
            )
            cv2.circle(overlay, particle, max(1, int(2 + pulse * 2)), color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)

    def _draw_sky_lightning(
        self,
        frame: np.ndarray,
        score: int,
        strength: float,
    ) -> None:
        rng = random.Random(score * 7919 + 17)
        overlay = frame.copy()
        width = frame.shape[1]
        height = frame.shape[0]
        bolt_count = 4 + (score % 3)

        # Add a bright storm cap near the top so bolts feel like they come from the sky.
        cloud_height = max(40, int(height * 0.16))
        cv2.rectangle(overlay, (0, 0), (width, cloud_height), (215, 235, 255), -1)
        cv2.addWeighted(overlay, 0.06 * strength, frame, 1.0 - (0.06 * strength), 0.0, frame)

        for bolt_index in range(bolt_count):
            start_x = rng.randint(int(width * 0.05), int(width * 0.95))
            start = (start_x, 0)
            segments = 7 + rng.randint(0, 4)
            current = start
            points = [start]
            for segment in range(segments):
                next_x = int(
                    np.clip(
                        current[0] + rng.randint(-70, 70) + math.sin((segment + 1) * 0.85 + bolt_index) * 16,
                        0,
                        width - 1,
                    )
                )
                next_y = int(
                    min(
                        height - 1,
                        (segment + 1) * (height * (0.085 + 0.028 * rng.random())),
                    )
                )
                current = (next_x, next_y)
                points.append(current)

            for index in range(1, len(points)):
                # Fat glow, white-hot core, and cool electric rim.
                cv2.line(overlay, points[index - 1], points[index], (120, 210, 255), 12, cv2.LINE_AA)
                cv2.line(overlay, points[index - 1], points[index], (210, 240, 255), 7, cv2.LINE_AA)
                cv2.line(overlay, points[index - 1], points[index], (255, 255, 255), 3, cv2.LINE_AA)

            branch_count = 3 + rng.randint(0, 3)
            for _ in range(branch_count):
                branch_root_index = rng.randint(1, max(1, len(points) - 2))
                branch_root = points[branch_root_index]
                branch_angle = rng.uniform(-1.6, 1.6)
                branch_length = rng.randint(36, 120)
                branch_target = (
                    int(np.clip(branch_root[0] + math.cos(branch_angle) * branch_length, 0, width - 1)),
                    int(np.clip(branch_root[1] + abs(math.sin(branch_angle)) * branch_length, 0, height - 1)),
                )
                cv2.line(overlay, branch_root, branch_target, (150, 220, 255), 5, cv2.LINE_AA)
                cv2.line(overlay, branch_root, branch_target, (255, 255, 255), 2, cv2.LINE_AA)

            impact = points[-1]
            impact_radius = int(24 + 16 * strength)
            cv2.circle(overlay, impact, impact_radius + 8, (120, 210, 255), 4, cv2.LINE_AA)
            cv2.circle(overlay, impact, impact_radius, (210, 240, 255), 3, cv2.LINE_AA)
            cv2.circle(overlay, impact, max(4, impact_radius // 3), (255, 255, 255), -1, cv2.LINE_AA)
            for spark_index in range(7):
                spark_angle = rng.uniform(0.0, math.tau)
                spark_len = rng.randint(14, 34)
                spark_end = (
                    int(np.clip(impact[0] + math.cos(spark_angle) * spark_len, 0, width - 1)),
                    int(np.clip(impact[1] + math.sin(spark_angle) * spark_len, 0, height - 1)),
                )
                cv2.line(overlay, impact, spark_end, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(overlay, impact, spark_end, (150, 220, 255), 4, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.62 * strength, frame, 1.0 - (0.62 * strength), 0.0, frame)
