from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import permutations
from typing import Deque

import numpy as np

from src.detectors.hand_detector import HandObservation


@dataclass
class TrackedHandVisual:
    key: str
    handedness: str
    gesture: str
    center: tuple[float, float]
    trail: list[tuple[float, float]]
    phase: str
    motion: float


@dataclass
class ComboSnapshot:
    score: int = 0
    streak: int = 0
    phase: str = "idle"
    active_hand: str | None = None
    active_gesture: str = "67_seesaw"
    activated: bool = False
    heat: float = 0.0
    last_scored_at: float = -999.0
    tracked_hands: list[TrackedHandVisual] = field(default_factory=list)


@dataclass
class _HandState:
    key: str
    handedness: str
    center: tuple[float, float]
    gesture: str = "unknown"
    motion: float = 0.0
    last_seen: float = 0.0
    trail: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=18))


class ComboTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._slots: dict[str, _HandState | None] = {"Left": None, "Right": None}
        self._score = 0
        self._streak = 0
        self._phase = "locked"
        self._active_hand: str | None = None
        self._active_gesture = "67_seesaw"
        self._activated = False
        self._activation_started_at = 0.0
        self._activation_hold_seconds = 0.18
        self._last_scored_at = -999.0
        self._previous_score_at: float | None = None
        self._heat = 0.0

        self._trail_stale_seconds = 0.70
        self._missing_grace_seconds = 0.35
        self._armed_orientation: str | None = None
        self._orientation_threshold = 0.075
        self._release_threshold = 0.035
        self._min_cycle_motion = 0.010

    def update(self, hands: list[HandObservation], timestamp: float) -> ComboSnapshot:
        detections = [
            {
                "center": self._center(hand.landmarks),
                "gesture": hand.gesture,
                "handedness": hand.handedness or "Unknown",
            }
            for hand in hands[:2]
        ]
        self._update_slots(detections, timestamp)
        self._update_activation_state(timestamp)
        if self._activated:
            self._update_cycle_state(timestamp)

        if (timestamp - self._last_scored_at) > 0.8 and self._phase == "scored":
            self._phase = "idle"

        self._heat *= 0.985
        if (timestamp - self._last_scored_at) > 1.4:
            self._heat *= 0.97

        tracked_hands = [
            TrackedHandVisual(
                key=state.key,
                handedness=slot_name,
                gesture=state.gesture,
                center=state.center,
                trail=list(state.trail),
                phase=self._phase,
                motion=state.motion,
            )
            for slot_name, state in self._slots.items()
            if state is not None and (timestamp - state.last_seen) <= self._trail_stale_seconds
        ]

        return ComboSnapshot(
            score=self._score,
            streak=self._streak,
            phase=self._phase,
            active_hand=self._active_hand,
            active_gesture=self._active_gesture,
            activated=self._activated,
            heat=float(np.clip(self._heat, 0.0, 1.0)),
            last_scored_at=self._last_scored_at,
            tracked_hands=tracked_hands,
        )

    def _update_activation_state(self, timestamp: float) -> None:
        if self._activated:
            return

        left_state = self._slots["Left"]
        right_state = self._slots["Right"]
        if left_state is None or right_state is None:
            self._phase = "locked"
            self._active_hand = None
            self._activation_started_at = 0.0
            return

        left_recent = (timestamp - left_state.last_seen) <= self._missing_grace_seconds
        right_recent = (timestamp - right_state.last_seen) <= self._missing_grace_seconds
        if not (left_recent and right_recent):
            self._phase = "locked"
            self._active_hand = None
            self._activation_started_at = 0.0
            return

        left_match = left_state.gesture == "number_six"
        right_match = right_state.gesture == "number_seven"
        if left_match and right_match:
            self._phase = "activation_pose"
            self._active_hand = "both"
            if self._activation_started_at == 0.0:
                self._activation_started_at = timestamp
            elif (timestamp - self._activation_started_at) >= self._activation_hold_seconds:
                self._activated = True
                self._phase = "ready"
                self._active_hand = "both"
                self._activation_started_at = 0.0
                self._armed_orientation = None
                left_state.motion = 0.0
                right_state.motion = 0.0
            return

        self._phase = "locked"
        self._active_hand = None
        self._activation_started_at = 0.0

    def _update_slots(self, detections: list[dict[str, object]], timestamp: float) -> None:
        if not detections:
            return

        if len(detections) == 1:
            slot_name = self._choose_slot_for_single_detection(
                detections[0]["center"],  # type: ignore[index]
                timestamp,
            )
            self._apply_detection(slot_name, detections[0], timestamp)
            return

        detections = sorted(
            detections,
            key=lambda item: float(item["center"][0]),  # type: ignore[index]
        )
        current_slots = {
            slot_name: state
            for slot_name, state in self._slots.items()
            if state is not None and (timestamp - state.last_seen) <= self._trail_stale_seconds
        }

        if len(current_slots) < 2:
            self._apply_detection("Left", detections[0], timestamp)
            self._apply_detection("Right", detections[1], timestamp)
            return

        best_assignment: tuple[str, str] | None = None
        best_cost = float("inf")
        slot_names = ("Left", "Right")
        for order in permutations((0, 1), 2):
            cost = 0.0
            for slot_name, det_index in zip(slot_names, order):
                state = current_slots[slot_name]
                center = np.array(state.center, dtype=np.float32)
                target = np.array(detections[det_index]["center"], dtype=np.float32)  # type: ignore[index]
                cost += float(np.linalg.norm(center - target))
            if cost < best_cost:
                best_cost = cost
                best_assignment = ("Left", "Right") if order == (0, 1) else ("Right", "Left")

        if best_assignment is None:
            best_assignment = ("Left", "Right")

        self._apply_detection(best_assignment[0], detections[0], timestamp)
        self._apply_detection(best_assignment[1], detections[1], timestamp)

    def _choose_slot_for_single_detection(
        self,
        center: tuple[float, float],
        timestamp: float,
    ) -> str:
        left_state = self._slots["Left"]
        right_state = self._slots["Right"]
        target = np.array(center, dtype=np.float32)

        left_recent = left_state is not None and (timestamp - left_state.last_seen) <= self._trail_stale_seconds
        right_recent = right_state is not None and (timestamp - right_state.last_seen) <= self._trail_stale_seconds

        if left_recent and not right_recent:
            return "Left"
        if right_recent and not left_recent:
            return "Right"
        if left_recent and right_recent:
            left_distance = float(
                np.linalg.norm(target - np.array(left_state.center, dtype=np.float32))
            )
            right_distance = float(
                np.linalg.norm(target - np.array(right_state.center, dtype=np.float32))
            )
            return "Left" if left_distance <= right_distance else "Right"

        return "Left" if center[0] <= 0.5 else "Right"

    def _apply_detection(
        self,
        slot_name: str,
        detection: dict[str, object],
        timestamp: float,
    ) -> None:
        center = detection["center"]  # type: ignore[index]
        state = self._slots[slot_name]
        if state is None:
            state = _HandState(
                key=f"slot_{slot_name.lower()}",
                handedness=slot_name,
                center=center,  # type: ignore[arg-type]
                last_seen=timestamp,
            )

        if state.trail:
            previous = np.array(state.trail[-1], dtype=np.float32)
            current = np.array(center, dtype=np.float32)
            state.motion += float(np.linalg.norm(current - previous))

        state.center = center  # type: ignore[assignment]
        state.gesture = str(detection["gesture"])
        state.handedness = slot_name
        state.last_seen = timestamp
        state.trail.append(center)  # type: ignore[arg-type]
        self._slots[slot_name] = state

    def _update_cycle_state(self, timestamp: float) -> None:
        left_state = self._slots["Left"]
        right_state = self._slots["Right"]
        if left_state is None or right_state is None:
            self._reset_cycle("idle", None)
            return

        left_recent = (timestamp - left_state.last_seen) <= self._missing_grace_seconds
        right_recent = (timestamp - right_state.last_seen) <= self._missing_grace_seconds
        if not (left_recent and right_recent):
            if self._phase == "scored":
                return
            if self._armed_orientation is not None:
                self._phase = "tracking_hold"
                self._active_hand = "both"
                return
            self._reset_cycle("idle", None)
            return

        self._active_hand = "both"

        gap = left_state.center[1] - right_state.center[1]
        abs_gap = abs(gap)
        total_motion = left_state.motion + right_state.motion

        if abs_gap < self._release_threshold:
            if self._phase != "scored":
                self._phase = "ready"
            return

        orientation: str | None = None
        if gap <= -self._orientation_threshold:
            orientation = "left_up"
        elif gap >= self._orientation_threshold:
            orientation = "right_up"

        if orientation is None:
            if self._phase != "scored":
                self._phase = "ready"
            return

        self._phase = orientation
        self._active_hand = "Left" if orientation == "left_up" else "Right"

        if self._armed_orientation is None:
            self._armed_orientation = orientation
            return

        if orientation != self._armed_orientation and total_motion >= self._min_cycle_motion:
            self._commit_score(left_state, right_state, timestamp)
            self._armed_orientation = orientation
            return

        self._armed_orientation = orientation

    def _commit_score(
        self,
        left_state: _HandState,
        right_state: _HandState,
        timestamp: float,
    ) -> None:
        self._score += 1
        self._streak += 1
        self._phase = "scored"
        self._active_hand = self._active_hand
        if self._previous_score_at is None:
            speed_boost = 0.32
        else:
            delta = max(0.05, timestamp - self._previous_score_at)
            speed_boost = float(np.clip(0.9 / delta, 0.14, 0.72))
        self._heat = float(np.clip(self._heat * 0.72 + speed_boost, 0.0, 1.0))
        self._previous_score_at = timestamp
        self._last_scored_at = timestamp
        left_state.motion = 0.0
        right_state.motion = 0.0

    def _reset_cycle(self, phase: str, active_hand: str | None) -> None:
        self._phase = phase
        self._active_hand = active_hand
        self._armed_orientation = None

    @staticmethod
    def _center(landmarks: list[tuple[float, float, float]]) -> tuple[float, float]:
        points = np.array([landmarks[0][:2], landmarks[5][:2], landmarks[17][:2]], dtype=np.float32)
        center = np.mean(points, axis=0)
        return float(center[0]), float(center[1])
