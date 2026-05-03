const stateElements = {
  statusText: document.getElementById("status-text"),
  statusDot: document.getElementById("status-dot"),
  fps: document.getElementById("fps-value"),
  face: document.getElementById("face-value"),
  hands: document.getElementById("hands-value"),
  comboMode: document.getElementById("combo-mode-value"),
  score: document.getElementById("score-value"),
  comboPhase: document.getElementById("combo-phase-value"),
  expressions: document.getElementById("expressions-value"),
  gestures: document.getElementById("gestures-value"),
  comboGesture: document.getElementById("combo-gesture-value"),
  backend: document.getElementById("backend-value"),
  detectSize: document.getElementById("detect-size-value"),
  error: document.getElementById("error-text"),
  vcamStatus: document.getElementById("vcam-status"),
};

async function refreshState() {
  try {
    const response = await fetch("/api/state");
    const state = await response.json();
    renderState(state);
  } catch (error) {
    stateElements.statusText.textContent = "Offline";
    stateElements.statusDot.style.background = "#ff6a88";
    stateElements.error.textContent = "Failed to reach the Python backend.";
  }
}

function renderState(state) {
  const healthy = state.status === "running";
  stateElements.statusText.textContent = state.status;
  stateElements.statusDot.style.background = healthy ? "#7ff0a5" : "#ff8f70";
  stateElements.fps.textContent = `${state.metrics.fps.toFixed(1)}`;
  stateElements.face.textContent = state.metrics.face_detected ? "Yes" : "No";
  stateElements.hands.textContent = `${state.metrics.hand_count}`;
  stateElements.comboMode.textContent = state.metrics.combo_activated ? "on" : "locked";
  stateElements.score.textContent = `${state.metrics.score}`;
  stateElements.comboPhase.textContent = state.metrics.combo_phase || "idle";
  stateElements.expressions.textContent = state.metrics.expressions.join(", ") || "none";
  stateElements.gestures.textContent = state.metrics.gestures.join(", ") || "none";
  stateElements.comboGesture.textContent = state.metrics.combo_gesture || "67_seesaw";
  stateElements.backend.textContent = state.metrics.backend || "unknown";
  stateElements.detectSize.textContent = state.metrics.detection_size || "-";
  stateElements.error.textContent = state.last_error || "No errors reported.";
  stateElements.vcamStatus.textContent = state.virtual_camera_active
    ? "Virtual camera active"
    : state.virtual_camera_requested
      ? "Starting virtual camera..."
      : "Idle";
}

async function setVirtualCamera(action) {
  await fetch(`/api/virtual-camera/${action}`, { method: "POST" });
  await refreshState();
}

document.getElementById("start-vcam").addEventListener("click", () => setVirtualCamera("start"));
document.getElementById("stop-vcam").addEventListener("click", () => setVirtualCamera("stop"));

refreshState();
setInterval(refreshState, 1000);
