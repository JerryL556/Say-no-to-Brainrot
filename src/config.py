from dataclasses import asdict, dataclass
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
FRONTEND_DIR = ROOT_DIR / "src" / "web" / "static"


@dataclass
class EffectSettings:
    smile_hearts: bool = True
    mouth_fire: bool = True
    brow_flash: bool = True
    peace_sparkles: bool = True
    thumbs_burst: bool = True
    open_palm_aura: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass
class EngineConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 60
    max_faces: int = 1
    max_hands: int = 2
    detection_max_width: int = 960
    preview_max_width: int = 1280
    detection_interval: int = 1
    jpeg_quality: int = 85
    opencv_threads: int = max(1, os.cpu_count() or 4)
    face_model_path: Path = MODEL_DIR / "face_landmarker.task"
    hand_model_path: Path = MODEL_DIR / "hand_landmarker.task"
