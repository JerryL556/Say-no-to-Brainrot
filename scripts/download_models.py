from pathlib import Path
from urllib.request import urlretrieve


MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def main() -> None:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in MODEL_URLS.items():
        destination = models_dir / filename
        if destination.exists():
            print(f"Skipping {filename}; already present.")
            continue
        print(f"Downloading {filename}...")
        urlretrieve(url, destination)
        print(f"Saved to {destination}")


if __name__ == "__main__":
    main()

