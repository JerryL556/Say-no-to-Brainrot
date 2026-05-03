from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import EngineConfig, FRONTEND_DIR
from src.engine import ARFilterEngine
from src.state import AppState


class SettingsPatch(BaseModel):
    smile_hearts: bool | None = None
    mouth_fire: bool | None = None
    brow_flash: bool | None = None
    peace_sparkles: bool | None = None
    thumbs_burst: bool | None = None
    open_palm_aura: bool | None = None


config = EngineConfig()
state = AppState()
engine = ARFilterEngine(config=config, state=state)


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="AR Filter Webcam Prototype", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/state")
async def get_state() -> JSONResponse:
    return JSONResponse(state.snapshot())


@app.post("/api/settings")
async def update_settings(patch: SettingsPatch) -> JSONResponse:
    state.update_settings(patch.model_dump(exclude_none=True))
    return JSONResponse(state.snapshot())


@app.post("/api/virtual-camera/start")
async def start_virtual_camera() -> JSONResponse:
    state.request_virtual_camera(True)
    return JSONResponse(state.snapshot())


@app.post("/api/virtual-camera/stop")
async def stop_virtual_camera() -> JSONResponse:
    state.request_virtual_camera(False)
    return JSONResponse(state.snapshot())


def mjpeg_stream():
    last_frame_id = -1
    while True:
        frame_id, jpeg_bytes = state.get_frame()
        if jpeg_bytes and frame_id != last_frame_id:
            last_frame_id = frame_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        else:
            time.sleep(1 / 30)


@app.get("/stream.mjpg")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

