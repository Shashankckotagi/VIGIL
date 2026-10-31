"""FastAPI server. Paste a URL, watch the timeline fill.

POST /api/start   {url, tau?, escalation?, provider?, bounded?, audio?}
POST /api/stop
GET  /api/status
WS   /ws          pushes every pipeline event to the browser

Run:  uvicorn server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vigil.config import Config
from vigil.pipeline import Pipeline

logging.basicConfig(
    level=os.environ.get("VIGIL_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("vigil.server")

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="VIGIL", version="0.1.0")


class StartRequest(BaseModel):
    url: str
    tau: float | None = None
    escalation: bool | None = None
    provider: str | None = None
    bounded: bool | None = None
    audio: bool = True
    gate: bool | None = None
    run_id: str | None = None
    api_key: str | None = None


class Hub:
    """Fans pipeline events out to every connected browser.

    Pipeline subscribers are plain callables invoked from the event loop, so
    they must not block. Each event is handed to asyncio and sent later.
    """

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.pipeline: Pipeline | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._recent: list[dict] = []

    def attach_loop(self) -> None:
        self._loop = asyncio.get_running_loop()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        for ev in self._recent[-40:]:
            try:
                await ws.send_json(ev)
            except Exception:
                break

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    def publish(self, event: dict) -> None:
        # Frame events are high-rate; keep other observable action events in recent history
        if event.get("type") != "frame":
            self._recent.append(event)
            self._recent = self._recent[-200:]
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._send(event), self._loop)

    async def _send(self, event: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


hub = Hub()


@app.on_event("startup")
async def _startup() -> None:
    hub.attach_loop()
    os.makedirs("logs", exist_ok=True)


@app.get("/")
async def index() -> Any:
    path = os.path.join(STATIC, "index.html")
    if not os.path.exists(path):
        return JSONResponse({"error": "static/index.html missing"}, 500)
    return FileResponse(path)


@app.post("/api/start")
async def start(req: StartRequest) -> Any:
    if hub.pipeline and hub.pipeline._running:
        log.info("stopping previous pipeline before starting new one")
        await hub.pipeline.stop()
        await asyncio.sleep(0.3)

    cfg = Config.from_env()
    if req.tau is not None:
        cfg.ssim_tau = float(req.tau)
    if req.escalation is not None:
        cfg.escalation_enabled = bool(req.escalation)
    if req.provider:
        cfg.provider = req.provider
    if req.bounded is not None:
        cfg.memory_bounded = bool(req.bounded)
    if req.gate is not None:
        cfg.gate_enabled = bool(req.gate)
    if req.api_key:
        cfg.gemini_api_key = req.api_key.strip()
    cfg.run_id = req.run_id or f"ui-{int(time.time())}"

    try:
        pipeline = Pipeline(req.url, cfg, with_audio=req.audio)
    except Exception as e:
        log.exception("failed to construct pipeline")
        return JSONResponse({"error": str(e)}, 400)

    pipeline.subscribe(hub.publish)
    hub.pipeline = pipeline
    try:
        await pipeline.start()
    except Exception as e:
        log.exception("failed to start")
        return JSONResponse({"error": str(e)}, 500)
    return {"ok": True, "run_id": cfg.run_id, "status": pipeline.status()}


@app.get("/api/feed")
async def video_feed() -> Any:
    """Stream live captured video frames as MJPEG directly to the browser."""
    import cv2

    async def frame_generator():
        while True:
            frame = None
            if hub.pipeline and hub.pipeline.source:
                frame = getattr(hub.pipeline.source, "latest_frame", None)
            if frame is not None:
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            await asyncio.sleep(0.06)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> Any:
    """Accept video file upload and return its local server path."""
    upload_dir = os.path.join(HERE, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    clean_filename = f"{int(time.time())}_{file.filename}"
    dest_path = os.path.join(upload_dir, clean_filename)
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)
    return {"ok": True, "path": dest_path, "filename": file.filename}


@app.post("/api/stop")
async def stop() -> Any:
    if not hub.pipeline:
        return JSONResponse({"error": "nothing running"}, 400)
    await hub.pipeline.stop()
    return {"ok": True, "status": hub.pipeline.status()}


@app.get("/api/status")
async def status() -> Any:
    if not hub.pipeline:
        return {"running": False}
    return hub.pipeline.status()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keepalive; client sends pings
    except WebSocketDisconnect:
        hub.disconnect(ws)
    except Exception:
        hub.disconnect(ws)


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
