"""ローカル専用の Web サーバー。127.0.0.1 でのみ待ち受ける。

ブラウザから受け取ったコピーを一時フォルダで処理するため、元の音源には触れない。
"""
import argparse
import json
import shutil
import socket
import tempfile
import threading
import urllib.request
import uuid
import webbrowser
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import dsp

HOST = "127.0.0.1"
PORT = 8765
APP_ID = "mastering-tool"
WORK_DIR = Path(tempfile.gettempdir()) / "mastering_tool_work"
STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".wav", ".aif", ".aiff"}
MAX_UPLOAD_BYTES = 4 * 1024 ** 3


@dataclass
class Track:
    id: str
    name: str
    path: Path
    info: dict
    preview: Path | None = None
    result_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Result:
    id: str
    path: Path
    filename: str


@dataclass
class Job:
    id: str
    state: str = "running"
    progress: float = 0.0
    message: str = "待機中"
    error: str | None = None
    results: list = field(default_factory=list)
    response: list = field(default_factory=list)
    target_lufs: float | None = None


tracks: dict[str, Track] = {}
results: dict[str, Result] = {}
jobs: dict[str, Job] = {}
store_lock = threading.Lock()
# 重い処理は1件ずつ順番に行う
worker = ThreadPoolExecutor(max_workers=1)


def new_id():
    return uuid.uuid4().hex


def reset_work_dir():
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_app):
    reset_work_dir()
    yield
    worker.shutdown(wait=False, cancel_futures=True)
    shutil.rmtree(WORK_DIR, ignore_errors=True)


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[HOST, "localhost"])


class MasterRequest(BaseModel):
    track_id: str
    hpf: bool = True
    hpf_freq: float = Field(30.0, ge=20.0, le=120.0)
    tone: bool = True
    low_db: float = Field(0.0, ge=-6.0, le=6.0)
    mid_db: float = Field(0.0, ge=-6.0, le=6.0)
    high_db: float = Field(0.0, ge=-6.0, le=6.0)
    glue: bool = True
    glue_db: float = Field(2.0, ge=0.0, le=6.0)
    release: bool = True
    target_lufs: float = Field(-12.0, ge=-24.0, le=-6.0)
    format: Literal["cd", "hires"] = "cd"


class AlignRequest(BaseModel):
    track_ids: list[str] = Field(min_length=2, max_length=6)
    mode: Literal["average", "fixed"] = "average"
    target_lufs: float = Field(-14.0, ge=-30.0, le=-6.0)
    format: Literal["cd", "hires"] = "cd"


@app.exception_handler(dsp.AudioError)
async def audio_error(_request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def get_track(track_id):
    track = tracks.get(track_id)
    if track is None:
        raise HTTPException(404, "ファイルが見つかりません。もう一度ドロップしてください。")
    return track


def source_stats(track):
    return {"lufs": track.info["lufs"], "tp": track.info["tp"]}


def track_payload(track):
    return {"id": track.id, "name": track.name, "output_name": dsp.output_name(track.name), **track.info}


def store_result(track, path, filename, processed):
    """結果を登録し、同じ曲の古い結果は削除する。"""
    result = Result(new_id(), path, filename)
    with store_lock:
        old = results.pop(track.result_id, None) if track.result_id else None
        results[result.id] = result
        track.result_id = result.id
    if old is not None:
        old.path.unlink(missing_ok=True)
    return {
        "result_id": result.id,
        "track_id": track.id,
        "name": track.name,
        "filename": filename,
        "url": f"/api/results/{result.id}/audio",
        "download_url": f"/api/results/{result.id}/audio?download=1",
        "source_url": f"/api/tracks/{track.id}/preview.wav",
        "source": processed.source,
        "output": processed.output,
        "glue_db": processed.glue_db,
        "reduction_db": processed.reduction_db,
        "warnings": processed.warnings,
        "wave_before": track.info["waveform"],
        "wave_after": processed.waveform,
    }


def run_job(job, task):
    try:
        task(job)
        job.progress, job.message, job.state = 1.0, "完了", "done"
    except dsp.AudioError as exc:
        job.state, job.error = "error", str(exc)
    except MemoryError:
        job.state, job.error = "error", "メモリが足りません。他のアプリを閉じてもう一度お試しください。"
    except Exception as exc:  # 画面に理由を返すため、ここで全て受け止める
        job.state, job.error = "error", f"処理に失敗しました: {exc}"


def start_job(task, job=None):
    job = job or Job(new_id())
    with store_lock:
        jobs[job.id] = job
    worker.submit(run_job, job, task)
    return {"job_id": job.id}


def set_progress(job, fraction, message):
    job.progress, job.message = fraction, message


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"app": APP_ID}


@app.post("/api/upload")
async def upload(request: Request, name: str):
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "WAV または AIFF のファイルをドロップしてください。")
    track_id = new_id()
    path = WORK_DIR / f"{track_id}{Path(name).suffix.lower()}"
    size = 0
    accepted = False
    try:
        with path.open("wb") as file:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(400, "ファイルが大きすぎます（上限 4 GB）。")
                file.write(chunk)
        info = await run_in_threadpool(dsp.probe, path)
        accepted = True
    finally:
        if not accepted:
            path.unlink(missing_ok=True)
    track = Track(track_id, Path(name.replace("\\", "/")).name, path, info)
    with store_lock:
        tracks[track_id] = track
    return track_payload(track)


@app.delete("/api/tracks/{track_id}")
def delete_track(track_id: str):
    with store_lock:
        track = tracks.pop(track_id, None)
        result = results.pop(track.result_id, None) if track and track.result_id else None
    if track:
        track.path.unlink(missing_ok=True)
        if track.preview:
            track.preview.unlink(missing_ok=True)
    if result:
        result.path.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/tracks/{track_id}/preview.wav")
def track_preview(track_id: str):
    track = get_track(track_id)
    with track.lock:
        if track.preview is None:
            preview = WORK_DIR / f"{track.id}_preview.wav"
            x, rate = dsp.read_audio(track.path)
            dsp.write_preview(preview, x, rate)
            track.preview = preview
    return FileResponse(track.preview, media_type="audio/wav")


@app.get("/api/results/{result_id}/audio")
def result_audio(result_id: str, download: bool = False):
    result = results.get(result_id)
    if result is None or not result.path.exists():
        raise HTTPException(404, "処理結果が見つかりません。もう一度処理してください。")
    if download:
        return FileResponse(result.path, media_type="audio/wav", filename=result.filename)
    return FileResponse(result.path, media_type="audio/wav")


@app.post("/api/master")
def master(request: MasterRequest):
    track = get_track(request.track_id)
    options = dsp.MasterOptions(**request.model_dump(exclude={"track_id"}))

    def task(job):
        path = WORK_DIR / f"{new_id()}.wav"
        processed = dsp.master_file(track.path, path, options, lambda f, m: set_progress(job, f, m), source_stats(track))
        job.response = [store_result(track, path, dsp.output_name(track.name), processed)]

    return start_job(task)


def unique_names(names):
    """同じ曲名が並んでも出力名が重ならないようにする。"""
    seen, unique = {}, []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 1:
            unique.append(name)
        else:
            unique.append(f"{Path(name).stem}_{seen[name]}.wav")
    return unique


@app.post("/api/align")
def align(request: AlignRequest):
    selected = [get_track(track_id) for track_id in request.track_ids]
    target = dsp.align_target([t.info["lufs"] for t in selected], request.mode, request.target_lufs)
    names = unique_names([dsp.output_name(t.name) for t in selected])

    def task(job):
        job.target_lufs = target
        count = len(selected)
        for index, (track, filename) in enumerate(zip(selected, names)):
            def report(fraction, message, index=index, name=track.name):
                set_progress(job, (index + fraction) / count, f"{index + 1}/{count} 曲目 {name}: {message}")

            path = WORK_DIR / f"{new_id()}.wav"
            processed = dsp.align_file(track.path, path, target, request.format, report, source_stats(track))
            job.response.append(store_result(track, path, filename, processed))

    return start_job(task)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "処理が見つかりません。")
    payload = {"state": job.state, "progress": job.progress, "message": job.message, "error": job.error}
    if job.state == "done":
        payload["results"] = job.response
        payload["target_lufs"] = job.target_lufs
        if len(job.response) > 1:
            payload["zip_url"] = f"/api/jobs/{job.id}/zip"
    return payload


@app.get("/api/jobs/{job_id}/zip")
def job_zip(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.state != "done":
        raise HTTPException(404, "処理結果が見つかりません。")
    archive = WORK_DIR / f"{job.id}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:
        for item in job.response:
            result = results.get(item["result_id"])
            if result is None:
                archive.unlink(missing_ok=True)
                raise HTTPException(404, "処理結果が古くなっています。もう一度処理してください。")
            bundle.write(result.path, result.filename)
    return FileResponse(archive, media_type="application/zip", filename="mastering_album.zip")


def existing_instance():
    """同じポートの状態を返す。'ours' / 'other' / None。"""
    with socket.socket() as probe_socket:
        probe_socket.settimeout(0.3)
        if probe_socket.connect_ex((HOST, PORT)) != 0:
            return None
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/api/health", timeout=1) as response:
            return "ours" if json.load(response).get("app") == APP_ID else "other"
    except (OSError, ValueError):
        return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://{HOST}:{PORT}/"

    state = existing_instance()
    if state == "ours":
        print(f"すでに起動しています: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return
    if state == "other":
        raise SystemExit(f"ポート {PORT} は別のアプリが使用しています。そのアプリを終了してください。")

    print(f"マスタリングツールを起動しました: {url}")
    print("終了するには Ctrl+C を押してください。作業用の一時ファイルも削除されます。")
    print("ウィンドウを閉じた場合、一時ファイルは次回の起動時に削除されます。")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
