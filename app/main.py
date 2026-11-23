"""HTTP API + web UI.  Run:  uv run uvicorn app.main:app --port 8000"""
from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path

import trimesh
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import accuracy, config, refpiece
from .jobs import JobStore, Worker
from .pipeline import export
from .sizes import parse_size
from . import scanmat
from .watcher import InboxWatcher

STATIC = Path(__file__).parent / "static"
store = JobStore()
worker = Worker(store)
watcher = InboxWatcher(store, worker)


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.quiet_crashes()
    worker.start()
    watcher.start()
    yield
    watcher.stop()
    worker.stop()


app = FastAPI(title="scan3d", lifespan=lifespan)


def _safe_name(name: str) -> str:
    name = Path(name).name  # drop any directory part from folder uploads
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "file"


def _get(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/health")
def health():
    tools = {
        "colmap": Path(config.COLMAP_BIN).exists(),
        "openmvs": Path(config.openmvs("DensifyPointCloud")).exists(),
        "ffmpeg": Path(config.FFMPEG_BIN).exists(),
    }
    return {"ok": all(tools.values()), "tools": tools, "inbox": str(config.INBOX_DIR),
            "outbox": str(config.OUTBOX_DIR), "qualities": list(config.QUALITIES),
            "install_hint": config.INSTALL_HINT}


@app.post("/api/jobs", status_code=201)
async def create_job(files: list[UploadFile] = File(...), name: str = Form(""),
                     quality: str = Form(config.DEFAULT_QUALITY), mask: bool = Form(False),
                     size_mm: str = Form(""), size_dimension: str = Form("longest")):
    if quality not in config.QUALITIES:
        raise HTTPException(400, f"quality must be one of {list(config.QUALITIES)}")
    try:
        size = parse_size(size_mm, size_dimension)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    allowed = config.IMAGE_EXTS | config.VIDEO_EXTS
    media = [f for f in files if Path(f.filename or "").suffix.lower() in allowed]
    if not media:
        raise HTTPException(400, "No supported images or videos in upload")
    name = name.strip() or Path(media[0].filename or "scan").stem
    job_id = store.create(name, quality, mask, "upload", size)
    dest = store.job_dir(job_id) / "input"
    for i, f in enumerate(media):
        with (dest / f"{i:04d}_{_safe_name(f.filename or 'file')}").open("wb") as out:
            while chunk := await f.read(1 << 20):
                out.write(chunk)
    store.update(job_id, status="queued", stage="Queued")
    worker.notify()
    return _get(job_id)


@app.get("/api/scan-mat")
def scan_mat(paper: str = "a4"):
    """Printable ArUco scan mat (PDF). Print at 100 % / actual size."""
    if paper not in scanmat.PAPERS:
        raise HTTPException(400, f"paper must be one of {list(scanmat.PAPERS)}")
    cache = config.DATA_DIR / f"scan_mat_{paper}.pdf"
    if not cache.exists():
        cache.write_bytes(scanmat.render_pdf(paper))
    return Response(cache.read_bytes(), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="scan3d_mat_{paper}.pdf"'})


@app.get("/api/jobs")
def list_jobs():
    return store.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _get(job_id)


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str, tail: int = 200):
    _get(job_id)
    log = store.job_dir(job_id) / "log.txt"
    if not log.exists():
        return ""
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:])


@app.get("/api/jobs/{job_id}/thumb")
def get_thumb(job_id: str):
    path = store.job_dir(job_id) / "thumb.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_file(job_id: str, filename: str):
    job = _get(job_id)
    if filename not in job["files"]:
        raise HTTPException(404, "File not found")
    path = store.job_dir(job_id) / "output" / filename
    download = f"{_safe_name(job['name'])}{Path(filename).suffix}" if filename.startswith("model") else filename
    return FileResponse(path, filename=download)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _get(job_id)
    if job["status"] == "queued":
        store.update(job_id, status="cancelled", stage="Cancelled")
    elif job["status"] == "running":
        worker.cancel(job_id)
    return _get(job_id)


@app.post("/api/jobs/{job_id}/scale")
def scale_job(job_id: str, dimension: str = Body("longest"), mm: float | None = Body(None)):
    """Set (or clear, with mm=null) the known real dimension and re-export instantly."""
    job = _get(job_id)
    if job["status"] in ("running", "queued"):
        raise HTTPException(409, "Wait until the job has finished")
    if not job["can_rescale"]:
        raise HTTPException(409, "This scan predates real-size support; re-run it once to enable sizing")
    try:
        size = parse_size(mm, dimension)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    job_dir = store.job_dir(job_id)
    stats = job["stats"] | export.write_outputs(job_dir / "base", job_dir / "output", size)
    store.update(job_id, size=size, stats=accuracy.update_stats(job_dir, stats, job))
    return _get(job_id)


@app.get("/api/test-piece")
def test_piece(format: str = "stl"):
    """The accuracy test piece: print it, scan the print, then check the scan against it."""
    if format not in ("stl", "3mf", "glb"):
        raise HTTPException(400, "format must be stl, 3mf or glb")
    piece = refpiece.build()
    if format == "glb":  # preview for the page's 3D viewer: Y-up, metres
        piece.apply_transform(accuracy.Z_UP_TO_Y_UP)
        piece.apply_scale(1 / 1000.0)
        piece.unmerge_vertices()
        piece.visual = trimesh.visual.ColorVisuals(piece, face_colors=[205, 205, 200, 255])
    return Response(piece.export(file_type=format), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="scan3d_test_piece.{format}"'})


@app.post("/api/jobs/{job_id}/accuracy")
async def check_accuracy(job_id: str, reference: UploadFile | None = File(None)):
    """Compare the scan with the model it was printed from: an uploaded STL/3MF/OBJ/PLY
    (millimetres, standing as it stood on the mat), or the built-in test piece when no file is sent."""
    job = _get(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "Wait until the job has finished")
    if not job["stats"].get("scale_source"):
        raise HTTPException(409, "The scan has no real size yet: scan on the mat or set a size first")
    job_dir = store.job_dir(job_id)
    try:
        if reference is not None and reference.filename:
            accuracy.set_reference(job_dir, reference.filename, await reference.read())
        else:
            accuracy.set_reference(job_dir)
        # Several seconds of CPU: off the event loop, so the page keeps updating meanwhile.
        report = await run_in_threadpool(accuracy.refresh, job_dir, job["stats"], job)
    except ValueError as e:
        accuracy.clear_reference(job_dir)
        store.update(job_id, stats={k: v for k, v in job["stats"].items() if k != "accuracy"})
        raise HTTPException(400, str(e)) from None
    store.update(job_id, stats=job["stats"] | {"accuracy": report})
    return _get(job_id)


@app.delete("/api/jobs/{job_id}/accuracy")
def remove_accuracy(job_id: str):
    job = _get(job_id)
    accuracy.clear_reference(store.job_dir(job_id))
    store.update(job_id, stats={k: v for k, v in job["stats"].items() if k != "accuracy"})
    return _get(job_id)


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str, quality: str | None = None, mask: bool | None = None):
    job = _get(job_id)
    if job["status"] in ("running", "queued"):
        raise HTTPException(409, "Job is already queued or running")
    fields = {"status": "queued", "stage": "Queued", "progress": 0, "error": None}
    if quality in config.QUALITIES:
        fields["quality"] = quality
    if mask is not None:
        fields["mask"] = int(mask)
    store.update(job_id, **fields)
    worker.notify()
    return _get(job_id)


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    job = _get(job_id)
    if job["status"] == "running":
        raise HTTPException(409, "Cancel the job before deleting it")
    store.delete(job_id)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
