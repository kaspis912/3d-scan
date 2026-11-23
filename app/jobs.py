"""SQLite-backed job store + a single background worker.

Reconstruction saturates every CPU core, so jobs run one at a time; the rest wait in the queue.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config
from .pipeline import PipelineCancelled, PipelineError, Reconstruction

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,          -- queued | running | done | failed | cancelled
    stage TEXT NOT NULL DEFAULT '',
    progress REAL NOT NULL DEFAULT 0,
    quality TEXT NOT NULL,
    mask INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL,          -- upload | inbox | cli
    error TEXT,
    stats TEXT NOT NULL DEFAULT '{}',
    size TEXT,                     -- JSON: known real dimension {"dimension": "height", "mm": 95}
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class JobStore:
    def __init__(self, db_path: Path = config.DB_PATH):
        config.ensure_dirs()
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(jobs)")}
        if "size" not in cols:  # databases created before real-size support
            self._db.execute("ALTER TABLE jobs ADD COLUMN size TEXT")
        # Jobs interrupted by a restart go back into the queue.
        self._db.execute("UPDATE jobs SET status='queued', stage='Re-queued after restart' WHERE status='running'")
        self._db.commit()

    def job_dir(self, job_id: str) -> Path:
        return config.JOBS_DIR / job_id

    def create(self, name: str, quality: str, mask: bool, source: str, size: dict | None = None) -> str:
        job_id = uuid.uuid4().hex[:10]
        now = time.time()
        self.job_dir(job_id).joinpath("input").mkdir(parents=True)
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, name, status, stage, quality, mask, source, size, created_at, updated_at) "
                "VALUES (?, ?, 'new', 'Receiving files', ?, ?, ?, ?, ?, ?)",
                (job_id, name, quality, int(mask), source, json.dumps(size) if size else None, now, now))
            self._db.commit()
        return job_id

    def update(self, job_id: str, **fields) -> None:
        if "stats" in fields:
            fields["stats"] = json.dumps(fields["stats"])
        if "size" in fields:
            fields["size"] = json.dumps(fields["size"]) if fields["size"] else None
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
            self._db.commit()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._row(r) for r in rows]

    def next_queued(self) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        return self._row(row) if row else None

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self._db.commit()
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def _row(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["stats"] = json.loads(d["stats"] or "{}")
        d["mask"] = bool(d["mask"])
        d["size"] = json.loads(d["size"]) if d.get("size") else None
        d["can_rescale"] = (self.job_dir(d["id"]) / "base" / "meta.json").exists()
        out = self.job_dir(d["id"]) / "output"
        d["files"] = sorted(p.name for p in out.iterdir()) if out.exists() else []
        d["has_thumb"] = (self.job_dir(d["id"]) / "thumb.jpg").exists()
        return d


class Worker(threading.Thread):
    def __init__(self, store: JobStore):
        super().__init__(daemon=True, name="scan3d-worker")
        self.store = store
        self.current: str | None = None
        self._cancel = threading.Event()
        self._wake = threading.Event()
        self._halt = threading.Event()

    def notify(self) -> None:
        self._wake.set()

    def cancel(self, job_id: str) -> bool:
        if self.current == job_id:
            self._cancel.set()
            return True
        return False

    def stop(self) -> None:
        self._halt.set()
        self._cancel.set()
        self._wake.set()

    def run(self) -> None:
        while not self._halt.is_set():
            job = self.store.next_queued()
            if job is None:
                self._wake.wait(2)
                self._wake.clear()
                continue
            self._process(job)

    def _process(self, job: dict) -> None:
        job_id = job["id"]
        self.current = job_id
        self._cancel.clear()
        self.store.update(job_id, status="running", stage="Starting", progress=0, error=None)

        def progress(stage: str, pct: float) -> None:
            self.store.update(job_id, stage=stage, progress=pct)

        rec = Reconstruction(self.store.job_dir(job_id), job["quality"], mask=job["mask"],
                             size=job["size"], on_progress=progress, cancel=self._cancel)
        try:
            stats = rec.execute()
            self.store.update(job_id, status="done", stage="Done", progress=100, stats=stats)
            if job["source"] == "inbox":  # drop the result next to the inbox for folder-based use
                glb = self.store.job_dir(job_id) / "output" / "model.glb"
                shutil.copy2(glb, config.OUTBOX_DIR / f"{job['name']}.glb")
        except PipelineCancelled:
            self.store.update(job_id, status="cancelled", stage="Cancelled", stats=rec.stats)
        except PipelineError as e:
            rec.log(f"ERROR: {e}")
            self.store.update(job_id, status="failed", stage="Failed", error=str(e), stats=rec.stats)
        except Exception as e:  # unexpected bug: keep the service alive, record the traceback
            rec.log(traceback.format_exc())
            self.store.update(job_id, status="failed", stage="Failed", error=f"Internal error: {e}",
                              stats=rec.stats)
        finally:
            self.current = None
