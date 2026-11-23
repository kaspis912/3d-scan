"""Watches the inbox folder and turns new content into jobs.

  inbox/
    my_mug/              <- a folder of photos and/or videos = one 3D model
    IMG_1.jpg IMG_2.jpg  <- loose photos in the inbox itself = one 3D model together
      IMG_0001.HEIC ...
      scan3d.json        <- optional: {"quality": "high", "mask": true,
                                       "size": {"dimension": "height", "mm": 95}}
    shoe.mov             <- a loose video = one 3D model

A folder or file is picked up once it has stopped changing for INBOX_SETTLE_SECONDS
(so half-copied AirDrop transfers are not processed early). It is then moved into the job.
Finished models are copied to the outbox as <name>.glb.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from . import config
from .sizes import parse_size
from .jobs import JobStore, Worker


def _signature(path: Path) -> tuple:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return tuple(sorted((str(p), p.stat().st_size, p.stat().st_mtime) for p in files))


def _has_media(path: Path) -> bool:
    exts = config.IMAGE_EXTS | config.VIDEO_EXTS
    if path.is_file():
        return path.suffix.lower() in config.VIDEO_EXTS
    return any(p.suffix.lower() in exts for p in path.rglob("*") if p.is_file())


class InboxWatcher(threading.Thread):
    def __init__(self, store: JobStore, worker: Worker, inbox: Path = config.INBOX_DIR):
        super().__init__(daemon=True, name="scan3d-inbox")
        self.store, self.worker, self.inbox = store, worker, inbox
        self._seen: dict[Path, tuple[tuple, float]] = {}
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.wait(3):
            try:
                self.scan()
            except Exception as e:  # never let the watcher die
                print(f"[inbox] scan error: {e}")

    def scan(self) -> None:
        now = time.time()
        present = set()
        candidates = [e for e in self.inbox.iterdir()
                      if not e.name.startswith((".", "_")) and _has_media(e)]
        # Photos dropped straight into the inbox (no subfolder) together form one scan.
        loose = sorted(e for e in self.inbox.iterdir()
                       if e.is_file() and not e.name.startswith(".") and e.suffix.lower() in config.IMAGE_EXTS)
        for entry in candidates + ([self.inbox] if loose else []):
            present.add(entry)
            sig = _signature_files(loose) if entry == self.inbox else _signature(entry)
            prev = self._seen.get(entry)
            if prev is None or prev[0] != sig:
                self._seen[entry] = (sig, now)
            elif now - prev[1] >= config.INBOX_SETTLE_SECONDS:
                if entry == self.inbox:
                    self._ingest_loose(loose)
                else:
                    self._ingest(entry)
                self._seen.pop(entry, None)
        for gone in set(self._seen) - present:
            self._seen.pop(gone, None)

    def _ingest_loose(self, files: list[Path]) -> None:
        name = time.strftime("scan_%Y%m%d_%H%M%S")
        opts = _read_settings(self.inbox / "scan3d.json")
        job_id = self._create(name, opts)
        dest = self.store.job_dir(job_id) / "input"
        for f in files:
            shutil.move(f, dest / f.name)
        self._queue(job_id, name, len(files))

    def _ingest(self, entry: Path) -> None:
        opts = _read_settings(entry / "scan3d.json") if entry.is_dir() else {}
        name = entry.stem if entry.is_file() else entry.name
        job_id = self._create(name, opts)
        dest = self.store.job_dir(job_id) / "input"
        if entry.is_file():
            shutil.move(entry, dest / entry.name)
        else:
            for child in entry.iterdir():
                shutil.move(child, dest / child.name)
            entry.rmdir()
        self._queue(job_id, name, len(list(dest.iterdir())))

    def _create(self, name: str, opts: dict) -> str:
        quality = opts.get("quality", config.DEFAULT_QUALITY)
        if quality not in config.QUALITIES:
            quality = config.DEFAULT_QUALITY
        try:
            size = parse_size(opts.get("size"))
        except ValueError as e:
            print(f"[inbox] ignoring size for '{name}': {e}")
            size = None
        return self.store.create(name, quality, bool(opts.get("mask", False)), "inbox", size)

    def _queue(self, job_id: str, name: str, n_files: int) -> None:
        self.store.update(job_id, status="queued", stage="Queued")
        self.worker.notify()
        print(f"[inbox] queued '{name}' ({n_files} files) as job {job_id}")


def _signature_files(files: list[Path]) -> tuple:
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime) for p in files)


def _read_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        print(f"[inbox] ignoring invalid {path}")
        return {}
