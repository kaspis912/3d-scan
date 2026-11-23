"""Paths, binaries and quality presets. Everything can be overridden with env vars."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SCAN3D_DATA", ROOT / "data"))
INBOX_DIR = Path(os.environ.get("SCAN3D_INBOX", DATA_DIR / "inbox"))
OUTBOX_DIR = Path(os.environ.get("SCAN3D_OUTBOX", DATA_DIR / "outbox"))
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "jobs.sqlite3"

IS_WINDOWS = os.name == "nt"
EXE = ".exe" if IS_WINDOWS else ""
# Where the install script puts the native tools (COLMAP, OpenMVS, FFmpeg on Windows; OpenMVS on macOS).
TOOLS_DIR = Path(os.environ.get("SCAN3D_TOOLS") or (
    Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "scan3d" if IS_WINDOWS
    else Path.home() / ".local/share/scan3d"))


def _find_tool(name: str, env_var: str | None = None, folder: str | None = None) -> str:
    """A tool's executable: the env var if set, else the one on PATH, else the one the install
    script unpacked under TOOLS_DIR/<folder> (searched, so the archive's layout does not matter)."""
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    found = shutil.which(name)
    if found and not found.lower().endswith((".bat", ".cmd")):  # a wrapper script cannot be cancelled cleanly
        return found
    root = TOOLS_DIR / (folder or name)
    if root.is_dir():
        hits = sorted(root.rglob(name + EXE), key=lambda p: len(p.parts))
        if hits:
            return str(hits[0])
    return found or name


COLMAP_BIN = _find_tool("colmap", "COLMAP_BIN")
FFMPEG_BIN = _find_tool("ffmpeg", "FFMPEG_BIN")
FFPROBE_BIN = _find_tool("ffprobe", "FFPROBE_BIN", folder="ffmpeg")
if os.environ.get("OPENMVS_BIN_DIR"):
    OPENMVS_DIR = Path(os.environ["OPENMVS_BIN_DIR"])
elif IS_WINDOWS:
    OPENMVS_DIR = Path(_find_tool("DensifyPointCloud", folder="openmvs")).parent
else:
    OPENMVS_DIR = TOOLS_DIR / "openmvs-install/bin/OpenMVS"
INSTALL_HINT = ("Run scripts\\install_windows.bat" if IS_WINDOWS else "Run ./scripts/install_macos.sh")
# COLMAP's GPU feature matching runs on OpenGL, so it also works on AMD and Intel GPUs. Off by
# default (the CPU path behaves the same everywhere); SCAN3D_SIFT_GPU=1 tries the GPU first and
# falls back to the CPU if that fails.
SIFT_GPU = os.environ.get("SCAN3D_SIFT_GPU", "0") == "1"

# Seconds a folder in the inbox must stay unchanged before it is picked up.
INBOX_SETTLE_SECONDS = float(os.environ.get("SCAN3D_INBOX_SETTLE", "8"))
THREADS = os.cpu_count() or 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class Quality:
    name: str
    max_image_size: int  # longest side of images fed to COLMAP / OpenMVS
    video_frames: int  # how many frames to keep from a video
    densify_resolution_level: int  # 0 = full res, 1 = half, 2 = quarter
    densify_max_resolution: int
    refine_mesh: bool
    max_faces: int  # mesh decimation target before texturing
    texture_size: int  # max texture atlas side
    poisson_depth: int  # octree depth of the watertight print solid (detail vs. noise)
    solid_resolution: int  # voxels along the longest side of the print solid


QUALITIES: dict[str, Quality] = {
    "preview": Quality("preview", 1200, 50, 2, 1000, False, 60_000, 2048, 8, 200),
    "balanced": Quality("balanced", 2000, 90, 0, 2000, False, 200_000, 4096, 9, 320),
    "high": Quality("high", 3200, 150, 0, 2600, True, 500_000, 8192, 10, 448),
}
DEFAULT_QUALITY = "balanced"


def openmvs(tool: str) -> str:
    return str(OPENMVS_DIR / (tool + EXE))


def tool_env(binary: str) -> dict[str, str]:
    """Environment for running a native tool. The Windows builds of COLMAP and OpenMVS keep their
    DLLs (and COLMAP its Qt plugins) next to or near the executable, which the launcher .bat
    normally puts on PATH; we call the .exe directly so a cancelled job really stops."""
    env = os.environ.copy()
    if not IS_WINDOWS:
        return env
    exe_dir = Path(binary).resolve().parent
    root = exe_dir.parent
    extra = list(dict.fromkeys(str(d) for d in (exe_dir, root / "lib", root / "bin") if d.is_dir()))
    env["PATH"] = os.pathsep.join([*extra, env.get("PATH", "")])
    for plugins in (root / "lib" / "plugins", root / "plugins", exe_dir / "plugins"):
        if plugins.is_dir():
            env["QT_PLUGIN_PATH"] = str(plugins)
            break
    return env


def quiet_crashes() -> None:
    """Windows: a crashing child process must fail with an exit code, not wait on an error dialog
    (the pipeline retries some steps, e.g. Poisson, when they crash). Child processes inherit this."""
    if IS_WINDOWS:
        import ctypes

        SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SEM_NOOPENFILEERRORBOX = 0x0001, 0x0002, 0x8000
        ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)


def ensure_dirs() -> None:
    for d in (DATA_DIR, INBOX_DIR, OUTBOX_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def imread_gray(path: Path):
    """cv2.imread that also works when the path has non-English characters on Windows
    (OpenCV opens files through the legacy code page there). Returns None if unreadable."""
    import cv2
    import numpy as np

    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except (OSError, cv2.error):
        return None


_colmap_caps: dict | None = None


def colmap_caps() -> dict:
    """What the installed COLMAP offers, asked once: its version line, its commands, and whether
    it uses the 4.x option names (--FeatureExtraction.* / --FeatureMatching.*) or the older
    --SiftExtraction.* / --SiftMatching.*. Lets the same pipeline drive either generation."""
    global _colmap_caps
    if _colmap_caps is None:
        import subprocess

        def ask(*args: str) -> str:
            try:
                out = subprocess.run([COLMAP_BIN, *args], capture_output=True, text=True, encoding="utf-8",
                                     errors="replace", stdin=subprocess.DEVNULL, env=tool_env(COLMAP_BIN),
                                     timeout=60, creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0)
                return out.stdout + out.stderr
            except (OSError, subprocess.SubprocessError):
                return ""

        general = ask("help")
        extractor = ask("feature_extractor", "-h")
        _colmap_caps = {
            "version": next((l.strip() for l in general.splitlines() if l.strip().startswith("COLMAP")), ""),
            "commands": {w for l in general.splitlines() for w in l.split()[:1]},
            "new_options": "--FeatureExtraction.use_gpu" in extractor,
            "runs": bool(general),
        }
    return _colmap_caps
