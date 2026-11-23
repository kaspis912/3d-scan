"""Is this machine ready to scan?

    uv run scan3d-doctor              check the tools, Python packages, GPU and folders
    uv run scan3d-doctor --selftest   also run a complete scan of the bundled test photos (a
                                      60 x 40 x 90 mm box on the scan mat) and compare the result
                                      with the true box

Exit code 0 when everything required works. The self-test is the proof that an installation
really works end to end: every tool runs, the scale comes from the mat, and the print solid matches.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import config

SELFTEST_PHOTOS = config.ROOT / "selftest" / "box"
SELFTEST_BOX_MM = (60.0, 40.0, 90.0)  # width, depth, height of the box in the test photos
COLMAP_COMMANDS = ("feature_extractor", "sequential_matcher", "exhaustive_matcher", "mapper",
                   "image_undistorter", "model_converter", "poisson_mesher")
OPENMVS_TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh")
DLL_NOT_FOUND = 0xC0000135  # Windows exit code when a program's DLL is missing


class Report:
    def __init__(self) -> None:
        self.failed = 0
        self.warned = 0

    def ok(self, what: str, detail: str = "") -> None:
        print(f"  [ok]   {what}" + (f": {detail}" if detail else ""))

    def warn(self, what: str, detail: str) -> None:
        self.warned += 1
        print(f"  [warn] {what}: {detail}")

    def fail(self, what: str, detail: str) -> None:
        self.failed += 1
        print(f"  [FAIL] {what}: {detail}")


def _starts(binary: str, *args: str) -> tuple[bool, str]:
    """Does the program start at all (all its DLLs / libraries load)? -> (yes, first output line)."""
    try:
        with tempfile.TemporaryDirectory() as scratch:  # OpenMVS writes a log file wherever it starts
            out = subprocess.run([binary, *args], capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", stdin=subprocess.DEVNULL, env=config.tool_env(binary),
                                 timeout=60, cwd=scratch,
                                 creationflags=subprocess.CREATE_NO_WINDOW if config.IS_WINDOWS else 0)
    except FileNotFoundError:
        return False, "not found"
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if out.returncode & 0xFFFFFFFF == DLL_NOT_FOUND:
        return False, "a required DLL is missing (install the Microsoft Visual C++ Redistributable)"
    text = (out.stdout + out.stderr).strip()
    if out.returncode < 0 or out.returncode > 255:
        return False, f"crashed on start (exit code {out.returncode})"
    return True, text.splitlines()[0] if text else ""


def check(rep: Report) -> None:
    print(f"scan3d doctor  -  {platform.platform()}  -  Python {platform.python_version()}")
    print(f"CPU threads: {config.THREADS}    tools folder: {config.TOOLS_DIR}")

    print("\nFolders")
    for label, path in (("project", config.ROOT), ("data", config.DATA_DIR)):
        if config.IS_WINDOWS and not str(path).isascii():
            rep.warn(label, f"{path} has non-English characters; COLMAP and OpenMVS can fail on such "
                            "paths on Windows. Move the project to a plain path like C:\\scan3d")
        else:
            rep.ok(label, str(path))
    try:
        config.ensure_dirs()
        probe = config.DATA_DIR / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        free_gb = shutil.disk_usage(config.DATA_DIR).free / 1e9
        (rep.ok if free_gb > 10 else rep.warn)("disk space", f"{free_gb:.0f} GB free (a scan needs a few GB while it runs)")
    except OSError as e:
        rep.fail("data folder", f"cannot write to {config.DATA_DIR}: {e}")

    print("\nNative tools")
    caps = config.colmap_caps()
    if not caps["runs"]:
        rep.fail("COLMAP", f"{config.COLMAP_BIN} does not run. {config.INSTALL_HINT}")
    else:
        rep.ok("COLMAP", f"{caps['version'] or 'version unknown'}  ({config.COLMAP_BIN})")
        missing = [c for c in COLMAP_COMMANDS if c not in caps["commands"]]
        if missing:
            rep.fail("COLMAP commands", "missing " + ", ".join(missing))
        if "global_mapper" not in caps["commands"]:
            rep.warn("COLMAP", "no global_mapper (older than 4.0): the incremental mapper is used instead, slower")
        if not caps["new_options"]:
            rep.ok("COLMAP options", "pre-4.0 option names (handled)")
    for tool in OPENMVS_TOOLS:
        path = config.openmvs(tool)
        if not Path(path).exists():
            (rep.warn if tool == "RefineMesh" else rep.fail)(f"OpenMVS {tool}", f"not found at {path}. {config.INSTALL_HINT}")
            continue
        started, line = _starts(path, "-h")
        (rep.ok if started else rep.fail)(f"OpenMVS {tool}", line if not started else path)
    for name, binary in (("FFmpeg", config.FFMPEG_BIN), ("ffprobe", config.FFPROBE_BIN)):
        started, line = _starts(binary, "-version")
        (rep.ok if started else rep.fail)(name, line[:70] if started else f"{line}. {config.INSTALL_HINT}")

    print("\nPython packages")
    for mod in ("cv2", "trimesh", "manifold3d", "skimage", "scipy", "fastapi", "pillow_heif"):
        try:
            __import__(mod)
            rep.ok(mod)
        except Exception as e:
            rep.fail(mod, f"{e} (run: uv sync --extra masking --extra silhouette)")
    try:
        import rembg  # noqa: F401
        rep.ok("rembg", "background removal for turntable scans")
    except Exception as e:
        rep.warn("rembg", f"not available, background removal is off ({str(e).splitlines()[0][:80]})")
    try:
        import torch

        from .pipeline import silhouette
        backend = silhouette._backend()
        if backend is None:
            raise ImportError("neither transformers (with SAM 2) nor sam2 is installed")
        device = silhouette.torch_device()
        note = {"cpu": "on the CPU: works, about 1 s per frame", "mps": "on the Apple GPU",
                "cuda": f"on the GPU ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cuda'})"}
        rep.ok("SAM 2 (shape from outlines)", f"{backend}, torch {torch.__version__}, {note.get(device, device)}")
    except Exception as e:
        rep.warn("SAM 2", "not available, black / shiny objects will scan worse "
                          f"({str(e).splitlines()[0][:100]})")

    print("\nSettings")
    rep.ok("feature matching", "GPU first (OpenGL), CPU fallback" if config.SIFT_GPU
           else "CPU (set SCAN3D_SIFT_GPU=1 to try the GPU)")
    rep.ok("dense reconstruction", "CPU (OpenMVS uses a GPU only through NVIDIA CUDA)")


def selftest(rep: Report, quality: str, keep: Path | None) -> None:
    import trimesh

    from . import accuracy
    from .pipeline import PipelineError, Reconstruction

    print(f"\nSelf-test: full scan of {SELFTEST_PHOTOS.name} test photos at '{quality}' quality")
    photos = sorted(SELFTEST_PHOTOS.glob("*.jpg"))
    if len(photos) < 20:
        rep.fail("self-test", f"test photos not found in {SELFTEST_PHOTOS}")
        return
    job = Path(tempfile.mkdtemp(prefix="scan3d_selftest_", dir=config.DATA_DIR))
    (job / "input").mkdir()
    for p in photos:
        shutil.copy2(p, job / "input" / p.name)
    last = [""]

    def progress(stage: str, pct: float) -> None:
        if stage != last[0]:
            last[0] = stage
            print(f"  [{pct:3.0f}%] {stage}", flush=True)

    t0 = time.time()
    rec = Reconstruction(job, quality, on_progress=progress)
    try:
        stats = rec.execute()
    except PipelineError as e:
        rep.fail("self-test", f"{e}\n         full log: {rec.log_path}")
        return
    except Exception as e:
        rep.fail("self-test", f"unexpected error: {e}\n         full log: {rec.log_path}")
        return
    took = time.time() - t0

    reg, n = stats.get("registered_images", 0), stats.get("num_images", 0)
    (rep.ok if reg >= 0.8 * n else rep.fail)("camera poses", f"{reg}/{n} photos aligned")
    if stats.get("scale_source") == "scan_mat":
        rep.ok("scan mat", f"scale measured, fit +-{stats['scan_mat']['rms_mm']} mm")
    else:
        rep.fail("scan mat", "not detected in the test photos, so the result has no real size")
    pr = stats.get("print", {})
    (rep.ok if pr.get("printable") else rep.fail)("print solid", "closed solid with a flat base" if pr.get("printable")
                                                 else "the STL is not a closed solid")
    sil = stats.get("silhouette")
    if sil:
        rep.ok("outlines (SAM 2)", f"{sil['frames']} frames segmented"
               + (", scan complete so they only trim" if sil.get("mode") == "trim" else ""))
    else:
        rep.warn("shape from outlines", "did not run (see the log); scans still work without it")
    if stats.get("scale_source"):
        w, d, h = SELFTEST_BOX_MM
        box = trimesh.creation.box(extents=[w, d, h])
        box.apply_translation([0, 0, h / 2])
        try:
            acc, _ = accuracy.compare(accuracy.load_mesh(job / "output" / "model.stl"), box, heatmaps=False)
            dims = stats["dimensions"]
            line = (f"{dims['width']} x {dims['depth']} x {dims['height']} mm vs true {w:g} x {d:g} x {h:g}; "
                    f"95% of the surface within {acc['p95_mm']} mm, typical {acc['median_mm']} mm, "
                    f"scale error {acc['fit']['scale_error_pct']:+}%")
            good = acc["p95_mm"] < 2.5 and abs(acc["fit"]["scale_error_pct"]) < 2
            (rep.ok if good else rep.fail)("accuracy", line)
        except ValueError as e:
            rep.fail("accuracy", str(e))
    print(f"  took {took / 60:.1f} min")
    if keep:
        shutil.copytree(job, keep, dirs_exist_ok=True)
        print(f"  job files kept in {keep}")
    elif rep.failed:
        print(f"  job files kept for inspection in {job}")
        return
    shutil.rmtree(job, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="scan3d-doctor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run a complete scan of the bundled test photos")
    ap.add_argument("--quality", choices=list(config.QUALITIES), default="preview",
                    help="quality preset for the self-test (default: preview, the fastest)")
    ap.add_argument("--keep", type=Path, help="keep the self-test job folder (log, models) here")
    args = ap.parse_args()
    config.quiet_crashes()
    rep = Report()
    check(rep)
    if args.selftest:
        if rep.failed:
            print("\nSelf-test skipped: fix the failed checks above first.")
        else:
            selftest(rep, args.quality, args.keep)
    print()
    if rep.failed:
        print(f"{rep.failed} problem(s) found." + (f" {rep.warned} warning(s)." if rep.warned else ""))
        sys.exit(1)
    print("All good." + (f" {rep.warned} warning(s), see above." if rep.warned else "")
          + ("" if args.selftest else "  Run with --selftest for a complete test scan."))


if __name__ == "__main__":
    main()
