"""Process a folder (or video) straight from the terminal, without the web service.

    uv run scan3d ~/Pictures/mug_photos --quality high -o ~/Desktop/mug.glb
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from . import config
from .pipeline import PipelineError, Reconstruction


def main() -> None:
    ap = argparse.ArgumentParser(prog="scan3d", description="Photos/video -> textured 3D model (GLB)")
    ap.add_argument("source", type=Path, help="folder with photos/videos, or a single video file")
    ap.add_argument("-o", "--output", type=Path,
                    help="output path; .glb (textured) or .stl (for 3D printing). Default: <source>.glb")
    ap.add_argument("-q", "--quality", choices=list(config.QUALITIES), default=config.DEFAULT_QUALITY)
    ap.add_argument("--mask", action="store_true", help="remove background (turntable captures)")
    ap.add_argument("--size", type=float, metavar="MM",
                    help="a real measurement of the object in mm, to output true-to-size models")
    ap.add_argument("--size-of", choices=["height", "width", "depth", "longest"], default="longest",
                    help="which dimension --size refers to (default: longest side)")
    ap.add_argument("--keep", type=Path, help="keep the whole job folder (log, OBJ, point cloud) here")
    args = ap.parse_args()

    src = args.source.expanduser().resolve()
    if not src.exists():
        sys.exit(f"{src} does not exist")
    output = (args.output or src.with_suffix(".glb")).expanduser().resolve()

    job_dir = Path(tempfile.mkdtemp(prefix="scan3d_"))
    (job_dir / "input").mkdir()
    def link(p: Path) -> None:
        dst = job_dir / "input" / p.name
        try:
            dst.symlink_to(p)
        except OSError:  # Windows without developer mode: no symlinks, copy instead
            shutil.copy2(p, dst)

    if src.is_dir():
        for p in src.iterdir():
            if p.is_file():
                link(p)
    else:
        link(src)

    def progress(stage: str, pct: float) -> None:
        print(f"[{pct:5.1f}%] {stage}", flush=True)

    size = {"dimension": args.size_of, "mm": args.size} if args.size else None
    rec = Reconstruction(job_dir, args.quality, mask=args.mask, size=size, on_progress=progress)
    try:
        stats = rec.execute()
    except PipelineError as e:
        print(f"\nFailed: {e}\nLog: {rec.log_path}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        rec.cancel.set()
        sys.exit(130)

    output.parent.mkdir(parents=True, exist_ok=True)
    fmt = "stl" if output.suffix.lower() == ".stl" else "glb"
    shutil.copy2(job_dir / "output" / f"model.{fmt}", output)
    print(f"\nSaved {output}  ({stats.get('faces', 0):,} faces, "
          f"{stats.get('registered_images')}/{stats.get('num_images')} images aligned, {stats.get('seconds')}s)")
    d = stats.get("dimensions") or {}
    if d:
        print(f"Size: W {d['width']} x D {d['depth']} x H {d['height']} {d['unit']}")
    if args.keep:
        shutil.copytree(job_dir, args.keep.expanduser(), dirs_exist_ok=True)
        print(f"Job files kept in {args.keep}")
    shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
