"""Turn whatever the user dropped in (photos, HEIC, videos) into a clean, uniformly
sized set of JPEG frames that COLMAP can work with."""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .. import config

try:  # iPhone photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

MIN_IMAGES = 5


def collect_inputs(input_dir: Path) -> tuple[list[Path], list[Path]]:
    images, videos = [], []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        if ext in config.IMAGE_EXTS:
            images.append(p)
        elif ext in config.VIDEO_EXTS:
            videos.append(p)
    return images, videos


def sharpness(path: Path) -> float:
    img = config.imread_gray(path)
    if img is None:
        return 0.0
    h, w = img.shape
    scale = 640 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _save_image(src: Path, dst: Path, max_size: int) -> tuple[int, int]:
    with Image.open(src) as im:
        exif = im.getexif()
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if max(im.size) > max_size:
            im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        exif.pop(0x0112, None)  # orientation already applied
        im.save(dst, "JPEG", quality=95, exif=exif.tobytes())
        return im.size


def _video_duration(video: Path) -> float:
    out = subprocess.run(
        [config.FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if config.IS_WINDOWS else 0,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def _extract_video(video: Path, out_dir: Path, target: int, max_size: int, prefix: str,
                   run: Callable[[list[str], str], None]) -> list[Path]:
    """Extract ~2x the needed frames, then keep the sharpest frame of each pair."""
    tmp = out_dir / f"_{prefix}_raw"
    tmp.mkdir(parents=True, exist_ok=True)
    duration = max(_video_duration(video), 0.5)
    fps = min(30.0, (target * 2) / duration)
    scale = f"scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'"
    run([config.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={fps:.4f},{scale}", "-q:v", "2", str(tmp / "f_%05d.jpg")], "Extracting video frames")
    raw = sorted(tmp.glob("f_*.jpg"))
    kept: list[Path] = []
    for i in range(0, len(raw), 2):
        group = raw[i:i + 2]
        best = max(group, key=sharpness)
        dst = out_dir / f"{prefix}_{len(kept):05d}.jpg"
        shutil.move(best, dst)
        kept.append(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    return kept


def prepare_images(input_dir: Path, images_dir: Path, quality: config.Quality,
                   run: Callable[[list[str], str], None], log: Callable[[str], None]) -> dict:
    images_dir.mkdir(parents=True, exist_ok=True)
    images, videos = collect_inputs(input_dir)
    if not images and not videos:
        raise ValueError("No images or videos found in the input.")

    sizes: set[tuple[int, int]] = set()
    count = 0
    for src in images:
        try:
            sizes.add(_save_image(src, images_dir / f"img_{count:05d}.jpg", quality.max_image_size))
            count += 1
        except Exception as e:  # unreadable file: skip, but tell the user
            log(f"Skipping {src.name}: {e}")

    per_video = max(MIN_IMAGES, quality.video_frames // max(1, len(videos)))
    for vi, video in enumerate(videos):
        frames = _extract_video(video, images_dir, per_video, quality.max_image_size, f"vid{vi}", run)
        log(f"{video.name}: kept {len(frames)} sharpest frames")
        for f in frames[:1]:
            with Image.open(f) as im:
                sizes.add(im.size)
        count += len(frames)

    if count < MIN_IMAGES:
        raise ValueError(f"Need at least {MIN_IMAGES} images, got {count}. "
                         "Take 20-80 overlapping photos around the object.")

    # Drop nearly-black / extremely blurry photos (motion blur ruins matching).
    scores = {p: sharpness(p) for p in images_dir.glob("*.jpg")}
    median = float(np.median(list(scores.values())))
    dropped = [p for p, s in scores.items() if s < 0.15 * median]
    if len(scores) - len(dropped) >= MIN_IMAGES:
        for p in dropped:
            p.unlink()
        if dropped:
            log(f"Dropped {len(dropped)} very blurry images")

    return {
        "num_images": len(list(images_dir.glob("*.jpg"))),
        "num_photos": len(images),
        "num_videos": len(videos),
        # One shared camera model is far more stable when every frame comes from the same device.
        "single_camera": len(sizes) == 1,
    }


def masking_available() -> bool:
    try:
        import rembg  # noqa: F401
        return True
    except ImportError:
        return False


def make_masks(images_dir: Path, masks_dir: Path, log: Callable[[str], None]) -> bool:
    """Background removal for turntable captures. COLMAP ignores features where mask==0."""
    try:
        from rembg import new_session, remove
    except ImportError:
        log("Masking requested but rembg is not installed (uv sync --extra masking); skipping.")
        return False
    masks_dir.mkdir(parents=True, exist_ok=True)
    session = new_session("isnet-general-use")
    for img_path in sorted(images_dir.glob("*.jpg")):
        with Image.open(img_path) as im:
            mask = remove(im, session=session, only_mask=True)
        mask = mask.point(lambda v: 255 if v > 32 else 0)
        mask.save(masks_dir / f"{img_path.name}.png")
        # Also black out the background in the image itself so OpenMVS (dense + texture)
        # does not reconstruct the static background of a turntable capture.
        with Image.open(img_path) as im:
            exif = im.info.get("exif", b"")
            black = Image.new("RGB", im.size)
            Image.composite(im.convert("RGB"), black, mask).save(img_path, "JPEG", quality=95, exif=exif)
    return True


def make_thumbnail(images_dir: Path, dst: Path) -> None:
    first = next(iter(sorted(images_dir.glob("*.jpg"))), None)
    if first is None:
        return
    with Image.open(first) as im:
        im.thumbnail((512, 512))
        im.save(dst, "JPEG", quality=85)
