"""Printable scan mat with ArUco markers -> automatic real-world scale and floor plane.

Print the mat at 100 % ("Actual size"), put the object in the middle, and scan as usual.
The markers have known positions in millimetres, so after COLMAP has solved the cameras we:
  1. detect marker corners in the undistorted frames (OpenCV ArUco),
  2. triangulate each corner from all frames that see it,
  3. fit a similarity transform  mat (mm) -> model  =>  scale (mm per model unit),
     plus the mat plane = the floor the object stands on.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

DICTIONARY = cv2.aruco.DICT_4X4_50
DPI = 300
MARKER_MM = 32.0
MARGIN_MM = 12.0
MIN_GAP_MM = 8.0


@dataclass(frozen=True)
class Paper:
    name: str
    width_mm: float
    height_mm: float
    first_id: int  # each paper size uses its own marker ids, so detection knows the layout


PAPERS = {"a4": Paper("A4", 210.0, 297.0, 0), "letter": Paper("Letter", 215.9, 279.4, 25)}


def _positions(length: float, n_min: int = 2) -> list[float]:
    usable = length - 2 * MARGIN_MM
    n = max(n_min, int((usable + MIN_GAP_MM) // (MARKER_MM + MIN_GAP_MM)))
    step = (usable - MARKER_MM) / (n - 1)
    return [MARGIN_MM + i * step for i in range(n)]


def layout(paper: Paper) -> dict[int, np.ndarray]:
    """marker id -> its 4 corners (TL, TR, BR, BL as OpenCV reports them) in mm on the sheet.
    Sheet frame: origin top-left, x to the right, y down, z = 0."""
    xs, ys = _positions(paper.width_mm), _positions(paper.height_mm)
    slots = [(x, ys[0]) for x in xs] + [(xs[-1], y) for y in ys[1:-1]] + \
            [(x, ys[-1]) for x in reversed(xs)] + [(xs[0], y) for y in reversed(ys[1:-1])]
    s = MARKER_MM
    return {paper.first_id + i: np.array([[x, y, 0], [x + s, y, 0], [x + s, y + s, 0], [x, y + s, 0]], float)
            for i, (x, y) in enumerate(slots)}


# --------------------------------------------------------------------------- printable sheet
def render_pdf(paper_key: str = "a4") -> bytes:
    buf = BytesIO()
    render_image(paper_key).convert("RGB").save(buf, "PDF", resolution=DPI)
    return buf.getvalue()


def render_image(paper_key: str = "a4") -> Image.Image:
    paper = PAPERS[paper_key]
    px = lambda mm: int(round(mm / 25.4 * DPI))  # noqa: E731
    W, H = px(paper.width_mm), px(paper.height_mm)
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(7)

    # Light random speckle inside the free area: extra texture helps matching plain objects.
    x0, y0 = px(MARGIN_MM + MARKER_MM + 6), px(MARGIN_MM + MARKER_MM + 24)  # below the scale bar
    x1, y1 = W - x0, H - px(MARGIN_MM + MARKER_MM + 16)  # above the footer text
    for _ in range(1500):
        cx, cy = rng.integers(x0, x1), rng.integers(y0, y1)
        if abs(cy - H // 2) < px(4) and abs(cx - W // 2) < px(30):
            continue  # keep the centre label readable
        r = int(rng.integers(px(0.6), px(2.2)))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(rng.integers(150, 215)))

    aruco = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    for mid, corners in layout(paper).items():
        side = px(MARKER_MM)
        marker = cv2.aruco.generateImageMarker(aruco, mid, side, borderBits=1)
        img.paste(Image.fromarray(marker), (px(corners[0][0]), px(corners[0][1])))

    font = ImageFont.load_default(size=px(3.4))
    small = ImageFont.load_default(size=px(2.6))
    # 100 mm scale bar to verify the print scale.
    bx, by = px(paper.width_mm / 2 - 50), px(MARGIN_MM + MARKER_MM + 14)
    draw.rectangle([bx, by, bx + px(100), by + px(1.2)], fill=0)
    for t in range(0, 101, 10):
        draw.rectangle([bx + px(t) - 2, by - px(2 if t % 50 else 3.5), bx + px(t) + 2, by], fill=0)
    draw.text((bx, by + px(2.5)), "This bar must measure exactly 100 mm. Print at 100% / Actual size.",
              fill=0, font=small)
    cx = W // 2
    draw.text((cx, H // 2), "place the object here", fill=110, font=font, anchor="mm")
    draw.text((cx, H - px(MARGIN_MM + MARKER_MM + 10)),
              f"scan3d scan mat · {paper.name} · keep the sheet flat · film all the way around",
              fill=0, font=small, anchor="mm")
    return img


# --------------------------------------------------------------------------- measurement
def _read_colmap_txt(model_txt: Path) -> tuple[dict, dict]:
    cams = {}
    for line in (model_txt / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if p[1] != "PINHOLE":
            raise ValueError(f"expected undistorted PINHOLE cameras, got {p[1]}")
        fx, fy, cx, cy = map(float, p[4:8])
        cams[int(p[0])] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    images = {}
    lines = [l for l in (model_txt / "images.txt").read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    for header in lines[0::2]:
        p = header.split()
        if len(p) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        images[p[9]] = {"P": cams[int(p[8])] @ np.hstack([R, t[:, None]]), "center": -R.T @ t}
    return cams, images


def _quat_to_rot(w, x, y, z) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _triangulate(obs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float]:
    """DLT from >= 2 (P, uv) observations; drops observations with large reprojection error."""
    def solve(o):
        A = np.vstack([np.vstack([uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]]) for P, uv in o])
        X = np.linalg.svd(A)[2][-1]
        return X[:3] / X[3]

    def reproj(X, o):
        errs = []
        for P, uv in o:
            x = P @ np.append(X, 1)
            errs.append(np.linalg.norm(x[:2] / x[2] - uv))
        return np.array(errs)

    X = solve(obs)
    err = reproj(X, obs)
    keep = [o for o, e in zip(obs, err) if e < max(3.0, 2.5 * np.median(err))]
    if len(keep) >= 2 and len(keep) < len(obs):
        X = solve(keep)
        err = reproj(X, keep)
    return X, float(np.median(err))


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity dst ~ s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (a ** 2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s


def measure(model_txt: Path, images_dir: Path, log: Callable[[str], None]) -> dict | None:
    """Returns {"mm_per_unit", "rms_mm", "markers", "corners", "paper", "plane": {"normal","d"}}
    in the model's coordinate frame, or None if no usable mat is visible."""
    _, images = _read_colmap_txt(model_txt)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY),
                                       cv2.aruco.DetectorParameters())
    obs: dict[tuple[int, int], list] = {}
    for name, cam in images.items():
        gray = config.imread_gray(images_dir / name)
        if gray is None:
            continue
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for c, mid in zip(corners, ids.ravel()):
            for k in range(4):
                obs.setdefault((int(mid), k), []).append((cam["P"], c[0][k].astype(float)))
    if not obs:
        log("Scan mat: no markers found (scale must be set manually)")
        return None

    seen = {mid for mid, _ in obs}
    paper = max(PAPERS.values(), key=lambda p: len(seen & set(layout(p))))
    lay = layout(paper)

    src, dst = [], []
    for (mid, k), o in obs.items():
        if mid not in lay or len(o) < 3:
            continue
        X, err = _triangulate(o)
        if err < 4.0:
            src.append(lay[mid][k])
            dst.append(X)
    if len(src) < 8:
        log(f"Scan mat: only {len(src)} marker corners triangulated, need 8; ignoring the mat")
        return None
    src, dst = np.array(src), np.array(dst)

    for _ in range(3):  # fit, drop outlier corners, refit
        s, R, t = _umeyama(src, dst)
        res_mm = np.linalg.norm((dst - (s * (R @ src.T).T + t)) / s, axis=1)
        keep = res_mm < max(1.0, 3 * np.median(res_mm))
        if keep.all() or keep.sum() < 8:
            break
        src, dst = src[keep], dst[keep]
    rms = float(np.sqrt((res_mm ** 2).mean()))
    if rms > 3.0:
        log(f"Scan mat: fit too poor (rms {rms:.2f} mm); ignoring the mat. Is the sheet flat?")
        return None

    # Mat normal pointing towards the cameras = up.
    n = R[:, 2]
    centers = np.array([c["center"] for c in images.values()])
    if ((centers - t) @ n).mean() < 0:
        n = -n
    result = {"mm_per_unit": 1.0 / s, "rms_mm": round(rms, 3), "corners": int(len(src)),
              "paper": paper.name, "plane": {"normal": n.tolist(), "d": float(-n @ t)},
              # model -> sheet mm:  R^T (X - t) / s ;  "up" tells which sheet z is above the mat
              "frame": {"scale": s, "R": R.tolist(), "t": t.tolist(), "up": 1.0 if n @ R[:, 2] > 0 else -1.0,
                        "width_mm": paper.width_mm, "height_mm": paper.height_mm}}
    log(f"Scan mat ({paper.name}): {len(src)} corners, scale {1.0 / s:.4f} mm/unit, fit rms {rms:.2f} mm")
    return result


def to_sheet_mm(points: np.ndarray, mat: dict) -> np.ndarray:
    """Model-space points -> sheet coordinates in mm (x right, y down the sheet, z = height above it)."""
    f = mat["frame"]
    R, t = np.asarray(f["R"]), np.asarray(f["t"])
    uvw = (np.asarray(points, float) - t) @ R / f["scale"]
    uvw[:, 2] *= f["up"]
    return uvw


def above_mat(points: np.ndarray, mat: dict, margin_mm: float = 10.0, below_mm: float = 3.0) -> np.ndarray:
    """True for points over the sheet (plus a margin) and not below it: the object's volume.
    Everything else (floor around the sheet, walls, furniture) is background."""
    uvw = to_sheet_mm(points, mat)
    f = mat["frame"]
    return ((uvw[:, 0] > -margin_mm) & (uvw[:, 0] < f["width_mm"] + margin_mm) &
            (uvw[:, 1] > -margin_mm) & (uvw[:, 1] < f["height_mm"] + margin_mm) &
            (uvw[:, 2] > -below_mm) & (uvw[:, 2] < 2.0 * max(f["width_mm"], f["height_mm"])))


def visible_in(images_dir: Path, samples: int = 8) -> bool:
    """Quick check before the heavy work: do a few frames show scan-mat markers?"""
    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        return False
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY),
                                       cv2.aruco.DetectorParameters())
    hits = 0
    for f in frames[:: max(1, len(frames) // samples)][:samples]:
        gray = config.imread_gray(f)
        ids = detector.detectMarkers(gray)[1] if gray is not None else None
        hits += ids is not None and len(ids) >= 4
    return hits >= 2
