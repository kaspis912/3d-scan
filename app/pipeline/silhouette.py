"""Silhouette carving: the object's outer shape from its outline in every frame.

Photogrammetry needs visible surface detail; black, shiny or plain objects have little, so their
scans come out with holes and lumps. Outlines don't need any: SAM 2 segments the object in each
undistorted frame (prompted with a box around where the scan says the object is), and a voxel is
kept only if it projects inside the object's outline in (almost) every view that sees it.
The result (the "visual hull") is closed and contains the object; it cannot see hollows that are
never visible in outline (a recess in a top face), which still come from the dense scan.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a few SAM 2 ops lack MPS kernels

MODEL_ID = "facebook/sam2.1-hiera-small"


def _backend() -> str | None:
    """Which SAM 2 implementation is installed: the official Hugging Face `transformers` one
    (plain wheels on every platform, preferred) or the `sam2` package (source build)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    try:
        from transformers import Sam2Model, Sam2Processor  # noqa: F401
        return "transformers"
    except ImportError:
        pass
    try:
        import sam2  # noqa: F401
        return "sam2"
    except ImportError:
        return None


def available() -> bool:
    return _backend() is not None


def load_predictor(device: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """-> predict(image RGB uint8, points (n, 2), labels (n,), box (4,)) -> boolean mask (h, w)."""
    import torch

    if _backend() == "transformers":
        from transformers import Sam2Model, Sam2Processor

        model = Sam2Model.from_pretrained(MODEL_ID).to(device).eval()
        processor = Sam2Processor.from_pretrained(MODEL_ID)

        def predict(img: np.ndarray, pts: np.ndarray, labels: np.ndarray, box: np.ndarray) -> np.ndarray:
            inputs = processor(images=img, input_points=[[pts.astype(float).tolist()]],
                               input_labels=[[labels.astype(int).tolist()]],
                               input_boxes=[[box.astype(float).tolist()]], return_tensors="pt")
            sizes = inputs["original_sizes"]
            # float32 before moving: Apple's GPU has no float64
            batch = {k: (v.float() if v.is_floating_point() else v).to(device)
                     for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            with torch.inference_mode():
                out = model(**batch, multimask_output=False)
            masks = processor.post_process_masks(out.pred_masks.float().cpu(), sizes)[0]
            return np.asarray(masks[0, 0].numpy(), dtype=bool)

        return predict

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=device)

    def predict(img: np.ndarray, pts: np.ndarray, labels: np.ndarray, box: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            predictor.set_image(img)
            m, _, _ = predictor.predict(point_coords=pts, point_labels=labels, box=box, multimask_output=False)
        return m[0] > 0.5

    return predict


def torch_device() -> str:
    """Where SAM 2 runs: SCAN3D_TORCH_DEVICE if set, else a GPU PyTorch can use ("cuda" covers
    NVIDIA and AMD ROCm builds, "mps" Apple Silicon), else the CPU (~1-2 s per frame)."""
    import torch

    forced = os.environ.get("SCAN3D_TORCH_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_cameras(model_txt: Path) -> list[dict]:
    """Undistorted (PINHOLE) COLMAP text model -> [{name, P (3x4), w, h}]."""
    from ..scanmat import _quat_to_rot

    cams = {}
    for line in (model_txt / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if p[1] != "PINHOLE":
            raise ValueError(f"expected undistorted PINHOLE cameras, got {p[1]}")
        w, h = int(p[2]), int(p[3])
        fx, fy, cx, cy = map(float, p[4:8])
        cams[int(p[0])] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), w, h)
    out = []
    lines = [l for l in (model_txt / "images.txt").read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    for header in lines[0::2]:
        p = header.split()
        if len(p) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        K, w, h = cams[int(p[8])]
        Rt = np.hstack([_quat_to_rot(qw, qx, qy, qz), np.array([[tx], [ty], [tz]])])
        out.append({"name": p[9], "P": K @ Rt, "w": w, "h": h})
    return sorted(out, key=lambda c: c["name"])


def _project(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = X @ P[:, :3].T + P[:, 3]
    z = x[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = x[:, :2] / z[:, None]
    return uv, z


def _spread(points: np.ndarray, n: int) -> np.ndarray:
    """n points far apart from each other (farthest-point sampling)."""
    idx = [0]
    d = np.linalg.norm(points - points[0], axis=1)
    for _ in range(n - 1):
        idx.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(points - points[idx[-1]], axis=1))
    return points[idx]


def segment(images_dir: Path, cams: list[dict], object_pts: np.ndarray, floor_pts: np.ndarray,
            log: Callable[[str], None], debug_dir: Path | None = None,
            box_pts: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """SAM 2 mask of the object in every frame.

    object_pts: points on the scanned object (model coords). Their projection gives the prompt box,
    and a few of them are "object" clicks: any of them lands on an object pixel, even when hidden
    behind another part of the object. floor_pts: points on the mat/floor around the object; those
    that project outside the object's box are "not object" clicks, so SAM never takes the mat.
    box_pts: corners of a generous 3D box around the object (the scan can miss parts of a dark
    object, e.g. a whole plate); its projection is the prompt box and the limit for the mask."""
    from PIL import Image

    device = torch_device()
    predict = load_predictor(device)
    log(f"Silhouettes: SAM 2 ({_backend()}) on {device}")
    rng = np.random.default_rng(0)
    sample = object_pts[rng.choice(len(object_pts), min(len(object_pts), 4000), replace=False)]
    clicks = _spread(sample, 5)
    masks = {}
    for i, cam in enumerate(cams):
        uv, z = _project(cam["P"], sample)
        if (z <= 0).any():
            continue
        (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
        mx, my = 0.06 * (x1 - x0) + 4, 0.06 * (y1 - y0) + 4
        bx0, by0, bx1, by1 = x0 - mx, y0 - my, x1 + mx, y1 + my
        if box_pts is not None:
            buv, bz = _project(cam["P"], box_pts)
            if (bz > 0).all():
                bx0, by0 = np.minimum([bx0, by0], buv.min(0))
                bx1, by1 = np.maximum([bx1, by1], buv.max(0))
        bx0, by0 = max(0.0, bx0), max(0.0, by0)
        bx1, by1 = min(cam["w"] - 1.0, bx1), min(cam["h"] - 1.0, by1)
        if bx1 - bx0 < 16 or by1 - by0 < 16:
            continue
        pos, _ = _project(cam["P"], clicks)
        nuv, nz = _project(cam["P"], floor_pts)
        outside = ~((nuv[:, 0] > x0 - mx) & (nuv[:, 0] < x1 + mx) & (nuv[:, 1] > y0 - my) & (nuv[:, 1] < y1 + my))
        visible = (nz > 0) & (nuv[:, 0] >= 0) & (nuv[:, 0] < cam["w"]) & (nuv[:, 1] >= 0) & (nuv[:, 1] < cam["h"])
        neg = nuv[outside & visible]
        if len(neg) > 8:
            neg = neg[np.linspace(0, len(neg) - 1, 8).astype(int)]
        pts = np.vstack([pos, neg]) if len(neg) else pos
        labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        img = np.asarray(Image.open(images_dir / cam["name"]).convert("RGB"))
        mask = predict(img, pts, labels, np.array([bx0, by0, bx1, by1]))
        # Nothing of the object lies outside the prompt box: drop leaks into the background.
        clip = np.zeros_like(mask)
        clip[int(by0):int(np.ceil(by1)) + 1, int(bx0):int(np.ceil(bx1)) + 1] = True
        masks[cam["name"]] = mask & clip
        if debug_dir is not None and i % 15 == 0:
            debug_dir.mkdir(parents=True, exist_ok=True)
            over = img.copy()
            over[~masks[cam["name"]]] //= 3
            Image.fromarray(over).resize((img.shape[1] // 3, img.shape[0] // 3)).save(debug_dir / f"mask_{i:03d}.jpg")
    log(f"Silhouettes: segmented the object in {len(masks)}/{len(cams)} frames")
    return masks


def carve(cams: list[dict], masks: dict[str, np.ndarray], to_model: np.ndarray, lo: np.ndarray,
          hi: np.ndarray, pitch: float, misses: int = 2, grow_px: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Visual hull on a voxel grid spanning [lo, hi] (base coords, y up). to_model maps base
    coords -> COLMAP model coords for projection. A voxel is carved away once more than `misses`
    views see it outside the outline: a fixed count, not a share, because a gap (under a plate,
    inside a handle) is often visible in only a handful of frames, while one or two frames can have
    a bad mask. Masks are grown by grow_px first so thin edges survive slightly loose outlines.
    Returns (occupancy[x, y, z], origin)."""
    import cv2

    def clean(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8)
        # Close thin streaks SAM leaves in deep shadow (a few px wide) without filling real openings.
        r = max(3, int(0.004 * np.hypot(*m.shape)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        if grow_px:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1,) * 2))
        return m.astype(bool)

    views = [(c, clean(masks[c["name"]])) for c in cams if c["name"] in masks]
    dims = np.ceil((hi - lo) / pitch).astype(int) + 1
    gx, gy, gz = (lo[i] + pitch * np.arange(dims[i]) for i in range(3))
    occ = np.zeros(dims, dtype=bool)
    for ix0 in range(0, dims[0], 16):  # slabs keep memory small
        xs = gx[ix0:ix0 + 16]
        X = np.stack(np.meshgrid(xs, gy, gz, indexing="ij"), -1).reshape(-1, 3)
        Xm = X @ to_model[:3, :3].T + to_model[:3, 3]
        seen = np.zeros(len(X), np.int32)
        outside = np.zeros(len(X), np.int32)
        for cam, mask in views:
            uv, z = _project(cam["P"], Xm)
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            ok = (z > 0) & (u >= 0) & (u < cam["w"]) & (v >= 0) & (v < cam["h"])
            seen += ok
            idx = np.flatnonzero(ok)
            outside[idx] += ~mask[v[idx], u[idx]]
        keep = (seen >= 5) & (outside <= misses)
        occ[ix0:ix0 + len(xs)] = keep.reshape(len(xs), dims[1], dims[2])
    return occ, lo.astype(float)


def hull_on_grid(cams: list[dict], masks: dict[str, np.ndarray], to_model: np.ndarray,
                 origin: np.ndarray, pitch: float, dims: np.ndarray,
                 log: Callable[[str], None]) -> np.ndarray:
    """Two-pass visual hull on a given voxel grid (base coords).

    Pass 1 tolerates 2 disagreeing views, which protects the object from a frame or two with a bad
    mask. Its silhouette is projected back into every frame; frames whose mask misses clearly
    more of it than the others (typically thin dark streaks SAM leaves in deep shadow) are dropped,
    and pass 2 carves strictly with the rest, so gaps seen by only a few frames open up."""
    import cv2

    hi = origin + (np.asarray(dims) - 1) * pitch
    occ, _ = carve(cams, masks, to_model, origin, hi, pitch, misses=2, grow_px=2)
    idx = np.argwhere(occ)
    if len(idx) == 0:
        return occ
    X = origin + idx[:: max(1, len(idx) // 400_000)] * pitch
    Xm = X @ to_model[:3, :3].T + to_model[:3, 3]
    cover = {}
    for cam in cams:
        mask = masks.get(cam["name"])
        if mask is None:
            continue
        uv, z = _project(cam["P"], Xm)
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, cam["w"] - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, cam["h"] - 1)
        proj = np.zeros(mask.shape, np.uint8)
        proj[v, u] = 1
        proj = cv2.dilate(proj, np.ones((5, 5), np.uint8)).astype(bool)
        cover[cam["name"]] = (proj & mask).sum() / max(proj.sum(), 1)
    vals = np.array(list(cover.values()))
    thr = float(np.median(vals)) - 0.025
    good = {n: masks[n] for n, c in cover.items() if c >= thr}
    if len(good) < 0.8 * len(cover):  # never drop more than a fifth of the views
        good = {n: masks[n] for n in sorted(cover, key=cover.get, reverse=True)[: int(0.8 * len(cover))]}
    occ2, _ = carve(cams, good, to_model, origin, hi, pitch, misses=0, grow_px=2)
    log(f"Silhouettes: carved with {len(good)}/{len(cover)} frames "
        f"({len(cover) - len(good)} with unreliable outlines left out)")
    return occ2
