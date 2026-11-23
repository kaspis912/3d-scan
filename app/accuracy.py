"""Accuracy check: how far a scanned solid is from a model whose geometry is known exactly.

Print a known model (the built-in test piece, or any STL), scan the print, and compare. The scan
is placed on the reference the way it would sit on the printer: base on the base, turned about
the vertical axis and shifted sideways only. Tilt, height offset and scale are *not* fitted away
in the headline numbers, because a wrong floor or a wrong scale is a real error of the scan;
they are reported separately (`fit`), which tells where an error comes from.

Deviations are signed: + where the scan has material the reference does not (too big, filled-in
gap), - where the scan lacks material (hole in the scan, missing part). Millimetres, Z up.
"""
from __future__ import annotations

import itertools
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import config, refpiece

REF_DIR = "reference"
REF_EXTS = {".stl", ".3mf", ".obj", ".ply"}
HEATMAP_LIMIT_MM = 1.0  # colour scale of the deviation views: full blue / red at -/+ this
HISTORY = config.DATA_DIR / "accuracy_history.jsonl"
Z_UP_TO_Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])

DOF_PLACED = (2, 3, 4)  # turn about Z, shift in X and Y: the scan stays on its base
DOF_RIGID = (0, 1, 2, 3, 4, 5)
DOF_RIGID_SCALE = (0, 1, 2, 3, 4, 5, 6)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("the file has no triangles")
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


class Surface:
    """Exact signed distance from points to a closed triangle mesh, without a ray tracer.

    Every triangle carries a grid of sample points no further than SPACING apart, so each point
    of the surface has a sample of its own triangle within SPACING. A query looks up its nearest
    sample (distance ds): the closest triangle must then have a sample within ds + SPACING, so
    measuring the exact point-to-triangle distance to all triangles sampled inside that ball finds
    it. The sign comes from the triangle's outward normal; where several triangles tie (closest
    point on an edge or corner) the one whose plane is farthest from the point decides, which is
    the correct side."""

    SPACING = 0.5  # mm
    FAR = 3.0  # mm

    def __init__(self, mesh: trimesh.Trimesh):
        self.triangles = tri = np.asarray(mesh.triangles, dtype=np.float64)
        self.normals = np.asarray(mesh.face_normals)
        self._e1, self._e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        longest = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
        div = np.maximum(1, np.ceil(longest / self.SPACING)).astype(int)
        pts, fid = [], []
        for n in np.unique(div):  # barycentric grid with n divisions per edge
            faces = np.flatnonzero(div == n)
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            keep = i + j <= n
            u, v = i[keep] / n, j[keep] / n
            w = np.stack([1 - u - v, u, v], axis=1)  # (m, 3)
            pts.append(np.einsum("mk,fkc->fmc", w, tri[faces]).reshape(-1, 3))
            fid.append(np.repeat(faces, len(w)))
        self._fid = np.concatenate(fid)
        self._tree = cKDTree(np.vstack(pts))

    def signed(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (signed distance, + outside; outward unit normal of the closest triangle)."""
        dist = np.empty(len(points))
        normal = np.empty((len(points), 3))
        nf = len(self.triangles)
        for s in range(0, len(points), 20_000):
            p = np.ascontiguousarray(points[s:s + 20_000], dtype=np.float64)
            ds, near = self._tree.query(p, k=8, workers=-1)
            # Far from the surface the ball would hold thousands of samples and exactness to a
            # fraction of SPACING no longer matters: there the nearest few samples are enough.
            radius = np.where(ds[:, 0] < self.FAR, ds[:, 0] + self.SPACING * 1.01, 0.0)
            hits = self._tree.query_ball_point(p, radius, workers=-1)
            counts = np.fromiter(map(len, hits), dtype=np.int64, count=len(p))
            flat = np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64, count=int(counts.sum()))
            far = np.flatnonzero(radius == 0.0)
            flat = np.r_[flat, near[far].ravel()]
            counts = np.r_[counts, np.full(len(far), near.shape[1])]
            owner = np.r_[np.arange(len(p)), far]
            pair = np.unique(np.repeat(owner, counts) * nf + self._fid[flat])  # sorted by point
            pid, face = pair // nf, pair % nf
            off = p[pid] - trimesh.triangles.closest_point(self.triangles[face], p[pid])
            d = np.linalg.norm(off, axis=1)
            dot = np.einsum("ij,ij->i", off, self.normals[face])
            starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
            group = np.cumsum(np.r_[True, pid[1:] != pid[:-1]]) - 1
            dmin = np.minimum.reduceat(d, starts)
            score = np.where(d <= dmin[group] * (1 + 1e-7) + 1e-7, np.abs(dot), -1.0)
            winner = np.flatnonzero(score == np.maximum.reduceat(score, starts)[group])
            winner = winner[np.unique(group[winner], return_index=True)[1]]
            dist[s:s + len(p)] = np.where(dot[winner] < 0, -1.0, 1.0) * dmin
            normal[s:s + len(p)] = self.normals[face[winner]]
        return dist, normal

    def line_hits(self, origin: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every crossing of the infinite line origin + t * direction with the surface
        (Moller-Trumbore) -> (t, triangle index)."""
        h = np.cross(direction, self._e2)
        det = np.einsum("ij,ij->i", self._e1, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            t0 = origin - self.triangles[:, 0]
            u = np.einsum("ij,ij->i", t0, h) * inv
            q = np.cross(t0, self._e1)
            v = (q @ direction) * inv
            t = np.einsum("ij,ij->i", self._e2, q) * inv
            ok = (np.abs(det) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        idx = np.flatnonzero(ok)
        return t[idx], idx

    def along_normal(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Where this surface lies along each point's normal: + beyond the point, - behind it,
        NaN when the line never meets it. Only crossings that face the same way as the point's
        normal count (the matching face, not the far side of the part), and the nearest one wins.
        Unlike the closest distance, this is what a caliper reads on a flat face: a filled hole
        reads its full depth, not the distance to the nearest wall."""
        out = np.full(len(points), np.nan)
        for i, (p, n) in enumerate(zip(points, normals)):
            t, face = self.line_hits(p, n)
            t = t[self.normals[face] @ n > 0.3]
            if len(t):
                out[i] = t[np.abs(t).argmin()]
        return out


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def _surface_points(mesh: trimesh.Trimesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spread surface points with normals, leaving out the base (never scanned, cut flat)."""
    pts, fid = trimesh.sample.sample_surface(mesh, int(n * 1.3), seed=seed)
    nrm = mesh.face_normals[fid]
    base_z = mesh.bounds[0][2] + 1e-3 * max(mesh.extents.max(), 1.0)
    keep = ~((pts[:, 2] < base_z) & (nrm[:, 2] < -0.9))
    return np.asarray(pts[keep][:n]), np.asarray(nrm[keep][:n])


def _icp(src: np.ndarray, target: Surface, T: np.ndarray, dof: tuple[int, ...], iters: int = 40,
         keep: float = 0.9) -> np.ndarray:
    """Point-to-plane ICP over the chosen degrees of freedom (rotation x/y/z, shift x/y/z, scale).
    The worst-matching tenth of the points is ignored, so a lump or a missing part in the scan
    does not drag the placement."""
    T = T.copy()
    cols = list(dof)
    for _ in range(iters):
        p = _apply(T, src)
        d, n = target.signed(p)
        ok = np.abs(d) <= np.quantile(np.abs(d), keep)
        p, d, n = p[ok], d[ok], n[ok]
        J = np.hstack([np.cross(p, n), n, np.einsum("ij,ij->i", p, n)[:, None]])[:, cols]
        delta = np.zeros(7)
        delta[cols] = np.linalg.lstsq(J, -d, rcond=None)[0]
        step = np.eye(4)
        step[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() * np.exp(delta[6])
        step[:3, 3] = delta[3:6]
        T = step @ T
        if np.abs(delta[:3]).max() < 1e-6 and np.abs(delta[3:6]).max() < 1e-4 and abs(delta[6]) < 1e-6:
            break
    return T


def _place(scan_pts: np.ndarray, ref_pts: np.ndarray, target: Surface) -> np.ndarray:
    """Scan -> reference placement: try every 5 degrees about the vertical axis, refine the best."""
    s_mid = (scan_pts.min(0) + scan_pts.max(0)) / 2
    r_mid = (ref_pts.min(0) + ref_pts.max(0)) / 2
    rng = np.random.default_rng(0)
    s_sub = scan_pts[rng.choice(len(scan_pts), min(len(scan_pts), 3000), replace=False)]
    r_sub = ref_pts[rng.choice(len(ref_pts), min(len(ref_pts), 3000), replace=False)]
    s_tree, r_tree = cKDTree(scan_pts), cKDTree(ref_pts)

    def start(deg: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", deg, degrees=True).as_matrix()
        T[:2, 3] = r_mid[:2] - (T[:3, :3] @ s_mid)[:2]
        return T

    scored = []
    for deg in np.arange(0, 360, 5.0):
        T = start(deg)
        fwd = np.median(r_tree.query(_apply(T, s_sub))[0])
        back = np.median(s_tree.query(_apply(np.linalg.inv(T), r_sub))[0])
        scored.append((fwd + back, deg))
    best, best_err = None, np.inf
    tried: list[float] = []
    for _, deg in sorted(scored):
        if any(min(abs(deg - t), 360 - abs(deg - t)) < 20 for t in tried):
            continue
        tried.append(deg)
        T = _icp(s_sub, target, start(deg), DOF_PLACED, iters=30)
        err = np.mean(np.sort(np.abs(target.signed(_apply(T, s_sub))[0]))[: int(0.9 * len(s_sub))])
        if err < best_err:
            best, best_err = T, err
        if len(tried) == 4:
            break
    return _icp(scan_pts[:15_000], target, best, DOF_PLACED, iters=40)


def _spread(values: np.ndarray) -> dict:
    a = np.abs(values)
    return {"median_mm": round(float(np.median(a)), 3), "p95_mm": round(float(np.quantile(a, 0.95)), 3),
            "rms_mm": round(float(np.sqrt(np.mean(values ** 2))), 3), "max_mm": round(float(a.max()), 2),
            "bias_mm": round(float(values.mean()), 3)}


def _piece_checks(ref_pts: np.ndarray, ref_nrm: np.ndarray, dev: np.ndarray,
                  scan: Surface) -> tuple[list[dict], list[dict]]:
    """Per-feature deviation and dimension checks for the built-in test piece. A measured
    dimension is the nominal one plus where the scan's surface sits along the normal of the
    faces that define it (median of 60 probes per face), like a caliper would read it."""
    labels = refpiece.feature_of(ref_pts)
    features = []
    for name in refpiece.FEATURES:
        v = dev[labels == name]
        if len(v):
            features.append({"name": name, "bias_mm": round(float(v.mean()), 2),
                             "p95_mm": round(float(np.quantile(np.abs(v), 0.95)), 2)})
    x, y, z = ref_pts.T
    nx, ny, nz = ref_nrm.T
    rng = np.random.default_rng(0)

    def face(mask: np.ndarray) -> float | None:
        idx = np.flatnonzero(mask)
        if len(idx) < 20:
            return None
        idx = rng.choice(idx, min(len(idx), 60), replace=False)
        probe = scan.along_normal(ref_pts[idx], ref_nrm[idx])
        found = ~np.isnan(probe)
        return float(np.median(probe[found])) if found.mean() >= 0.5 else None

    checks = []

    def add(name: str, nominal: float, *offsets: float | None) -> None:
        if all(o is not None for o in offsets):
            checks.append({"name": name, "nominal_mm": nominal, "measured_mm": round(nominal + sum(offsets), 2)})
        else:
            checks.append({"name": name, "nominal_mm": nominal, "measured_mm": None})  # not in the scan

    w, d, _ = refpiece.SIZE_MM
    wall = (z > 1) & (z < refpiece.PLATE_Z - 1)
    add("Width", w, face(wall & (nx < -0.9)), face(wall & (nx > 0.9)))
    add("Depth", d, face(wall & (ny < -0.9)), face(wall & (ny > 0.9)))
    tops = {}
    for i, (x0, x1, top) in enumerate(refpiece.STEPS, 1):
        tops[top] = face((nz > 0.9) & (labels == "steps") & (np.abs(z - top) < 0.05) & (x > x0 + 1) & (x < x1 - 1))
        add(f"Step {i} height", top, tops[top])
    cx, cy, r, dz = refpiece.DOME
    add("Dome height", dz + r, face((labels == "dome") & (np.hypot(x - cx, y - cy) < 2.5)))
    add("Post diameter", 2 * r, face((labels == "cylinder") & (nx < 0)), face((labels == "cylinder") & (nx > 0)))
    hx, hy, hr, hz = refpiece.HOLE
    rim = refpiece.STEPS[1][2]
    bottom = face((nz > 0.9) & (labels == "hole") & (np.abs(z - hz) < 0.05) & (np.hypot(x - hx, y - hy) < hr - 1))
    # + deviation means material: on the hole's floor it makes the hole shallower.
    add("Hole depth", rim - hz, tops[rim], None if bottom is None else -bottom)
    _, _, _, _, atop = refpiece.ARCH
    _, _, ttop = refpiece.TUNNEL
    add("Arch height", atop, face((nz > 0.9) & (labels == "arch") & (np.abs(z - atop) < 0.05)))
    ceiling = face((labels == "arch gap") & (nz < -0.9))
    floor = face((labels == "arch gap") & (nz > 0.9))
    add("Arch opening", ttop - refpiece.PLATE_Z, None if ceiling is None else -ceiling,
        None if floor is None else -floor)
    groove = face((labels == "ribs") & (nz > 0.9) & (np.abs(z - refpiece.PLATE_Z) < 0.05)
                  & (x > refpiece.RIBS[0][1] + 0.3) & (x < refpiece.RIBS[-1][0] - 0.3))
    for x0, x1 in refpiece.RIBS:  # a ridge the scan blurred away reads lower than 4 mm
        top = face((labels == "ribs") & (nz > 0.9) & (np.abs(z - refpiece.RIB_Z) < 0.05) & (x > x0) & (x < x1))
        add(f"{x1 - x0:g} mm ridge height", refpiece.RIB_Z - refpiece.PLATE_Z, top,
            None if groove is None else -groove)
    return features, checks


def _colours(dev: np.ndarray) -> np.ndarray:
    """Blue (scan lacks material) - light grey (on the reference) - red (scan has extra)."""
    t = np.clip(dev / HEATMAP_LIMIT_MM, -1, 1)[:, None]
    mid, red, blue = np.array([232, 232, 226.0]), np.array([200, 40, 30.0]), np.array([35, 90, 200.0])
    rgb = np.where(t >= 0, mid + (red - mid) * t, mid + (blue - mid) * -t)
    return np.hstack([rgb, np.full((len(dev), 1), 255.0)]).astype(np.uint8)


def _refined(mesh: trimesh.Trimesh, edge: float) -> trimesh.Trimesh:
    """The same shape with no edge longer than `edge`, so vertex colours can show local detail."""
    try:
        import manifold3d as m3
        solid = m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                                    tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
        out = solid.refine_to_length(edge).to_mesh()
        if len(out.tri_verts):
            return trimesh.Trimesh(out.vert_properties[:, :3].astype(np.float64), out.tri_verts, process=False)
    except Exception:
        pass  # not a clean solid: fall back to plain subdivision
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=2 * edge, max_iter=12)
    return trimesh.Trimesh(v, f, process=False)


def _viewer_glb(mesh_mm_zup: trimesh.Trimesh, dev: np.ndarray, flat: bool) -> bytes:
    m = trimesh.Trimesh(mesh_mm_zup.vertices, mesh_mm_zup.faces, process=False)
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=_colours(dev))
    if flat:
        m.unmerge_vertices()  # sharp edges of a CAD model stay sharp in the viewer
    m.apply_transform(Z_UP_TO_Y_UP)
    m.apply_scale(1 / 1000.0)  # the viewer's GLBs are Y-up, in metres
    return m.export(file_type="glb")


def compare(scan: trimesh.Trimesh, ref: trimesh.Trimesh, builtin: bool = False,
            heatmaps: bool = True) -> tuple[dict, dict[str, bytes]]:
    """Both meshes: millimetres, Z up, standing on their base. -> (report, {filename: GLB})."""
    scan, ref = scan.copy(), ref.copy()
    ref.apply_translation([0, 0, -ref.bounds[0][2]])
    scan.apply_translation([0, 0, -scan.bounds[0][2]])
    target = Surface(ref)
    s_pts, _ = _surface_points(scan, 60_000, seed=1)
    r_pts, r_nrm = _surface_points(ref, 60_000, seed=2)

    T = _place(s_pts, r_pts, target)
    fit_pts = s_pts[:15_000]  # plenty for a placement good to a few microns
    typical = float(np.median(np.abs(target.signed(_apply(T, fit_pts))[0])))
    if typical > max(2.0, 0.05 * ref.extents.max()):
        raise ValueError(f"The scan does not match this reference (typically {typical:.0f} mm apart). Is it the "
                         "right model, in millimetres, standing the way the object stood on the mat?")
    placed = scan.copy()
    placed.apply_transform(T)
    scan_surface = Surface(placed)
    a = target.signed(_apply(T, s_pts))[0]  # scan surface seen from the reference
    b = -scan_surface.signed(r_pts)[0]  # reference surface seen from the scan
    both = np.r_[a, b]

    report: dict = _spread(both) | {
        "within_pct": {f"{t:g}": round(float((np.abs(both) <= t).mean() * 100), 1) for t in (0.2, 0.5, 1.0)},
        "scan_surface": _spread(a), "reference_surface": _spread(b),
        "size_mm": {"scan": [round(float(v), 2) for v in placed.extents],
                    "reference": [round(float(v), 2) for v in ref.extents]},
    }
    if scan.is_watertight and ref.is_watertight:
        vs, vr = abs(float(scan.volume)), abs(float(ref.volume))
        report["volume_cm3"] = {"scan": round(vs / 1000, 2), "reference": round(vr / 1000, 2),
                                "diff_pct": round((vs / vr - 1) * 100, 1)}

    # Where does the error come from? Let the placement also tilt and lift, then also scale.
    T_rigid = _icp(fit_pts, target, T, DOF_RIGID, iters=30)
    T_scale = _icp(fit_pts, target, T_rigid, DOF_RIGID_SCALE, iters=30)
    s = float(np.cbrt(np.linalg.det(T_scale[:3, :3])))
    base_mid = np.r_[(scan.bounds[0][:2] + scan.bounds[1][:2]) / 2, 0.0]
    report["fit"] = {
        "tilt_deg": round(float(np.degrees(np.arccos(np.clip(T_rigid[2, 2], -1, 1)))), 2),
        "lift_mm": round(float(_apply(T_rigid, base_mid[None])[0, 2]), 2),
        "scale_error_pct": round((1 / s - 1) * 100, 2),  # + = the scan is too big
        "p95_free_mm": round(float(np.quantile(np.abs(target.signed(_apply(T_rigid, s_pts))[0]), 0.95)), 3),
        "p95_free_scaled_mm": round(float(np.quantile(np.abs(target.signed(_apply(T_scale, s_pts))[0]), 0.95)), 3),
    }
    if builtin:
        report["features"], report["checks"] = _piece_checks(r_pts, r_nrm, b, scan_surface)

    files: dict[str, bytes] = {}
    if heatmaps:
        files["deviation.glb"] = _viewer_glb(scan, target.signed(_apply(T, scan.vertices))[0], flat=False)
        fine = _refined(ref, 1.0)
        dev = -scan_surface.signed(fine.vertices)[0]
        fine.apply_transform(np.linalg.inv(T))  # shown in the scan's pose, like the other views
        files["deviation_reference.glb"] = _viewer_glb(fine, dev, flat=True)
    return report, files


# --- per-job reference -------------------------------------------------------------------------

def set_reference(job_dir: Path, filename: str | None = None, data: bytes | None = None) -> None:
    """Attach a reference to a job: an uploaded model file, or (no file) the built-in test piece."""
    ref_dir = job_dir / REF_DIR
    meta = {"builtin": True, "name": refpiece.NAME}
    staged = None
    if data is not None:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in REF_EXTS:
            raise ValueError(f"Reference must be one of {sorted(REF_EXTS)}")
        staged = job_dir / f"_reference{suffix}"
        staged.write_bytes(data)
        try:
            load_mesh(staged)
        except Exception as e:
            staged.unlink(missing_ok=True)
            raise ValueError(f"Could not read {filename}: {e}") from None
        meta = {"builtin": False, "name": Path(filename).name, "file": f"model{suffix}"}
    shutil.rmtree(ref_dir, ignore_errors=True)
    ref_dir.mkdir(parents=True)
    if staged is not None:
        staged.rename(ref_dir / meta["file"])
    (ref_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def clear_reference(job_dir: Path) -> None:
    shutil.rmtree(job_dir / REF_DIR, ignore_errors=True)
    for name in ("deviation.glb", "deviation_reference.glb"):
        try:
            (job_dir / "output" / name).unlink(missing_ok=True)
        except PermissionError:  # Windows: the page is still loading it; harmless leftover
            pass


def has_reference(job_dir: Path) -> bool:
    return (job_dir / REF_DIR / "meta.json").exists()


def update_stats(job_dir: Path, stats: dict, job: dict | None = None,
                 log: Callable[[str], None] = lambda _: None) -> dict:
    """stats with a fresh "accuracy" entry (or without one). Called after every run and rescale,
    so a job with a reference re-measures itself; it never fails the job."""
    stats = {k: v for k, v in stats.items() if k != "accuracy"}
    try:
        report = refresh(job_dir, stats, job, log)
    except Exception as e:
        log(f"Accuracy check failed: {e}")
        report = None
    if report:
        stats["accuracy"] = report
    return stats


def refresh(job_dir: Path, stats: dict, job: dict | None = None,
            log: Callable[[str], None] = lambda _: None) -> dict | None:
    """Recompute the job's accuracy report from its current print solid (after a run or a
    rescale). Returns None when the job has no reference or the scan has no real size."""
    stl = job_dir / "output" / "model.stl"
    if not has_reference(job_dir) or not stl.exists() or not stats.get("scale_source"):
        return None
    meta = json.loads((job_dir / REF_DIR / "meta.json").read_text(encoding="utf-8"))
    ref = refpiece.build() if meta["builtin"] else load_mesh(job_dir / REF_DIR / meta["file"])
    t0 = time.time()
    report, files = compare(load_mesh(stl), ref, builtin=meta["builtin"])
    for name, data in files.items():
        (job_dir / "output" / name).write_bytes(data)
    report["reference"] = meta["name"]
    report["builtin"] = meta["builtin"]
    report["heatmap_limit_mm"] = HEATMAP_LIMIT_MM
    log(f"Accuracy vs {meta['name']}: 95% of the surface within {report['p95_mm']} mm, "
        f"typical {report['median_mm']} mm, worst {report['max_mm']} mm ({time.time() - t0:.0f}s)")
    with HISTORY.open("a", encoding="utf-8") as f:  # one line per measurement: the trend across pipeline changes
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "job": job_dir.name,
                            "name": (job or {}).get("name"), "quality": (job or {}).get("quality"),
                            "reference": meta["name"],
                            **{k: report[k] for k in ("p95_mm", "median_mm", "max_mm", "bias_mm")},
                            "scale_error_pct": report["fit"]["scale_error_pct"]}) + "\n")
    return report


# --- command line ------------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="scan3d-accuracy",
        description="Compare a scanned STL with the model it was printed from (both in mm, Z up).")
    ap.add_argument("scan", nargs="?", type=Path, help="the scan's model.stl")
    ap.add_argument("reference", nargs="?", type=Path, help="known model; default: the built-in test piece")
    ap.add_argument("--piece", type=Path, metavar="OUT", help="write the test piece (.stl or .3mf) and exit")
    ap.add_argument("--views", type=Path, metavar="DIR", help="also write the coloured deviation GLBs here")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args()
    if args.piece:
        refpiece.build().export(args.piece)
        print(f"Wrote {args.piece}: {' x '.join(f'{v:g}' for v in refpiece.SIZE_MM)} mm")
        return
    if not args.scan:
        ap.error("give a scan STL, or --piece OUT")
    ref = load_mesh(args.reference) if args.reference else refpiece.build()
    try:
        report, files = compare(load_mesh(args.scan), ref, builtin=args.reference is None, heatmaps=bool(args.views))
    except ValueError as e:
        sys.exit(str(e))
    if args.views:
        args.views.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (args.views / name).write_bytes(data)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        return
    fit = report["fit"]
    print(f"95% of the surface within {report['p95_mm']} mm   typical {report['median_mm']} mm   "
          f"worst {report['max_mm']} mm   average {report['bias_mm']:+} mm (+ = scan too big)")
    print(f"scale error {fit['scale_error_pct']:+}%   floor tilt {fit['tilt_deg']} deg   floor offset {fit['lift_mm']:+} mm")
    for c in report.get("checks", []):
        got = "not in the scan" if c["measured_mm"] is None else f"{c['measured_mm']:.2f}  ({c['measured_mm'] - c['nominal_mm']:+.2f})"
        print(f"  {c['name']:<28}{c['nominal_mm']:>7.2f} -> {got}")


if __name__ == "__main__":
    main()
