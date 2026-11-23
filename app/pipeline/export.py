"""Mesh cleanup, ground removal, real-world scaling and export
(GLB for web/AR, STL for printing, OBJ bundle, PLY point cloud)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

# COLMAP's world (after IMAGE-ORIENTATION alignment) has +Y pointing down, glTF is +Y up.
COLMAP_TO_GLTF = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
COLMAP_UP = np.array([0.0, -1.0, 0.0])
# Slicers / printers are Z-up; our models are Y-up (glTF). Applied to STL and 3MF only.
Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
DIMENSIONS = ("height", "width", "depth", "longest")


# --------------------------------------------------------------------------- ground plane
def _fit_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
    return n, -float(n @ c)


def find_ground(vertices: np.ndarray, up: np.ndarray = COLMAP_UP, max_tilt_deg: float = 45.0,
                seed: int = 0) -> dict | None:
    """Find the floor/table the object stands on: the lowest large, roughly horizontal plane
    that has (almost) the whole model above it. Returns {"normal", "d", "noise"} with the
    normal pointing up, or None."""
    rng = np.random.default_rng(seed)
    pts = vertices if len(vertices) <= 150_000 else vertices[rng.choice(len(vertices), 150_000, replace=False)]
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    thresh = 0.004 * diag
    cos_tilt = np.cos(np.radians(max_tilt_deg))

    remaining = pts
    planes = []
    for _ in range(3):  # peel off up to 3 horizontal planes
        if len(remaining) < 500:
            break
        best, best_count = None, 0
        for _ in range(400):
            a, b, c = remaining[rng.choice(len(remaining), 3, replace=False)]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n /= norm
            if abs(n @ up) < cos_tilt:
                continue
            count = int((np.abs(remaining @ n - n @ a) < thresh).sum())
            if count > best_count:
                best, best_count = (n, a), count
        if best is None or best_count < 0.04 * len(pts):
            break
        n, a = best
        inl = np.abs(remaining @ n - n @ a) < thresh
        n, d = _fit_plane(remaining[inl])  # least-squares refinement
        if n @ up < 0:
            n, d = -n, -d
        planes.append((n, d, int(inl.sum()), remaining[inl]))
        remaining = remaining[~inl]

    if not planes:
        return None
    biggest = max(p[2] for p in planes)
    # lowest candidate first (height of plane along `up` is -d)
    for n, d, count, inliers in sorted(planes, key=lambda p: -p[1]):
        if count < 0.25 * biggest:
            continue
        h = pts @ n + d
        off_plane = np.abs(h) >= thresh
        if off_plane.sum() == 0 or (h[off_plane] > 0).mean() < 0.9:
            continue  # a box top etc. -- the model is mostly *below* it, not a floor
        noise = float(np.std(inliers @ n + d))
        return {"normal": n.tolist(), "d": float(d), "noise": noise}
    return None


def clean_mesh(src: Path, dst: Path, max_faces: int, keep_ratio: float = 0.2,
               remove_ground: bool = True, mat: dict | None = None) -> dict:
    mesh = trimesh.load(src, force="mesh", process=False)
    stats: dict = {"faces_raw": len(mesh.faces)}

    ground = None
    if mat:  # scan mat: we know exactly where the object stands, cut away everything else
        from ..scanmat import above_mat
        inside = above_mat(mesh.vertices, mat)[mesh.faces].all(axis=1)
        if inside.sum() > 1000:
            mesh.update_faces(inside)
            mesh.remove_unreferenced_vertices()
            stats["outside_mat_removed"] = int((~inside).sum())
    if remove_ground and mat:  # exact plane from the scan mat
        n, d = np.array(mat["plane"]["normal"]), float(mat["plane"]["d"])
        h = mesh.vertices @ n + d
        near = np.abs(h) < 1.5 / mat["mm_per_unit"]  # the sheet surface, within 1.5 mm
        noise = float(np.std(h[near])) if near.sum() > 100 else 0.3 / mat["mm_per_unit"]
        # at least 0.6 mm: paper thickness + slight curl, so no sheet fragments stay attached
        noise = max(noise, 0.2 / mat["mm_per_unit"])
        ground = {"normal": n.tolist(), "d": d, "noise": noise, "source": "scan_mat"}
    elif remove_ground:
        ground = find_ground(mesh.vertices)
    if ground:
        n, d = np.array(ground["normal"]), ground["d"]
        height = mesh.vertices @ n + d
        band = max(3.0 * ground["noise"], 1e-9)
        on_or_below = (height < band)[mesh.faces].all(axis=1)
        if on_or_below.mean() < 0.98:  # never delete (almost) everything
            mesh.update_faces(~on_or_below)
            mesh.remove_unreferenced_vertices()
            stats["ground_faces_removed"] = int(on_or_below.sum())
            stats["ground"] = ground

    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    counts = np.bincount(labels)
    keep = np.flatnonzero(counts >= keep_ratio * counts.max())
    mesh.update_faces(np.isin(labels, keep))
    mesh.remove_unreferenced_vertices()
    stats.update(components_kept=len(keep), components_total=len(counts))

    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception:  # fast-simplification missing or failed: keep full res
            pass
    stats["faces"] = len(mesh.faces)
    mesh.export(dst)
    return stats


# --------------------------------------------------------------------------- orientation
def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(a @ b)
    if s < 1e-9:
        return np.eye(4) if c > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    return trimesh.transformations.rotation_matrix(np.arctan2(s, c), axis / s)


def orientation_transform(vertices: np.ndarray, ground: dict | None) -> np.ndarray:
    """COLMAP frame -> Y-up, levelled on the ground plane, footprint aligned to X (longer
    horizontal side) and Z, centred, standing on y = 0."""
    T = COLMAP_TO_GLTF.copy()
    if ground:
        n_up = COLMAP_TO_GLTF[:3, :3] @ np.array(ground["normal"])
        T = _rotation_between(n_up, np.array([0.0, 1.0, 0.0])) @ T
    v = trimesh.transform_points(vertices, T)

    # Align the footprint's minimum-area rectangle with the X/Z axes.
    (_, _), (w, h), angle = cv2.minAreaRect(v[:, [0, 2]].astype(np.float32))
    yaw = np.radians(angle) + (np.pi / 2 if h > w else 0.0)
    T = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0]) @ T
    v = trimesh.transform_points(vertices, T)

    lo, hi = v.min(0), v.max(0)
    if ground:  # put the floor (not the lowest surviving vertex) at y = 0
        p0 = -ground["d"] * np.array(ground["normal"])
        floor_y = trimesh.transform_points(p0[None], T)[0, 1]
    else:
        floor_y = lo[1]
    center = (lo + hi) / 2
    return trimesh.transformations.translation_matrix([-center[0], -floor_y, -center[2]]) @ T


# --------------------------------------------------------------------------- export
def export_base(textured_obj: Path, dense_ply: Path, base: Path, ground: dict | None,
                mm_per_unit: float | None = None) -> dict:
    """Write the oriented, *unscaled* model once. Scaled outputs are derived from it, so the
    real-world size can be changed later without re-running the reconstruction."""
    base.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(textured_obj, process=False)
    scene = scene if isinstance(scene, trimesh.Scene) else trimesh.Scene(scene)
    verts = np.vstack([g.vertices for g in scene.geometry.values()])
    T = orientation_transform(verts, ground)
    scene.apply_transform(T)
    (base / "model.glb").write_bytes(scene.export(file_type="glb"))

    meta = {"grounded": bool(ground), "mm_per_unit": mm_per_unit, "model_to_base": T.tolist()}
    if dense_ply.exists():
        try:
            xyz, rgb, normals = read_openmvs_cloud(dense_ply)
            xyz = trimesh.transform_points(xyz, T)
            trimesh.PointCloud(xyz, colors=rgb).export(base / "points.ply")
            if normals is not None:  # kept for the watertight (print) reconstruction
                normals = normals @ T[:3, :3].T
                np.savez_compressed(base / "points_normals.npz", xyz=xyz.astype(np.float32),
                                    normals=normals.astype(np.float32), rgb=rgb)
            meta["points"] = len(xyz)
        except Exception as e:  # the cloud is a bonus output; never fail the job over it
            meta["pointcloud_error"] = str(e)
    (base / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def raw_dimensions(scene: trimesh.Scene | trimesh.Trimesh, grounded: bool) -> dict:
    lo, hi = scene.bounds
    height = hi[1] - (min(lo[1], 0.0) if grounded else lo[1])
    return {"width": float(hi[0] - lo[0]), "depth": float(hi[2] - lo[2]), "height": float(height)}


def scale_factor_mm(raw: dict, size: dict | None) -> float | None:
    """mm per model unit from one known real dimension, e.g. {"dimension": "height", "mm": 95}."""
    if not size or not size.get("mm"):
        return None
    dim = size.get("dimension", "longest")
    ref = max(raw.values()) if dim == "longest" else raw[dim]
    return float(size["mm"]) / ref if ref > 0 else None


def write_outputs(base: Path, out: Path, size: dict | None = None) -> dict:
    """base/ -> output/: GLB in metres, STL/OBJ/PLY in millimetres (when a real size is known)."""
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    scene = trimesh.load(base / "model.glb", process=False)
    solid_raw = trimesh.load(base / "solid.ply", force="mesh", process=False) if (base / "solid.ply").exists() else None
    # Measure what gets printed: the solid when there is one, otherwise the textured surface.
    raw = raw_dimensions(solid_raw if solid_raw is not None else scene,
                         True if solid_raw is not None else meta.get("grounded", False))
    # Scale precedence: a size the user typed > the scan mat > unscaled.
    mm = scale_factor_mm(raw, size)
    source = "manual" if mm else None
    if not mm and meta.get("mm_per_unit"):
        mm, source = float(meta["mm_per_unit"]), "scan_mat"
    k_mm = mm or 1.0

    def scaled(k: float) -> trimesh.Scene:
        s = scene.copy()
        s.apply_scale(k)
        return s

    for f in out.glob("*"):
        try:
            f.unlink()
        except PermissionError:  # Windows: still being downloaded; it is overwritten below instead
            pass
    stats: dict = {"outputs": []}

    glb = scaled(k_mm / 1000.0 if mm else 1.0)  # glTF unit is the metre
    (out / "model.glb").write_bytes(glb.export(file_type="glb"))
    stats["faces"] = int(sum(len(g.faces) for g in glb.geometry.values()))

    mesh_mm = scaled(k_mm).to_geometry()

    # Print files: the watertight solid (falls back to the raw surface), mm, Z-up.
    from .printable import printability  # local import: printable imports this module's siblings
    if solid_raw is not None:
        solid = solid_raw.copy()
        solid.apply_scale(k_mm)
    else:
        solid = mesh_mm.copy()
        solid.merge_vertices()
    stats["print"] = printability(solid, scaled=bool(mm)) | {"solid": (base / "solid.ply").exists()}
    preview = solid.copy()  # Y-up, metres: what the viewer shows in "Print" mode
    preview.apply_scale(1 / 1000.0 if mm else 1.0)
    preview.visual = trimesh.visual.ColorVisuals(preview, face_colors=[205, 205, 200, 255])
    (out / "print_preview.glb").write_bytes(preview.export(file_type="glb"))
    solid_z = solid.copy()
    solid_z.apply_transform(Y_UP_TO_Z_UP)
    solid_z.export(out / "model.stl")  # STL has no colour or texture
    (out / "model.3mf").write_bytes(solid_z.export(file_type="3mf"))

    obj_text, textures = trimesh.exchange.obj.export_obj(
        mesh_mm, include_texture=True, return_texture=True, mtl_name="model.mtl")
    with zipfile.ZipFile(out / "model_obj.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.obj", obj_text)
        for name, data in textures.items():
            z.writestr(name, data)

    stats["outputs"] = ["model.glb", "model.stl", "model.3mf", "model_obj.zip"]
    if (base / "points.ply").exists():
        cloud = trimesh.load(base / "points.ply", process=False)
        cloud.apply_scale(k_mm)
        cloud.export(out / "pointcloud.ply")
        stats["outputs"].append("pointcloud.ply")
        stats["points"] = meta.get("points")

    unit = "mm" if mm else "units"
    stats["dimensions"] = {k: round(v * k_mm, 2) for k, v in raw.items()} | {"unit": unit}
    stats["size"] = size if source == "manual" else None
    stats["scale_source"] = source
    return stats


_PLY_TYPES = {"float32": "f4", "float": "f4", "float64": "f8", "double": "f8", "uint8": "u1",
              "uchar": "u1", "int8": "i1", "char": "i1", "uint16": "u2", "ushort": "u2", "int16": "i2",
              "short": "i2", "uint32": "u4", "uint": "u4", "int32": "i4", "int": "i4"}


def read_openmvs_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read xyz + rgb from OpenMVS's binary PLY, whose vertices carry variable-length
    `view_indices` / `view_weights` lists that generic PLY readers choke on."""
    data = path.read_bytes()
    end = data.index(b"end_header") + len(b"end_header")
    end += 2 if data[end:end + 2] == b"\r\n" else 1
    header = data[:end].decode("ascii", "replace").splitlines()
    if not any("binary_little_endian" in l for l in header):
        raise ValueError("unsupported PLY format")
    count, fixed, lists = 0, [], []  # fixed: (name, dtype) before the lists
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
        elif parts[:2] == ["property", "list"]:
            lists.append((np.dtype(_PLY_TYPES[parts[2]]), np.dtype(_PLY_TYPES[parts[3]])))
        elif parts[:1] == ["property"]:
            if lists:
                raise ValueError("scalar property after list property")
            fixed.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
    rec = np.dtype(fixed)
    buf = np.frombuffer(data, dtype=np.uint8, offset=end)
    if not lists:
        verts = np.frombuffer(buf[:count * rec.itemsize].tobytes(), dtype=rec)
    else:  # walk the records once to find where each one starts
        starts = np.empty(count, dtype=np.int64)
        off = 0
        for i in range(count):
            starts[i] = off
            off += rec.itemsize
            for cnt_t, item_t in lists:
                n = int(buf[off]) if cnt_t.itemsize == 1 else int.from_bytes(buf[off:off + cnt_t.itemsize], "little")
                off += cnt_t.itemsize + n * item_t.itemsize
        idx = starts[:, None] + np.arange(rec.itemsize)[None, :]
        verts = np.frombuffer(buf[idx].tobytes(), dtype=rec)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    if {"red", "green", "blue"} <= set(verts.dtype.names):
        rgb = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.uint8)
    else:
        rgb = np.full((count, 3), 200, np.uint8)
    normals = None
    if {"nx", "ny", "nz"} <= set(verts.dtype.names):
        normals = np.stack([verts["nx"], verts["ny"], verts["nz"]], axis=1).astype(np.float64)
    return xyz, rgb, normals

