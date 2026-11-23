"""Print-ready solid: a single closed (watertight, manifold) mesh with a flat base.

Photogrammetry surfaces are open (no bottom where the object touched the floor, holes where
the camera never looked) and often in several pieces, which slicers cannot print reliably.

  dense points near the object (outward normals)
     -> Screened Poisson reconstruction (COLMAP poisson_mesher, no trimming: closed surface)
     -> voxel fill (closes remaining gaps, seals the base at the floor) -> marching cubes
     -> manifold3d: exact flat base, simplify, keep the largest solid
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import manifold3d
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .. import config

PLA_DENSITY = 1.24  # g/cm3
RunFn = Callable[[list, str], None]


def build_volume_mm() -> tuple[float, float, float]:
    """Printer build volume, X x Y x Z in mm (default fits Bambu A1/P1/X1, Prusa MK4 is 250x210x220)."""
    raw = os.environ.get("SCAN3D_BUILD_VOLUME", "256x256x256")
    try:
        x, y, z = (float(v) for v in raw.lower().split("x"))
        return x, y, z
    except ValueError:
        return 256.0, 256.0, 256.0


def _write_oriented_ply(path: Path, xyz: np.ndarray, normals: np.ndarray, rgb: np.ndarray) -> None:
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                                     ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz.T
    rec["nx"], rec["ny"], rec["nz"] = normals.T
    rec["red"], rec["green"], rec["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def _to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    m = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    if m.status() != manifold3d.Error.NoError or m.is_empty():
        raise ValueError(f"mesh is not manifold ({m.status()})")
    return m


def _to_trimesh(m: manifold3d.Manifold) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def _oriented_points(base: Path, surface: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense points that lie on the object, normals flipped to agree with the (outward) surface."""
    diag = float(np.linalg.norm(surface.extents))
    tree = cKDTree(surface.vertices)
    vnormals = surface.vertex_normals
    npz = base / "points_normals.npz"
    if npz.exists():
        d = np.load(npz)
        xyz, normals, rgb = d["xyz"].astype(np.float64), d["normals"].astype(np.float64), d["rgb"]
        dist, idx = tree.query(xyz, distance_upper_bound=0.006 * diag)
        keep = np.isfinite(dist) & (xyz[:, 1] > 0.0)
        xyz, normals, rgb, idx = xyz[keep], normals[keep], rgb[keep], idx[keep]
        # Statistical outlier removal: drop specks far from their neighbours (they become bumps).
        if len(xyz) > 1000:
            knn = cKDTree(xyz).query(xyz, k=9)[0][:, 1:].mean(1)
            ok = knn < knn.mean() + 2.5 * knn.std()
            xyz, normals, rgb, idx = xyz[ok], normals[ok], rgb[ok], idx[ok]
        flip = (normals * vnormals[idx]).sum(1) < 0
        normals[flip] *= -1
        if len(xyz) >= 5000:
            return xyz, normals, rgb
    # Fallback: sample the surface mesh itself.
    pts, face_idx = trimesh.sample.sample_surface_even(surface, 200_000)
    return pts, surface.face_normals[face_idx], np.full((len(pts), 3), 200, np.uint8)


HullFn = Callable[[np.ndarray, float, np.ndarray], np.ndarray]
# Share of the scan's surface with measured points on it, from which the scan counts as complete.
# Measured: textured box 99.6 %, glossy earbud case 82 %, black 3D-printed stand 65-76 %.
COMPLETE_SCAN = 0.95


def _scan_coverage(shell: trimesh.Trimesh, xyz: np.ndarray, surface: trimesh.Trimesh) -> float:
    """How much of the closed (Poisson) surface is backed by measured dense points, rather than
    bridged over a hole: the share of it within 1 % of the object's diagonal of a point. The
    never-seen underside (lowest 3 % of the height) and anything outside the scan's box is left out."""
    lo, hi = surface.bounds
    diag = float(np.linalg.norm(hi - lo))
    pts, _ = trimesh.sample.sample_surface(shell, 100_000, seed=0)
    inside = np.all((pts > lo - 0.02 * diag) & (pts < hi + 0.02 * diag), axis=1) & (pts[:, 1] > 0.03 * (hi[1] - lo[1]))
    if not inside.any() or len(xyz) == 0:
        return 0.0
    dist, _ = cKDTree(xyz).query(pts[inside], distance_upper_bound=0.01 * diag)
    return float(np.isfinite(dist).mean())


def build_solid(base: Path, work: Path, run: RunFn, log: Callable[[str], None],
                depth: int = 10, resolution: int = 320, hull: HullFn | None = None) -> dict:
    """Writes base/solid.ply (oriented, unscaled, watertight). Returns a short report."""
    scene = trimesh.load(base / "model.glb", process=False)
    surface = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    surface.merge_vertices()
    xyz, normals, rgb = _oriented_points(base, surface)
    log(f"Print solid: {len(xyz):,} points on the object")

    pts_ply, poisson_ply = work / "print_points.ply", work / "print_poisson.ply"
    _write_oriented_ply(pts_ply, xyz, normals, rgb)
    shell = None
    # COLMAP's PoissonRecon randomly aborts ("Failed to close loop") in roughly 1 of 4 multi-threaded
    # runs on the same input; retrying usually works, single-threaded is steadier, coarser last.
    attempts = [(depth, config.THREADS), (depth, config.THREADS), (depth, 1), (depth - 1, 1)]
    for d, threads in attempts:
        try:
            run([config.COLMAP_BIN, "poisson_mesher", "--input_path", pts_ply, "--output_path", poisson_ply,
                 "--PoissonMeshing.depth", str(d), "--PoissonMeshing.trim", "0", "--PoissonMeshing.point_weight", "4",
                 "--PoissonMeshing.num_threads", str(threads)], "Watertight reconstruction")
            shell = trimesh.load(poisson_ply, force="mesh", process=False)
            break
        except Exception as e:
            log(f"Poisson (depth {d}, {threads} threads) failed, retrying: {str(e).splitlines()[0][:120]}")
    coverage = _scan_coverage(shell, xyz, surface) if shell is not None else 0.0
    if shell is None:  # last resort: the voxel fill closes the scanned surface's holes itself
        log("Building the print solid from the scanned surface instead")
        shell = surface
    # A complete scan is more exact than outlines, which over-size anything the cameras did not
    # see side-on (tested on a textured 60 x 40 x 90 mm box: +7 to +11 mm from outlines, under
    # 2 mm from the scan). So outlines add material only when the scan has holes.
    complete = coverage >= COMPLETE_SCAN
    if hull is not None:
        log(f"Scan completeness: {coverage:.0%} of the surface was measured -> outlines "
            + ("only trim what sticks out of the scan" if complete else "give the outer shape, the scan its hollows"))

    poisson = shell
    lo, hi = surface.bounds
    margin = 0.02 * float(np.linalg.norm(surface.extents))
    floor_y = 0.0 if lo[1] > -margin else lo[1]  # y = 0 is the detected floor
    glo, ghi = lo, hi
    if hull is not None and not complete:  # the outline may show the object reaching beyond the patchy scan
        glo, ghi = lo - 0.1 * (hi - lo), hi + np.array([0.1, 0.35, 0.1]) * (hi - lo)
    out = _voxel_solid(poisson, glo - margin, ghi + margin, floor_y, resolution, top_surface=surface,
                       hull=hull, trim_only=complete, log=log)
    # A solid much smaller than the scan means only part of it was closed (big holes): don't ship it.
    ratio = out.extents / np.maximum(hi - np.maximum(lo, floor_y), 1e-9)
    if ratio.min() < 0.85:
        raise ValueError(f"the closed solid covers only {ratio.min():.0%} of the scan (holes too large)")
    out.export(base / "solid.ply")
    report = {"watertight": bool(out.is_watertight), "faces": len(out.faces), "coverage": round(coverage, 3)}
    if hull is not None:
        report["outlines"] = "trim" if complete else "shape"
    log(f"Print solid: watertight={report['watertight']} faces={report['faces']:,}")
    return report


def _voxel_solid(shape: trimesh.Trimesh, lo: np.ndarray, hi: np.ndarray, floor_y: float,
                 resolution: int, top_surface: trimesh.Trimesh | None = None,
                 hull: HullFn | None = None, trim_only: bool = False,
                 log: Callable[[str], None] = print) -> trimesh.Trimesh:
    """Closed shell -> filled voxel volume -> smooth marching-cubes surface -> manifold solid
    with an exactly flat base at floor_y. Robust to small holes in the input shell."""
    from scipy import ndimage
    from skimage.measure import marching_cubes

    lo = lo.copy()
    lo[1] = max(lo[1], floor_y)
    pitch = float((hi - lo).max()) / resolution
    pad = 3
    origin = lo - pad * pitch
    dims = np.ceil((hi - lo) / pitch).astype(int) + 2 * pad + 1

    # Rasterise the shell by dense surface sampling (much faster than exact voxelisation).
    n = int(min(6_000_000, max(200_000, 4 * shape.area / pitch**2)))
    pts, _ = trimesh.sample.sample_surface(shape, n, seed=0)
    pts = np.vstack([pts, shape.vertices])
    ijk = np.floor((pts - origin) / pitch).astype(int)
    ok = np.all((ijk >= 0) & (ijk < dims), axis=1)
    shell = np.zeros(dims, dtype=bool)
    shell[tuple(ijk[ok].T)] = True

    floor_j = int(np.floor((floor_y - origin[1]) / pitch))
    occupied = np.flatnonzero(shell.any(axis=(0, 2)))
    low_j = max(floor_j, int(occupied[0])) if len(occupied) else floor_j
    filled = None
    for closing in (1, 2, 4, 8, 12):  # close gaps in the shell, more aggressively if needed
        s = ndimage.binary_dilation(shell, iterations=closing)
        # Seal the base: the scan's open bottom rim (its lowest few slices, which may sit a little
        # above the floor once the floor was cut away) is filled in and extended down to the floor.
        rim = s[:, low_j:low_j + closing + 2, :].any(axis=1)
        base = ndimage.binary_fill_holes(rim)
        for j in range(floor_j, low_j + closing + 2):
            s[:, j, :] |= base
        f = ndimage.binary_fill_holes(s)
        f = ndimage.binary_erosion(f, iterations=closing, border_value=0) | shell
        if f.sum() > 1.5 * shell.sum():  # interior got filled, not just the shell
            filled = f
            break
    if filled is None:
        raise ValueError("could not close the surface into a solid")
    filled[:, :floor_j + 1, :] = False  # everything at/below the floor goes
    labels, count = ndimage.label(filled)
    if count > 1:
        filled = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)

    if top_surface is not None:  # first undo Poisson sags, so they don't look like real hollows
        _support_top(filled, top_surface, origin, pitch, floor_j)
    if hull is not None:
        filled, shell = _apply_hull(filled, shell, hull(origin, pitch, dims), floor_j, log, trim_only)

    # Voxels the surface passes through are only partly inside: weight them 0.5 so the iso-surface
    # sits on the scanned surface instead of half a voxel outside it.
    occupancy = filled.astype(np.float32)
    occupancy[shell & filled] = 0.5
    field = np.pad(ndimage.gaussian_filter(occupancy, sigma=0.8), 2)  # padding: never an open surface
    verts, faces, _, _ = marching_cubes(field, level=0.5 - 1e-3)  # off-grid level: exact hits duplicate vertices
    verts = (verts - 2) * pitch + origin
    mesh = trimesh.Trimesh(verts, faces[:, ::-1], process=False)
    if mesh.volume < 0:
        mesh.invert()

    solid = _to_manifold(mesh)
    solid = solid.trim_by_plane((0.0, 1.0, 0.0), floor_y + 1.5 * pitch)  # exactly flat base
    solid = solid.translate((0.0, -(floor_y + 1.5 * pitch), 0.0))   # base at y = 0
    solid = solid.simplify(0.25 * pitch)
    pieces = solid.decompose()
    if not pieces:
        raise ValueError("solid became empty")
    return _to_trimesh(max(pieces, key=lambda p: p.volume()))


def _apply_hull(filled: np.ndarray, shell: np.ndarray, hull: np.ndarray, floor_j: int,
                log: Callable[[str], None], trim_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Silhouette hull as the outer shape; the scan only removes hollows the outline can't see.

    The hull is closed and correct wherever the outline shows the shape (it fills the scan's holes
    on black / shiny parts and drops its lumps), but it cannot see a hollow whose opening never
    shows in outline (a cup's inside, a recess in a top face). Those come from the scan: hull
    voxels far inside the scan's empty space are removed, where "far" (a fifth of the object's
    size) is more than any hole a patchy scan leaves in a surface."""
    from scipy import ndimage

    hull = hull.copy()
    hull[:, :floor_j + 1, :] = False
    if hull.sum() < 0.3 * filled.sum():  # outlines clearly failed (wrong object, bad prompt)
        log("Silhouettes: hull much smaller than the scan, ignoring it")
        return filled, shell
    if trim_only:
        # Complete scan: keep it, minus whatever lies outside the outlines (lumps, a skirt at the
        # floor). Two voxels of slack, so a slightly tight outline never shaves a measured surface.
        solid = filled & ndimage.binary_dilation(hull, iterations=2)
        labels, count = ndimage.label(solid)
        if count > 1:
            solid = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
        log(f"Silhouettes: scan kept, {(filled & ~solid).sum():,} of {filled.sum():,} voxels outside the outlines trimmed")
        edge = solid & ~ndimage.binary_erosion(solid)
        cut = edge & ndimage.binary_dilation(filled & ~solid)  # new surface where something was trimmed
        return solid, (shell & solid) | cut
    size = max(np.ptp(np.argwhere(hull), axis=0)) + 1
    far_empty = hull & (ndimage.distance_transform_edt(~filled) > 0.2 * size)
    # Only hollows that open to the outside are real (a cup's inside); a pocket sealed inside the
    # object is just where the scan saw nothing, and a scan cannot see into a closed object anyway.
    outside = ~hull
    outside[:, :floor_j + 1, :] = False  # the floor seals the bottom; it is not an opening
    labels, _ = ndimage.label(far_empty | outside)
    open_ids = np.unique(labels[outside])
    far_empty &= np.isin(labels, open_ids[open_ids > 0])
    solid = hull & ~far_empty
    labels, count = ndimage.label(solid)
    if count > 1:
        solid = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    log(f"Silhouettes: print solid = outline hull ({hull.sum():,} voxels), "
        f"{(hull & far_empty).sum():,} removed as hollows seen by the scan")
    # The hull's surface is exact (not a scanned shell): weight its boundary voxels like the shell.
    edge = solid & ~ndimage.binary_erosion(solid)
    return solid, (shell & solid) | edge


def _support_top(filled: np.ndarray, surface: trimesh.Trimesh, origin: np.ndarray, pitch: float,
                 floor_j: int) -> np.ndarray | None:
    """Never let the solid sag below the top the cameras actually saw.

    Where the scan has holes in its top (shiny or dark tops reflect instead of showing texture),
    Poisson bridges them with a surface that can dip deep into the object, and mirror-image points
    from reflections pull it further in. For every column whose solid ends clearly below the
    scanned top (holes in the top are interpolated from their surroundings), fill up to that top.
    Overhangs are untouched: under a handle the solid already reaches the scanned top.
    Returns the scanned top height per column (voxel index, holes interpolated; -1 = none)."""
    from scipy import ndimage

    nx, ny, nz = filled.shape
    pts, _ = trimesh.sample.sample_surface(surface, int(min(3_000_000, max(200_000, 2 * surface.area / pitch**2))),
                                           seed=1)
    ijk = np.floor((pts - origin) / pitch).astype(int)
    ok = np.all((ijk >= 0) & (ijk < filled.shape), axis=1) & (ijk[:, 1] > floor_j)
    ijk = ijk[ok]
    top = np.full((nx, nz), -1, dtype=int)
    np.maximum.at(top, (ijk[:, 0], ijk[:, 2]), ijk[:, 1])
    top = ndimage.median_filter(top, size=3)  # single-column spikes are noise

    footprint = filled.any(axis=1)
    known = top > floor_j
    holes = footprint & ~known
    if known.sum() < 50:
        return None
    if holes.any():  # inpaint missing tops with the nearest seen heights, then smooth that fill
        idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        guess = ndimage.uniform_filter(top[idx[0], idx[1]].astype(float), size=7)
        top = np.where(holes, np.round(guess).astype(int), top)

    has = filled.any(axis=1)
    solid_top = np.where(has, ny - 1 - np.argmax(filled[:, ::-1, :], axis=1), -1)
    # Only raise a column whose current top was never seen (a Poisson bridge over a hole). If the
    # scan has surface right at the solid's top, that top is real and the space above it is a
    # genuine gap (under a handle or an overhanging plate), so leave it open.
    seen = np.zeros(filled.shape, dtype=bool)
    seen[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    seen = ndimage.binary_dilation(seen, structure=np.ones((3, 5, 3), bool))
    ci, ck = np.nonzero(has)
    top_seen = np.zeros_like(has)
    top_seen[ci, ck] = seen[ci, solid_top[ci, ck], ck]
    # Only inside the footprint: on its rim, side-wall samples would thicken the walls.
    # Never near an overhang: columns with solid, then a gap, then solid again (a plate over a base,
    # a handle) mean the space under the upper part is real, and raising would build pillars there.
    runs = (filled[:, 1:, :] & ~filled[:, :-1, :]).sum(axis=1) + filled[:, 0, :]
    # (wide margin: a hole in an upper plate can be a fair fraction of the object across)
    reach = max(8, int(0.15 * max(filled.shape[0], filled.shape[2])))
    over = runs >= 2
    near_overhang = ndimage.distance_transform_edt(~over) <= reach if over.any() else over
    raise_cols = (ndimage.binary_erosion(footprint, iterations=3) & (top > solid_top + 2) & ~top_seen
                  & ~near_overhang)
    if not raise_cols.any():
        return top
    ii, kk = np.nonzero(raise_cols)
    ys = np.arange(ny)[None, :]
    lo_y = np.maximum(solid_top[ii, kk], floor_j)[:, None]
    fill = (ys > lo_y) & (ys <= top[ii, kk][:, None])
    for col, (i, k) in enumerate(zip(ii, kk)):
        filled[i, fill[col], k] = True
    return top


def printability(solid_mm: trimesh.Trimesh, scaled: bool) -> dict:
    """Facts a slicer would complain about, plus volume / material for real-size models."""
    ext = solid_mm.extents
    rep: dict = {"watertight": bool(solid_mm.is_watertight),
                 "winding_consistent": bool(solid_mm.is_winding_consistent),
                 "flat_base": bool(abs(solid_mm.bounds[0][1]) < 1e-6 * max(ext.max(), 1)),
                 "faces": int(len(solid_mm.faces))}
    if scaled and rep["watertight"]:
        vol_cm3 = abs(float(solid_mm.volume)) / 1000.0
        rep["volume_cm3"] = round(vol_cm3, 2)
        rep["pla_solid_g"] = round(vol_cm3 * PLA_DENSITY, 1)
    if scaled:
        bx, by, bz = build_volume_mm()
        # The model is Y-up; on the printer the base lies on X/Y and height is Z.
        w, h, d = ext
        fits = (max(w, d) <= max(bx, by) and min(w, d) <= min(bx, by)) and h <= bz
        rep["build_volume_mm"] = [bx, by, bz]
        rep["fits_build_volume"] = bool(fits)
    rep["printable"] = rep["watertight"] and rep["winding_consistent"]
    return rep
