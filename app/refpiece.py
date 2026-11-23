"""The scan3d test piece: a small part with exactly known geometry, to print and then scan.

Comparing the scan against this model gives the pipeline's real error in millimetres, per kind
of feature, each of which exercises a different weakness of photo scanning:

    steps     flat tops at 12 / 20 / 28 mm        height accuracy
    hole      blind hole, 8 mm wide, 8 mm deep    hollows that outlines cannot see
    dome      10 mm hemisphere on a cylinder      rounded tops (too tall without side-on views)
    arch      14 x 12 mm tunnel                   gaps under overhangs
    slope     45 degree ramp                      slanted faces
    ribs      1, 2 and 3 mm wide ridges           the smallest detail that survives
    plate     80 x 50 x 4 mm base                 overall width / depth

Z is up, millimetres, base on Z = 0. It has no symmetry, so a scan can only match it one way.
Prints without supports (the arch is a 14 mm bridge).
"""
from __future__ import annotations

import manifold3d as m3
import numpy as np
import trimesh

NAME = "scan3d test piece"
SIZE_MM = (80.0, 50.0, 28.0)

PLATE_Z = 4.0
STEPS = ((4.0, 16.0, 12.0), (16.0, 28.0, 20.0), (28.0, 40.0, 28.0))  # x0, x1, top z
STEP_Y = (4.0, 24.0)
HOLE = (22.0, 14.0, 4.0, 12.0)  # centre x, y, radius, bottom z (opens in the 20 mm step)
DOME = (60.0, 14.0, 10.0, 18.0)  # centre x, y, radius, z where the cylinder ends and the dome starts
ARCH = (4.0, 34.0, 30.0, 46.0, 24.0)  # x0, x1, y0, y1, top z
TUNNEL = (12.0, 26.0, 16.0)  # x0, x1, top z (runs through the arch along y)
SLOPE = (40.0, 62.0, 30.0, 46.0, 26.0)  # x0, x1, y0, y1, top z at x1
RIBS = ((66.0, 67.0), (69.0, 71.0), (73.0, 76.0))  # x0, x1 of each ridge
RIB_Y = (30.0, 46.0)
RIB_Z = 8.0

# Order of the per-feature rows in a report.
FEATURES = ("plate", "steps", "hole", "dome", "cylinder", "arch", "arch gap", "slope", "ribs")


def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> m3.Manifold:
    return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def build() -> trimesh.Trimesh:
    parts = [_box(0, SIZE_MM[0], 0, SIZE_MM[1], 0, PLATE_Z)]
    parts += [_box(x0, x1, *STEP_Y, PLATE_Z, top) for x0, x1, top in STEPS]
    cx, cy, r, z = DOME
    parts.append(m3.Manifold.cylinder(z - PLATE_Z, r, r, 128).translate([cx, cy, PLATE_Z]))
    parts.append(m3.Manifold.sphere(r, 128).translate([cx, cy, z]))
    ax0, ax1, ay0, ay1, atop = ARCH
    parts.append(_box(ax0, ax1, ay0, ay1, PLATE_Z, atop))
    sx0, sx1, sy0, sy1, stop = SLOPE
    parts.append(m3.Manifold.hull_points(np.array(
        [[sx0, y, PLATE_Z] for y in (sy0, sy1)] + [[sx1, y, PLATE_Z] for y in (sy0, sy1)]
        + [[sx1, y, stop] for y in (sy0, sy1)], dtype=np.float64)))
    parts += [_box(x0, x1, *RIB_Y, PLATE_Z, RIB_Z) for x0, x1 in RIBS]
    solid = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
    hx, hy, hr, hz = HOLE
    solid -= m3.Manifold.cylinder(30.0, hr, hr, 96).translate([hx, hy, hz])
    tx0, tx1, ttop = TUNNEL
    solid -= _box(tx0, tx1, ay0 - 1, ay1 + 1, PLATE_Z, ttop)
    mesh = solid.to_mesh()
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=False)


def feature_of(points: np.ndarray) -> np.ndarray:
    """Name of the feature each surface point (in the piece's own coordinates) belongs to."""
    x, y, z = points.T
    eps = 0.05
    out = np.full(len(points), "plate", dtype=object)
    above = z > PLATE_Z + eps

    in_steps = (x > STEPS[0][0] - eps) & (x < STEPS[-1][1] + eps) & (y > STEP_Y[0] - eps) & (y < STEP_Y[1] + eps)
    out[in_steps & above] = "steps"
    hx, hy, hr, _ = HOLE
    out[(np.hypot(x - hx, y - hy) < hr + eps) & (z < STEPS[1][2] - eps)] = "hole"

    cx, cy, r, dz = DOME
    near_post = np.hypot(x - cx, y - cy) < r + eps
    out[near_post & above] = "cylinder"
    out[near_post & (z > dz + eps)] = "dome"

    ax0, ax1, ay0, ay1, _ = ARCH
    in_arch = (x > ax0 - eps) & (x < ax1 + eps) & (y > ay0 - eps) & (y < ay1 + eps)
    out[in_arch & above] = "arch"
    tx0, tx1, ttop = TUNNEL
    out[in_arch & (x > tx0 - eps) & (x < tx1 + eps) & (z < ttop + eps) & (y > ay0 + eps) & (y < ay1 - eps)] = "arch gap"

    sx0, sx1, sy0, sy1, _ = SLOPE
    out[(x > sx0 - eps) & (x < sx1 + eps) & (y > sy0 - eps) & (y < sy1 + eps) & above] = "slope"
    # The plate between the ridges counts too: that is where a blurred scan fills in.
    out[(x > RIBS[0][0] - eps) & (x < RIBS[-1][1] + eps) & (y > RIB_Y[0] - eps) & (y < RIB_Y[1] + eps)
        & (z > PLATE_Z - eps)] = "ribs"
    return out
