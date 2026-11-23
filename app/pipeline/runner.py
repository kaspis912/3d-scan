"""Photogrammetry pipeline:  images -> COLMAP (SfM) -> OpenMVS (MVS, mesh, texture) -> GLB.

Every heavy step is an external open-source CLI, run as a subprocess so it can be
logged, cancelled and swapped out independently.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh


from .. import accuracy, config
from .. import scanmat
from . import export, prepare, printable, silhouette

ProgressFn = Callable[[str, float], None]


class PipelineError(RuntimeError):
    pass


class PipelineCancelled(RuntimeError):
    pass


# (stage label, progress % at the *start* of the stage)
STAGES = {
    "prepare": ("Preparing images", 1),
    "mask": ("Removing backgrounds", 6),
    "features": ("Detecting features", 10),
    "matching": ("Matching images", 20),
    "mapping": ("Solving camera poses", 34),
    "undistort": ("Undistorting images", 50),
    "scale": ("Looking for the scan mat", 52),
    "densify": ("Building dense point cloud", 53),
    "mesh": ("Reconstructing mesh", 74),
    "refine": ("Refining mesh", 80),
    "texture": ("Texturing", 87),
    "export": ("Exporting", 92),
    "outline": ("Tracing the object's outline", 93),
    "solid": ("Making print-ready solid", 96),
}


def _num_registered(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    if not images_bin.exists():
        return 0
    with images_bin.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


class Reconstruction:
    def __init__(self, job_dir: Path, quality: str = config.DEFAULT_QUALITY, *, mask: bool = False,
                 size: dict | None = None, on_progress: ProgressFn | None = None,
                 cancel: threading.Event | None = None):
        self.job_dir = job_dir
        self.input_dir = job_dir / "input"
        self.work = job_dir / "work"
        self.out = job_dir / "output"
        self.log_path = job_dir / "log.txt"
        self.q = config.QUALITIES.get(quality, config.QUALITIES[config.DEFAULT_QUALITY])
        self.mask = mask
        self.size = size  # known real dimension, e.g. {"dimension": "height", "mm": 95}
        self.base = job_dir / "base"
        self._ground: dict | None = None
        self._mat: dict | None = None
        self.on_progress = on_progress or (lambda stage, pct: None)
        self.cancel = cancel or threading.Event()
        self.stats: dict = {"quality": self.q.name}
        self._proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def stage(self, key: str) -> None:
        if self.cancel.is_set():
            raise PipelineCancelled("Cancelled")
        label, pct = STAGES[key]
        self.log(f"=== {label}")
        self.on_progress(label, pct)

    def run(self, args: list[str], what: str, cwd: Path | None = None) -> None:
        if self.cancel.is_set():
            raise PipelineCancelled("Cancelled")
        self.log("$ " + " ".join(str(a) for a in args))
        # Own process group / no console window, so the tool neither receives the service's
        # Ctrl+C nor pops up a window; cancelling terminates it explicitly below.
        detach = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                  if config.IS_WINDOWS else {"start_new_session": True})
        with self.log_path.open("ab") as logf:
            self._proc = subprocess.Popen([str(a) for a in args], cwd=cwd, stdout=logf,
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          env=config.tool_env(str(args[0])), **detach)
            while self._proc.poll() is None:
                if self.cancel.is_set():
                    self._proc.terminate()
                    try:
                        self._proc.wait(10)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                    raise PipelineCancelled("Cancelled")
                time.sleep(0.3)
            code = self._proc.returncode
        self._proc = None
        if code != 0:
            raise PipelineError(f"{what} failed (exit code {code}). {self._tail()}")

    def _tail(self, n: int = 6) -> str:
        try:
            lines = [l for l in self.log_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
            return " | ".join(lines[-n:])[-600:]
        except OSError:
            return ""

    # ------------------------------------------------------------------ pipeline
    def execute(self) -> dict:
        t0 = time.time()
        shutil.rmtree(self.out, ignore_errors=True)  # never show a previous run's model
        shutil.rmtree(self.base, ignore_errors=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.check_tools()

        images = self.work / "images"
        if images.exists():  # retry: start the image set from scratch
            shutil.rmtree(images)
        self.stage("prepare")
        try:
            info = prepare.prepare_images(self.input_dir, images, self.q, self.run, self.log)
        except ValueError as e:
            raise PipelineError(str(e)) from e
        self.stats.update(info)
        prepare.make_thumbnail(images, self.job_dir / "thumb.jpg")
        self.log(f"{info['num_images']} images ready")

        masks = None
        has_mat = scanmat.visible_in(images)
        if self.mask and has_mat:
            # Masking would black out the mat's markers; the mat already separates object and floor.
            self.log("Scan mat visible: skipping background removal (the mat isolates the object)")
        elif self.mask:
            self.stage("mask")
            if prepare.make_masks(images, self.work / "masks", self.log):
                masks = self.work / "masks"

        try:
            sparse_model = self.colmap(images, masks, info)
        except PipelineError as e:
            # Typical cause: the object was turned in front of a fixed camera, so the static
            # background disagrees with the object. Retry once on the object alone.
            if masks is not None or has_mat or not prepare.masking_available():
                raise
            self.log(f"{e}\nRetrying with background removed (object-only matching)")
            self.stage("mask")
            prepare.make_masks(images, self.work / "masks", self.log)
            masks = self.work / "masks"
            self.stats["auto_masked"] = True
            sparse_model = self.colmap(images, masks, info)
        self.stage("scale")
        self._mat = self.find_scan_mat(sparse_model)
        mesh_obj = self.openmvs(sparse_model)

        self.stage("export")
        meta = export.export_base(mesh_obj, self.work / "mvs" / "scene_dense.ply", self.base, self._ground,
                                  mm_per_unit=self._mat["mm_per_unit"] if self._mat else None)
        hull = self.outline_hull(sparse_model, np.array(meta["model_to_base"]))
        self.stage("solid")
        try:
            solid = printable.build_solid(self.base, self.work, self.run, self.log,
                                          depth=self.q.poisson_depth, resolution=self.q.solid_resolution, hull=hull)
            self.stats["scan_coverage"] = solid["coverage"]
            if "silhouette" in self.stats and "outlines" in solid:
                self.stats["silhouette"]["mode"] = solid["outlines"]
        except (PipelineError, ValueError) as e:  # still deliver the scan; the report shows it is not solid
            self.log(f"Print-ready solid failed, STL falls back to the raw surface: {e}")
        self.stats.update(export.write_outputs(self.base, self.out, self.size))
        # A job with a reference model (accuracy check) re-measures itself on every run.
        self.stats = accuracy.update_stats(self.job_dir, self.stats, {"quality": self.q.name}, self.log)
        self.stats["seconds"] = round(time.time() - t0)
        self.on_progress("Done", 100)
        self.log(f"Finished in {self.stats['seconds']}s")
        if not os.environ.get("SCAN3D_KEEP_WORK"):
            for heavy in ("mvs", "dense", "database.db", "masks", "print_points.ply", "print_poisson.ply"):
                p = self.work / heavy
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        return self.stats

    def check_tools(self) -> None:
        config.quiet_crashes()
        missing = [t for t in (config.COLMAP_BIN, config.FFMPEG_BIN) if not shutil.which(t)]
        missing += [t for t in ("DensifyPointCloud", "ReconstructMesh", "TextureMesh", "InterfaceCOLMAP")
                    if not Path(config.openmvs(t)).exists()]
        if missing:
            raise PipelineError("Missing tools: " + ", ".join(missing) + ". " + config.INSTALL_HINT)

    # ------------------------------------------------------------------ COLMAP
    def colmap(self, images: Path, masks: Path | None, info: dict) -> Path:
        db = self.work / "database.db"
        threads = str(config.THREADS)
        n = info["num_images"]

        caps = config.colmap_caps()
        # COLMAP 4 renamed --SiftExtraction.* / --SiftMatching.* to --FeatureExtraction.* / --FeatureMatching.*
        extraction, matching = (("FeatureExtraction", "FeatureMatching") if caps["new_options"]
                                else ("SiftExtraction", "SiftMatching"))

        def features_and_matches(gpu: bool) -> None:
            db.unlink(missing_ok=True)
            self.stage("features")
            args = [config.COLMAP_BIN, "feature_extractor", "--database_path", db, "--image_path", images,
                    "--ImageReader.camera_model", "OPENCV" if n >= 30 else "SIMPLE_RADIAL",
                    "--ImageReader.single_camera", "1" if info["single_camera"] else "0",
                    f"--{extraction}.use_gpu", "1" if gpu else "0", f"--{extraction}.num_threads", threads,
                    f"--{extraction}.max_image_size", str(min(self.q.max_image_size, 3200))]
            if masks:
                args += ["--ImageReader.mask_path", masks]
            self.run(args, "Feature extraction")

            self.stage("matching")
            common = ["--database_path", db, f"--{matching}.use_gpu", "1" if gpu else "0",
                      f"--{matching}.num_threads", threads]
            if info.get("num_photos", n) == 0 or n > 300:
                # Video frames are in order: match each with its 20 neighbours plus 2^k-apart frames
                # (closes the loop / links the rings). Exhaustive matching on video is 10x slower for
                # no gain, especially with repetitive patterns like the scan mat.
                self.run([config.COLMAP_BIN, "sequential_matcher", *common,
                          "--SequentialMatching.overlap", "20", "--SequentialMatching.quadratic_overlap", "1"],
                         "Feature matching")
            else:  # unordered photos: try every pair (robust, fast enough for a few hundred)
                self.run([config.COLMAP_BIN, "exhaustive_matcher", *common,
                          f"--{matching}.guided_matching", "1"], "Feature matching")

        if config.SIFT_GPU:
            try:
                features_and_matches(gpu=True)
                self.stats["sift_gpu"] = True
            except PipelineError as e:  # no usable OpenGL context, driver trouble, ...
                self.log(f"GPU feature matching failed, using the CPU instead: {e}")
                features_and_matches(gpu=False)
        else:
            features_and_matches(gpu=False)

        self.stage("mapping")
        sparse = self.work / "sparse"
        if "global_mapper" in caps["commands"]:
            best = self._map(sparse, "global_mapper", db, images)
            if _num_registered(best) < 0.6 * n:
                self.log(f"Global mapper registered {_num_registered(best)}/{n}; trying incremental mapper")
                alt = self._map(self.work / "sparse_incremental", "mapper", db, images)
                if _num_registered(alt) > _num_registered(best):
                    best = alt
        else:  # older COLMAP without the built-in global mapper (GLOMAP)
            best = self._map(sparse, "mapper", db, images)
        reg = _num_registered(best)
        self.stats["registered_images"] = reg
        self.log(f"Camera poses solved for {reg}/{n} images ({best.name})")
        if reg < prepare.MIN_IMAGES:
            raise PipelineError(
                f"Only {reg} of {n} images could be aligned. Use more overlap between photos, "
                "avoid shiny/plain surfaces, and keep the object static (or enable masking for turntables).")

        # Put gravity "down" so the model is upright in the viewer.
        aligned = self.work / "sparse_aligned"
        shutil.rmtree(aligned, ignore_errors=True)
        aligned.mkdir()
        try:
            self.run([config.COLMAP_BIN, "model_orientation_aligner", "--image_path", images,
                      "--input_path", best, "--output_path", aligned, "--method", "IMAGE-ORIENTATION"],
                     "Orientation alignment")
            best = aligned
        except PipelineError:
            self.log("Orientation alignment failed; keeping original orientation")

        self.stage("undistort")
        dense = self.work / "dense"
        shutil.rmtree(dense, ignore_errors=True)
        self.run([config.COLMAP_BIN, "image_undistorter", "--image_path", images, "--input_path", best,
                  "--output_path", dense, "--output_type", "COLMAP",
                  "--max_image_size", str(self.q.max_image_size)], "Image undistortion")
        return dense

    def _map(self, out: Path, cmd: str, db: Path, images: Path) -> Path:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        try:
            self.run([config.COLMAP_BIN, cmd, "--database_path", db, "--image_path", images,
                      "--output_path", out], f"COLMAP {cmd}")
        except PipelineError as e:
            self.log(str(e))
        models = [d for d in out.iterdir() if d.is_dir()]
        if not models:
            return out
        return max(models, key=_num_registered)

    def find_scan_mat(self, dense: Path) -> dict | None:
        """Real-world scale + floor plane from the printed ArUco mat, if it is in the frames."""
        txt = self.work / "dense_txt"
        shutil.rmtree(txt, ignore_errors=True)
        txt.mkdir()
        try:
            self.run([config.COLMAP_BIN, "model_converter", "--input_path", dense / "sparse",
                      "--output_path", txt, "--output_type", "TXT"], "Model export")
            mat = scanmat.measure(txt, dense / "images", self.log)
        except (PipelineError, ValueError, OSError) as e:
            self.log(f"Scan mat detection skipped: {e}")
            return None
        if mat:
            self.stats["scan_mat"] = {k: mat[k] for k in ("paper", "rms_mm", "corners")}
        return mat

    def outline_hull(self, dense: Path, model_to_base: np.ndarray):
        """Silhouette carving for the print solid (see silhouette.py). Needs the scan mat, which
        tells us where the object stands so SAM 2 can be pointed at it in every frame."""
        if not self._mat or os.environ.get("SCAN3D_SILHOUETTE", "1") == "0":
            return None
        if not silhouette.available():
            self.log("Silhouette carving skipped: SAM 2 not installed (uv sync --extra silhouette)")
            return None
        self.stage("outline")
        try:
            to_model = np.linalg.inv(model_to_base)
            tm = lambda X: X @ to_model[:3, :3].T + to_model[:3, 3]  # noqa: E731
            scene = trimesh.load(self.base / "model.glb", process=False)
            surface = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
            lo, hi = surface.bounds
            ext = hi - lo
            ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            r = 0.625 * float(np.hypot(ext[0], ext[2]))
            floor = tm(np.c_[r * np.cos(ang), np.zeros_like(ang), r * np.sin(ang)])
            cams = silhouette.read_cameras(self.work / "dense_txt")
            # Generous box: the scan of a dark object can miss whole parts (a plate, the top).
            blo = np.array([lo[0] - 0.15 * ext[0], 0.0, lo[2] - 0.15 * ext[2]])
            bhi = np.array([hi[0] + 0.15 * ext[0], hi[1] + 0.35 * ext[1], hi[2] + 0.15 * ext[2]])
            box = np.array([[x, y, z] for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1]) for z in (blo[2], bhi[2])])
            debug = self.work / "outline_debug" if os.environ.get("SCAN3D_KEEP_WORK") else None
            masks = silhouette.segment(dense / "images", cams, tm(np.asarray(surface.vertices)), floor, self.log,
                                       debug_dir=debug, box_pts=tm(box))
            if len(masks) < 0.5 * len(cams):
                self.log("Silhouette carving skipped: the object was found in too few frames")
                return None
            # How low did the camera go? Outlines pin down the height of a rounded top only from
            # near side-on views; without them the top comes out too tall.
            centers = np.array([-np.linalg.inv(c["P"][:, :3]) @ c["P"][:, 3] for c in cams])
            uvw = scanmat.to_sheet_mm(centers, self._mat)
            f = self._mat["frame"]
            elev = np.degrees(np.arctan2(uvw[:, 2], np.hypot(uvw[:, 0] - f["width_mm"] / 2,
                                                             uvw[:, 1] - f["height_mm"] / 2)))
            self.stats["silhouette"] = {"frames": len(masks), "lowest_view_deg": round(float(elev.min()))}
            if elev.min() > 20:
                self.log(f"Note: lowest camera view was {elev.min():.0f} deg above the table; film one "
                         "circle at table height so the top of rounded objects is not over-estimated")
        except Exception as e:  # never fail the scan over the optional outline step
            self.log(f"Silhouette carving skipped: {e}")
            return None
        return lambda origin, pitch, dims: silhouette.hull_on_grid(cams, masks, to_model, origin, pitch, dims,
                                                                   self.log)

    # ------------------------------------------------------------------ OpenMVS
    def openmvs(self, dense: Path) -> Path:
        mvs = self.work / "mvs"
        shutil.rmtree(mvs, ignore_errors=True)
        mvs.mkdir()
        threads = str(config.THREADS)
        omvs = config.openmvs

        self.run([omvs("InterfaceCOLMAP"), "-i", dense, "-o", mvs / "scene.mvs",
                  "--image-folder", dense / "images"], "Import into OpenMVS", cwd=mvs)

        self.stage("densify")
        self.run([omvs("DensifyPointCloud"), "scene.mvs", "-o", "scene_dense.mvs",
                  "--resolution-level", str(self.q.densify_resolution_level),
                  "--max-resolution", str(self.q.densify_max_resolution),
                  # No region of interest from the sparse points: dark or plain objects have almost
                  # none, so the estimated box can slice the object off. We crop the mesh ourselves.
                  "--estimate-roi", "0", "--crop-to-roi", "0",
                  "--max-threads", threads], "Dense reconstruction", cwd=mvs)

        self.stage("mesh")
        self.run([omvs("ReconstructMesh"), "scene_dense.mvs", "-o", "mesh_raw.ply",
                  "--max-threads", threads], "Mesh reconstruction", cwd=mvs)
        # Keep the object, drop floating background islands, cap the face count.
        mesh_stats = export.clean_mesh(mvs / "mesh_raw.ply", mvs / "mesh_clean.ply", self.q.max_faces,
                                       mat=self._mat)
        self._ground = mesh_stats.pop("ground", None)
        self.stats["ground_removed"] = bool(self._ground)
        self.log(f"Mesh cleanup: {mesh_stats}; ground plane {'found and removed' if self._ground else 'not found'}")
        mesh = "mesh_clean.ply"

        if self.q.refine_mesh:
            self.stage("refine")
            try:
                self.run([omvs("RefineMesh"), "scene_dense.mvs", "-m", mesh, "-o", "mesh_refined.ply",
                          "--resolution-level", "1", "--max-threads", threads], "Mesh refinement", cwd=mvs)
                mesh = "mesh_refined.ply"
            except PipelineError as e:
                self.log(f"Refinement skipped: {e}")

        self.stage("texture")
        self.run([omvs("TextureMesh"), "scene_dense.mvs", "-m", mesh, "-o", "model.obj",
                  "--export-type", "obj", "--max-texture-size", str(self.q.texture_size),
                  "--max-threads", threads], "Texturing", cwd=mvs)
        return mvs / "model.obj"
