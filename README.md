# scan3d — photos / video → textured 3D model, locally on your Mac or Windows PC

Upload photos or a short video of an object (or drop a folder into the inbox) and get a
textured 3D model back as **GLB** (web/AR/Blender), **STL** (3D printing, geometry only), an **OBJ** bundle and a **point cloud**.
Everything runs on your machine with open-source tools. No cloud, no CUDA needed.

## How it works

```
photos / HEIC / video
   │  ffmpeg + OpenCV: extract sharpest video frames, fix orientation, resize, drop blurry shots
   ▼
COLMAP 4 (BSD)        SIFT features → matching → global SfM (GLOMAP) → camera poses
   │                  gravity alignment → undistortion
   ▼
OpenMVS (AGPL)        dense point cloud → mesh → (refine) → texture atlas   [Metal / CPU]
   │
   ▼
trimesh               remove the floor/table plane, keep the object, decimate, level on the floor,
                      align footprint to X/Z, scale to your real measurement → GLB / OBJ / PLY
   │
   ▼
print solid           Poisson (COLMAP) → voxel fill → marching cubes → manifold3d flat base
                      → watertight STL / 3MF, Z-up, millimetres
```

| Component | Role | License |
|---|---|---|
| [COLMAP](https://colmap.github.io) | Structure-from-Motion (camera poses) | BSD |
| [OpenMVS](https://github.com/cdcseacave/openMVS) | Multi-view stereo, meshing, texturing | AGPL-3.0 |
| [FFmpeg](https://ffmpeg.org) | Video frame extraction | LGPL/GPL |
| [trimesh](https://trimesh.org) | Mesh cleanup + GLB/STL/3MF export | MIT |
| [manifold3d](https://github.com/elalish/manifold) | Guaranteed-solid geometry, flat base cut | Apache-2.0 |
| [scikit-image](https://scikit-image.org) / SciPy | Voxel fill + marching cubes | BSD |
| [rembg](https://github.com/danielgatis/rembg) (optional) | Background removal for turntables | MIT |
| [SAM 2](https://github.com/facebookresearch/sam2) via `transformers` (optional) | Object outlines for silhouette carving | Apache-2.0 |
| [`<model-viewer>`](https://modelviewer.dev) | 3D preview in the browser | Apache-2.0 |
| FastAPI / SQLite | API, job queue | MIT / public domain |

## Install (once)

**macOS (Apple Silicon)**

```bash
./scripts/install_macos.sh
```

Installs COLMAP, FFmpeg and uv via Homebrew, compiles OpenMVS (its dependencies are built by
vcpkg into `~/.local/share/scan3d`, ~20–40 min the first time), and creates the Python env.

**Windows 10 / 11** (any CPU and GPU, including AMD; no CUDA needed): double-click
`scripts\install_windows.bat`. It downloads prebuilt COLMAP, OpenMVS and FFmpeg, sets up Python,
and finishes with a complete test scan. Full instructions: **[WINDOWS.md](WINDOWS.md)**.

### Check an installation

```bash
uv run scan3d-doctor              # every tool, Python package, GPU and folder
uv run scan3d-doctor --selftest   # plus a complete scan of bundled test photos (a 60 × 40 × 90 mm
                                  # box on the mat), compared with the true box
```

On Windows: `scripts\doctor_windows.bat` (add `--selftest`).

## Use it

### Web UI
```bash
./scripts/run.sh              # Windows: double-click scripts\run_windows.bat
```
Open http://localhost:8000, drop photos / a folder / a video, pick a quality, and watch progress.
Finished models render in the page and can be downloaded as GLB, OBJ or PLY.

### Inbox folder (hands-free)
While the service runs, copy a folder into `data/inbox/`:

```
data/inbox/
  coffee_mug/          ← folder of photos and/or videos = one model
    IMG_0001.HEIC …
    scan3d.json        ← optional: {"quality": "high", "mask": true}
  sneaker.mov          ← a loose video = one model
  IMG_1.jpg IMG_2.jpg  ← loose photos in the inbox itself = one model together
```
Once the folder stops changing for ~8 s it is moved into a job. The result is written to
`data/outbox/<name>.glb`. Point the inbox elsewhere with `SCAN3D_INBOX=/path/to/folder`.

### Command line
```bash
uv run scan3d ~/Pictures/mug_photos --quality high -o ~/Desktop/mug.glb
uv run scan3d ~/Movies/shoe.mov
uv run scan3d ~/Pictures/mug_photos -o ~/Desktop/mug.stl   # STL for printing
```

### API
| Method | Path | |
|---|---|---|
| `POST` | `/api/jobs` | multipart: `files[]`, `name`, `quality` (`preview`/`balanced`/`high`), `mask` |
| `GET` | `/api/jobs`, `/api/jobs/{id}` | status, stage, progress %, stats, output files |
| `GET` | `/api/jobs/{id}/files/model.stl` | also `model.3mf`, `model.glb`, `model_obj.zip`, `pointcloud.ply`, `print_preview.glb` |
| `GET` | `/api/jobs/{id}/log` | pipeline log |
| `POST` | `/api/jobs/{id}/cancel`, `/api/jobs/{id}/retry?quality=high` | |
| `POST` / `DELETE` | `/api/jobs/{id}/accuracy` | compare with a known model (multipart `reference`, or none for the test piece) / remove |
| `GET` | `/api/test-piece?format=stl` | the accuracy test piece (`stl`, `3mf`) |
| `DELETE` | `/api/jobs/{id}` | |

Interactive docs: http://localhost:8000/docs

## 3D printing

Every scan produces **print-ready** `model.stl` and `model.3mf`:

- a single **closed, manifold solid** (no holes, no loose pieces), rebuilt from the scan with
  Poisson reconstruction and a voxel fill, so even the never-seen bottom is closed;
- an exactly **flat base** cut at the detected floor, sitting at Z = 0;
- **Z-up and in millimetres** (when a real size is set), so it lands upright and true-to-size
  in Bambu Studio, PrusaSlicer, OrcaSlicer or Cura.

The viewer's **Print view** shows exactly this solid; the textured view shows the colour scan.
Under the model you get a printability check: closed solid, flat base, volume, approximate PLA
weight (at 100 % infill, i.e. an upper bound) and whether it fits your printer.
Set your build volume with `SCAN3D_BUILD_VOLUME=256x256x256` (X×Y×Z mm; default fits Bambu A1/P1/X1).

### Black, shiny or plain objects: shape from outlines

Photogrammetry needs visible detail on the surface. Black plastic, glossy cases and plain
surfaces show little, so their scans come out with holes and lumps. When the scan mat is in
the frames (and the `silhouette` extra is installed), the print solid also uses the object's
**outline**: [SAM 2](https://github.com/facebookresearch/sam2) (Apache-2.0; the Hugging Face
`transformers` build, on the Mac GPU or on the CPU elsewhere, about 1 s per frame)
segments the object in every frame, pointed at it by the scan, and the outlines are intersected
from all camera positions (*silhouette carving*). That fills holes, removes lumps and keeps real
gaps (under an overhang, through a handle) open; hollows the outline cannot see (a cup's inside)
still come from the scan.

Outlines over-size whatever the cameras did not see side-on, so they add material only when the
scan needs it. The pipeline measures how much of the scan's surface has real measured points on
it: from 95 % up the scan counts as **complete**, and outlines then only trim what sticks out of
it. (A textured test box measures 99.6 %, a glossy earbud case 82 %, a black 3D print 65–76 %.
On the box, outlines alone were 7–11 mm too big; the scan is within 2 mm.)

The height of a **rounded top** is only pinned down by views close to side-on: film one circle
with the phone at table height. If the lowest view was higher than 20°, the page warns that the
top may come out too tall. The first run downloads the SAM 2 weights (~180 MB) from Hugging Face.

Detail is limited by the scan: the solid is built at ~1/320 of the object's longest side on
*balanced* (1/200 preview, 1/448 high), e.g. ~0.3 mm for a 95 mm object.

## Measuring accuracy (the benchmark)

How close is a scan to the real object? Scan something whose exact shape is known and compare:

1. Download the **test piece** from the web UI (or `GET /api/test-piece?format=stl|3mf`, or
   `uv run scan3d-accuracy --piece piece.stl`) and print it: 80 × 50 × 28 mm, no supports,
   flat on the bed. Matte, light filament gives the best case; black or glossy the worst case.
2. Scan the print on the scan mat, standing the way it was printed.
3. On the finished scan press **Check accuracy (test piece)**. To check against any other printed
   model, use **Compare with an STL…** (millimetres, standing as the object stood on the mat).

The scan is placed on the model base-on-base (turned and shifted only, never tilted or scaled),
because a wrong floor or a wrong scale is a real error of the scan. The report shows:

- **95 % of the surface within ±X mm**, plus typical (median), worst and average deviation.
  Deviations are signed: + where the scan has extra material, − where material is missing.
- **Where the error comes from:** scale error in %, base tilt, base offset, and the error left
  once those are fitted away ("shape only").
- **Dimension checks** (test piece only), read the way a caliper would: width, depth, the three
  step heights, dome height (rounded tops), post diameter, hole depth (hollows that outlines
  cannot see), arch opening (gaps under overhangs) and the 1 / 2 / 3 mm ridges (smallest detail
  that survives).
- Two coloured views, **Errors on scan** and **Errors on model**: blue = missing, red = extra,
  full colour at ±1 mm.

A scan with a reference re-measures itself on every **Re-run** and rescale, so after a pipeline
or capture change, re-running the same job shows whether the numbers got better. Every
measurement is also appended to `data/accuracy_history.jsonl`.

```bash
uv run scan3d-accuracy data/jobs/<id>/output/model.stl                # vs the test piece
uv run scan3d-accuracy scan.stl original.stl --views out/             # vs any model, + coloured GLBs
```

The comparison itself was checked against deliberately wrong copies of the test piece (2 % too
big, tilted 1°, steps 0.5 mm too tall, filled hole, lowered arch): each error is reported at its
true value. It measures the scan against the *model*, so it includes your printer's own error
(typically 0.1–0.2 mm).

## True-to-size models (exact dimensions)

Photogrammetry recovers shape but not absolute scale. Two ways to fix that:

### Automatic: the scan mat (recommended)

1. Download the mat from the web UI (or `GET /api/scan-mat?paper=a4|letter`) and print it at
   **100 % / Actual size**. Check that the printed scale bar measures exactly 100 mm.
2. Lay it **flat** (tape the corners), put the object in the middle, and film/photograph all the
   way around so the markers around the edge are visible in most frames.
3. Leave *Real size* empty. The pipeline finds the ArUco markers, triangulates them and fits the
   mat's known layout, which gives the scale (typically ±0.1–0.3 mm fit) and the exact floor plane
   used to level the model and remove the floor.

The mat also tells the pipeline *where the object is*: everything that is not above the sheet
(the floor or table around it, walls, furniture) is cut away from the mesh, and the sheet itself
is removed exactly at its plane, even if the object only fills a small part of the frame. Keep the object
inside the marker border. Background removal is skipped automatically when the mat is visible
(it would erase the markers). The mat's speckled centre also gives plain objects extra texture.
In tests with a synthetic 60 × 40 × 90 mm box the mat scale was within 0.03 %, and the printed
solid's faces were within ~0.5 mm (sharp edges can overshoot by up to ~1 mm).

### Manual: one measurement

Give it **one real measurement** (measure the object with a ruler or calipers). A manual size
always overrides the mat; *Use scan mat* switches back.

- **Web UI:** fill *Real size* (Height / Width / Depth / Longest side, in mm) when uploading — or
  on a finished scan use *Set size*; it re-exports instantly, no re-run needed.
- **Inbox:** `scan3d.json` → `{"size": {"dimension": "height", "mm": 95}}`
- **CLI:** `uv run scan3d video.mov --size 95 --size-of height -o box.stl`
- **API:** `size_mm` + `size_dimension` form fields on upload, or `POST /api/jobs/{id}/scale`
  with `{"dimension": "height", "mm": 95}` (`mm: null` clears it).

The model is levelled on the detected floor and its footprint is aligned to the axes, so
**width** = longer horizontal side (X), **depth** = shorter horizontal side (Z), **height** = from
the floor up (Y). Height is the most reliable reference because it is measured from the floor plane.

Units once scaled: **STL / OBJ / PLY in millimetres** (what slicers expect), **GLB in metres**
(the glTF standard). Expect ~1–2 % accuracy on well-captured, matte, textured objects; shiny,
thin or poorly covered parts (e.g. the top if you never filmed it) are less accurate.

## Quality presets

| Preset | Image size | Video frames | Dense res | Mesh refine | Max faces | Typical time* |
|---|---|---|---|---|---|---|
| preview | 1200 px | 50 | ¼ | no | 60k | 1–3 min |
| balanced | 2000 px | 90 | full | no | 200k | 4–12 min |
| high | 3200 px | 150 | full | yes | 500k | 15–45 min |

\*40–80 photos on an M-series Mac. Jobs run one at a time (each uses every core).

## Capture tips (this matters more than any setting)

- **Minimum 5 photos**, but 5 only works for easy, well-textured objects. Aim for **20–80**.
- **Overlap:** 20–80 photos, moving ~10–15° around the object between shots; do 2–3 rings at different heights.
- **Video (iPhone):** record **4K 30 fps** with **HDR Video off** (Settings › Camera › Record Video),
  long-press to **lock focus/exposure** (AE/AF Lock), walk slowly 30–60 s, and do **two circles**:
  one with the phone **at table height** (side-on; this pins down the height) and one higher,
  looking down ~45° so the top is covered.
- **Object stays still, you move.** Don't flip or rotate the object between shots in front of a fixed camera: the pipeline will reconstruct the static background instead. If you use a turntable, enable *Remove background*.
- **Light:** even, diffuse (overcast / lightbox). Avoid hard shadows.
- **Surfaces:** shiny, transparent, black or completely plain objects don't reconstruct well: the
  cameras need visible detail on the object itself. Black 3D prints and glossy plastic come out
  patchy (holes, lumps). A light dusting of talc / cornstarch, dry shampoo or a 3D-scanning spray
  (wipes or sublimates off) fixes this more than any setting; a few pencil or tape marks also help.
- Fill the frame with the object; keep it in focus.

## Layout

```
app/
  main.py            FastAPI app (API + static UI)
  jobs.py            SQLite job store + background worker
  watcher.py         inbox folder watcher
  cli.py             `scan3d` command
  config.py          paths, binaries, quality presets (env-overridable)
  pipeline/
    prepare.py       inputs → clean frames, masks, thumbnail
    runner.py        COLMAP + OpenMVS orchestration, progress, cancel
    export.py        mesh cleanup, GLB/OBJ/PLY export
    printable.py     watertight print solid (Poisson, voxel fill, flat base)
    silhouette.py    SAM 2 outlines + silhouette carving
  scanmat.py         printable ArUco scan mat, scale + floor from it
  refpiece.py        the accuracy test piece (known geometry)
  accuracy.py        scan vs. known model: placement, deviations, dimension checks
  doctor.py          `scan3d-doctor`: installation check + end-to-end test scan
  static/index.html  web UI
scripts/             install_macos.sh, run.sh; install_windows.bat/.ps1, run_windows.bat,
                     doctor_windows.bat; package.sh (zip for another computer)
selftest/box/        36 synthetic photos of a 60 × 40 × 90 mm box on the mat (for the test scan)
data/                jobs, inbox, outbox, SQLite db (created on first run)
```

Env overrides: `SCAN3D_DATA`, `SCAN3D_INBOX`, `SCAN3D_OUTBOX`, `COLMAP_BIN`, `OPENMVS_BIN_DIR`,
`FFMPEG_BIN`, `SCAN3D_TOOLS` (folder the install script puts the native tools in),
`SCAN3D_KEEP_WORK=1` (keep intermediate dense/MVS files for debugging), `SCAN3D_SILHOUETTE=0`
(turn off silhouette carving), `SCAN3D_TORCH_DEVICE=cpu|cuda|mps` (where SAM 2 runs),
`SCAN3D_SIFT_GPU=1` (try COLMAP's OpenGL feature matching first; falls back to the CPU).
