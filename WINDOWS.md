# scan3d on Windows

Setup for a Windows 10 or 11 PC (64-bit). Any CPU and any graphics card work, including AMD:
nothing in scan3d needs NVIDIA or CUDA.

## What you need

| | |
|---|---|
| Windows | 10 or 11, 64-bit |
| Memory | 16 GB works; 32 GB recommended for *High* quality |
| Disk | about 8 GB free for the install, plus a few GB of temporary files per scan |
| Internet | for the install (about 3.5 GB of downloads); scanning itself runs offline |
| Printer | to print the scan mat on A4 or Letter paper |

## 1. Put the project on the PC

Copy the project folder to a **short, plain path** such as `C:\scan3d`.

- On the Mac, `./scripts/package.sh` writes `dist/scan3d.zip` with everything needed. Copy that
  zip over and unpack it to `C:\scan3d`.
- Avoid folders with non-English letters (for example a user name like `C:\Users\Žygis\...`) and
  avoid OneDrive folders. COLMAP and OpenMVS can fail on such paths.

## 2. Install

Double-click **`scripts\install_windows.bat`**.

- If Windows shows "Windows protected your PC", choose **More info → Run anyway** (the file came
  from another computer, so Windows asks).
- Windows asks once for permission to install the Microsoft Visual C++ runtime: choose **Yes**.
- The first run takes 10–20 minutes, mostly downloads. It is safe to run again; finished steps
  are skipped.

What it installs, all without touching the system PATH:

| What | Where | Why |
|---|---|---|
| Microsoft Visual C++ runtime | Windows | COLMAP and OpenMVS need it |
| COLMAP 4.2.1 (no-CUDA build) | `%LOCALAPPDATA%\scan3d\colmap` | camera poses, watertight meshing |
| OpenMVS 2.4.0 | `%LOCALAPPDATA%\scan3d\openmvs` | dense point cloud, mesh, texture |
| FFmpeg | `%LOCALAPPDATA%\scan3d\ffmpeg` | video frames |
| uv | `%USERPROFILE%\.local\bin` | manages Python |
| Python 3.12 + packages | `.venv` in the project folder | the service, SAM 2, background removal |

The install ends with two checks:

1. **`scan3d-doctor`** starts every tool once and lists what works.
2. **A test scan**: a complete scan of 36 bundled photos of a 60 × 40 × 90 mm box on the scan mat,
   compared with the true box. It takes 3–10 minutes and downloads the SAM 2 model (180 MB) the
   first time. It must end with `All good.`

If the last line is `Installation finished.`, the PC is ready.

## 3. Start

Double-click **`scripts\run_windows.bat`**. A window opens and stays open while scan3d runs, and
the browser opens **http://127.0.0.1:8000**. Close the window to stop scan3d.

To use another port, open a Command Prompt in the project folder and run:

```bat
set PORT=8080
scripts\run_windows.bat
```

## 4. Scan

1. In the page, download the **scan mat** (A4 or Letter) and print it at **100 % / actual size**.
   Check that the printed scale bar measures exactly 100 mm.
2. Lay the mat flat, put the object in the middle, and film or photograph all the way around
   (capture tips are in [README.md](README.md#capture-tips-this-matters-more-than-any-setting)).
3. Drop the video or photos into the page and press **Create 3D model**.
4. Download the **STL** or **3MF** and open it in your slicer.

You can also copy a folder of photos or a video into `data\inbox` inside the project folder; it is
picked up automatically.

## What uses the graphics card

Most of the work runs on the CPU, on every platform. The open-source dense reconstruction
(OpenMVS, and COLMAP's own) can use a GPU only through NVIDIA CUDA, so an AMD card does not speed
it up. More CPU cores help most.

| Step | Default | With an AMD GPU |
|---|---|---|
| Feature detection and matching (COLMAP) | CPU | optional, through OpenGL (see below) |
| Camera poses (COLMAP) | CPU | CPU |
| Dense point cloud, mesh, texture (OpenMVS) | CPU | CPU |
| Object outlines (SAM 2) | CPU, about 1 s per frame | optional, through AMD ROCm (see below) |

### Optional: GPU feature matching

COLMAP's GPU feature matching runs on OpenGL, which AMD cards support. Turn it on by setting
`SCAN3D_SIFT_GPU=1` before starting; if the GPU path fails, scan3d falls back to the CPU on its
own. This has not been tested on an AMD card by us.

```bat
set SCAN3D_SIFT_GPU=1
scripts\run_windows.bat
```

### Optional: SAM 2 on the AMD GPU

AMD publishes a PyTorch build for Radeon cards on Windows (ROCm). It needs **Windows 11 25H2**
and a supported card (RX 7600 and newer, RX 9000 series; not RX 6000). It only speeds up the
outline step, from roughly 1–2 minutes to a few seconds per scan, so it is rarely worth it.
We have not tested it. If you want to try, in a Command Prompt in the project folder:

```bat
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-all]==2.15.0+rocm10.2.0" "torchvision[device-all]==0.29.2+rocm10.2.0"
set SCAN3D_NO_SYNC=1
scripts\run_windows.bat
```

`SCAN3D_NO_SYNC=1` keeps that build instead of restoring the standard one at every start. Check
with `scripts\doctor_windows.bat`: the SAM 2 line should say `on the GPU`. To go back, start
without `SCAN3D_NO_SYNC`.

## If something does not work

Double-click **`scripts\doctor_windows.bat`**. It lists every tool with `[ok]`, `[warn]` or
`[FAIL]` and says what to do. For a complete test scan:

```bat
scripts\doctor_windows.bat --selftest
```

| Problem | Fix |
|---|---|
| `a required DLL is missing` | Install the Visual C++ runtime from https://aka.ms/vc14/vc_redist.x64.exe, then run the doctor again |
| `Missing tools: ...` in the page | Run `scripts\install_windows.bat` again |
| The page does not open | Look at the black window for an error. If the port is taken, start with another `PORT` |
| A scan fails right at the start with odd path errors | Move the project to a plain path such as `C:\scan3d` |
| Antivirus slows or blocks the tools | Allow `%LOCALAPPDATA%\scan3d` and the project folder |
| A scan failed | Press **Log** on the scan in the page; the last lines say which step failed |

Each scan's full log is also in `data\jobs\<id>\log.txt`.

## Update or remove

- **Update:** replace the project folder's files with the new version (keep `data`), then run
  `scripts\install_windows.bat` again.
- **Remove:** delete the project folder, `%LOCALAPPDATA%\scan3d`, and the downloaded models in
  `%USERPROFILE%\.cache\huggingface`. The Visual C++ runtime and uv can stay or be removed in
  Windows' *Installed apps*.

## What has and has not been tested

scan3d was developed on a Mac. For Windows, these things were checked without a Windows PC:

- the installer script parses in PowerShell and uses only commands Windows' built-in PowerShell 5.1 has;
- every download link in it responds, and the COLMAP and OpenMVS archives have the folder layout the code looks for;
- the Windows COLMAP (4.2.1) and OpenMVS (2.4.0) are the same versions the Mac uses, and every
  option the pipeline passes exists in the Windows programs;
- every Python package has a ready-made Windows build, so no compiler is needed;
- the same code, including the test scan, passes on the Mac.

**Not tested: an actual run on Windows.** The first real proof is the test scan at the end of the
install. The project includes a GitHub Actions workflow (`.github/workflows/windows.yml`) that
runs the installer and the test scan on a Windows machine whenever the project is pushed to GitHub.
