<#
Installs everything scan3d needs on Windows 10 / 11 (64-bit). Works with any CPU and GPU
(AMD, Intel or NVIDIA): nothing here needs CUDA.

  - Microsoft Visual C++ runtime   needed by COLMAP and OpenMVS (Windows asks for permission once)
  - COLMAP 4.2.1 (no-CUDA build)   camera poses, undistortion, watertight meshing
  - OpenMVS 2.4.0                  dense point cloud, mesh, texture
  - FFmpeg                         video frames
  - uv + Python 3.12 + packages    the service itself, SAM 2 and background removal

The tools go to %LOCALAPPDATA%\scan3d (override with -ToolsDir or SCAN3D_TOOLS). Nothing is
added to the system PATH. Safe to run again: finished steps are skipped.

Run it by double-clicking scripts\install_windows.bat, or:
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_windows.ps1
#>
param(
    [string]$ToolsDir = $(if ($env:SCAN3D_TOOLS) { $env:SCAN3D_TOOLS } else { Join-Path $env:LOCALAPPDATA 'scan3d' }),
    [switch]$SkipSelfTest
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is many times faster without the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# Pinned versions: the release lists of both projects also carry model / dataset tags, so "latest" is not safe.
$ColmapUrl   = 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-nocuda.zip'
$OpenMvsUrl  = 'https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip'
$FfmpegUrl   = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
$VcRedistUrl = 'https://aka.ms/vc14/vc_redist.x64.exe'

$Root = Split-Path -Parent $PSScriptRoot
$Extras = @('--extra', 'masking', '--extra', 'silhouette')

function Step($message) {
    Write-Host ''
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Get-File($url, $dest) {
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L --fail --retry 3 --silent --show-error -o $dest $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
    } else {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    }
}

function Find-Tool($folder, $exe) {
    Get-ChildItem -Path (Join-Path $ToolsDir $folder) -Recurse -Filter $exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Install-Zip($name, $url, $folder, $exe) {
    $found = Find-Tool $folder $exe
    if ($found) {
        Write-Host "$name is already installed: $($found.FullName)"
        return
    }
    $dest = Join-Path $ToolsDir $folder
    $zip = Join-Path $env:TEMP "scan3d_$folder.zip"
    Write-Host "Downloading $name ..."
    Get-File $url $zip
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Write-Host 'Unpacking ...'
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
        & tar.exe -xf $zip -C $dest
        if ($LASTEXITCODE -ne 0) { throw "Could not unpack $zip" }
    } else {
        Expand-Archive -Path $zip -DestinationPath $dest -Force
    }
    Remove-Item $zip -Force
    $found = Find-Tool $folder $exe
    if (-not $found) { throw "$exe was not found after unpacking $name into $dest" }
    Write-Host "Installed: $($found.FullName)"
}

# ---------------------------------------------------------------------------------------------
Step 'Checking this computer'
if (-not [Environment]::Is64BitOperatingSystem) { throw 'scan3d needs 64-bit Windows.' }
Write-Host "Project folder: $Root"
Write-Host "Tools folder:   $ToolsDir"
if ("$Root$ToolsDir" -match '[^\x00-\x7F]') {
    Write-Warning 'A folder name contains non-English characters. COLMAP and OpenMVS can fail on such paths.'
    Write-Warning 'If scans fail, move the project to a plain folder such as C:\scan3d and run this again'
    Write-Warning '(for the tools folder: set SCAN3D_TOOLS=C:\scan3d-tools before installing and before starting).'
}
if ($Root -match 'OneDrive') {
    Write-Warning 'The project is inside OneDrive. Scans write several GB of temporary files; a local folder such as C:\scan3d is better.'
}
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

Step 'Microsoft Visual C++ runtime (needed by COLMAP and OpenMVS)'
$vcKey = 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
$vc = Get-ItemProperty -Path $vcKey -ErrorAction SilentlyContinue
if ($vc -and $vc.Installed -eq 1 -and ($vc.Major -gt 14 -or ($vc.Major -eq 14 -and $vc.Minor -ge 50))) {
    Write-Host "Already installed (version $($vc.Major).$($vc.Minor))."
} else {
    $vcExe = Join-Path $env:TEMP 'scan3d_vc_redist.x64.exe'
    Get-File $VcRedistUrl $vcExe
    Write-Host 'Installing. Windows will ask for permission: choose Yes.'
    try {
        $proc = Start-Process -FilePath $vcExe -ArgumentList '/install', '/quiet', '/norestart' -Wait -PassThru
        # 0 = installed, 1638 = a newer version is already there, 3010 = installed, restart pending
        if (@(0, 1638, 3010) -notcontains $proc.ExitCode) {
            Write-Warning "The runtime installer returned code $($proc.ExitCode). If the check below reports a missing DLL, install it by hand from $VcRedistUrl"
        }
    } catch {
        Write-Warning "The runtime was not installed ($($_.Exception.Message)). If the check below reports a missing DLL, install it by hand from $VcRedistUrl"
    }
    Remove-Item $vcExe -Force -ErrorAction SilentlyContinue
}

Step 'COLMAP'
Install-Zip 'COLMAP 4.2.1' $ColmapUrl 'colmap' 'colmap.exe'

Step 'OpenMVS'
Install-Zip 'OpenMVS 2.4.0' $OpenMvsUrl 'openmvs' 'DensifyPointCloud.exe'

Step 'FFmpeg'
Install-Zip 'FFmpeg' $FfmpegUrl 'ffmpeg' 'ffmpeg.exe'

Step 'uv (Python environment manager)'
$env:Path = (Join-Path $env:USERPROFILE '.local\bin') + ';' + $env:Path
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' | Invoke-Expression
    $env:Path = (Join-Path $env:USERPROFILE '.local\bin') + ';' + $env:Path
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv could not be installed. Install it from https://docs.astral.sh/uv/ and run this script again.'
}
& uv --version

Step 'Python 3.12 and packages (about 3 GB of downloads the first time)'
Set-Location $Root
$env:PYTHONUTF8 = '1'
& uv sync @Extras
if ($LASTEXITCODE -ne 0) { throw 'Installing the Python packages failed (see above).' }

Step 'Checking the installation'
& uv run @Extras scan3d-doctor
if ($LASTEXITCODE -ne 0) { throw 'The check found problems (see the [FAIL] lines above).' }

if (-not $SkipSelfTest) {
    Step 'Test scan (a full scan of bundled test photos, 3 to 10 minutes; also downloads the SAM 2 model, 180 MB)'
    & uv run @Extras scan3d-doctor --selftest
    if ($LASTEXITCODE -ne 0) { throw 'The test scan did not pass (see the [FAIL] lines above).' }
}

Write-Host ''
Write-Host 'Done. Start scan3d by double-clicking scripts\run_windows.bat' -ForegroundColor Green
