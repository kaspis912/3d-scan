#!/usr/bin/env bash
# Installs everything the 3D reconstruction service needs on Apple Silicon macOS.
#   - COLMAP   (Structure-from-Motion: camera poses + sparse cloud)       -> Homebrew
#   - OpenMVS  (dense cloud, mesh, texture; Metal/CPU, no CUDA needed)    -> built from source via vcpkg
#   - uv       (Python env manager) + Python deps                         -> Homebrew / uv
# Safe to re-run: finished steps are skipped. First run takes ~20-40 min (OpenMVS deps compile).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# vcpkg cannot build in paths containing spaces, so native deps live outside the project.
DEPS="${SCAN3D_DEPS:-$HOME/.local/share/scan3d}"
PREFIX="$DEPS/openmvs-install"
JOBS="$(sysctl -n hw.ncpu)"

echo "==> Homebrew packages"
brew install colmap uv ffmpeg cmake ninja pkgconf autoconf automake autoconf-archive libtool nasm libomp vcpkg

mkdir -p "$DEPS"
cd "$DEPS"

if [[ -x "$PREFIX/bin/OpenMVS/DensifyPointCloud" ]]; then
  echo "==> OpenMVS already installed in $PREFIX"
else
  echo "==> Fetching OpenMVS, VCGLib and vcpkg sources"
  [[ -d vcglib ]]  || git clone --depth 1 -b master https://github.com/cdcseacave/VCG.git vcglib
  [[ -d openMVS ]] || git clone --depth 1 -b develop --recurse-submodules https://github.com/cdcseacave/openMVS.git openMVS
  [[ -d vcpkg ]]   || git clone https://github.com/microsoft/vcpkg.git vcpkg
  # Use Homebrew's vcpkg binary instead of downloading one via bootstrap.
  ln -sf "$(brew --prefix)/bin/vcpkg" vcpkg/vcpkg
  export VCPKG_ROOT="$DEPS/vcpkg"

  echo "==> Building OpenMVS (dependencies are compiled by vcpkg, be patient)"
  cmake -S openMVS -B openmvs-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_RPATH="@executable_path/../../lib/OpenMVS" \
    -DVCG_ROOT="$DEPS/vcglib" \
    -DVCPKG_TARGET_TRIPLET=arm64-osx-release \
    -DVCPKG_HOST_TRIPLET=arm64-osx-release \
    -DOpenMVS_USE_CUDA=OFF \
    -DOpenMVS_USE_PYTHON=OFF \
    -DOpenMVS_BUILD_VIEWER=OFF \
    -DOpenMVS_USE_BREAKPAD=OFF \
    -DOpenMVS_ENABLE_TESTS=OFF \
    -DOpenMP_ROOT="$(brew --prefix libomp)"
  cmake --build openmvs-build -j "$JOBS"
  cmake --install openmvs-build
fi

echo "==> Python environment"
cd "$ROOT"
uv sync --extra masking --extra silhouette

echo
echo "Done. Start the service with:  ./scripts/run.sh"
