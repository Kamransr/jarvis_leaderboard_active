#!/usr/bin/env bash
# Reproduce V7_COREPP_exfoliation using a checksum-pinned external package.
# Requires Linux/WSL, Bash 4+, Git, Python 3.10 with venv, and network access.
# The submitted prediction ZIP is a reference and is never overwritten.
set -euo pipefail

PACKAGE_REPO="https://github.com/AcuVizionAI/V7_COREPP_exfoliation.git"
PACKAGE_COMMIT="f676749b857ed03d76a91ba1469a4fae1c9c05f0"
OUTPUT_NAME="AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae.csv"
SUBMITTED_SHA256="fb6300aefa036fa7c57a603dcb3613aa8d6409be3aa9cbb088c81f6feda59205"
declare -A ASSET_SHA256=(
  ["model/RAW_fold1.pth"]="23ac6a5850d740de277e88e3726af31e588f868504099895350e0bc6d35e9168"
  ["model/preproc.json"]="9046e3b5318879cf8fc469e8c691efac99a96dc2d2f4fb84df7748447006e8b1"
  ["model/REG_STATS_fold1.json"]="bc653aeb589b77f7a6756f56a6302d9e9f642e89ecc6521ceef2942b86795195"
  ["featurize.py"]="4c1fd86f30a50d175297b5b02367520abb5fcfe35d522331131963503cea5899"
  ["model_inference.py"]="55ef617d72918e16f456b570c4f8a6af567cb0d2ba94a70fd536baf569a986ff"
  ["run.py"]="ed6da59a7ae928cbf274f9b64c22b249541c6ad045a04112fa3b5903a57cc831"
  ["test_ids.json"]="57a5ec29321564a7e784bef4f32c2fff7e6fb969f536e90ca5b006589108d338"
  ["requirements.txt"]="90bdc6e4cfdac1a571e96f62ab15801dacafde71ad9ed01d6f6623d6bfbbd386"
  ["verify.py"]="8b03cb2d078b3672c00ec00f91d7cefeecec5a115f26e551d8052e0cbd21d169"
)

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
unset PYTHONPATH
log() { printf '[run.sh] %s\n' "$*" >&2; }
fail() { log "$*"; exit 1; }
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# Paths may be supplied; the device, activation profile and tolerance stay fixed.
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --data-zip|--cache-dir)
      (($# >= 2)) || fail "Missing value for $1"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --data-zip=*|--cache-dir=*) EXTRA_ARGS+=("$1"); shift ;;
    --help|-h)
      printf 'Usage: bash run.sh [--data-zip PATH] [--cache-dir PATH]\n'
      printf 'Optional environment: PYTHON_BIN, V7_WORK_DIR\n'
      exit 0
      ;;
    *) fail "Unsupported argument: $1; use --data-zip or --cache-dir" ;;
  esac
done

SUBMITTED="$HERE/$OUTPUT_NAME.zip"
[ -f "$SUBMITTED" ] || fail "Missing submitted archive: $SUBMITTED"
[ "$(sha256_of "$SUBMITTED")" = "$SUBMITTED_SHA256" ] || fail "Submitted ZIP checksum mismatch"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10, or set PYTHON_BIN to its executable."'

WORK_BASE="${V7_WORK_DIR:-$HERE/.v7_work}"
mkdir -p "$WORK_BASE"
WORK="$(cd -- "$WORK_BASE" && pwd)"
RUN="$(mktemp -d "$WORK/replay.XXXXXX")"
PKG="$RUN/package"
OUT="$RUN/predictions"
VENV="$WORK/venv"
log "work directory: $WORK"

# Fetch a fresh checkout, then verify all executable code and model assets.
git init -q "$PKG"
git -C "$PKG" remote add origin "$PACKAGE_REPO"
git -C "$PKG" fetch -q --depth 1 origin "$PACKAGE_COMMIT"
git -C "$PKG" checkout -q --detach FETCH_HEAD
[ "$(git -C "$PKG" rev-parse HEAD)" = "$PACKAGE_COMMIT" ] || fail "Package commit mismatch"
for rel in "${!ASSET_SHA256[@]}"; do
  [ -f "$PKG/$rel" ] || fail "Missing package file: $rel"
  [ "$(sha256_of "$PKG/$rel")" = "${ASSET_SHA256[$rel]}" ] || fail "SHA-256 mismatch: $rel"
done
log "package $PACKAGE_COMMIT: all ${#ASSET_SHA256[@]} checksums verified"

# Re-run installation if an earlier setup failed before the ready marker.
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
READY="$VENV/.v7-requirements-sha256"
EXPECTED_ENV="${ASSET_SHA256[requirements.txt]}-torch-cpu-py310"
if [ ! -f "$READY" ] || [ "$(cat "$READY")" != "$EXPECTED_ENV" ]; then
  log "installing pinned dependencies in isolated Python 3.10 environment"
  # Install only the CPU wheel from PyTorch; resolve dependencies from PyPI below.
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q --no-deps torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q -r "$PKG/requirements.txt"
  printf '%s\n' "$EXPECTED_ENV" > "$READY"
fi
"$VENV/bin/python" -c 'import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.__version__ == "2.5.1+cpu", torch.__version__'

log "running CPU inference -> $OUT"
"$VENV/bin/python" "$PKG/run.py" --device cpu --profile portable_tanh \
  --cache-dir "$WORK/cache" --out-dir "$OUT" "${EXTRA_ARGS[@]}"

# Compare with the frozen submission before announcing successful reproduction.
"$VENV/bin/python" "$PKG/verify.py" "$SUBMITTED" "$OUT/$OUTPUT_NAME.zip" --atol 1e-4
[ "$(sha256_of "$SUBMITTED")" = "$SUBMITTED_SHA256" ] || fail "Submitted ZIP changed during replay"
log "verified reproduction: $OUT/$OUTPUT_NAME.zip"
log "submitted archive preserved: $SUBMITTED"
