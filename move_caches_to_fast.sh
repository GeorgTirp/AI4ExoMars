#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Phase 1 of moving the 512 caches off home: COPY and VERIFY only.
#
# Home is at 423/500 GiB, over its 400 GiB soft quota; these two caches are
# ~49 GB of it. /fast has a 9.6 TiB quota and is mounted read-write on the
# compute nodes (the 1024 build writes there from g132).
#
# This script never deletes or relinks anything. The swap (original -> .old,
# symlink to /fast, verify a real load through the symlink, remove .old) is a
# separate step, done only after this reports COPY VERIFIED and no running job
# has the originals open.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
mkdir -p job_outputs/move_caches

SRC=data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived
DST=/fast/gtirpitz/ai4exomars/derived_512
mkdir -p "$DST"

[ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ] && source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Host: $(hostname)  started $(date -Is)"
for c in seg_crop_cache_full seg_context_cache_full; do
  if [ -L "$SRC/$c" ]; then echo "$c is already a symlink -- nothing to copy"; continue; fi
  echo "=== $c ==="
  # rsync checks each transferred file with a whole-file checksum and retries
  # on mismatch, so a clean exit already means the bytes arrived intact.
  rsync -a "$SRC/$c/" "$DST/$c/"
  if diff <(cd "$SRC/$c" && find . -type f -printf '%P %s\n' | sort) \
          <(cd "$DST/$c" && find . -type f -printf '%P %s\n' | sort) >/dev/null; then
    echo "  file list and sizes identical"
  else
    echo "  MISMATCH in file list or sizes -- stopping" >&2; exit 1
  fi
done

# Independent content check, through the same memmap path training uses.
SRC="$SRC" DST="$DST" python - <<'PY'
import os, numpy as np
src, dst = os.environ["SRC"], os.environ["DST"]
rng = np.random.default_rng(0)
for c, arrays in (("seg_crop_cache_full", ("images.npy", "labels.npy")),
                  ("seg_context_cache_full", ("context.npy",))):
    for a in arrays:
        s = np.load(f"{src}/{c}/{a}", mmap_mode="r"); d = np.load(f"{dst}/{c}/{a}", mmap_mode="r")
        assert s.shape == d.shape and s.dtype == d.dtype, f"{c}/{a}: shape/dtype differ"
        idx = rng.choice(s.shape[0], 128, replace=False)
        assert all(np.array_equal(s[i], d[i]) for i in idx), f"{c}/{a}: row mismatch"
        print(f"  {c}/{a}: {s.shape} {s.dtype} -- 128 random rows bitwise identical")
PY
echo "COPY VERIFIED $(date -Is)"
