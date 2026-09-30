#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Remove the two data zips ONLY where they are provably redundant.
#
# Both archives were extracted next to themselves under data/. A size match
# (already confirmed for every file) is strong evidence but not proof, so each
# archived file's CRC32 -- stored in the zip's central directory -- is compared
# with a CRC32 computed over the extracted file on disk. The zip is deleted only
# if EVERY file matches. Any mismatch means the extracted copy was changed after
# unzipping and the archive may hold the only original, so that zip is kept.
# Decided per zip, independently.
# ---------------------------------------------------------------------------

cd /lustre/home/gtirpitz/AI4ExoMars/data
echo "Host: $(hostname)  started $(date -Is)"

python3 - <<'PY'
import os, sys, time, zipfile, zlib

def crc_of(path, chunk=8 << 20):
    crc = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return crc & 0xFFFFFFFF
            crc = zlib.crc32(block, crc)

verdicts = {}
for z in ("2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic.zip", "hirise_simmim_crops.zip"):
    t0 = time.time()
    files = [i for i in zipfile.ZipFile(z).infolist()
             if not i.is_dir() and not i.filename.startswith("__MACOSX")]
    bad = []
    for n, info in enumerate(files, 1):
        path = info.filename
        if not os.path.isfile(path) or os.path.getsize(path) != info.file_size:
            bad.append(f"{path} (missing or size differs)")
        elif crc_of(path) != info.CRC:
            bad.append(f"{path} (CRC32 differs)")
        if n % 5000 == 0:
            print(f"  {z}: {n:,}/{len(files):,} checked, {len(bad)} mismatches", flush=True)
    verdicts[z] = not bad
    print(f"{z}: {len(files):,} files, {len(bad)} mismatches, {time.time() - t0:.0f}s", flush=True)
    for b in bad[:10]:
        print(f"    MISMATCH {b}", flush=True)

with open("/lustre/home/gtirpitz/AI4ExoMars/job_outputs/zip_verdicts.txt", "w") as f:
    for z, ok in verdicts.items():
        f.write(f"{'REDUNDANT' if ok else 'KEEP'} {z}\n")
PY

while read -r verdict zip; do
  if [ "$verdict" = "REDUNDANT" ]; then
    size=$(du -sh "$zip" | cut -f1)
    rm -f -- "$zip" && echo "DELETED $zip ($size) -- every file CRC-identical to its extracted copy"
  else
    echo "KEPT    $zip -- at least one file differs from its extracted copy"
  fi
done < /lustre/home/gtirpitz/AI4ExoMars/job_outputs/zip_verdicts.txt
echo "finished $(date -Is)"
