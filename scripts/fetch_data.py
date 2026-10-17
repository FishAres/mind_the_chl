"""Download IO.zarr from the public GCS bucket to local disk (~/mind_the_chl_gap/data).

Anonymous access, no credentials needed. ~24.7 GB, 3,997 files, ~2.5 min on SageMaker.
Skips the download if a complete copy already exists.

Run:  python scripts/fetch_data.py
"""
import time

import gcsfs

from io_summaries import DATA, SRC

BUCKET_PATH = "nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO.zarr"
N_FILES = 3997  # object count in the bucket as of 2026-09-29


def n_local_files():
    return sum(1 for p in SRC.rglob("*") if p.is_file()) if SRC.exists() else 0


def main():
    if n_local_files() == N_FILES:
        print(f"{SRC} already complete ({N_FILES} files)")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token="anon")
    print(f"downloading gs://{BUCKET_PATH} -> {SRC} (~24.7 GB)...")
    t = time.time()
    fs.get(BUCKET_PATH, str(SRC), recursive=True, batch_size=64)
    n = n_local_files()
    print(f"done in {(time.time() - t) / 60:.1f} min: {n} files" + ("" if n == N_FILES else f" (expected {N_FILES}!)"))


if __name__ == "__main__":
    main()
