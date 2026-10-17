"""Score models/spatial/<run>/preds.npz and the DOY climatology on hidden val pixels by region x season.
Usage: python model_by_region.py [run_dir ...]   (default: every models/spatial/blk_*/preds.npz)
preds.npz: val_idx (store day indices), val_pred (n,H,W) float16 ln CHL on the full grid, or on the ocean-bbox crop if config.json has crop=true.
Runs whose config.json store != blk (e.g. Indian Ocean) or whose file is missing are reported and skipped (the regions are Black Sea only)."""
import sys, json, glob
import numpy as np
from common import *


def crop_slices(ocean):
    rows, cols = np.flatnonzero(ocean.any(1)), np.flatnonzero(ocean.any(0))
    return (slice(max(rows[0] - 1, 0), min(rows[-1] + 2, ocean.shape[0])), slice(max(cols[0] - 1, 0), min(cols[-1] + 2, ocean.shape[1])))


def score_run(run_dir, ctx):
    run_dir = Path(run_dir); f = run_dir / "preds.npz"
    if not f.exists():
        print(f"[{run_dir.name}] no preds.npz; skipped"); return None
    cfg = json.load(open(run_dir / "config.json")) if (run_dir / "config.json").exists() else {}
    if cfg.get("store", "blk") != "blk":
        print(f"[{run_dir.name}] store={cfg.get('store')} is not blk; skipped (regions are Black Sea only)"); return None
    z = np.load(f); vidx, pred = z["val_idx"], z["val_pred"].astype("float32")
    H, W = ctx["reg"].shape
    if pred.shape[1:] != (H, W):
        if cfg.get("crop"):
            ys, xs = crop_slices(ctx["ocean"]); full = np.full((len(vidx), H, W), np.nan, "float32"); full[:, ys, xs] = pred; pred = full
        else:
            print(f"[{run_dir.name}] pred grid {pred.shape[1:]} != {(H, W)}; skipped"); return None
    pos = np.searchsorted(ctx["va"], vidx)
    if not np.array_equal(ctx["va"][pos], vidx):
        print(f"[{run_dir.name}] val_idx not within val days; skipped"); return None
    y, hid, clim, b = ctx["y"][vidx], ctx["hid"][pos], ctx["clim"], ctx["bin"][vidx]
    ok = hid & np.isfinite(pred); print(f"[{run_dir.name}] {len(vidx)} val days, hidden px {int(hid.sum())}, non-finite pred among them {int(hid.sum() - ok.sum())}")
    res_m = score_by_region_season(np.abs(pred - y), ok, ctx["reg"], ctx["times"][vidx], ctx["names"])
    res_c = score_by_region_season(np.abs(clim[b] - y), ok, ctx["reg"], ctx["times"][vidx], ctx["names"])
    print(md_table(res_m, ctx["names"], f"{run_dir.name}: model hidden-pixel MAE (ln CHL); cell = MAE (n)")); print()
    print(md_table(res_c, ctx["names"], f"Climatology on the same pixels")); print()
    return res_m, res_c


def main(dirs):
    d = S.load_store(); reg, names = load_regions(); c = train_climatology(d)
    ctx = dict(reg=reg, names=names, ocean=d["ocean"], **c)
    out = {}
    for r in dirs:
        s = score_run(r, ctx)
        if s: out[Path(r).name] = {f"{k[0]}|{k[1]}": v for k, v in s[0].items()}
    return out


if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(str(Path(p).parent) for p in glob.glob(str(ROOT / "models/spatial/blk_*/preds.npz")))
    if not dirs:
        print("No models/spatial/blk_*/preds.npz exist yet.")
        dirs = [str(ROOT / "models/spatial/blk_does_not_exist")]   # structural check: missing file must not crash
    main(dirs)
