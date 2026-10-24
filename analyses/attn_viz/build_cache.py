"""Day-weight cache of one attention gap-filler for the "Attention models" section of notebooks/progress.ipynb (2026-10-23).

Runs the trainer's own eval-only path (chl_cnn.train_pinn.main --eval-only --init-from RUN, arguments rebuilt from the run's
config.json) with the existing recorders switched on, on every --val-stride-th val day (the stride gives --days val days, evenly
spaced in time; the test pass runs too and is not recorded):
  --t3d-stem attn    analyses/io3_attn/attn_eval.Recorder (chl_cnn/tattn.py EVAL_HOOKS["recorder"]): pooling weights (mean over
                     heads) over the T window days, day t included
  --t3d-stem patch   analyses/io3_attn/ptok_eval.Rec (chl_cnn/ptok.py HOOKS["recorder"]): per-pixel day weights over the T days + "no
                     day", patch score (interpolated) and pixel term, local confidence conv(m) (lavg)
  --t3d-stem tokdec  analyses/io4_tokdec/tokdec_eval.Rec (chl_cnn/tokdec.py HOOKS["recorder"]): pooling weights per ocean patch over the
                     T days (day t included), per-patch visible fraction; with --tokdec-blend also the per-pixel blend weights
Nothing is written under models/: the eval run's folder is a temporary directory, removed afterwards.

Saved to analyses/attn_viz/cache/<run>.npz (float16 maps, land 0): days (store index), dates, offsets, t_index, ocean, lat, lon,
vis (n, T, H, W) visible in the stem input (donor-masked), hid (n, H, W) val hidden pixels (observed and not visible on day t), and per
stem the arrays above. Also pred_check: max |CPU - saved preds.npz| over ocean px of the cached val days (ln CHL).

  python -m analyses.attn_viz.build_cache --run io3_ptok4 --days 12                  # CPU (CUDA hidden), default
  flock ~/mind_the_chl_gap/logs/gpu.lock python -m analyses.attn_viz.build_cache --run io4_tokdec --device cuda
"""
import argparse
import os
import sys

if __name__ == "__main__":                                     # hide the GPU before torch is imported (CPU is the default)
    _pre = argparse.ArgumentParser(add_help=False); _pre.add_argument("--device", default="cpu")
    if _pre.parse_known_args()[0].device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import math
import shutil
import tempfile
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SKIP = {"run", "eval_only", "init_from", "continue_from", "continue_weights_only", "resume_best", "fig_every", "eval_batch", "val_stride",
        "leak_day", "leak_offset", "leak_delta", "leak_what", "help"}


def trainer_parser():
    """The argparse parser of chl_cnn.train_pinn.parse (captured from one call)."""
    from chl_cnn import train_pinn
    box, orig = {}, argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        box["p"] = self
        return orig(self, args, namespace)
    argparse.ArgumentParser.parse_args = grab
    try:
        train_pinn.parse(["--run", "x"])
    finally:
        argparse.ArgumentParser.parse_args = orig
    return box["p"]


def argv_from_config(cfg):
    """Trainer arguments that reproduce the run's settings (config.json holds vars(a)); keys the config lacks keep their defaults."""
    argv = []
    for act in trainer_parser()._actions:
        if act.dest in SKIP or act.dest not in cfg or not act.option_strings:
            continue
        v, opt = cfg[act.dest], act.option_strings[0]
        if act.dest == "dilations" and v == [1, 2, 4, 8, 16, 1, 2, 4]:   # config stores the resolved default; the argument default is None
            continue
        if isinstance(act, argparse._StoreTrueAction):
            argv += [opt] if v else []
        elif isinstance(act, argparse._StoreFalseAction):
            argv += [opt] if v is False else []
        elif isinstance(act, argparse.BooleanOptionalAction):
            argv += [opt if v else "--no-" + opt[2:]]
        elif v is None:
            continue
        elif isinstance(v, list):
            argv += ([opt] + [str(x) for x in v]) if act.nargs in ("+", "*") else [opt, ",".join(str(x) for x in v)]
        else:
            argv += [opt, str(v)]
    return argv


def cpu_torch():
    """A stand-in for the torch module inside chl_cnn.train_pinn only: torch.device(...) -> cpu, torch.cuda.* memory / RNG calls no-ops."""
    import torch
    proxy = types.ModuleType("torch_cpu_proxy")
    proxy.__getattr__ = lambda k: getattr(torch, k)
    proxy.device = lambda *a, **k: torch.device("cpu")
    proxy.cuda = types.SimpleNamespace(reset_peak_memory_stats=lambda *a: None, max_memory_allocated=lambda *a: 0,
                                       max_memory_reserved=lambda *a: 0, get_rng_state=lambda *a: torch.zeros(0, dtype=torch.uint8),
                                       set_rng_state=lambda *a: None, is_available=lambda: False)
    return proxy


class _LazyObs:
    """obs[days] (store-observed and ocean) for attn_eval.Recorder, read per call from the memory-mapped store."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __getitem__(self, days):
        return np.stack([self.ctx["obs"](int(i)) for i in np.atleast_1d(days)])


def build(run, n_days=12, device="cpu", eval_batch=None, threads=4, out=None):
    import torch
    from chl_cnn import train_pinn, viz
    from chl_cnn.runs import find_run
    src = find_run(run)
    cfg = json.loads((src / "config.json").read_text())
    stem = cfg.get("t3d_stem")
    assert cfg.get("model") == "temporal3d" and stem in ("attn", "patch", "tokdec"), f"{run}: not an attention stem ({stem})"
    assert (src / "model.pt").exists(), f"{run}: no model.pt yet (training not finished)"
    if device == "cpu":
        assert not torch.cuda.is_available(), "CPU mode: CUDA must be hidden (run as python -m analyses.attn_viz.build_cache)"
        torch.set_num_threads(threads)
    ctx = viz.context_for(src)
    va = ctx["va"]
    stride = math.ceil(len(va) / n_days)
    va_s = va[::stride]
    n = len(va_s)
    tmp = Path(tempfile.mkdtemp(prefix=f"attnviz_{run}_"))
    argv = ["--run", f"attnviz_{run}"] + argv_from_config(cfg) + ["--eval-only", "--init-from", run, "--val-stride", str(stride),
                                                                  "--eval-batch", str(eval_batch or (2 if device == "cpu" else 8)), "--fig-every", "0"]
    ns = train_pinn.parse(argv)                                # self-check: the rebuilt arguments give the run's settings
    diff = {k: (cfg[k], v) for k, v in vars(ns).items() if k in cfg and k not in SKIP and cfg[k] != v
            and not (k == "dilations" and (v is None or v == ",".join(map(str, cfg[k]))))}
    assert not diff, f"rebuilt trainer arguments differ from config.json: {diff}"
    print(f"{run}: stem {stem}, {n} val days (stride {stride}), device {device}; trainer args {' '.join(argv)}", flush=True)

    if stem == "attn":
        from chl_cnn import tattn
        from analyses.io3_attn.attn_eval import Recorder
        rec = Recorder(va_s, _LazyObs(ctx), ctx["ocean"], 1)
        tattn.EVAL_HOOKS["recorder"] = rec
    elif stem == "patch":
        from chl_cnn import ptok
        from analyses.io3_attn.ptok_eval import Rec
        rec = Rec(list(range(n)), n)
        ptok.HOOKS["recorder"] = rec
    else:
        from chl_cnn import tokdec
        from analyses.io4_tokdec.tokdec_eval import Rec
        rec = Rec(list(range(n)), n)
        tokdec.HOOKS["recorder"] = rec

    saved = dict(run_dir=train_pinn.run_dir, torch=train_pinn.torch)
    train_pinn.run_dir = lambda name, trainer, model: tmp / name
    if device == "cpu":
        train_pinn.torch = cpu_torch()
    t0 = time.time()
    try:
        train_pinn.main(argv)
    finally:
        train_pinn.run_dir, train_pinn.torch = saved["run_dir"], saved["torch"]
        for mod, hk in (("tattn", "EVAL_HOOKS"), ("ptok", "HOOKS"), ("tokdec", "HOOKS")):
            m = sys.modules.get(f"chl_cnn.{mod}")
            if m is not None:
                getattr(m, hk)["recorder"] = None
    wall = time.time() - t0

    ocean = ctx["ocean"].astype(bool)
    f16 = lambda a: np.where(ocean, a, 0).astype("float16")
    res = dict(run=run, stem=stem, value=cfg.get("ptok_value") or "", blend=bool(cfg.get("tokdec_blend")), days=va_s,
               dates=np.array([str(ctx["times"][i])[:10] for i in va_s]), ocean=ocean, lat=ctx["lat"], lon=ctx["lon"],
               val_stride=stride, device=device, wall_s=wall)
    if stem == "attn":
        assert rec.k == n and rec.days == list(map(int, va_s)), (rec.k, rec.days)
        res.update(t_index=rec.t_index, w=f16(np.stack(rec.maps).astype("float32")), vis=np.stack(rec.vis), hid=np.stack(rec.hid))
    else:
        assert rec.k == n and sorted(rec.out) == list(range(n)), (rec.k, sorted(rec.out))
        o = [rec.out[i] for i in range(n)]
        vis = np.stack([d["vis"] for d in o])
        T = vis.shape[1]
        ti = T // 2 if cfg.get("context_before") == cfg.get("context_after") else int(cfg["context_before"])
        obs = np.stack([ctx["obs"](int(i)) for i in va_s])
        res.update(t_index=ti, vis=vis, hid=obs & ~vis[:, ti])
        for k in ("w", "s_patch", "s_pixel", "conf"):
            if k in o[0]:
                res[k] = f16(np.stack([d[k] for d in o]).astype("float32"))
        if stem == "tokdec":
            res.update(pool_w=np.concatenate(rec.pool).astype("float16"), visfrac=np.concatenate(rec.visf).astype("float16"),
                       patch_idx=rec.meta["patch_idx"], nPy=rec.meta["nPy"], nPx=rec.meta["nPx"], P=rec.meta["P"])
    res["offsets"] = np.arange(res["vis"].shape[1]) - res["t_index"]
    # check: CPU predictions on the cached val days vs the run's saved preds.npz (GPU, AMP)
    z = np.load(tmp / f"attnviz_{run}" / "preds.npz")
    pr = viz.load_preds(src)
    mine = {int(i): k for k, i in enumerate(z["idx"])}
    d_ = [np.nanmax(np.abs(z["pred"][mine[int(i)]].astype("float32") - pr["pred"][pr["pos"][int(i)]].astype("float32"))[ocean]) for i in va_s]
    res["pred_check"] = np.array(d_)
    shutil.rmtree(tmp, ignore_errors=True)
    out = Path(out or CACHE / f"{run}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), eval {wall / 60:.1f} min; max |pred - saved preds| per day (ln CHL): "
          f"{np.round(res['pred_check'], 4).tolist()}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, nargs="+")
    p.add_argument("--days", type=int, default=12, help="val days to cache (every ceil(n_val / days)-th val day)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="cpu (CUDA hidden, default) or cuda (run under flock gpu.lock)")
    p.add_argument("--eval-batch", type=int, default=None, help="days per forward (default 2 on CPU, 8 on GPU)")
    p.add_argument("--threads", type=int, default=4, help="CPU threads (torch.set_num_threads)")
    p.add_argument("--out-dir", default=None, help="write <run>.npz here instead of analyses/attn_viz/cache (tests)")
    a = p.parse_args()
    for r in a.run:
        build(r, a.days, a.device, a.eval_batch, a.threads, out=(Path(a.out_dir) / f"{r}.npz" if a.out_dir else None))


if __name__ == "__main__":
    main()
