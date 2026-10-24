"""Where training runs live, and how to find them.

Layout (since 2026-10-03; runs trained before were moved here, nothing else changed):
  models/cnn/<run>         data-only runs from train.py / train2.py with a CNN  (baseline_*, full_*, perf_*, smoke_ctx)
  models/unet/<run>        data-only U-Net runs from train2.py
  models/pinn/cnn/<run>    train_pinn.py runs with the dilated CNN (pinn_*, nochl_*), any lambda incl. 0
  models/pinn/unet/<run>   train_pinn.py runs with a U-Net (unet_*)
  models/pinn/deq/<run>    train_pinn.py runs with --model deq (deep equilibrium fill, deq.py)
  models/pinn/t3d/<run>    train_pinn.py runs with --model temporal3d (3D-conv stem over adjacent days, models.py)
  models/pinn/upstream/<run>  train_pinn.py runs with --store io_up (2024 hackweek protocol, store_io_up.py; 2026-10-20)
  models/spatial/<run>     train_spatial.py runs (steady-state anisotropic diffusion, spatial_node.py)
  models/ude/<run>         train_ude.py runs (daily UDE: semi-Lagrangian transport + learned source, ude.py)
  models/tracer/<run>      train_tracer.py runs (learned Lagrangian tracer, velocity terms; tracer.py)
  models/history/<category>/<run>  archived runs (archive_runs.py): config/history/metrics only, weights and preds deleted

CATEGORIES maps a short category name to its folder; find_run() locates a run by name in any folder (live first, then
history). config.json is written when a run ends; run_config() falls back to the flags of the live trainer process, then to
the run's log header (~/mind_the_chl_gap/logs/**/<run>.log), so running runs can be described too.
"""
import json
import os
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
HISTORY = MODELS / "history"
LOGS = Path.home() / "mind_the_chl_gap" / "logs"
CATEGORIES = {"cnn": MODELS / "cnn", "unet": MODELS / "unet", "pinn/cnn": MODELS / "pinn" / "cnn", "pinn/unet": MODELS / "pinn" / "unet", "pinn/deq": MODELS / "pinn" / "deq", "pinn/t3d": MODELS / "pinn" / "t3d", "pinn/upstream": MODELS / "pinn" / "upstream", "spatial": MODELS / "spatial", "ude": MODELS / "ude", "tracer": MODELS / "tracer"}


def run_dir(name, trainer, model):
    """Folder for a new run. trainer: 'train' | 'train2' | 'pinn'; model: 'simple' | 'dilated' | 'unet' | 'deq' | 'temporal3d'."""
    arch = "unet" if model == "unet" else ("deq" if model == "deq" else ("t3d" if model == "temporal3d" else "cnn"))
    cat = f"pinn/{arch}" if trainer == "pinn" else arch
    return CATEGORIES[cat] / name


def find_run(name):
    """Path of an existing run by name (searches every category, then the legacy flat layout, then history)."""
    for d in list(CATEGORIES.values()) + [MODELS] + [HISTORY / c for c in CATEGORIES]:
        if (d / name / "config.json").exists() or (d / name / "history.json").exists():
            return d / name
    raise FileNotFoundError(f"run {name!r} not found under {MODELS}")


def list_runs(categories=None, pattern="*", archived=False):
    """Run folders (with a history.json) in the given categories (default all), sorted by name. archived=True adds the
    runs in models/history/<category>/ that have no live folder of the same name."""
    cats = list(CATEGORIES) if categories is None else list(categories)
    out = []
    for c in cats:
        live = [p.parent for p in CATEGORIES[c].glob(f"{pattern}/history.json")]
        names = {p.name for p in live}
        out += live
        if archived:
            out += [p.parent for p in (HISTORY / c).glob(f"{pattern}/history.json") if p.parent.name not in names]
    return sorted(set(out), key=lambda p: p.name)


SIDE = ("smoke", "leak", "eval", "regcheck", "_maps")          # name parts of smoke tests, leak checks and eval-only re-scores


def is_side_run(path):
    return any(k in Path(path).name for k in SIDE)


def is_archived(path):
    return HISTORY.resolve() in Path(path).resolve().parents


def category_of(path):
    path = Path(path).resolve()
    for c, d in CATEGORIES.items():
        if d.resolve() in path.parents or (HISTORY / c).resolve() in path.parents:
            return c
    return "legacy"


def run_label(path):
    """'pinn/t3d/io3_ptok4', or 'history/pinn/cnn/pinn_lam0' for archived runs."""
    return f"{'history/' if is_archived(path) else ''}{category_of(path)}/{Path(path).name}"


def _parse_flags(tokens):
    """['--t3d-stem', 'patch', '--lam', '0', '--clim'] -> {'t3d_stem': 'patch', 'lam': 0, 'clim': True}; last occurrence wins."""
    def num(v):
        for f in (int, float):
            try:
                return f(v)
            except ValueError:
                pass
        return v
    out, key, vals = {}, None, []
    for t in list(tokens) + ["--"]:
        if t.startswith("--"):
            if key:
                out[key] = True if not vals else (vals[0] if len(vals) == 1 else vals)
            key, vals = t[2:].replace("-", "_") or None, []
        elif key:
            vals.append(num(t))
    return out


def _live_cmdline(name):
    """Flags of the live trainer process training run `name` (Linux /proc), or None."""
    for d in Path("/proc").glob("[0-9]*"):
        try:
            a = (d / "cmdline").read_bytes().decode(errors="ignore").split("\0")
        except OSError:
            continue
        if any(t.startswith("chl_cnn.train") for t in a) and "--run" in a and a[a.index("--run") + 1:a.index("--run") + 2] == [name]:
            return a[a.index("--run"):]
    return None


def is_running(path):
    """True while a trainer process is training this run (live folders only)."""
    return not is_archived(path) and _live_cmdline(Path(path).name) is not None


_CFG = {}                                                      # (config.json path, mtime) -> parsed config


def run_config(path):
    """config.json of a run; for runs without one (still training, or stopped early) the trainer flags from the live process,
    else from the run's log header ('=== <date> train <run> <flags>'), with '_partial': True. {} when nothing is found."""
    path = Path(path)
    f = path / "config.json"
    try:
        key = (str(f), f.stat().st_mtime)
    except FileNotFoundError:
        key = None
    if key:
        if key not in _CFG:
            _CFG[key] = json.loads(f.read_text())
        return _CFG[key]
    a = _live_cmdline(path.name)
    if a is None and LOGS.exists():
        for lg in LOGS.rglob(f"{path.name}.log"):
            with open(lg, errors="ignore") as fh:
                head = fh.readline().split()
            if path.name in head and head[0] == "===":
                a = head[head.index(path.name) - 1:]
                a[0] = "--run"
                break
    if a is None:
        return {}
    c = _parse_flags(a)
    if c.get("t3d_stem") and not c.get("model"):
        c["model"] = "temporal3d"                     # log headers carry only the per-run flags; a stem implies the 3D model
    c["_partial"] = True
    return c
