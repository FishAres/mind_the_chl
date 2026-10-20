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

CATEGORIES maps a short category name to its folder; find_run() locates a run by name in any folder.
"""
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
CATEGORIES = {"cnn": MODELS / "cnn", "unet": MODELS / "unet", "pinn/cnn": MODELS / "pinn" / "cnn", "pinn/unet": MODELS / "pinn" / "unet", "pinn/deq": MODELS / "pinn" / "deq", "pinn/t3d": MODELS / "pinn" / "t3d", "pinn/upstream": MODELS / "pinn" / "upstream", "spatial": MODELS / "spatial", "ude": MODELS / "ude", "tracer": MODELS / "tracer"}


def run_dir(name, trainer, model):
    """Folder for a new run. trainer: 'train' | 'train2' | 'pinn'; model: 'simple' | 'dilated' | 'unet' | 'deq' | 'temporal3d'."""
    arch = "unet" if model == "unet" else ("deq" if model == "deq" else ("t3d" if model == "temporal3d" else "cnn"))
    cat = f"pinn/{arch}" if trainer == "pinn" else arch
    return CATEGORIES[cat] / name


def find_run(name):
    """Path of an existing run by name (searches every category, then the legacy flat layout)."""
    for d in list(CATEGORIES.values()) + [MODELS]:
        if (d / name / "config.json").exists() or (d / name / "history.json").exists():
            return d / name
    raise FileNotFoundError(f"run {name!r} not found under {MODELS}")


def list_runs(categories=None, pattern="*"):
    """Run folders (with a history.json) in the given categories (default all), sorted by name."""
    cats = list(CATEGORIES) if categories is None else list(categories)
    out = []
    for c in cats:
        out += [p.parent for p in CATEGORIES[c].glob(f"{pattern}/history.json")]
    return sorted(set(out), key=lambda p: p.name)


def category_of(path):
    path = Path(path).resolve()
    for c, d in CATEGORIES.items():
        if d.resolve() in path.parents:
            return c
    return "legacy"
