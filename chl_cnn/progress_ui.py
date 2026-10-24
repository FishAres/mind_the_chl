"""ipywidgets panels for notebooks/progress.ipynb. Each function displays its controls and figure output in the calling cell.

  val_curves()    val hidden-pixel MAE and physics residual by epoch for any runs (chl_cnn/watch.py), live and archived
  days()          output visualizer: the same 32 val days for any run with preds.npz (chl_cnn/viz.py)
  Attention()     attention stems: .curves() (a), .offsets() (b), .maps() (c) (chl_cnn/attn_viz.py)
  tracer()        learned-tracer runs (chl_cnn/tracer_viz.py)

Runs are found with chl_cnn/runs.py (models/<category>/<run>, and models/history/<category>/<run> when asked). Running runs
have no config.json yet; runs.run_config reads their flags from the live trainer process or the log header instead.
The notebook calls plt.ioff() so figures appear only inside the Output widgets.
"""
import time
from pathlib import Path

import ipywidgets as W
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from .runs import CATEGORIES, list_runs, run_label, is_side_run


class _Busy:
    """Set while a panel refills its selectors, so value observers skip the intermediate states (changing a selector's
    options resets its value) and the panel draws once at the end."""
    def __init__(self):
        self.on = False

    def __enter__(self):
        self.on = True

    def __exit__(self, *_):
        self.on = False


def _recent(paths, n):
    """The n most recently updated runs (history.json mtime)."""
    return sorted(paths, key=lambda p: (Path(p) / "history.json").stat().st_mtime)[-n:]


# ---------------------------------------------------------------- validation curves of any runs
def val_curves(categories=("pinn/t3d",), n_default=6):
    cat_sel = W.SelectMultiple(options=list(CATEGORIES), value=tuple(categories), rows=6, description="category")
    pattern = W.Text(value="*", description="name glob", layout=W.Layout(width="260px"))
    archived = W.Checkbox(value=False, description="include archived (models/history)", indent=False)
    hide_side = W.Checkbox(value=True, description="hide smoke / leak / eval runs", indent=False)
    all_runs = W.Checkbox(value=False, description="plot every listed run", indent=False)
    refresh = W.Button(description="Refresh", icon="refresh")
    run_sel = W.SelectMultiple(options=[], rows=14, description="runs", layout=W.Layout(width="620px"))
    out = W.Output()
    from .watch import draw_runs

    def _runs():
        rs = list_runs(cat_sel.value, pattern.value or "*", archived.value)
        return [p for p in rs if not (hide_side.value and is_side_run(p))]

    busy = _Busy()

    def _fill(*_):
        rs = _runs()
        keep = set(run_sel.value)
        with busy:
            run_sel.options = [(run_label(p), str(p)) for p in rs]
            run_sel.value = tuple(v for _, v in run_sel.options if v in keep) or tuple(str(p) for p in _recent(rs, n_default))
        _draw()

    def _draw(*_):
        if busy.on:
            return
        with out:
            clear_output(wait=True)
            sel = _runs() if all_runs.value else [Path(v) for v in run_sel.value]
            if not sel:
                print("no runs selected")
                return
            rows, fig = draw_runs(sel, title_suffix=f"  ({time.strftime('%H:%M:%S')})")
            fig.set_size_inches(16, 5.6)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            for cat, name, n, best, st in rows:
                print(f"{cat:13s} {name:44s} {st:10s} epochs {n:3d}  best val {best:.4f}")

    for w in (cat_sel, pattern, archived, hide_side):
        w.observe(_fill, "value")
    run_sel.observe(_draw, "value")
    all_runs.observe(_draw, "value")
    refresh.on_click(_fill)
    display(W.HBox([cat_sel, W.VBox([pattern, archived, hide_side, all_runs, refresh])]), run_sel, out)
    _fill()


# ---------------------------------------------------------------- output visualizer
def days(category="pinn/t3d", n=32):
    from .viz import context_for, sample_days, draw_day

    cat = W.Dropdown(options=list(CATEGORIES), value=category, description="category")
    pattern = W.Text(value="*", description="name glob", layout=W.Layout(width="260px"))
    hide_side = W.Checkbox(value=True, description="hide smoke / leak / eval runs", indent=False)
    run = W.Dropdown(options=[], description="run", layout=W.Layout(width="520px"))
    idx = W.IntSlider(value=0, min=0, max=n - 1, description="day", continuous_update=False)
    play = W.Play(value=0, min=0, max=n - 1, interval=2500, description="cycle")
    W.jslink((play, "value"), (idx, "value"))
    out = W.Output()
    sampled = {}                                               # run -> its n sampled val days
    busy = _Busy()

    def _fill(*_):
        rs = [p for p in list_runs([cat.value], pattern.value or "*")
              if (p / "preds.npz").exists() and (p / "config.json").exists() and not (hide_side.value and is_side_run(p))]
        keep = run.value
        with busy:
            run.options = [(p.name, str(p)) for p in rs]
            vals = [v for _, v in run.options]
            run.value = keep if keep in vals else (str(_recent(rs, 1)[0]) if vals else None)
        if not vals:
            with out:
                clear_output()
                print(f"no runs with preds.npz in {cat.value} matching {pattern.value!r}")
        elif run.value != keep:
            _draw()

    def _draw(*_):
        if busy.on or not run.value:
            return
        with out:
            clear_output(wait=True)
            r = run.value
            if r not in sampled:
                print("loading store context (first time per store: a few seconds, ~30 s for the 2 km Black Sea)")
                sampled[r] = sample_days(context_for(r), r, n=n)
            ds = sampled[r]
            idx.max = play.max = len(ds) - 1
            fig = draw_day(r, ds[min(idx.value, len(ds) - 1)])
            clear_output(wait=True)
            display(fig)
            plt.close(fig)

    for w in (cat, pattern, hide_side):
        w.observe(_fill, "value")
    run.observe(_draw, "value")
    idx.observe(_draw, "value")
    display(W.HBox([cat, pattern, hide_side]), W.VBox([run, W.HBox([play, idx]), out]))
    _fill()


# ---------------------------------------------------------------- attention stems
class Attention:
    """Shared run selection for the three attention panels: .curves() (a), .offsets() (b), .maps() (c)."""
    DEFAULT = ("io2_tattn8", "io2_tattn8_s30", "io3_ptok4", "io3_ptok4_lavg", "io4_tokdec", "io4_tokdec_blend")

    def __init__(self, store="io"):
        from . import attn_viz as AV
        self.AV = AV
        self.store = W.Dropdown(options=["io", "blk4", "blk", "io_up"], value=store, description="store")
        self.hide_side = W.Checkbox(value=True, description="hide smoke / leak / eval runs", indent=False)
        self.a_sel = W.SelectMultiple(options=[], rows=9, description="attention", layout=W.Layout(width="460px"))
        self.r_sel = W.SelectMultiple(options=[], rows=9, description="references", layout=W.Layout(width="460px"))
        self.x = W.ToggleButtons(options=[("epoch", "epoch"), ("optimiser steps", "steps"), ("training days seen", "days")], description="x-axis")
        self.refresh = W.Button(description="Refresh", icon="refresh")
        self.a_out, self.b_out, self.c_out = W.Output(), W.Output(), W.Output()
        self.b_redraw = W.Button(description="Redraw", icon="refresh")
        self._listeners = []                                     # panel redraws, called when the attention selection changes
        self._busy = _Busy()
        self.a_sel.observe(lambda *_: self._changed(), "value")
        for w in (self.store, self.hide_side):
            w.observe(lambda *_: self.fill(), "value")
        self.refresh.on_click(lambda *_: self.fill())
        self.fill()

    def fill(self):
        AV = self.AV
        kw = dict(store=self.store.value, main_only=self.hide_side.value)
        keep_a, keep_r = set(self.a_sel.value), set(self.r_sel.value)
        with self._busy:
            self.a_sel.options = [(f"{p.name}  [{AV.kind_label(p)}]", str(p)) for p in AV.attn_runs(**kw)]
            self.r_sel.options = [(f"{p.name}  [{AV.kind_label(p)}]", str(p)) for p in AV.ref_runs(**kw)]
            self.a_sel.value = tuple(v for _, v in self.a_sel.options if v in keep_a) or \
                tuple(v for _, v in self.a_sel.options if Path(v).name in self.DEFAULT)
            self.r_sel.value = tuple(v for _, v in self.r_sel.options if v in keep_r) or \
                tuple(v for _, v in self.r_sel.options if Path(v).name in AV.REFS)
        self._changed()

    def _changed(self):
        if self._busy.on:
            return
        for f in self._listeners:
            f()

    # (a) training curves
    def curves(self):
        def _draw(*_):
            if self._busy.on:
                return
            with self.a_out:
                clear_output(wait=True)
                if not self.a_sel.value and not self.r_sel.value:
                    print("no runs selected")
                    return
                fig, rows = self.AV.draw_curves([Path(v) for v in self.a_sel.value], [Path(v) for v in self.r_sel.value], x=self.x.value)
                fig.set_size_inches(15, 6.6)
                fig.tight_layout()
                display(fig)
                plt.close(fig)

        self.r_sel.observe(_draw, "value")
        self.x.observe(_draw, "value")
        self._listeners.append(_draw)
        display(W.HBox([self.a_sel, self.r_sel, W.VBox([self.store, self.hide_side, self.refresh])]), self.x, self.a_out)
        _draw()

    def _caches(self, quiet=False):
        """Weight caches of the selected attention runs; prints the build command for runs without one."""
        AV, out = self.AV, []
        for v in self.a_sel.value:
            name = Path(v).name
            if not (Path(v) / "config.json").exists():
                if not quiet:
                    print(f"{name}: still training (no config.json); build its cache once it has finished")
                continue
            z = AV.load_cache(v)
            if z is None:
                if not quiet:
                    print(f"{name}: no weight cache; build it with\n  {AV.cache_cmd(v)}")
                continue
            if AV.cache_stale(v, z) and not quiet:
                print(f"{name}: cache older than model.pt; rebuild with\n  {AV.cache_cmd(v)}")
            out.append(z)
        return out

    # (b) pooling weight by day offset
    def offsets(self):
        def _draw(*_):
            with self.b_out:
                clear_output(wait=True)
                cs = self._caches()
                if not cs:
                    print("no selected attention run has a weight cache")
                    return
                fig = self.AV.draw_offsets(cs)
                display(fig)
                plt.close(fig)

        self.b_redraw.on_click(_draw)
        self._listeners.append(_draw)
        display(self.b_redraw, self.b_out)
        _draw()

    # (c) day-weight maps
    def maps(self):
        AV = self.AV
        run = W.Dropdown(options=[], description="run", layout=W.Layout(width="440px"))
        day = W.Dropdown(options=[], description="val day", layout=W.Layout(width="260px"))
        offs = W.SelectMultiple(options=[(f"t{k:+d}", k) for k in range(-8, 9) if k], value=(-4, -2, -1, 1, 2, 4), rows=8, description="offsets")
        with_t = W.Checkbox(value=True, description="day t")
        with_none = W.Checkbox(value=True, description="no day")
        busy = _Busy()

        def _fill(*_):
            keep = run.value
            with busy:
                run.options = [(Path(v).name, v) for v in self.a_sel.value if AV.cache_path(v).exists()]
                vals = [v for _, v in run.options]
                run.value = keep if keep in vals else (vals[0] if vals else None)
            _days()

        def _days(*_):
            if busy.on:
                return
            z = AV.load_cache(run.value) if run.value else None
            keep = day.value
            with busy:
                day.options = [] if z is None else [(f"{d}  ({int(h):,} hidden px)", i) for i, (d, h) in enumerate(zip(z["dates"], z["hid"].sum((1, 2))))]
                vals = [v for _, v in day.options]
                day.value = keep if keep in vals else (vals[len(vals) // 2] if vals else None)
            _draw()

        def _draw(*_):
            if busy.on:
                return
            with self.c_out:
                clear_output(wait=True)
                if not run.value:
                    print("no selected attention run has a weight cache (see (b) for the build command)")
                    return
                if day.value is None:
                    return
                fig = AV.draw_maps(AV.load_cache(run.value), day.value, offs.value, with_t=with_t.value, with_none=with_none.value)
                display(fig)
                plt.close(fig)

        self._listeners.append(_fill)
        self.b_redraw.on_click(_fill)
        run.observe(_days, "value")
        for w in (day, offs, with_t, with_none):
            w.observe(_draw, "value")
        display(W.HBox([W.VBox([run, day, W.HBox([with_t, with_none])]), offs]), self.c_out)
        _fill()


# ---------------------------------------------------------------- tracer
def tracer(default=()):
    from . import tracer_viz as TV

    sel = W.SelectMultiple(options=[], rows=8, description="tracer runs", layout=W.Layout(width="520px"))
    refresh = W.Button(description="Refresh", icon="refresh")
    out = W.Output()
    busy = _Busy()

    def _fill(*_):
        keep = set(sel.value)
        with busy:
            sel.options = [(p.name, str(p)) for p in TV.tracer_runs()]
            sel.value = tuple(v for _, v in sel.options if v in keep or Path(v).name in default)
        _draw()

    def _draw(*_):
        if busy.on:
            return
        with out:
            clear_output(wait=True)
            runs = [Path(v) for v in sel.value]
            if not runs:
                print("no tracer runs selected (Ctrl/Cmd-click to pick several)")
                return
            for fn in (TV.draw_val_by_lead, TV.draw_best_epoch, TV.draw_factors, TV.draw_speed, TV.draw_factor_maps,
                       TV.draw_source_by_epoch, TV.draw_source_maps):
                fig = fn(runs)
                if fig is not None:
                    display(fig)
                    plt.close(fig)

    sel.observe(_draw, "value")
    refresh.on_click(_fill)
    display(W.HBox([sel, refresh]), out)
    _fill()
