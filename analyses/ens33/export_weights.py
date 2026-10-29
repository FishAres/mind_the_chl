"""Bridge #35: record of the ens33 ensembles (members, weights, scores). Nothing frozen; reads results/ens33.json only.

Writes results/weights.csv (sea, scheme, fit, bin, member, weight; every member incl. zeros) and results/ENSEMBLES.md.
Run: python -m analyses.ens33.export_weights
"""
import csv
import json
from pathlib import Path

RES = Path("analyses/ens33/results")
SEAS = {"io": "step2_io", "blk4": "step3_blk4"}
SCHEMES = ["w_keybins", "w_distbins", "w_global"]


def mae(s, split):
    return f"{s[split]['mae']:.5f}"


def main():
    d = json.loads((RES / "ens33.json").read_text())
    rows, md = [], ["# ens33 ensembles: members, weights, scores (eval only, not frozen)", "",
                    "Source: results/ens33.json (analyses/ens33/report.py). Weights per bin are ≥ 0 and sum to 1, fit on one val "
                    "half (A or B) to minimise hidden-px MAE. oos val = half A scored with the B fit + half B with the A fit. "
                    "Inputs: ~/mind_the_chl_gap/data/ens33/{io,blk4}_hidden.npz (per-hidden-px member predictions, from "
                    "analyses/ens33/prep.py); member preds: IO via score.load_preds(run), Black Sea via <run>/preds.npz. "
                    "Full weight table: results/weights.csv.", ""]
    for sea, key in SEAS.items():
        s = d[key]
        md += [f"## {sea}", "", "Members: " + ", ".join(s["members"]), "",
               "| ensemble | val | test (fit A / fit B) | val A | val B |", "|---|---|---|---|---|"]
        plain = {"ens11_mean": s["ens11_mean"]} if sea == "io" else s["plain"]
        for name, p in plain.items():
            sc = p.get("scores", p)
            md.append(f"| {name} | {mae(sc, 'val')} | {mae(sc, 'test')} | {mae(sc, 'A')} | {mae(sc, 'B')} |")
        for sch in SCHEMES:
            sc = s[sch]["scores"]
            md.append(f"| {sch} (oos) | {mae(sc['oos'], 'val')} | {mae(sc['fitA'], 'test')} / {mae(sc['fitB'], 'test')} | "
                      f"{mae(sc['oos'], 'A')} | {mae(sc['oos'], 'B')} |")
        if sea == "blk4":
            md += ["", "Plain-mean members: " + "; ".join(f"{k}: {', '.join(v['members'])}" for k, v in plain.items()
                                                         if k != "mean_all9")]
        for sch in SCHEMES:
            w = s[sch]["weights"]
            bins = s[sch]["bins"]
            md += ["", f"### {sea} {sch} weights (fit A / fit B)", "",
                   "| member | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
            for m in s["members"]:
                md.append(f"| {m} | " + " | ".join(f"{w['A'][b][m]:.2f} / {w['B'][b][m]:.2f}" for b in bins) + " |")
                rows += [(sea, sch, f, b, m, round(w[f][b][m], 6)) for f in "AB" for b in bins]
        md.append("")
    with open(RES / "weights.csv", "w", newline="") as f:
        csv.writer(f).writerows([("sea", "scheme", "fit", "bin", "member", "weight")] + rows)
    (RES / "ENSEMBLES.md").write_text("\n".join(md))
    print(RES / "weights.csv", len(rows), "rows;", RES / "ENSEMBLES.md")


if __name__ == "__main__":
    main()
