"""Append the dated io6 entry to analyses/FINDINGS.md from analyses/io6_kriging/results/io6_summary.json (report.py) and the check JSONs.
Observations only (no interpretation section). Skips if an io6 entry is already present (prints it instead with --print).

Usage (project root): python -m analyses.io6_kriging.findings [--print]
"""
import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FINDINGS = HERE.parent / "FINDINGS.md"
TITLE = "## Learned kriging and spatial token attention, io6"
NAME = {"io6_tokdec_krig": "K1 (io6_tokdec_krig)", "io6_tokdec_krig_signed": "K1-signed (io6_tokdec_krig_signed)",
        "io6_tokdec_blend_spatial": "spatial (io6_tokdec_blend_spatial)", "io4_tokdec_blend": "io4_tokdec_blend", "io3_ptok4": "io3_ptok4",
        "io2_tattn8": "io2_tattn8", "io2_base_eval": "conv ±4 seed 0", "io2_seed1": "conv seed 1", "io2_seed2": "conv seed 2", "ens3_mean": "3-seed conv ensemble"}
IO6 = ("io6_tokdec_krig", "io6_tokdec_blend_spatial", "io6_tokdec_krig_signed")
RUN_CFG = {"io6_tokdec_krig": "krig", "io6_tokdec_blend_spatial": "spatial", "io6_tokdec_krig_signed": "krig_signed"}


def f6(x):
    return "n/a" if x is None else f"{x:.6f}"


def sg(x, n=4):
    return "n/a" if x is None else f"{x:+.{n}f}"


def entry(S):
    R = {r["run"]: r for r in S["rows"]}
    done = [k for k in IO6 if k in R]
    L = [f"{TITLE} (IO; {time.strftime('%Y-%m-%d', time.gmtime())}; chl_cnn/krig.py, chl_cnn/krig_triton.py, chl_cnn/tokdec.py --tokdec-spatial, "
         "train_pinn.py --tokdec-krig [--krig-signed] / --tokdec-spatial, analyses/io6_kriging/)"]
    L += ["- Set-up (design: analyses/io6_kriging/DESIGN.md, revision 2026-10-24):",
          "  - K1 io6_tokdec_krig: io4 tokens + decoder; the own-pixel blend replaced by the learned-kriging stem. Stencil |dt| <= 4, 484 offsets "
          "(dt 0: disc r3 excluding the px, 28; |dt| 1-2: disc r5, 81 each; |dt| 3-4: 5 x 5 block + 8 points at r6, 33 each). Masked softmax over the "
          "visible keys (after donor masking) + a no-key slot per head; query / key / value features of DESIGN.md section 2 (advected offset with "
          "learned s, init 0.25; local visible density); relative-position bias per (offset, head); D 32, 4 heads, dv 16. 20 channels (attention "
          "output 16, estimate, spread, mass, entropy) -> backbone input 32 + 20 + 11 + 7 = 70 ch. No back-advected centring (--krig-advect "
          "implemented, not run).",
          "  - K1-signed io6_tokdec_krig_signed: K1 + a signed head pair, w = softmax(s+) − lambda softmax(s−), lambda = sigmoid (init 0.1); "
          "3 more channels (signed estimate, sum softmax(s+), lambda sum softmax(s−)) -> 73 ch.",
          "  - spatial io6_tokdec_blend_spatial: io4_tokdec_blend + 2 pre-norm spatial self-attention blocks over the 1,164 pooled ocean-patch tokens "
          "(global, fused SDPA, 4 heads, D 128, learned relative-position bias per (patch offset, head); output projections zero-initialised).",
          "  - Recipe as io4_tokdec_blend: batch 8, 60-epoch cosine, lr 1e-3, wd 1e-4, --ckpt-from 10, single seed 0.",
          "  - Kriging implementation: fused Triton kernels (online softmax over all offsets per 16-px block, flash-style backward; tl.dot in tf32); "
          "the PyTorch streamed path (chunks, checkpointing) is the reference."]
    L.append("- Checks:")
    ut = S.get("unit_tests")
    if ut:
        dense = {k: v for k, v in ut.items() if k.startswith("dense_vs_stream")}
        L.append(f"  - CPU unit tests (analyses/io6_kriging/test_krig.py): {sum(1 for k, v in ut.items() if isinstance(v, dict) and v.get('passed'))} of "
                 f"{sum(1 for v in ut.values() if isinstance(v, dict))} passed; streamed vs dense masked softmax max |Δ| outputs "
                 f"{max(v['max_abs_out'] for v in dense.values()):.1e}, gradients {max(v['max_rel_grad'] for v in dense.values()):.1e} (relative); "
                 f"signed weights incl. no-key slots sum to 1 − lambda within {ut['signed_weights_sum']['max_dev_dense']:.1e}.")
    gc = S.get("gpu_check")
    if gc:
        ag = gc["agree"]
        L.append("  - GPU check (gpu_check.py, random inputs, batch 2): Triton vs PyTorch reference, max |Δ| outputs / max relative |Δ| gradients: "
                 + "; ".join(f"{k} {v['max_abs_out']:.1e} / {v['max_rel_grad']:.1e}" for k, v in ag.items()) + ".")
    for k, v in sorted(S.get("leak", {}).items()):
        c = v["cases"]
        L.append(f"  - leak test {k[5:]} (val day {v['day']}, {v['n_hidden']} hidden px): +1 on day t's hidden px and on t+2's donor-hidden observed px: "
                 f"max change {max(max(c['hidden'][x] for x in c['hidden'] if x.startswith('max')), max(c['masked_t+2'][x] for x in c['masked_t+2'] if x.startswith('max'))):.1e} "
                 f"over every recorded tensor and the hidden-px predictions; positive control (all observed px of t+2): hidden-px predictions change by up to "
                 f"{c['obs_t+2']['max_abs_change_pred_hidden']:.3f}; no_leak {v['no_leak']}.")
    for g in S.get("gates", []):
        L.append(f"  - gate: {g}")
    if not done:
        L.append("- No io6 run finished; see the gates above.")
        return "\n".join(L) + "\n"
    b0, bb = R["io2_base_eval"], R["io4_tokdec_blend"]
    L.append(f"- Val / test hidden-px MAE (val n = {b0['val']['n_days']:,} days, {b0['val']['n_hidden']:,} px; test {b0['test']['n_days']:,} days, "
             f"{b0['test']['n_hidden']:,} px):")
    for k in done:
        r = R[k]
        L.append(f"  - {NAME[k]} {f6(r['val']['mae'])} / {f6(r['test']['mae'])} (best epoch {r.get('best_epoch', 'n/a')}); vs io4_tokdec_blend "
                 f"{sg(r['val']['mae'] - bb['val']['mae'], 5)} / {sg(r['test']['mae'] - bb['test']['mae'], 5)}; vs conv seed 0 "
                 f"{sg(r['val']['mae'] - b0['val']['mae'], 5)} / {sg(r['test']['mae'] - b0['test']['mae'], 5)}; non-finite predictions "
                 f"{r['val']['n_nonfinite']} / {r['test']['n_nonfinite']};")
    L.append("  - references: " + "; ".join(f"{NAME[k]} {f6(R[k]['val']['mae'])} / {f6(R[k]['test']['mae'])}" for k in
                                            ("io4_tokdec_blend", "io3_ptok4", "io2_tattn8", "io2_base_eval", "ens3_mean") if k in R) + ".")
    L.append("- RMSE val / test, bias val / test: " + "; ".join(f"{NAME[k]} {R[k]['val']['rmse']:.4f} / {R[k]['test']['rmse']:.4f}, "
                                                           f"{R[k]['val']['bias']:+.4f} / {R[k]['test']['bias']:+.4f}" for k in done + ["io4_tokdec_blend"]) + ".")
    L.append(f"- Gradient ratio / corr, val, lags 1 / 2 / 4 px ({b0['val']['grad']['1']['n_pairs'] / 1e6:.2f} M lag-1 pairs):")
    for k in done + ["io4_tokdec_blend", "io2_base_eval"]:
        g = R[k]["val"]["grad"]
        L.append(f"  - {NAME[k]} " + ", ".join(f"{g[x]['ratio']:.3f}/{g[x]['corr']:.3f}" for x in ("1", "2", "4")) + ";")
    ep = S.get("epochs", {})
    L.append("- Best val over epochs 11-30 / 11-60: " + "; ".join(
        f"{NAME[k]} {ep[k]['best_11_30']:.5f} (ep {ep[k]['epoch_11_30']}) / " + (f"{ep[k]['best_11_60']:.5f} (ep {ep[k]['epoch_11_60']})" if ep[k]['best_11_60'] else "n/a")
        for k in done + ["io4_tokdec_blend"] if k in ep) + " (S30 0.0015, S60 0.00024 from the conv seeds).")
    L.append("- Cost at batch 8 (median s / epoch incl. the val pass, peak GB allocated / reserved, params): " + "; ".join(
        f"{NAME[k]} {R[k].get('s_per_epoch', float('nan')):.0f} s, {R[k].get('peak_gpu_gb') or float('nan'):.1f} / {R[k].get('peak_gpu_reserved_gb') or float('nan'):.1f} GB, "
        f"{R[k].get('n_params', 'n/a'):,}" for k in done + ["io4_tokdec_blend"]) + ".")
    sm = S.get("smokes", {})
    if sm:
        L.append("- Smokes (2 epochs, batch 8): " + "; ".join(f"{k} {', '.join(f'{x:.0f}' for x in v['s_per_epoch'])} s/epoch, {v['peak_gpu_gb']:.1f} / "
                                                          f"{v['peak_gpu_reserved_gb']:.1f} GB, {v['n_params']:,} params" for k, v in sm.items()) + ".")
    L.append("- Val MAE gain over conv seed 0, by bin (" + " / ".join(NAME[k] for k in done + ["io4_tokdec_blend"]) + "):")
    for key, lab in (("by_dist", "distance 1 / 2-3 / 4-7 / 8-15 / 16+ px"), ("by_age2", "two-sided age 1 d / 2-3 d / 4-8 d / none"),
                     ("by_age", "past-only age 1 d / 2-3 d / 4-8 d / none")):
        bins = list(b0["val"][key])
        ns = ", ".join(f"{b0['val'][key][b]['n'] / 1e6:.2f} M" for b in bins)
        L.append(f"  - {lab} (n {ns}): " + "; ".join(" / ".join(sg(b0['val'][key][b]['mae'] - R[k]['val'][key][b]['mae']) for b in bins)
                                                      for k in done + ["io4_tokdec_blend"] if key in R[k]['val']) + ".")
    kr = next((R[k] for k in done if "by_key" in R[k]["val"]), None)
    if kr:
        bk = kr["val"]["by_key"]
        L.append(f"- Val hidden px by visible kriging-stencil keys: >= 1 key n {bk['>=1 key']['run']['n']:,}, no key n {bk['no key']['run']['n']:,}; keys per hidden px "
                 f"mean {bk['n_keys_hidden_px']['mean']:.1f}, median {bk['n_keys_hidden_px']['median']:.0f}. MAE >= 1 key / no key:")
        for k in done:
            if "by_key" in R[k]["val"]:
                b = R[k]["val"]["by_key"]
                L.append(f"  - {NAME[k]} {b['>=1 key']['run']['mae']:.5f} / {b['no key']['run']['mae']:.5f};")
        L.append(f"  - io4_tokdec_blend {bk['>=1 key']['io4_tokdec_blend']['mae']:.5f} / {bk['no key']['io4_tokdec_blend']['mae']:.5f}; conv seed 0 "
                 f"{bk['>=1 key']['conv_seed0']['mae']:.5f} / {bk['no key']['conv_seed0']['mae']:.5f}.")
    for k, v in S.get("kriging", {}).items():
        lp = S.get("learned", {}).get(k, {})
        L.append(f"- Kriging weights, {NAME[k]} ({v['n_days']} val days, {v['n_hidden_px']:,} hidden px, {v['n_hidden_px_with_key']:,} with >= 1 key; mean over the value heads):")
        L.append("  - weight share by dt −4..+4: " + " / ".join(f"{v['share_by_dt'][str(t)] if str(t) in v['share_by_dt'] else v['share_by_dt'][t]:.3f}"
                                                            for t in range(-4, 5)) + f"; total on real keys {v['share_total']:.3f};")
        wv = v["w_per_visible_by_dt"]
        L.append("  - mean weight per visible key by dt −4..+4: " + " / ".join(f"{wv[str(t)] if str(t) in wv else wv[t]:.4f}" for t in range(-4, 5)) + ";")
        sst = v["sst"]
        L.append("  - mean weight per visible key by |dSST| 0 / 0.25 / 0.5 / 0.75 / 1.0 / 1.25-1.5 / 1.75-2 / 2.25-3 / >3 K (store SST quantised to 0.25 K): "
                 + " / ".join(f"{x:.4f}" for x in sst["w_per_visible"]) + " (n " + ", ".join(f"{x / 1e6:.2f} M" for x in sst["n"]) + ");")
        if lp:
            L.append(f"  - learned s {lp['s']:.3f} (init {lp['s_init']})" + (f"; lambda {lp['lam']:.3f} (init 0.1)" if "lam" in lp else "")
                     + "; no-key logits " + ", ".join(f"{x:+.2f}" for x in lp["none_logit"]) + ".")
        if "signed_share_by_dt" in v:
            L.append("  - signed pair, weight share by dt (softmax(s+) / softmax(s−)): " + " / ".join(
                f"{v['signed_share_by_dt'][str(t)][0]:.3f}:{v['signed_share_by_dt'][str(t)][1]:.3f}" for t in range(-4, 5)) + ".")
    for k in done:
        s = R[k]["val"].get("spread")
        if s:
            L.append(f"- Kriging spread vs |error|, {NAME[k]} (val hidden px with >= 1 key, n {s['n']:,}): Spearman {s['spearman_all']:.3f}; MAE by spread decile "
                     f"{s['mae_by_decile'][0]:.4f} (lowest) ... {s['mae_by_decile'][-1]:.4f} (highest); Spearman by distance bin "
                     + " / ".join(f"{x['spearman']:.3f}" if x['spearman'] is not None else "n/a" for x in s["by_dist"].values()) + ".")
    L += ["- Outputs:",
          "  - analyses/io6_kriging/results/{io6_summary, " + ", ".join(done) + ", gpu_check, unit_tests, leak_*}.json;",
          "  - figures io6_curves, io6_bins, io6_variogram_lines_<run>, io6_variogram_maps_<run>, io6_weight_vs_sst, io6_learned_params, "
          "io6_spread_vs_error (.png);",
          "  - recorder data ~/mind_the_chl_gap/data/io6/<run>_{krigall,diag}.npz; queue ~/mind_the_chl_gap/logs/io6/queue.sh."]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--print", action="store_true")
    a = p.parse_args()
    S = json.loads((RES / "io6_summary.json").read_text())
    txt = entry(S)
    if a.print:
        print(txt)
        return
    cur = FINDINGS.read_text()
    if TITLE in cur:
        print("io6 entry already in FINDINGS.md; not appended")
        print(txt)
        return
    with open(FINDINGS, "a") as f:
        f.write(("\n" if not cur.endswith("\n\n") else "") + txt)
    print("appended to", FINDINGS)
    print(txt)


if __name__ == "__main__":
    main()
