"""Build regions.npy (int8 HxW, 0 = not ocean) and region_names.json for the Black Sea store. Later rules overwrite earlier ones."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from common import *

NAMES = {1: "NW shelf", 2: "Western open sea", 3: "Eastern open sea", 4: "Anatolian shelf",
         5: "Crimea/Kerch/Caucasus shelf", 6: "Azov Sea + Kerch strait", 7: "Marmara/Bosporus corner"}


def build():
    d = S.load_store(); ocean = d["ocean"].astype(bool); depth = np.asarray(d["depth"]); lat, lon = d["lat"], d["lon"]
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    dep = np.where(np.isfinite(depth), depth, np.nan)
    shelf = dep <= 100; deep = dep > 100   # NaN depth (ocean px without depth) -> neither
    reg = np.zeros(ocean.shape, np.int8)
    # order of assignment resolves overlaps: broad rules first, specific last
    reg[ocean & shelf & (LO < 33.5)] = 1
    reg[ocean & deep & (LO < 34)] = 2
    reg[ocean & deep & (LO >= 34)] = 3
    reg[ocean & shelf & (LA < 42.5) & (LO >= 28.5)] = 4
    reg[ocean & shelf & (LA >= 42.5) & (LO >= 33.5)] = 5
    reg[ocean & (LA > 45.2) & (LO > 34.5) & (dep <= 20)] = 6
    reg[ocean & (LA < 41.3) & (LO < 29.5)] = 7
    return d, ocean, reg, dep


if __name__ == "__main__":
    d, ocean, reg, dep = build()
    np.save(HERE / "regions.npy", reg); json.dump(NAMES, open(HERE / "region_names.json", "w"), indent=1)
    print(f"ocean px {int(ocean.sum())}; unassigned ocean px (reg 0 but ocean): {int((ocean & (reg == 0)).sum())}; ocean px with NaN depth: {int((ocean & ~np.isfinite(dep)).sum())}")
    print("| region | pixels |\n|---|---|")
    for k, v in NAMES.items():
        print(f"| {k} {v} | {int((reg == k).sum())} |")
    lat, lon = d["lat"], d["lon"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    cols = ["#FFFFFF"] + OKABE_ITO[:7]
    ax.imshow(reg, origin="lower", cmap=ListedColormap(cols), vmin=-0.5, vmax=7.5, extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto", interpolation="nearest")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols[k], label=f"{k} {NAMES[k]} (n={int((reg == k).sum())})") for k in NAMES], loc="upper left", fontsize=8)
    ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)"); ax.set_title("Black Sea region map (ocean pixels, 2 km grid)")
    fig.tight_layout(); fig.savefig(FIG / "regions_map.png", dpi=130)
