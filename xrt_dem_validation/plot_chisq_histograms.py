"""
Chi-square histograms (v2) for the corrected-intensity 162-DEM campaign.

Two overlaid figures (IDL orange vs XRTpy blue), log10(chi2) x-axis, with a
dashed median line per solver:

  1. BASE      -- 162 IDL + 162 XRTpy base chi-square (one per DEM)
                  plots/idl_xrtpy_chi_square_histograms/chisq_hist_base.png
  2. ALL MC    -- ~162,000 IDL + ~162,000 XRTpy MC chi-square (all draws)
                  plots/idl_xrtpy_chi_square_histograms/chisq_hist_mc_all.png

Sources (each solver from its own clean file, no paired CSV):
  Python base : data/xrtpy_monte_carlo_dem_results/*_MC1000.npz  -> mc_chisq[0]  (fallback *_base.npz)
  Python MC   : data/xrtpy_monte_carlo_dem_results/*_MC1000.npz  -> mc_chisq[1:]
  IDL base    : data/idl_base_chisq.csv        (dem_id, chisq)
  IDL MC      : data/idl_mc_chisq_long.csv     (dem_id, mc_run, chisq); mc_run>=1

Usage (from xrt_dem_validation/):
    python plot_chisq_histograms.py
"""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
XRTPY_DIR = BASE_DIR / "data" / "xrtpy_monte_carlo_dem_results"
IDL_BASE_CSV = BASE_DIR / "data" / "idl_base_chisq.csv"
IDL_MC_CSV = BASE_DIR / "data" / "idl_mc_chisq_long.csv"
PLOT_DIR = BASE_DIR / "plots" / "idl_xrtpy_chi_square_histograms"

N_MC = 1000
COLOR_IDL = "orange"
COLOR_XRTPY = "#1E90FF"
CHI_FLOOR = 1e-6   # clip for log10 so zero/negative never breaks the axis


# ── Python loaders (from npz) ─────────────────────────────────────────────────
def _dem_id(p):
    return int(re.search(r"idx(\d+)_", p.name).group(1))


def xrtpy_base_and_mc():
    """Return (base_vals[162], mc_vals[~162000]) from the MC npz files."""
    base, mc = [], []
    files = sorted(XRTPY_DIR.glob(f"xrtpy_dem_idx*_allfilters_MC{N_MC}.npz"),
                   key=_dem_id)
    if not files:
        raise FileNotFoundError(f"No XRTpy MC npz in {XRTPY_DIR}")
    for p in files:
        d = np.load(p, allow_pickle=True)
        chi = np.asarray(d["mc_chisq"], dtype=float)  # [0]=base, [1:]=MC
        base.append(chi[0])
        mc.append(chi[1:])
    return np.array(base), np.concatenate(mc)


# ── IDL loaders (from CSV) ────────────────────────────────────────────────────
def idl_base():
    vals = []
    with open(IDL_BASE_CSV) as f:
        for row in csv.DictReader(f):
            vals.append(float(row["chisq"]))
    return np.array(vals)


def idl_mc():
    """All MC chi-square (mc_run >= 1), skipping the base rows."""
    vals = []
    with open(IDL_MC_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for dem_id, mc_run, chisq in reader:
            if int(mc_run) >= 1:
                vals.append(float(chisq))
    return np.array(vals)


# ── Histogram ─────────────────────────────────────────────────────────────────
def make_hist(idl_vals, py_vals, title, out_png, xlabel, subtitle=None):
    idl_vals = np.asarray(idl_vals, dtype=float)
    py_vals = np.asarray(py_vals, dtype=float)

    # Drop inf/NaN so they can't break the bin edges; report how many.
    n_bad_idl = int((~np.isfinite(idl_vals)).sum())
    n_bad_py = int((~np.isfinite(py_vals)).sum())
    idl_vals = idl_vals[np.isfinite(idl_vals)]
    py_vals = py_vals[np.isfinite(py_vals)]
    if n_bad_idl or n_bad_py:
        print(f"  dropped non-finite chi-square: IDL {n_bad_idl}, XRTpy {n_bad_py}")

    log_idl = np.log10(np.clip(idl_vals, CHI_FLOOR, None))
    log_py = np.log10(np.clip(py_vals, CHI_FLOOR, None))

    lo = np.floor(min(log_idl.min(), log_py.min()))
    hi = np.ceil(max(log_idl.max(), log_py.max()))
    bins = np.linspace(lo, hi, 60)

    med_idl = float(np.median(idl_vals))
    med_py = float(np.median(py_vals))

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(log_idl, bins=bins, color=COLOR_IDL, alpha=0.55,
            label=f"IDL   (N={len(idl_vals):,})")
    ax.hist(log_py, bins=bins, color=COLOR_XRTPY, alpha=0.55,
            label=f"XRTpy (N={len(py_vals):,})")

    ax.axvline(np.log10(max(med_idl, CHI_FLOOR)), color=COLOR_IDL,
               linestyle="--", linewidth=2.0,
               label=f"IDL median χ² = {med_idl:.2f}")
    ax.axvline(np.log10(max(med_py, CHI_FLOOR)), color=COLOR_XRTPY,
               linestyle="--", linewidth=2.0,
               label=f"XRTpy median χ² = {med_py:.2f}")

    ttl = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(ttl, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # console summary
    print(f"  {out_png.name}")
    print(f"    IDL   : N={len(idl_vals):,}  median={med_idl:.3f}  "
          f"max={idl_vals.max():.3g}")
    print(f"    XRTpy : N={len(py_vals):,}  median={med_py:.3f}  "
          f"max={py_vals.max():.3g}")


# ── Side-by-side panel (base | all-MC) ───────────────────────────────────────
def draw_panel(ax, idl_vals, py_vals, title, xlabel):
    idl_vals = np.asarray(idl_vals, dtype=float)
    py_vals = np.asarray(py_vals, dtype=float)
    idl_vals = idl_vals[np.isfinite(idl_vals)]
    py_vals = py_vals[np.isfinite(py_vals)]

    log_idl = np.log10(np.clip(idl_vals, CHI_FLOOR, None))
    log_py = np.log10(np.clip(py_vals, CHI_FLOOR, None))

    lo = np.floor(min(log_idl.min(), log_py.min()))
    hi = np.ceil(max(log_idl.max(), log_py.max()))
    bins = np.linspace(lo, hi, 60)

    med_idl = float(np.median(idl_vals))
    med_py = float(np.median(py_vals))

    ax.hist(log_idl, bins=bins, color=COLOR_IDL, alpha=0.55)
    ax.hist(log_py, bins=bins, color=COLOR_XRTPY, alpha=0.55)
    ax.axvline(np.log10(max(med_idl, CHI_FLOOR)), color=COLOR_IDL,
               linestyle="--", linewidth=2.0)
    ax.axvline(np.log10(max(med_py, CHI_FLOOR)), color=COLOR_XRTPY,
               linestyle="--", linewidth=2.0)

    ax.text(0.97, 0.80,
            f"IDL median χ² = {med_idl:.2f}\nXRTpy median χ² = {med_py:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(visible=True, alpha=0.3)


def make_panel(idl_b, py_base, idl_m, py_mc, out_png):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6.5))

    draw_panel(
        ax_left, idl_b, py_base,
        title=f"Base DEM χ²  —  {len(idl_b)} synthetic DEMs",
        xlabel=r"log$_{10}$($\chi^2$)  (base DEM)",
    )
    draw_panel(
        ax_right, idl_m, py_mc,
        title=f"All-MC χ²  —  162 DEMs × 1000 MC  "
              f"({len(idl_m):,} draws each)",
        xlabel=r"log$_{10}$($\chi^2$)  (every MC draw)",
    )

    handles = [
        Patch(fc=COLOR_IDL, alpha=0.55, label="IDL"),
        Patch(fc=COLOR_XRTPY, alpha=0.55, label="XRTpy"),
        Line2D([], [], color=COLOR_IDL, ls="--", lw=2, label="IDL median"),
        Line2D([], [], color=COLOR_XRTPY, ls="--", lw=2, label="XRTpy median"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_png.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading XRTpy chi-square from npz...")
    py_base, py_mc = xrtpy_base_and_mc()
    print("Loading IDL chi-square from CSV...")
    idl_b = idl_base()
    idl_m = idl_mc()

    print("\nFigures:")
    make_hist(
        idl_b, py_base,
        title=f"Base DEM χ² Distribution — {len(idl_b)} synthetic DEMs",
        out_png=PLOT_DIR / "chisq_hist_base.png",
        xlabel=r"log$_{10}$($\chi^2$)  (base DEM)",
    )
    make_hist(
        idl_m, py_mc,
        title="All-MC χ² Distribution — 162 synthetic DEMs, 1000 MC each",
        out_png=PLOT_DIR / "chisq_hist_mc_all.png",
        xlabel=r"log$_{10}$($\chi^2$)  (every MC draw)",
        subtitle=f"{len(idl_m):,} IDL draws vs {len(py_mc):,} XRTpy draws",
    )
    make_panel(
        idl_b, py_base, idl_m, py_mc,
        out_png=PLOT_DIR / "chisq_hist_panel_base_and_mc.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()