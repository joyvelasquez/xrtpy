"""
Compare synthetic truth, IDL base DEM results, and XRTpy base DEM results.

The script performs two parts of the synthetic validation workflow:

1. Compare the recovered base DEM from IDL and XRTpy against the known
   synthetic truth for every DEM index available in both result sets
   (one overlay plot per DEM).
2. Write a table of the paired IDL/XRTpy base chi-square values.
   (Chi-square histograms are made by plot_chisq_histograms.py.)

The script automatically discovers DEM indices that have both an IDL base SAV
file and an XRTpy base NPZ file, so partial result sets are handled without a
hardcoded index list.

Inputs
------
IDL base results:
    data/IDL_synthetic_dem_data/
    idl_synthetic_dem_idx{N}_allfilters_base.sav

XRTpy base results:
    data/xrtpy_base_dem_results/
    xrtpy_dem_idx{N}_allfilters_base.npz

Synthetic truth DEMs:
    data/synthetic_dems_data/
    DEM_{N}.txt

Outputs
-------
Comparison table:
    data/chi_square_comparison_tables/base_dem_comparison.csv

Per-DEM overlays:
    plots/idl_xrtpy_synthetic_base_dem_comparison/per_dem_overlays/
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import readsav


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent

IDL_BASE_DIR = BASE_DIR / "data" / "IDL_synthetic_dem_data"
IDL_BASE_PATTERN = "idl_synthetic_dem_idx{n}_allfilters_base.sav"

XRTPY_BASE_DIR = BASE_DIR / "data" / "xrtpy_base_dem_results"
XRTPY_BASE_PATTERN = "xrtpy_dem_idx{n}_allfilters_base.npz"

TRUE_DEM_DIR = BASE_DIR / "data" / "synthetic_dems_data"
TRUE_DEM_PATTERN = "DEM_{n}.txt"

OUT_CSV = (
    BASE_DIR
    / "data"
    / "chi_square_comparison_tables"
    / "base_dem_comparison.csv"
)

PLOT_DIR = (
    BASE_DIR
    / "plots"
    / "idl_xrtpy_synthetic_base_dem_comparison"
)

OVERLAY_DIR = PLOT_DIR / "per_dem_overlays"

DEM_FLOOR = 1e-99

# ============================================================
# Discovery
# ============================================================

def _extract_index(filename: str, pattern: str) -> int | None:
    """Turn a glob-matched filename into its DEM index using `pattern`."""
    regex = re.escape(pattern).replace(r"\{n\}", r"(\d+)")
    m = re.match(regex, filename)
    return int(m.group(1)) if m else None


def discover_common_indices() -> list[int]:
    """
    Find DEM indices that have BOTH an IDL base .sav and an XRTpy base .npz.
    This is deliberately auto-discovering rather than hardcoded 0..161,
    since IDL runs may skip cases (nosolve, etc.).
    """
    idl_indices = set()
    for f in IDL_BASE_DIR.glob(IDL_BASE_PATTERN.format(n="*")):
        idx = _extract_index(f.name, IDL_BASE_PATTERN)
        if idx is not None:
            idl_indices.add(idx)

    xrtpy_indices = set()
    for f in XRTPY_BASE_DIR.glob(XRTPY_BASE_PATTERN.format(n="*")):
        idx = _extract_index(f.name, XRTPY_BASE_PATTERN)
        if idx is not None:
            xrtpy_indices.add(idx)

    common = sorted(idl_indices & xrtpy_indices)
    idl_only = sorted(idl_indices - xrtpy_indices)
    xrtpy_only = sorted(xrtpy_indices - idl_indices)

    print(f"IDL base files found:    {len(idl_indices)}")
    print(f"XRTpy base files found:  {len(xrtpy_indices)}")
    print(f"Common indices (usable): {len(common)}")
    if idl_only:
        print(f"  IDL-only (missing XRTpy):  {idl_only}")
    if xrtpy_only:
        print(f"  XRTpy-only (missing IDL):  {xrtpy_only}")

    return common


# ============================================================
# Loaders
# ============================================================

def load_idl_base(idx: int) -> dict:
    path = IDL_BASE_DIR / IDL_BASE_PATTERN.format(n=idx)
    d = readsav(str(path), python_dict=True)
    return {
        "logT": np.asarray(d["logt_out"], dtype=float).ravel(),
        "dem": np.asarray(d["dem_out"], dtype=float).ravel(),
        "chisq": float(np.asarray(d["chisq"]).ravel()[0]),
    }


def load_xrtpy_base(idx: int) -> dict:
    path = XRTPY_BASE_DIR / XRTPY_BASE_PATTERN.format(n=idx)
    d = np.load(path)
    return {
        "logT": np.asarray(d["logT"], dtype=float).ravel(),
        "dem": np.asarray(d["dem"], dtype=float).ravel(),
        "chisq": float(np.asarray(d["chisq"]).ravel()[0]),
    }


def load_true_dem(idx: int) -> dict:
    path = TRUE_DEM_DIR / TRUE_DEM_PATTERN.format(n=idx)
    arr = np.loadtxt(path)
    return {
        "logT": arr[:, 0].astype(float),
        "dem": arr[:, 1].astype(float),
    }


def _log10_dem(dem: np.ndarray, floor: float = DEM_FLOOR) -> np.ndarray:
    return np.log10(np.maximum(dem, floor))


# ============================================================
# Step 1: Base DEM comparison (build table + per-DEM plots)
# ============================================================


def build_comparison_table(indices: list[int]) -> pd.DataFrame:
    """Build the paired IDL/XRTpy base chi-square comparison table."""
    rows = []

    for dem_id in indices:
        try:
            idl = load_idl_base(dem_id)
            xrtpy = load_xrtpy_base(dem_id)

            # Load the truth DEM to ensure the corresponding synthetic case
            # is present before including it in the comparison.
            load_true_dem(dem_id)

        except FileNotFoundError as error:
            print(f"  [skip] DEM {dem_id}: {error}")
            continue

        if not np.allclose(
            idl["logT"],
            xrtpy["logT"],
            atol=1e-8,
        ):
            print(
                f"  [warn] DEM {dem_id}: "
                "IDL and XRTpy logT grids differ."
            )

        rows.append(
            {
                "dem_id": dem_id,
                "chisq_idl": idl["chisq"],
                "chisq_xrtpy": xrtpy["chisq"],
                "delta_chisq": xrtpy["chisq"] - idl["chisq"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("dem_id")
        .reset_index(drop=True)
    )



def plot_base_overlay(
    dem_id: int,
    outdir: Path = OVERLAY_DIR,
) -> None:
    """Plot synthetic truth, IDL base DEM, and XRTpy base DEM."""
    outdir.mkdir(parents=True, exist_ok=True)

    truth = load_true_dem(dem_id)
    idl = load_idl_base(dem_id)
    xrtpy = load_xrtpy_base(dem_id)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.step(
        truth["logT"],
        _log10_dem(truth["dem"]),
        where="mid",
        color="red",
        linewidth=2.2,
        label="Synthetic DEM (truth)",
    )

    ax.step(
        idl["logT"],
        _log10_dem(idl["dem"]),
        where="mid",
        color="orange",
        linewidth=2.0,
        label="IDL base",
    )

    ax.step(
        xrtpy["logT"],
        _log10_dem(xrtpy["dem"]),
        where="mid",
        color="blue",
        linestyle="--",
        linewidth=2.0,
        label="XRTpy base",
    )

    ax.set_xlabel(r"$\log_{10} T$  [K]")
    ax.set_ylabel(r"$\log_{10}$ DEM  [cm$^{-5}$ K$^{-1}$]")

    ax.set_title(
        f"DEM {dem_id}  |  "
        f"IDL $\\chi^2$={idl['chisq']:.3g}   "
        f"XRTpy $\\chi^2$={xrtpy['chisq']:.3g}"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_png = outdir / f"DEM_{dem_id}_base_overlay.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    """Run the full base DEM comparison workflow."""
    print("Discovering usable DEM indices...")
    indices = discover_common_indices()

    if not indices:
        raise RuntimeError(
            "No common IDL and XRTpy base DEM results were found. "
            "Check IDL_BASE_DIR and XRTPY_BASE_DIR."
        )

    print("\nBuilding chi-square comparison table...")
    comparison = build_comparison_table(indices)

    if comparison.empty:
        raise RuntimeError(
            "No valid base DEM comparisons could be built from the "
            "discovered result files."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV} ({len(comparison)} rows)")

    print("\nGenerating per-DEM overlay plots...")

    for dem_id in comparison["dem_id"]:
        plot_base_overlay(int(dem_id))

    print(
        f"Saved {len(comparison)} overlay plots to: "
        f"{OVERLAY_DIR}"
    )

    n_xrtpy_lower = (
        comparison["delta_chisq"] < 0
    ).sum()

    print("\nSummary:")
    print(f"  N = {len(comparison)}")
    print(
        "  IDL   median chisq = "
        f"{comparison['chisq_idl'].median():.4g}"
    )
    print(
        "  XRTpy median chisq = "
        f"{comparison['chisq_xrtpy'].median():.4g}"
    )
    print(
        "  Cases where XRTpy chisq < IDL chisq: "
        f"{n_xrtpy_lower} / {len(comparison)}"
    )


if __name__ == "__main__":
    main()