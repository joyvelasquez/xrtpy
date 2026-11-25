"""
File readers for the Delta validation.

Everything that opens a file lives here, so ``delta_metric.py`` stays pure
math. Every loader returns plain 64-bit numpy arrays in a dict, with the
same key names for XRTpy and IDL.

Data folders (relative to ``xrt_dem_validation/data/``):

    synthetic_dems_data/DEM_{n}.txt
        True DEM. Two columns, no header: logT, linear DEM.
    xrtpy_base_dem_results/xrtpy_dem_idx{n}_allfilters_base.npz
        XRTpy base solve (base runner, monte_carlo_runs=0).
    xrtpy_monte_carlo_dem_results/xrtpy_dem_idx{n}_allfilters_MC1000.npz
        XRTpy MC solve. mc_dem is (1001, 26): row 0 = unperturbed solve,
        rows 1-1000 = Monte Carlo.
    IDL_synthetic_dem_data/idl_synthetic_dem_idx{n}_allfilters_base.sav
        IDL base solve (mc_iter=0). The IDL base always comes from this
        file, never from row 0 of the MC file.
    IDL_monte_carlo_synthetic_dem_data/idl_synthetic_dem_idx{n}_allfilters_MC1000.sav
        IDL MC solve. dem_out is (1001, 26); rows 1-1000 = Monte Carlo.
"""

from pathlib import Path

import numpy as np
from scipy.io import readsav

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRUE_DEM_DIR = DATA_DIR / "synthetic_dems_data"
XRTPY_BASE_DIR = DATA_DIR / "xrtpy_base_dem_results"
XRTPY_MC_DIR = DATA_DIR / "xrtpy_monte_carlo_dem_results"
IDL_BASE_DIR = DATA_DIR / "IDL_synthetic_dem_data"
IDL_MC_DIR = DATA_DIR / "IDL_monte_carlo_synthetic_dem_data"

TRUE_DEM_PATTERN = "DEM_{n}.txt"
XRTPY_BASE_PATTERN = "xrtpy_dem_idx{n}_allfilters_base.npz"
XRTPY_MC_PATTERN = "xrtpy_dem_idx{n}_allfilters_MC1000.npz"
IDL_BASE_PATTERN = "idl_synthetic_dem_idx{n}_allfilters_base.sav"
IDL_MC_PATTERN = "idl_synthetic_dem_idx{n}_allfilters_MC1000.sav"


def load_true_dem(dem_id, folder=TRUE_DEM_DIR):
    """Load the true DEM. Returns {"logT", "dem"}, both shape (n_T,), dem linear."""
    data = np.loadtxt(Path(folder) / TRUE_DEM_PATTERN.format(n=dem_id))
    return {"logT": data[:, 0].astype(float), "dem": data[:, 1].astype(float)}


def load_xrtpy_base(dem_id, folder=XRTPY_BASE_DIR):
    """Load the XRTpy base solve (the base runner's .npz)."""
    path = Path(folder) / XRTPY_BASE_PATTERN.format(n=dem_id)
    with np.load(path) as d:
        return {
            "logT": np.asarray(d["logT"], dtype=float),
            "dem": np.asarray(d["dem"], dtype=float),
            "chisq": float(d["chisq"]),
            "modeled_intensities": np.asarray(d["modeled_intensities"], dtype=float),
            "filters": [str(f) for f in d["filters"]],
        }


def load_xrtpy_mc(dem_id, folder=XRTPY_MC_DIR):
    """Load the XRTpy Monte Carlo rows 1..N (row 0 is dropped here)."""
    path = Path(folder) / XRTPY_MC_PATTERN.format(n=dem_id)
    with np.load(path) as d:
        return {
            "logT": np.asarray(d["logT"], dtype=float),
            "mc_dems": np.asarray(d["mc_dem"], dtype=float)[1:],
        }


def _decode(names):
    """IDL strings come back as bytes; return a list of plain str."""
    return [
        (n.decode() if isinstance(n, bytes) else str(n)).strip()
        for n in np.atleast_1d(names)
    ]


def _runs_by_T(dem, n_T):
    """Return a DEM array shaped (n_runs, n_T), whichever way IDL saved it."""
    dem = np.atleast_2d(np.asarray(dem, dtype=float))
    if dem.shape[1] == n_T:
        return dem
    if dem.shape[0] == n_T:
        return dem.T
    raise ValueError(f"Cannot orient DEM array {dem.shape} with n_T={n_T}")


def load_idl_base(dem_id, folder=IDL_BASE_DIR):
    """Load the IDL base solve (the separate _base.sav, never MC row 0)."""
    d = readsav(str(Path(folder) / IDL_BASE_PATTERN.format(n=dem_id)))
    logT = np.asarray(d["logt_out"], dtype=float)
    return {
        "logT": logT,
        "dem": _runs_by_T(d["dem_out"], logT.size)[0],
        "chisq": float(np.ravel(d["chisq"])[0]),
        "modeled_intensities": np.asarray(d["mod_obs"], dtype=float).ravel(),
        "filters": _decode(d["obs_index"]),
    }


def load_idl_mc(dem_id, folder=IDL_MC_DIR):
    """Load the IDL Monte Carlo rows 1..N (row 0 is dropped here)."""
    d = readsav(str(Path(folder) / IDL_MC_PATTERN.format(n=dem_id)))
    logT = np.asarray(d["logt_out"], dtype=float)
    return {
        "logT": logT,
        "mc_dems": _runs_by_T(d["dem_out"], logT.size)[1:],
    }
