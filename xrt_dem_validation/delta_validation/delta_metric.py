"""
DEM goodness-of-fit parameter (Delta) from Reeves & Weber (draft, Eq. 3).

    Delta = (1 / n_temp) * sqrt( sum_i (DEM_true_i - DEM_calc_i)^2 / sigma_i^2 )

The sum runs only over the "good" temperature bins (band_mask == True),
and n_temp is the number of those bins.

Note on scale: n_temp sits OUTSIDE the square root, so if every bin is off
by exactly 1 sigma, Delta = 1 / sqrt(n_temp) (not 1). This is implemented
exactly as written in the paper.

All values are in linear DEM space (cm^-5 K^-1). DEM and sigma must share
units, so Delta is dimensionless.
"""

import numpy as np


def compute_delta(dem_true, dem_calc, sigma, band_mask):
    """
    Compute the DEM goodness-of-fit parameter Delta (Eq. 3).

    Parameters
    ----------
    dem_true : array-like, shape (n_T,)
        Known model ("true") DEM on the solver temperature grid.
    dem_calc : array-like, shape (n_T,)
        DEM returned by the solver (XRTpy or IDL) on the same grid.
    sigma : array-like, shape (n_T,)
        Per-bin DEM uncertainty (from the Monte Carlo spread).
    band_mask : array-like of bool, shape (n_T,)
        True for the "good" temperature bins to include in the sum.

    Returns
    -------
    delta : float
        The Delta value. NaN if no bins are selected.
    n_temp : int
        Number of temperature bins used.

    Raises
    ------
    ValueError
        If shapes differ, inputs in the band are not finite, or any
        sigma in the band is <= 0.
    """
    dem_true = np.asarray(dem_true, dtype=float)
    dem_calc = np.asarray(dem_calc, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    band_mask = np.asarray(band_mask, dtype=bool)

    if not (dem_true.shape == dem_calc.shape == sigma.shape == band_mask.shape):
        raise ValueError(
            "Shape mismatch: "
            f"dem_true={dem_true.shape}, dem_calc={dem_calc.shape}, "
            f"sigma={sigma.shape}, band_mask={band_mask.shape}"
        )

    n_temp = int(band_mask.sum())
    if n_temp == 0:
        return np.nan, 0

    t = dem_true[band_mask]
    c = dem_calc[band_mask]
    s = sigma[band_mask]

    if not (
        np.all(np.isfinite(t)) and np.all(np.isfinite(c)) and np.all(np.isfinite(s))
    ):
        raise ValueError("Non-finite values found inside the band.")

    if np.any(s <= 0):
        bad = np.flatnonzero(band_mask)[s <= 0]
        raise ValueError(f"sigma <= 0 in band bins {bad.tolist()}; cannot divide.")

    chi2_dem = np.sum(((t - c) / s) ** 2)
    delta = np.sqrt(chi2_dem) / n_temp
    return float(delta), n_temp


def sigma_from_mc(mc_dems):
    """
    Per-bin DEM uncertainty from the Monte Carlo solutions.

        sigma_i = median( |DEM_mc,i - median(DEM_mc,i)| )

    This is the plain median absolute deviation (no 1.4826 factor), as the
    paper describes it.

    Pass only the Monte Carlo rows (rows 1..N of the saved arrays). Row 0
    is the unperturbed solve and is not part of the spread.

    A Monte Carlo run with any non-finite value (inf or NaN) is a failed
    solve. The whole run is dropped before the median is taken, and the
    number of runs actually used is returned so the drop is visible.

    Parameters
    ----------
    mc_dems : array-like, shape (n_runs, n_T)
        Monte Carlo DEM solutions, linear DEM units.

    Returns
    -------
    sigma : ndarray, shape (n_T,)
        Median absolute deviation in each temperature bin.
    n_used : int
        Number of Monte Carlo runs used (finite runs only).

    Raises
    ------
    ValueError
        If the input is not 2-D or no run is fully finite.
    """
    mc_dems = np.asarray(mc_dems, dtype=float)
    if mc_dems.ndim != 2:
        raise ValueError(f"mc_dems must be 2-D (n_runs, n_T); got {mc_dems.shape}")

    finite_runs = np.all(np.isfinite(mc_dems), axis=1)
    n_used = int(finite_runs.sum())
    if n_used == 0:
        raise ValueError("No Monte Carlo run is fully finite; cannot compute sigma.")

    good = mc_dems[finite_runs]
    median = np.median(good, axis=0)
    sigma = np.median(np.abs(good - median), axis=0)
    return sigma, n_used
