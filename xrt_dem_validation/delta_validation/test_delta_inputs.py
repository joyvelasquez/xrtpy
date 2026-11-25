"""
Tests for the file readers in delta_inputs.py.

Most tests write small fake files into a temporary folder (pytest's
``tmp_path``), so they do not depend on the real campaign data. scipy cannot
write IDL .sav files, so the IDL loaders are tested two ways: their helpers
(_decode, _runs_by_T) with fake values, and the loaders themselves on the
real files, skipped if those files are not on this machine.
"""

import numpy as np
import pytest
from delta_inputs import (
    IDL_BASE_DIR,
    IDL_BASE_PATTERN,
    XRTPY_BASE_DIR,
    XRTPY_BASE_PATTERN,
    XRTPY_MC_PATTERN,
    _decode,
    _runs_by_T,
    build_forward_matrix,
    forward_check_dex,
    load_idl_base,
    load_idl_mc,
    load_true_dem,
    load_xrtpy_base,
    load_xrtpy_mc,
)

# ---------------------------------------------------------------------------
# Truth and XRTpy loaders, with fake files
# ---------------------------------------------------------------------------


def test_load_true_dem_two_columns(tmp_path):
    (tmp_path / "DEM_5.txt").write_text("5.5 1.0e22\n5.6 2.0e20\n5.7 0.0\n")

    out = load_true_dem(5, folder=tmp_path)

    np.testing.assert_allclose(out["logT"], [5.5, 5.6, 5.7])
    np.testing.assert_allclose(out["dem"], [1e22, 2e20, 0.0])


def test_load_xrtpy_base_keys_and_types(tmp_path):
    np.savez(
        tmp_path / XRTPY_BASE_PATTERN.format(n=7),
        logT=np.array([5.5, 5.6]),
        dem=np.array([1e21, 2e21]),
        chisq=np.float64(0.5),
        modeled_intensities=np.array([3.0, 4.0]),
        filters=np.array(["Al-mesh", "Be-thin"]),
    )

    out = load_xrtpy_base(7, folder=tmp_path)

    assert sorted(out) == ["chisq", "dem", "filters", "logT", "modeled_intensities"]
    assert out["chisq"] == 0.5
    assert isinstance(out["chisq"], float)
    assert out["filters"] == ["Al-mesh", "Be-thin"]
    assert all(type(f) is str for f in out["filters"])


def test_load_xrtpy_mc_drops_row0(tmp_path):
    mc_dem = np.arange(4 * 3, dtype=float).reshape(4, 3)  # row 0 + 3 MC rows
    np.savez(
        tmp_path / XRTPY_MC_PATTERN.format(n=7),
        logT=np.array([5.5, 5.6, 5.7]),
        mc_dem=mc_dem,
    )

    out = load_xrtpy_mc(7, folder=tmp_path)

    np.testing.assert_array_equal(out["mc_dems"], mc_dem[1:])


# ---------------------------------------------------------------------------
# IDL helpers, with fake values
# ---------------------------------------------------------------------------


def test_decode_bytes_and_padding():
    assert _decode([b"Al-mesh ", "Be-thin", b"Al-poly/Ti-poly"]) == [
        "Al-mesh",
        "Be-thin",
        "Al-poly/Ti-poly",
    ]


def test_runs_by_T_keeps_runs_by_bins():
    dem = np.zeros((1001, 26))
    assert _runs_by_T(dem, 26).shape == (1001, 26)


def test_runs_by_T_flips_bins_by_runs():
    dem = np.arange(26 * 3, dtype=float).reshape(26, 3)  # saved the other way
    out = _runs_by_T(dem, 26)

    assert out.shape == (3, 26)
    np.testing.assert_array_equal(out, dem.T)


def test_runs_by_T_single_dem_becomes_one_row():
    assert _runs_by_T(np.ones(26), 26).shape == (1, 26)


def test_runs_by_T_wrong_size_raises():
    with pytest.raises(ValueError, match="Cannot orient"):
        _runs_by_T(np.zeros((10, 12)), 26)


# ---------------------------------------------------------------------------
# Real data (skips if not on this machine)
# ---------------------------------------------------------------------------

REAL_ID = 159
real_idl = IDL_BASE_DIR / IDL_BASE_PATTERN.format(n=REAL_ID)


@pytest.mark.skipif(not real_idl.exists(), reason="IDL files not on this machine")
def test_real_idl_matches_xrtpy_layout():
    idl = load_idl_base(REAL_ID)
    xrtpy = load_xrtpy_base(REAL_ID)
    mc = load_idl_mc(REAL_ID)

    assert sorted(idl) == sorted(xrtpy)  # same keys for both solvers
    assert idl["filters"] == xrtpy["filters"]  # same filters, same order
    assert idl["dem"].shape == (26,)
    assert mc["mc_dems"].shape == (1000, 26)  # row 0 dropped
    assert idl["dem"].dtype == np.float64  # 32-bit / big-endian converted
    np.testing.assert_allclose(idl["logT"], xrtpy["logT"], atol=1e-5)


# ---------------------------------------------------------------------------
# Forward matrix K
# ---------------------------------------------------------------------------


def test_forward_check_zero_when_model_matches():
    K = np.array([[1.0, 2.0], [0.5, 0.5]])
    dem = np.array([2.0, 1.0])

    assert forward_check_dex(K, dem, K @ dem) == pytest.approx(0.0)
    assert forward_check_dex(K, dem, 10 * (K @ dem)) == pytest.approx(1.0)


def test_forward_check_nothing_to_check_is_nan():
    K = np.ones((2, 2))
    assert np.isnan(forward_check_dex(K, np.ones(2), np.zeros(2)))


real_xrtpy_base = XRTPY_BASE_DIR / XRTPY_BASE_PATTERN.format(n=REAL_ID)


@pytest.mark.skipif(
    not real_xrtpy_base.exists(), reason="XRTpy base files not on this machine"
)
def test_real_forward_matrix_reproduces_xrtpy_model():
    # K @ DEM must give back the modeled intensities the solver saved.
    # If this fails, the 2% band would be built from the wrong responses.
    base = load_xrtpy_base(REAL_ID)
    K = build_forward_matrix(base["filters"], base["logT"])

    assert K.shape == (14, 26)
    assert forward_check_dex(K, base["dem"], base["modeled_intensities"]) < 1e-6
