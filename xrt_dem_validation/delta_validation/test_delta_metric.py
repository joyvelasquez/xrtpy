import numpy as np
import pytest
from delta_metric import compute_delta, sigma_from_mc


def test_every_bin_off_by_one_sigma():
    # 4 bins, each off by exactly 1 sigma -> sqrt(4) / 4 = 0.5
    true = np.array([10.0, 20.0, 30.0, 40.0])
    sigma = np.array([1.0, 2.0, 3.0, 4.0])
    calc = true + sigma
    mask = np.ones(4, dtype=bool)

    delta, n = compute_delta(true, calc, sigma, mask)

    assert n == 4
    assert delta == pytest.approx(0.5)


def test_perfect_match_is_zero():
    true = np.array([1e20, 5e21, 3e20])
    sigma = np.array([1e19, 1e20, 1e19])
    mask = np.ones(3, dtype=bool)

    delta, n = compute_delta(true, true, sigma, mask)

    assert n == 3
    assert delta == 0.0


def test_only_band_bins_are_used():
    # Bins outside the band are wildly wrong and have sigma=0;
    # they must be ignored completely.
    true = np.array([5.0, 10.0, 10.0, 5.0])
    calc = np.array([999.0, 12.0, 8.0, -999.0])
    sigma = np.array([0.0, 2.0, 2.0, 0.0])
    mask = np.array([False, True, True, False])

    delta, n = compute_delta(true, calc, sigma, mask)

    # Each band bin is off by 1 sigma -> sqrt(2) / 2
    assert n == 2
    assert delta == pytest.approx(np.sqrt(2) / 2)


def test_empty_band_returns_nan():
    x = np.ones(3)
    delta, n = compute_delta(x, x, x, np.zeros(3, dtype=bool))

    assert n == 0
    assert np.isnan(delta)


def test_zero_sigma_in_band_raises():
    x = np.ones(3)
    sigma = np.array([1.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="sigma <= 0"):
        compute_delta(x, x, sigma, np.ones(3, dtype=bool))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_delta(np.ones(3), np.ones(4), np.ones(3), np.ones(3, dtype=bool))


# ---------------------------------------------------------------------------
# sigma_from_mc
# ---------------------------------------------------------------------------


def test_sigma_is_median_absolute_deviation():
    # Bin 0: values 1, 2, 3, 4, 100 -> median 3 -> |dev| 2, 1, 0, 1, 97 -> MAD 1
    # Bin 1: all equal -> MAD 0
    mc = np.array(
        [[1.0, 7.0], [2.0, 7.0], [3.0, 7.0], [4.0, 7.0], [100.0, 7.0]],
    )

    sigma, n_used = sigma_from_mc(mc)

    assert n_used == 5
    np.testing.assert_allclose(sigma, [1.0, 0.0])


def test_sigma_has_no_1p4826_factor():
    # Normal draws: plain MAD is about 0.674 * std, not about 1.0 * std.
    rng = np.random.default_rng(0)
    mc = rng.normal(loc=50.0, scale=2.0, size=(20000, 1))

    sigma, _ = sigma_from_mc(mc)

    assert sigma[0] == pytest.approx(0.6745 * 2.0, rel=0.03)


def test_sigma_drops_whole_run_with_non_finite_value():
    # Run 2 has inf in bin 0 only, run 3 has NaN in bin 1 only.
    # Both whole runs are dropped -> rows [1, 2, 4] remain in BOTH bins.
    mc = np.array(
        [[1.0, 1.0], [2.0, 2.0], [np.inf, 3.0], [3.0, np.nan], [4.0, 4.0]],
    )

    sigma, n_used = sigma_from_mc(mc)

    assert n_used == 3
    np.testing.assert_allclose(sigma, [1.0, 1.0])


def test_sigma_all_runs_non_finite_raises():
    mc = np.full((3, 2), np.inf)

    with pytest.raises(ValueError, match="No Monte Carlo run"):
        sigma_from_mc(mc)


def test_sigma_needs_2d_input():
    with pytest.raises(ValueError, match="must be 2-D"):
        sigma_from_mc(np.ones(5))
