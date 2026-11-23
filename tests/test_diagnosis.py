"""Milestone 5 diagnosis tools: ablation masks, ranking arithmetic, spread and causes."""

import numpy as np
import pytest
import torch

from cazaplanetas import diagnosis as D
from cazaplanetas import exominer_tables as ex
from cazaplanetas import features as F
from cazaplanetas import training


def _inputs(n=6, as_torch=False):
    rng = np.random.default_rng(0)
    x = {name: rng.normal(size=(n, 12)).astype(np.float32) + 3.0 for name in F.VIEW_INPUTS}
    x["scalars"] = rng.normal(size=(n, len(F.SCALAR_COLUMNS))).astype(np.float32) + 3.0
    return {k: torch.from_numpy(v) for k, v in x.items()} if as_torch else x


# --- Groups and ablations ------------------------------------------------------------


def test_groups_partition_the_allowed_inputs():
    scalars = [c for cols in F.SCALAR_GROUPS.values() for c in cols]
    assert sorted(scalars) == sorted(F.SCALAR_COLUMNS) and len(scalars) == len(set(scalars))
    views = [v for names in F.VIEW_GROUPS.values() for v in names]
    assert sorted(views) == sorted(F.VIEW_INPUTS) and len(views) == len(set(views))
    F.assert_allowed(scalars + views)


def test_ablations_only_name_known_groups_and_main_ones_cover_every_group():
    for name, a in D.ABLATIONS.items():
        assert set(a.views) <= set(F.VIEW_GROUPS) and set(a.scalars) <= set(F.SCALAR_GROUPS), name
        F.assert_allowed(list(a.view_names()) + list(a.scalar_columns()))
    assert D.MAIN_ABLATIONS[0] == "full" and len(D.MAIN_ABLATIONS) == 11
    single = [D.ABLATIONS[n] for n in D.MAIN_ABLATIONS if n not in ("full", "views_only", "scalars_only")]
    assert {g for a in single for g in a.views} == set(F.VIEW_GROUPS)
    assert {g for a in single for g in a.scalars} == set(F.SCALAR_GROUPS)
    assert set(D.ABLATIONS["views_only"].scalar_columns()) == set(F.SCALAR_COLUMNS)
    assert set(D.ABLATIONS["scalars_only"].view_names()) == set(F.VIEW_INPUTS)


@pytest.mark.parametrize("as_torch", [False, True])
@pytest.mark.parametrize("name", list(D.ABLATIONS))
def test_mask_inputs_zeroes_exactly_the_ablated_inputs(name, as_torch):
    x = _inputs(as_torch=as_torch)
    original = {k: (v.clone() if as_torch else v.copy()) for k, v in x.items()}
    a = D.ABLATIONS[name]
    out = D.mask_inputs(x, a)
    cols = list(F.SCALAR_COLUMNS)
    removed = [cols.index(c) for c in a.scalar_columns()]
    kept = [i for i in range(len(cols)) if i not in removed]
    for k in F.VIEW_INPUTS:
        got, ref = np.asarray(out[k]), np.asarray(original[k])
        assert np.array_equal(got, np.zeros_like(ref) if k in a.view_names() else ref), k
    s, ref = np.asarray(out["scalars"]), np.asarray(original["scalars"])
    assert (s[:, removed] == 0).all() and np.array_equal(s[:, kept], ref[:, kept])
    for k, v in x.items():  # the caller's arrays are untouched
        assert np.array_equal(np.asarray(v), np.asarray(original[k]))


# --- Ranking arithmetic ----------------------------------------------------------------


def _toy():
    """200 planets and 4 negatives ranked among them: 2 near the top, 2 near the bottom."""
    y = np.r_[np.ones(200), np.zeros(4)].astype(int)
    score = np.r_[np.linspace(1.0, 0.5, 200), [0.99, 0.98, 0.52, 0.51]]
    return y, score


def test_zero_fixes_is_the_plain_recall_at_precision():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.1).astype(int)
    score = np.clip(0.55 * y + rng.normal(0.3, 0.2, 3000), 0, 1)
    for precision in (0.99, 0.95, 0.9):
        r = D.best_recall_with_fixes(y, score, 5, precision)
        assert r[0] == pytest.approx(training.recall_at_precision(y, score, precision))
        assert (np.diff(r) >= 0).all()
    assert D.operating_point(y, score, 0.9)["recall"] == pytest.approx(training.recall_at_precision(y, score, 0.9))


def test_fixes_needed_on_a_case_solved_by_hand():
    y, score = _toy()
    # Two negatives sit above almost every planet: at most 1 negative per 99 planets is allowed.
    # No fix: the best threshold takes all 200 planets and all 4 negatives (precision 0.980): not valid;
    # the valid ones have <= 2 negatives and need 198 planets above -> 2 negatives allowed at 198+.
    assert training.recall_at_precision(y, score) == pytest.approx(D.best_recall_with_fixes(y, score, 0)[0])
    r = D.best_recall_with_fixes(y, score, 4)
    assert r[4] == 1.0  # every negative removed
    assert D.fixes_needed(y, score, 1.0) == 2  # 200 planets allow 2 negatives: remove 2 of the 4
    assert D.fixes_needed(y, score, r[0]) == 0
    assert D.fixes_needed(y, score, 1.0, max_fixes=1) == -1


def test_fixing_a_planet_counts_as_one_fix():
    # 99 planets on top, one negative, then one planet far below: recall 0.99 needs nothing,
    # recall 1.0 needs either the negative removed or... one fix in any case.
    y = np.r_[np.ones(99), 0, 1].astype(int)
    score = np.r_[np.linspace(1, 0.9, 99), 0.8, 0.1]
    assert D.best_recall_with_fixes(y, score, 1).tolist() == pytest.approx([1.0, 1.0])  # 100 planets allow 1 negative
    y = np.r_[np.ones(50), 0, 1].astype(int)
    score = np.r_[np.linspace(1, 0.9, 50), 0.8, 0.1]
    r = D.best_recall_with_fixes(y, score, 1)
    assert r[0] == pytest.approx(50 / 51) and r[1] == 1.0


def test_fixes_for_mean_shares_the_fixes_among_folds():
    curves = [np.array([0.5, 0.9, 1.0]), np.array([0.8, 0.85, 0.9]), np.array([0.9, 0.9, 0.9])]
    total, alloc = D.fixes_for_mean(curves, 0.86)  # (0.9 + 0.8 + 0.9) / 3 = 0.867 with one fix in fold 0
    assert total == 1 and alloc == [1, 0, 0]
    total, alloc = D.fixes_for_mean(curves, 0.93)  # (1.0 + 0.9 + 0.9) / 3 = 0.933 needs 2 + 2
    assert total == 4 and alloc == [2, 2, 0]
    assert D.fixes_for_mean(curves, 0.99) == (-1, [])
    assert D.fixes_for_mean(curves, 0.7) == (0, [0, 0, 0])


def test_rounded_precision_is_more_lenient_than_exact():
    # 197 planets above 3 negatives' worth of ranking: precision 197/200 = 0.985 rounds to 0.99 (numpy rounds half to even: 0.98).
    y = np.r_[np.ones(100), 0, np.ones(97), 0, 0, np.ones(3)].astype(int)
    score = np.linspace(1, 0, len(y))
    exact = training.recall_at_precision(y, score, 0.99)
    rounded = D.recall_at_rounded_precision(y, score, 0.99)
    assert rounded >= exact
    assert exact == pytest.approx(197 / 200)  # 197 planets, 1 negative: 0.9949
    y2 = np.r_[np.ones(100), 0, 0, np.ones(100)].astype(int)  # all 200 planets with 2 negatives: 0.990099 exact
    assert training.recall_at_precision(y2, np.linspace(1, 0, 202), 0.99) == 1.0
    y3 = np.r_[np.ones(100), 0, 0, 0, np.ones(100)].astype(int)  # 200 planets, 3 negatives: 0.98522 -> 0.99 rounded, not exact
    s3 = np.linspace(1, 0, 203)
    assert training.recall_at_precision(y3, s3, 0.99) == 0.5
    assert D.recall_at_rounded_precision(y3, s3, 0.99) == 1.0
    assert D.recall_at_rounded_precision(y3, s3, 0.975) == training.recall_at_precision(y3, s3, 0.975)


def test_interpolated_pr_auc_agrees_with_the_exact_area_on_spread_scores():
    rng = np.random.default_rng(2)
    y = (rng.random(20000) < 0.2).astype(int)
    score = np.clip(0.25 * y + rng.normal(0.4, 0.15, 20000), 0, 1)
    assert D.interpolated_pr_auc(y, score) == pytest.approx(training.average_precision(y, score), abs=3e-3)
    assert D.interpolated_pr_auc(y, y.astype(float)) == pytest.approx(1.0, abs=1e-6)


# --- Spread ------------------------------------------------------------------------------


def test_variance_components_recovers_known_spreads():
    rng = np.random.default_rng(3)
    fold = rng.normal(0, 0.05, size=(400, 1))
    table = 0.8 + fold + rng.normal(0, 0.1, size=(400, 10))
    v = D.variance_components(table)
    assert v["seed_sd"] == pytest.approx(0.1, rel=0.05) and v["fold_sd"] == pytest.approx(0.05, rel=0.15)
    assert v["fold_share"] == pytest.approx(0.2, abs=0.05)
    flat = D.variance_components(np.full((5, 4), 0.7))
    assert flat["seed_sd"] == 0 and flat["fold_sd"] == 0 and flat["fold_share"] == 0


# --- Causes --------------------------------------------------------------------------------


def test_negative_cause_takes_the_first_rule_that_applies():
    base = {"label_class": "AFP", "koi_disposition": "FALSE POSITIVE", "rv_flags": "0000", "rv_comment": "", "fpwg_eb": 0, "fpwg_offset": 0,
            "fpwg_perep": 0, "same_period_sibling": False, "mes": 20.0}
    assert D.negative_cause({**base, "koi_disposition": "CANDIDATE", "fpwg_eb": 1}) == "dubious_label"
    assert D.negative_cause({**base, "label_class": "NTP", "rv_comment": "IS_SEC_TCE---EPHEM_MATCH", "rv_flags": "1101"}) == "secondary_tce"
    assert D.negative_cause({**base, "label_class": "NTP", "same_period_sibling": True}) == "secondary_tce"
    assert D.negative_cause({**base, "same_period_sibling": True, "fpwg_eb": 1}) == "eclipsing_binary"  # a KOI: the primary of the pair
    assert D.negative_cause({**base, "fpwg_perep": 1, "fpwg_offset": 1}) == "ephemeris_match"
    assert D.negative_cause({**base, "rv_flags": "0010"}) == "centroid_offset"
    assert D.negative_cause({**base, "rv_flags": "0100"}) == "eclipsing_binary"
    assert D.negative_cause({**base, "label_class": "NTP", "mes": 8.0, "fpwg_eb": None, "fpwg_offset": None, "fpwg_perep": None}) == "low_snr"
    assert D.negative_cause({**base, "label_class": "NTP", "rv_flags": "1000"}) == "other_false_alarm"
    assert set(D.NEGATIVE_CAUSES) >= {D.negative_cause(base)}


def test_planet_cause_takes_the_first_rule_that_applies():
    base = {"tce_period": 10.0, "koi_period": 10.0, "controversial": False, "kepmag": 14.0, "rv_flags": "0000", "rv_comment": "",
            "n_transits": 100, "mes": 30.0, "prad": 2.0, "depth": 400.0}
    assert D.planet_cause(base) == "unclear"
    assert D.planet_cause({**base, "tce_period": 20.0, "kepmag": 9.0}) == "wrong_period"
    assert D.planet_cause({**base, "controversial": True, "mes": 8.0}) == "controversial"
    assert D.planet_cause({**base, "rv_comment": "CENT_SATURATED", "rv_flags": "0100"}) == "saturated"
    assert D.planet_cause({**base, "kepmag": 10.0}) == "saturated"
    assert D.planet_cause({**base, "rv_comment": "MOD_SEC_DV---PLANET_OCCULT_DV"}) == "secondary"
    assert D.planet_cause({**base, "rv_flags": "0010"}) == "centroid"
    assert D.planet_cause({**base, "n_transits": 4, "mes": 9.0}) == "few_transits"
    assert D.planet_cause({**base, "mes": 9.0}) == "low_snr"
    assert D.planet_cause({**base, "rv_flags": "1000"}) == "not_transit_like"
    assert D.planet_cause({**base, "prad": 13.0}) == "giant"
    assert D.planet_cause({**base, "tce_period": 0.7, "koi_period": 0.7}) == "short_period"
    assert D.planet_cause({**base, "koi_period": float("nan")}) == "unclear"
    assert set(D.PLANET_CAUSES) >= {"wrong_period", "saturated", "unclear"}


def test_same_period_siblings():
    kepid = np.array([5, 7, 5, 7, 9, 5])
    period = np.array([10.0, 3.0, 10.05, 6.0, 4.0, 20.0])
    assert D.same_period_siblings(kepid, period).tolist() == [True, False, True, False, False, False]
    assert not D.same_period_siblings(np.array([1, 2]), np.array([5.0, 5.0])).any()  # different stars


# --- ExoMiner's published tables (offline: tiny files with the same layout) -----------------


def test_exominer_tables_are_read_and_aligned(tmp_path):
    t3 = tmp_path / "table_3.csv"
    t3.write_text("target_id,tce_plnt_num,tce_period,mes,original_label,binary_label,astronet,exonet,exominer,exominer_tce\n"
                  "100,1,3.5,20.1,PC,1,0.8,0.9,0.99,0.98\n200,2,7.0,9.0,NTP,0,0.1,0.2,0.01,0.02\n")
    t13 = tmp_path / "table_13.csv"
    t13.write_text("target_id,tce_plnt_num,tce_period,mes,exominer,exominer_prior\n300,1,5.0,12.0,0.7,0.8\n")
    lab, unl = ex.read_labelled(t3), ex.read_unlabelled(t13)
    assert lab["label"].tolist() == [1, 0] and lab["label_class"].tolist() == ["PC", "NTP"] and lab["exominer"].tolist() == [0.99, 0.01]
    kepid, plnt = np.array([200, 300, 100, 100]), np.array([2, 1, 1, 2])
    got = ex.align(kepid, plnt, lab, "exominer")
    assert got[0] == 0.01 and np.isnan(got[1]) and got[2] == 0.99 and np.isnan(got[3])
    assert ex.align(kepid, plnt, lab, "label", missing=-1).tolist() == [0, -1, 1, -1]
    assert ex.align(kepid, plnt, unl, "exominer")[1] == 0.7
    folds = tmp_path / "folds.csv"
    folds.write_text("kepid,tce_plnt_num,fold\n100,1,3\n200,2,9\n")
    assert ex.read_folds(folds) == {(100, 1): 3, (200, 2): 9}
    assert set(ex.TABLES) == {"table_3.csv", "table_13.csv"} and len(ex.REPO_COMMIT) == 40


def test_weaker_signal_of_a_same_period_pair():
    kepid = np.array([5, 5, 5, 7, 7])
    period = np.array([10.0, 10.02, 3.0, 4.0, 4.0])
    mes = np.array([100.0, 20.0, 50.0, 9.0, 9.0])
    assert D.same_period_siblings(kepid, period, strength=mes).tolist() == [False, True, False, False, False]


def test_ap_loss_per_positive_adds_up_to_one_minus_auc_pr():
    rng = np.random.default_rng(3)
    y = rng.random(400) < 0.2
    score = np.round(rng.random(400) * 0.5 + 0.4 * y, 2)  # rounded: many ties
    score[:30] = 0.0
    loss = D.ap_loss_per_positive(y, score)
    assert np.isclose(loss.sum(), 1.0 - training.average_precision(y, score))
    assert (loss[~y] == 0).all() and (loss >= 0).all()
    # A perfect ranking loses nothing; one negative on top costs each positive 1/(k+1)/P.
    y2 = np.array([0, 1, 1, 0, 0], bool)
    assert np.allclose(D.ap_loss_per_positive(y2, np.array([0.1, 0.9, 0.8, 0.2, 0.3])), 0.0)
    got = D.ap_loss_per_positive(y2, np.array([0.95, 0.9, 0.8, 0.2, 0.3]))
    assert np.allclose(got, [0.0, (1 - 1 / 2) / 2, (1 - 2 / 3) / 2, 0.0, 0.0])
