"""No column that carries the answer may reach a model (milestone 4)."""

import numpy as np
import pytest
import torch

from cazaplanetas import features as F
from cazaplanetas import models
from cazaplanetas import views as V

# Columns of the DR25 TCE table (and of the label tables) that give the answer away.
LEAKS = [
    "av_training_set",
    "av_pred_class",
    "av_pp_pc",
    "av_vf_pc",
    "av_vf_afp_err",
    "tce_nkoi",
    "tce_rogue_flag",
    "tce_steff_prov",
    "tce_sradius_prov",
    "tce_steff_err",
    "tce_slogg_err",
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "fpwg_disp_status",
    "robovetter_score",
    "robovetter_pc",
    "label",
    "label_class",
    "kepid",
    "tce_plnt_num",
    "ra",
    "dec",
    "tce_delivname",
    "tce_datalink_dvr",
    "rowupdate",
]


@pytest.mark.parametrize("column", LEAKS)
def test_leaking_columns_are_forbidden(column):
    assert F.is_forbidden(column)
    with pytest.raises(F.ForbiddenColumnError):
        F.assert_allowed([column])


def test_allow_list_has_no_forbidden_column():
    for column in [*F.SCALAR_COLUMNS, *F.VIEW_INPUTS]:
        assert not F.is_forbidden(column), column
    F.assert_allowed([*F.SCALAR_COLUMNS, *F.VIEW_INPUTS])


def test_columns_outside_the_allow_list_are_rejected():
    with pytest.raises(F.ForbiddenColumnError):
        F.assert_allowed(["tce_period", "tce_incl"])  # harmless but not on the list
    with pytest.raises(F.ForbiddenColumnError):
        F.scalar_matrix({"av_training_set": np.zeros(3)}, columns={"av_training_set": "lin"})


def _poisoned_table(n=300, seed=0):
    """A fake DR25 table: allowed columns are noise, forbidden ones *are* the label."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    table = {c: rng.lognormal(size=n) for c in F.SCALAR_COLUMNS}
    table["boot_fap"] = rng.uniform(0, 1, n)
    for c in LEAKS:
        table[c] = label.astype(float)
    return table, label


def test_forbidden_columns_cannot_change_the_scalar_inputs():
    table, label = _poisoned_table()
    x = F.scalar_matrix(table)
    assert x.shape == (len(label), len(F.SCALAR_COLUMNS))
    flipped = {**table, **{c: 1.0 - table[c] for c in LEAKS}}
    dropped = {c: v for c, v in table.items() if c not in LEAKS}
    np.testing.assert_array_equal(x, F.scalar_matrix(flipped))
    np.testing.assert_array_equal(x, F.scalar_matrix(dropped))
    # and none of the input columns is the label
    z = F.Standardizer().fit(x).transform(x)
    for j in range(z.shape[1]):
        assert abs(np.corrcoef(z[:, j], label)[0, 1]) < 0.5


def test_model_inputs_are_only_allowed_ones():
    """Every model reads only allowed views, plus the scalars built by scalar_matrix."""
    for cls in (models.AstronetCNN, models.PocketAstronet, models.PocketExo):
        F.assert_allowed([n for n in cls.INPUTS if n != "scalars"])
    model = models.PocketExo(len(F.SCALAR_COLUMNS))
    assert model.scalar_branch[0].in_features == len(F.SCALAR_COLUMNS)
    import inspect

    assert tuple(inspect.signature(model.forward).parameters) == models.PocketExo.INPUTS


def test_view_inputs_only_return_allowed_views():
    n = 4
    views = {k: np.random.default_rng(1).normal(size=(n, V.VIEW_LENGTHS[k])).astype(np.float32) for k in V.VIEW_NAMES}
    views["global_view"][:, 1000] = -5
    views["local_view"][:, 100] = -5
    out = F.view_inputs(views)
    assert set(out) == set(F.VIEW_INPUTS)
    with pytest.raises(F.ForbiddenColumnError):
        F.view_inputs({**views, "label": np.zeros((n, 3))}, names=(*F.VIEW_INPUTS, "label"))
    model = models.PocketExo(len(F.SCALAR_COLUMNS))
    batch = {k: torch.from_numpy(v) for k, v in out.items()}
    batch["scalars"] = torch.zeros(n, len(F.SCALAR_COLUMNS))
    assert model(**batch).shape == (n,)


def test_transforms():
    x = np.array([-1.0, 0.0, 1e-3, 1.0, 1e3])
    np.testing.assert_allclose(F._transform(x, "slog")[3:], np.log10([2.0, 1001.0]))
    assert np.isnan(F._transform(x, "log")[:2]).all()
    fap = F._transform(np.array([-1.0, 0.0, 1e-3]), "fap")
    assert np.isnan(fap[0]) and fap[1] == 50 and np.isclose(fap[2], 3)


def test_standardizer_uses_training_rows_and_fills_missing():
    train = np.array([[1.0, np.nan], [2.0, 10.0], [3.0, 20.0], [4.0, 30.0]])
    s = F.Standardizer().fit(train)
    z = s.transform(np.array([[2.5, np.nan], [1e9, 20.0]]))
    assert z[0, 0] == 0 and z[0, 1] == 0  # median, and missing -> 0
    assert z[1, 0] == s.clip  # clipped
    assert z.dtype == np.float32
