"""Offline checks for the view classifiers, the metrics and the TFRecord decoder.

Everything runs on CPU with random inputs in a few seconds, so it needs neither
the Astronet download nor a GPU.
"""

import struct

import numpy as np
import pytest
import torch

from cazaplanetas import astronet_data, models, training


@pytest.mark.parametrize("name", ["astronet", "pocket"])
def test_input_and_output_shapes(name):
    model = models.MODELS[name]().eval()
    with torch.no_grad():
        out = model(torch.randn(3, 2001), torch.randn(3, 201))
        out_channels = model(torch.randn(3, 1, 2001), torch.randn(3, 1, 201))
    assert out.shape == (3,)
    assert out_channels.shape == (3,)
    assert torch.isfinite(out).all()


def test_parameter_budgets():
    astronet = models.count_parameters(models.AstronetCNN())
    pocket = models.count_parameters(models.PocketAstronet())
    # Astronet "local_global": ~9.9 M, almost all in the first dense layer.
    assert 9_500_000 < astronet < 10_500_000
    # Plan, milestone 2: 20k-80k parameters and at most 1/50 of Astronet.
    assert 20_000 <= pocket <= 80_000
    assert pocket * 50 <= astronet


@pytest.fixture
def one_cpu_thread():
    """Depthwise convolutions on CPU are far slower with many threads (see docs/hito2.md)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("name,lr", [("astronet", 1e-4), ("pocket", 1e-3)])
def test_trains_a_few_iterations(name, lr, one_cpu_thread):
    """A short run on an easy synthetic task: no errors and the loss goes down."""
    rng = np.random.default_rng(0)
    n = 64
    label = rng.integers(0, 2, n)
    g = rng.normal(0, 0.1, (n, 2001)).astype(np.float32)
    loc = rng.normal(0, 0.1, (n, 201)).astype(np.float32)
    g[label == 1, 990:1010] -= 1.0  # planets get a dip in the middle
    loc[label == 1, 90:110] -= 1.0
    split = astronet_data.Split(g, loc, label, np.arange(n), np.ones(n, dtype=np.int64), np.array(["PC"] * n))
    data = training.Tensors.from_split(split, torch.device("cpu"))

    training.seed_everything(0)
    model = models.MODELS[name]()
    cfg = training.TrainConfig(epochs=4, batch_size=16, lr=lr, reverse_prob=0.5)
    result = training.train(model, data, data, cfg, seed=0, log_every=0)

    losses = [h["train_loss"] for h in result["history"]]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert training.predict(model, data).shape == (n,)


def test_metrics_on_known_cases():
    y = np.array([0, 0, 1, 1])
    assert training.roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert training.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert training.average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    m = training.classification_metrics(y, np.array([0.9, 0.1, 0.7, 0.2]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["accuracy"] == pytest.approx(0.5)


def _pb_field(number: int, wire: int, payload: bytes) -> bytes:
    def varint(x: int) -> bytes:
        out = b""
        while True:
            b, x = x & 0x7F, x >> 7
            out += bytes([b | (0x80 if x else 0)])
            if not x:
                return out

    key = varint(number << 3 | wire)
    return key + (varint(len(payload)) + payload if wire == 2 else payload)


def test_parse_example_and_tfrecord(tmp_path):
    """Hand-encode a tf.train.Example like Astronet's and decode it back."""
    floats = np.array([1.5, -2.0, 0.25], dtype="<f4")

    def entry(key: str, feature: bytes) -> bytes:
        return _pb_field(1, 2, _pb_field(1, 2, key.encode()) + _pb_field(2, 2, feature))

    float_feature = _pb_field(2, 2, _pb_field(1, 2, floats.tobytes()))  # packed FloatList
    int_feature = _pb_field(3, 2, _pb_field(1, 2, bytes([0xD9, 0xB2, 0xBA, 0x05])))  # packed varint 11442521
    bytes_feature = _pb_field(1, 2, _pb_field(1, 2, b"PC"))
    record = _pb_field(1, 2, entry("global_view", float_feature) + entry("kepid", int_feature) + entry("av_training_set", bytes_feature))

    path = tmp_path / "test-00000-of-00001"
    with open(path, "wb") as f:
        for _ in range(2):
            f.write(struct.pack("<Q", len(record)) + b"\0" * 4 + record + b"\0" * 4)

    records = list(astronet_data.read_tfrecord(path))
    assert len(records) == 2
    ex = astronet_data.parse_example(records[0])
    np.testing.assert_array_equal(ex["global_view"], floats)
    assert int(ex["kepid"][0]) == 11442521
    assert ex["av_training_set"] == [b"PC"]


# --- Milestone 5, step 2: configurable pocket model and recipe options -----------------


def _vet_batch(model, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {k: torch.randn(n, 2001 if k.endswith("global_view") else 201, generator=g) for k in model.INPUTS if k != "scalars"}
    out["scalars"] = torch.randn(n, model.scalar_branch[0].in_features, generator=g)
    return out


def test_pocket_vet_with_default_inputs_is_pocket_exo():
    torch.manual_seed(0)
    exo = models.PocketExo(32).eval()
    torch.manual_seed(0)
    vet = models.PocketVet(32).eval()
    assert models.count_parameters(vet) == models.count_parameters(exo) == 38_441
    assert set(vet.INPUTS) == set(models.PocketExo.INPUTS)
    batch = _vet_batch(vet)
    with torch.no_grad():
        torch.testing.assert_close(vet(**batch), exo(**batch))


def test_pocket_vet_extra_channels_and_budget():
    vet = models.PocketVet(43, global_inputs=(*models.PocketVet.GLOBAL_INPUTS, "half_global_view"),
                           local_inputs=(*models.PocketVet.LOCAL_INPUTS, "sibling_view")).eval()
    assert "sibling_view" in vet.INPUTS and "half_global_view" in vet.INPUTS
    assert models.count_parameters(vet) <= 50_000  # milestone 5: pocket cap
    batch = _vet_batch(vet)
    with torch.no_grad():
        assert vet(**batch).shape == (8,)
    with pytest.raises(TypeError):
        vet(**{k: v for k, v in batch.items() if k != "sibling_view"})
    with pytest.raises(ValueError):
        models.PocketVet(3, local_inputs=("global_view",))


def test_shift_rolls_each_example_and_view_group():
    batch = {"local_view": torch.arange(5.0).repeat(2, 1), "global_view": torch.arange(7.0).repeat(2, 1), "scalars": torch.ones(2, 3)}
    out = training._shift(batch, {"local": torch.tensor([1, -2]), "global": torch.tensor([0, 3])})
    assert out["local_view"][0].tolist() == [4, 0, 1, 2, 3]
    assert out["local_view"][1].tolist() == [2, 3, 4, 0, 1]
    assert out["global_view"][1].tolist() == [4, 5, 6, 0, 1, 2, 3]
    assert out["scalars"] is batch["scalars"]


def test_loss_options():
    logit, label = torch.tensor([2.0, -1.0, 0.5]), torch.tensor([1.0, 0.0, 1.0])
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logit, label)
    cfg = training.TrainConfig()
    assert training._plain(cfg)
    torch.testing.assert_close(training._loss(logit, label, None, cfg), plain)
    assert training._loss(logit, label, None, training.TrainConfig(pos_weight=3.0)) > plain
    assert training._loss(logit, label, None, training.TrainConfig(focal_gamma=2.0)) < plain
    soft = torch.tensor([1.0, 0.0, 1.0])
    torch.testing.assert_close(training._loss(logit, label, soft, training.TrainConfig(distill_alpha=1.0)), plain)


@pytest.mark.parametrize("options", [dict(pos_weight=3.0, shift_local=3, shift_global=10), dict(focal_gamma=2.0, noise=0.05),
                                     dict(ema=0.9, label_smoothing=0.05), dict(distill_alpha=0.5)])
def test_recipe_options_train(options, one_cpu_thread):
    """A short run with each group of options on the easy synthetic task: no errors, the
    loss goes down and the validation selection still works."""
    rng = np.random.default_rng(0)
    n = 64
    label = rng.integers(0, 2, n)
    model = models.PocketVet(4, local_inputs=(*models.PocketVet.LOCAL_INPUTS, "sibling_view"))
    inputs = {k: rng.normal(0, 0.1, (n, 2001 if k.endswith("global_view") else 201)).astype(np.float32) for k in model.INPUTS if k != "scalars"}
    inputs["global_view"][label == 1, 990:1010] -= 1.0
    inputs["local_view"][label == 1, 90:110] -= 1.0
    inputs["scalars"] = rng.normal(0, 1, (n, 4)).astype(np.float32)
    data = training.Tensors.from_arrays(inputs, label, torch.device("cpu"))
    data.soft = torch.from_numpy(np.clip(label + rng.normal(0, 0.1, n), 0, 1).astype(np.float32))
    cfg = training.TrainConfig(epochs=4, batch_size=16, lr=1e-3, select="best_val", select_metric="auc_pr", **options)
    training.seed_everything(0)
    result = training.train(model, data, data, cfg, seed=0, log_every=0)
    losses = [h["train_loss"] for h in result["history"]]
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]
    assert 1 <= result["selected_epoch"] <= 4
    assert data.subset(np.arange(5)).soft.shape == (5,)


def test_distillation_needs_the_teacher():
    model = models.PocketAstronet()
    data = training.Tensors.from_arrays({"global_view": np.zeros((4, 2001)), "local_view": np.zeros((4, 201))}, np.array([0, 1, 0, 1]),
                                        torch.device("cpu"))
    with pytest.raises(ValueError):
        training.train(model, data, None, training.TrainConfig(epochs=1, distill_alpha=0.5), seed=0, log_every=0)


def test_pocket_branches_one_branch_per_test():
    model = models.PocketBranches(43).eval()
    assert set(model.INPUTS) == {v for vs in models.PocketBranches.BRANCHES.values() for v in vs} | {"scalars"}
    assert models.count_parameters(model) <= 50_000
    batch = _vet_batch(model)
    with torch.no_grad():
        assert model(**batch).shape == (8,)
    with pytest.raises(ValueError):
        models.PocketBranches(3, branches={"mixed": ("global_view", "local_view")})
