"""Transit classifiers that read Astronet's global (2001 bins) and local (201 bins) views.

* :class:`AstronetCNN` re-implements in PyTorch the best "local_global"
  configuration of Shallue & Vanderburg (2018), as described in the paper
  (Fig. 7, Sec. 5.2) and in the Apache-2.0 exoplanet-ml configuration files
  (``astro_cnn_model/configurations.py``). Written from that description, not
  copied.
* :class:`PocketAstronet` is this project's small model: one depthwise-separable
  1D CNN branch per view, global mean + max pooling instead of large dense layers,
  and a 32-unit head.

Both take ``(global_view, local_view)`` tensors of shape ``(batch, 2001)`` and
``(batch, 201)`` (a channel axis is optional) and return one logit per example.
"""

from __future__ import annotations

import torch
from torch import nn


def _as_channels(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1) if x.dim() == 2 else x


# --- Astronet ----------------------------------------------------------------


def _astronet_column(num_blocks: int, pool_size: int, block_size: int = 2, filters: int = 16, kernel: int = 5) -> nn.Sequential:
    """Conv blocks with the same number of filters inside a block, doubling between
    blocks, "same" padding, ReLU, and a "valid" max-pool (stride 2) after each block."""
    layers: list[nn.Module] = []
    c_in = 1
    for i in range(num_blocks):
        c_out = filters * 2**i
        for _ in range(block_size):
            layers += [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.ReLU()]
            c_in = c_out
        layers.append(nn.MaxPool1d(pool_size, stride=2))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


class AstronetCNN(nn.Module):
    """Astronet "local_global" CNN: 5 conv blocks (16..256 filters, pool 5) on the
    global view, 2 blocks (16, 32 filters, pool 7) on the local view, then 4 dense
    layers of 512 ReLU units and a sigmoid output. No dropout."""

    INPUTS = ("global_view", "local_view")

    def __init__(self, global_len: int = 2001, local_len: int = 201, hidden: int = 512, num_hidden: int = 4):
        super().__init__()
        self.global_column = _astronet_column(num_blocks=5, pool_size=5)
        self.local_column = _astronet_column(num_blocks=2, pool_size=7)
        with torch.no_grad():
            n = self.global_column(torch.zeros(1, 1, global_len)).shape[1] + self.local_column(torch.zeros(1, 1, local_len)).shape[1]
        dense: list[nn.Module] = []
        for _ in range(num_hidden):
            dense += [nn.Linear(n, hidden), nn.ReLU()]
            n = hidden
        dense.append(nn.Linear(n, 1))
        self.head = nn.Sequential(*dense)
        # TensorFlow/Keras defaults used by the original: Glorot-uniform kernels, zero biases.
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        g = self.global_column(_as_channels(global_view))
        loc = self.local_column(_as_channels(local_view))
        return self.head(torch.cat([g, loc], dim=1)).squeeze(1)


# --- Pocket Astronet ---------------------------------------------------------


class SeparableConv(nn.Sequential):
    """Depthwise conv (one filter per channel) + pointwise 1x1 mix + BatchNorm + ReLU."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 5):
        super().__init__(
            nn.Conv1d(c_in, c_in, kernel, padding=kernel // 2, groups=c_in, bias=False),
            nn.Conv1d(c_in, c_out, 1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
        )


class GlobalMeanMax(nn.Module):
    """Average and max over the length axis, concatenated: (B, C, L) -> (B, 2C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)


def _pocket_branch(channels: tuple[int, ...], stem: int, kernel: int, pool: int, in_channels: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(in_channels, stem, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(stem),
        nn.ReLU(),
    ]
    c_in = stem
    for c in channels:
        layers += [SeparableConv(c_in, c, kernel), SeparableConv(c, c, kernel), nn.MaxPool1d(pool, stride=2, padding=pool // 2)]
        c_in = c
    layers.append(GlobalMeanMax())
    return nn.Sequential(*layers)


class PocketAstronet(nn.Module):
    """Small two-branch separable CNN (~3e4 parameters with the defaults)."""

    INPUTS = ("global_view", "local_view")

    def __init__(
        self,
        global_channels: tuple[int, ...] = (16, 32, 64, 64),
        local_channels: tuple[int, ...] = (16, 32, 64),
        stem: int = 8,
        kernel: int = 5,
        hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.global_branch = _pocket_branch(global_channels, stem, kernel, pool=5)
        self.local_branch = _pocket_branch(local_channels, stem, kernel, pool=3)
        n = 2 * (global_channels[-1] + local_channels[-1])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        g = self.global_branch(_as_channels(global_view))
        loc = self.local_branch(_as_channels(local_view))
        return self.head(torch.cat([g, loc], dim=1)).squeeze(1)


class PocketExo(nn.Module):
    """Pocket model with every input a vetter looks at (milestone 4).

    * Global branch, 2 channels: flux and centroid shift over the whole orbit.
    * Local branch, 5 channels: transit, odd transits, even transits, the secondary
      eclipse (all on the transit's depth scale, so a deeper odd than even transit or a
      visible secondary shows up) and the centroid shift during the transit.
    * Scalar branch: the allowed DR25 diagnostics (see ``features.SCALAR_COLUMNS``).

    Same separable blocks and mean+max pooling as :class:`PocketAstronet`."""

    INPUTS = (
        "global_view",
        "local_view",
        "odd_view",
        "even_view",
        "secondary_view",
        "centroid_global_view",
        "centroid_local_view",
        "scalars",
    )

    def __init__(
        self,
        n_scalars: int,
        global_channels: tuple[int, ...] = (16, 32, 64, 64),
        local_channels: tuple[int, ...] = (16, 32, 64),
        stem: int = 8,
        kernel: int = 5,
        scalar_hidden: int = 32,
        hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.global_branch = _pocket_branch(global_channels, stem, kernel, pool=5, in_channels=2)
        self.local_branch = _pocket_branch(local_channels, stem, kernel, pool=3, in_channels=5)
        self.scalar_branch = nn.Sequential(nn.Linear(n_scalars, scalar_hidden), nn.ReLU())
        n = 2 * (global_channels[-1] + local_channels[-1]) + scalar_hidden
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
        odd_view: torch.Tensor,
        even_view: torch.Tensor,
        secondary_view: torch.Tensor,
        centroid_global_view: torch.Tensor,
        centroid_local_view: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        g = self.global_branch(torch.stack([global_view, centroid_global_view], dim=1))
        loc = self.local_branch(torch.stack([local_view, odd_view, even_view, secondary_view, centroid_local_view], dim=1))
        return self.head(torch.cat([g, loc, self.scalar_branch(scalars)], dim=1)).squeeze(1)


class PocketVet(nn.Module):
    """:class:`PocketExo` with the views of each branch chosen at construction
    (milestone 5, step 2), so that rebuilt views (the sibling signal's, the global view at
    P/2) can be added as channels without touching the milestone 4 model.

    With the default inputs it has the same layers, in the same order, as
    :class:`PocketExo`. ``scalar_hidden``, ``hidden`` and the branch widths set the size;
    ``dropout`` applies before the head."""

    GLOBAL_INPUTS = ("global_view", "centroid_global_view")
    LOCAL_INPUTS = ("local_view", "odd_view", "even_view", "secondary_view", "centroid_local_view")

    def __init__(
        self,
        n_scalars: int,
        global_inputs: tuple[str, ...] = GLOBAL_INPUTS,
        local_inputs: tuple[str, ...] = LOCAL_INPUTS,
        global_channels: tuple[int, ...] = (16, 32, 64, 64),
        local_channels: tuple[int, ...] = (16, 32, 64),
        stem: int = 8,
        kernel: int = 5,
        scalar_hidden: int = 32,
        hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not all(n.endswith("global_view") for n in global_inputs) or any(n.endswith("global_view") for n in local_inputs):
            raise ValueError("global branch views must end in 'global_view', local ones must not")
        self.global_inputs, self.local_inputs = tuple(global_inputs), tuple(local_inputs)
        self.INPUTS = (*self.global_inputs, *self.local_inputs, "scalars")
        self.global_branch = _pocket_branch(global_channels, stem, kernel, pool=5, in_channels=len(global_inputs))
        self.local_branch = _pocket_branch(local_channels, stem, kernel, pool=3, in_channels=len(local_inputs))
        self.scalar_branch = nn.Sequential(nn.Linear(n_scalars, scalar_hidden), nn.ReLU())
        n = 2 * (global_channels[-1] + local_channels[-1]) + scalar_hidden
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        missing = set(self.INPUTS) - set(inputs)
        if missing:
            raise TypeError(f"missing inputs: {sorted(missing)}")
        g = self.global_branch(torch.stack([inputs[k] for k in self.global_inputs], dim=1))
        loc = self.local_branch(torch.stack([inputs[k] for k in self.local_inputs], dim=1))
        return self.head(torch.cat([g, loc, self.scalar_branch(inputs["scalars"])], dim=1)).squeeze(1)


class PocketBranches(nn.Module):
    """One small separable branch per diagnostic test, as ExoMiner does (its Sec. 3.6),
    each ending in a narrow bottleneck, plus the scalar branch (milestone 5, step 2).

    ``branches`` maps a branch name to the views it stacks as channels; views ending in
    ``global_view`` get the global settings (pool 5), the others the local ones (pool 3)."""

    BRANCHES = {
        "flux_global": ("global_view", "half_global_view"),
        "centroid_global": ("centroid_global_view",),
        "flux_local": ("local_view", "odd_view", "even_view"),
        "secondary": ("secondary_view", "sibling_view"),
        "centroid_local": ("centroid_local_view",),
    }

    def __init__(
        self,
        n_scalars: int,
        branches: dict[str, tuple[str, ...]] | None = None,
        global_channels: tuple[int, ...] = (8, 16, 32, 32),
        local_channels: tuple[int, ...] = (8, 16, 32),
        stem: int = 8,
        kernel: int = 5,
        bottleneck: int = 8,
        scalar_hidden: int = 32,
        hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branches_views = dict(branches or self.BRANCHES)
        self.INPUTS = (*(v for views in self.branches_views.values() for v in views), "scalars")
        self.branch = nn.ModuleDict()
        for name, views in self.branches_views.items():
            is_global = views[0].endswith("global_view")
            if any(v.endswith("global_view") != is_global for v in views):
                raise ValueError(f"branch {name} mixes global and local views")
            ch = global_channels if is_global else local_channels
            self.branch[name] = nn.Sequential(
                _pocket_branch(ch, stem, kernel, pool=5 if is_global else 3, in_channels=len(views)),
                nn.Linear(2 * ch[-1], bottleneck), nn.ReLU())
        self.scalar_branch = nn.Sequential(nn.Linear(n_scalars, scalar_hidden), nn.ReLU())
        n = bottleneck * len(self.branches_views) + scalar_hidden
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        missing = set(self.INPUTS) - set(inputs)
        if missing:
            raise TypeError(f"missing inputs: {sorted(missing)}")
        feats = [self.branch[name](torch.stack([inputs[v] for v in views], dim=1)) for name, views in self.branches_views.items()]
        return self.head(torch.cat([*feats, self.scalar_branch(inputs["scalars"])], dim=1)).squeeze(1)


MODELS = {"astronet": AstronetCNN, "pocket": PocketAstronet}


def example_inputs(model: nn.Module, global_len: int = 2001, local_len: int = 201) -> dict[str, torch.Tensor]:
    """One all-zero example with the inputs ``model`` expects (for counting operations)."""
    out = {}
    for name in getattr(model, "INPUTS", ("global_view", "local_view")):
        if name == "scalars":
            out[name] = torch.zeros(1, model.scalar_branch[0].in_features)
        else:
            out[name] = torch.zeros(1, global_len if name.endswith("global_view") else local_len)
    return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, global_len: int = 2001, local_len: int = 201) -> int:
    """Multiply-accumulates for one example (conv and linear layers only)."""
    total = 0

    def hook(m, _inp, out):
        nonlocal total
        if isinstance(m, nn.Conv1d):
            total += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(**example_inputs(model, global_len, local_len))
    model.train(was_training)
    for h in handles:
        h.remove()
    return total
