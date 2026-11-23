"""Training, evaluation and timing for the view-based classifiers.

The whole dataset (~140 MB as float32) is moved to the device once and batches
are sliced there, so the GPU (Apple MPS) is never waiting on a data loader.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from cazaplanetas.astronet_data import Split


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds MPS and CUDA


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# --- Metrics -----------------------------------------------------------------


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties count one half)."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, like scikit-learn)."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    s, yy = np.asarray(score)[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(~yy)
    last = np.r_[np.diff(s) != 0, True]  # one point per distinct threshold
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def recall_at_precision(y: np.ndarray, score: np.ndarray, target: float = 0.99) -> float:
    """Highest recall reachable with precision >= target (ExoMiner's R@P0.99)."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    s, yy = np.asarray(score)[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(~yy)
    last = np.r_[np.diff(s) != 0, True]
    tp, fp = tp[last], fp[last]
    ok = tp / (tp + fp) >= target
    return float(tp[ok].max() / y.sum()) if ok.any() else 0.0


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """Decision threshold that maximises F1 on (validation) data.

    Returned as the midpoint between the chosen score and the next lower one, so
    that ``prob > threshold`` reproduces the chosen operating point."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(prob, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    s, yy = p[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(~yy)
    last = np.r_[np.diff(s) != 0, True]
    idx = np.flatnonzero(last)
    f1 = 2 * tp[idx] / (tp[idx] + fp[idx] + y.sum())
    j = idx[int(np.argmax(f1))]
    lower = s[j + 1] if j + 1 < len(s) else s[j] - 1e-6
    return float(0.5 * (s[j] + lower))


def classification_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(prob) > threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {
        "auc": roc_auc(y, prob),
        "auc_pr": average_precision(y, prob),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / len(y),
        "recall_at_p099": recall_at_precision(y, prob, 0.99),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(y),
    }


# --- Training ----------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 0.0
    optimizer: str = "adam"  # "adam" or "adamw"
    schedule: str = "constant"  # "constant" or "cosine"
    reverse_prob: float = 0.5  # random time reversal of both views (Astronet's augmentation)
    select: str = "last"  # "last" epoch, or "best_val" (early stopping on the validation part)
    select_metric: str = "auc"  # what "best_val" maximises: "auc" or "auc_pr"
    # Milestone 5, step 2. The defaults leave the loss, the augmentations and the random
    # draws exactly as before, so earlier runs reproduce.
    pos_weight: float = 1.0  # weight of the positive class in the cross-entropy
    focal_gamma: float = 0.0  # focal loss (Lin et al. 2017) when > 0
    label_smoothing: float = 0.0  # targets become eps/2 and 1 - eps/2
    distill_alpha: float = 0.0  # weight of the teacher's score in the target (needs Tensors.soft)
    shift_local: int = 0  # random shift of the local views, up to +-n bins (all local views together)
    shift_global: int = 0  # the same for the global views
    noise: float = 0.0  # Gaussian noise added to the flux views (in units of the transit depth)
    ema: float = 0.0  # exponential moving average of the weights (decay per step) when > 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Tensors:
    """Model inputs on one device: ``inputs`` maps each forward() argument name to a
    tensor whose first axis is the example (views are (n, bins) or (n, channels, bins);
    ``scalars`` is (n, k)). Names ending in ``_view`` are flipped by time reversal."""

    inputs: dict[str, torch.Tensor]
    label: torch.Tensor
    soft: torch.Tensor | None = None  # a teacher's scores, for distillation

    @classmethod
    def from_split(cls, split: Split, device: torch.device) -> "Tensors":
        return cls.from_arrays({"global_view": split.global_view, "local_view": split.local_view}, split.label, device)

    @classmethod
    def from_arrays(cls, inputs: dict[str, np.ndarray], label: np.ndarray, device: torch.device) -> "Tensors":
        return cls(
            {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(device) for k, v in inputs.items()},
            torch.from_numpy(np.asarray(label).astype(np.float32)).to(device),
        )

    @property
    def global_view(self) -> torch.Tensor:
        return self.inputs["global_view"]

    @property
    def local_view(self) -> torch.Tensor:
        return self.inputs["local_view"]

    def batch(self, idx) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.inputs.items()}

    def subset(self, idx: np.ndarray) -> "Tensors":
        i = torch.as_tensor(np.asarray(idx), device=self.label.device)
        return Tensors({k: v[i] for k, v in self.inputs.items()}, self.label[i], None if self.soft is None else self.soft[i])

    def to(self, device: torch.device) -> "Tensors":
        return Tensors({k: v.to(device) for k, v in self.inputs.items()}, self.label.to(device),
                       None if self.soft is None else self.soft.to(device))

    def __len__(self) -> int:
        return len(self.label)


def _reverse(batch: dict[str, torch.Tensor], flip: torch.Tensor) -> dict[str, torch.Tensor]:
    """Time reversal of every view for the examples where ``flip`` (n,) is True."""
    out = {}
    for k, v in batch.items():
        if k.endswith("_view"):
            f = flip.view(-1, *([1] * (v.dim() - 1)))
            out[k] = torch.where(f, v.flip(-1), v)
        else:
            out[k] = v
    return out


FLUX_VIEWS = ("global_view", "half_global_view", "local_view", "odd_view", "even_view", "secondary_view", "sibling_view")


def _shift(batch: dict[str, torch.Tensor], shifts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Roll every view by a per-example number of bins: ``shifts`` maps "global" and
    "local" to an (n,) integer tensor; views ending in ``global_view`` use the first."""
    out = {}
    for k, v in batch.items():
        group = "global" if k.endswith("global_view") else "local"
        if not k.endswith("_view") or group not in shifts:
            out[k] = v
            continue
        length = v.shape[-1]
        idx = (torch.arange(length, device=v.device)[None, :] - shifts[group][:, None]) % length
        out[k] = torch.gather(v, -1, idx)
    return out


def _loss(logit: torch.Tensor, label: torch.Tensor, soft: torch.Tensor | None, cfg: "TrainConfig") -> torch.Tensor:
    target = label
    if cfg.label_smoothing:
        target = target * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
    if cfg.distill_alpha and soft is not None:
        target = (1 - cfg.distill_alpha) * target + cfg.distill_alpha * soft
    pos_weight = torch.tensor(cfg.pos_weight, device=logit.device) if cfg.pos_weight != 1.0 else None
    loss = nn.functional.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight, reduction="none")
    if cfg.focal_gamma:
        p = torch.sigmoid(logit)
        p_true = p * label + (1 - p) * (1 - label)
        loss = loss * (1 - p_true).pow(cfg.focal_gamma)
    return loss.mean()


def _plain(cfg: "TrainConfig") -> bool:
    return not (cfg.pos_weight != 1.0 or cfg.focal_gamma or cfg.label_smoothing or cfg.distill_alpha)


@torch.no_grad()
def predict(model: nn.Module, data: Tensors, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(data), batch_size):
        out.append(torch.sigmoid(model(**data.batch(slice(i, i + batch_size)))))
    return torch.cat(out).float().cpu().numpy()


def train(
    model: nn.Module,
    train_data: Tensors,
    val_data: Tensors | None,
    cfg: TrainConfig,
    seed: int,
    log_every: int = 10,
) -> dict:
    """Train in place. Returns timing and the validation history."""
    device = train_data.label.device
    model.to(device)
    opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = (len(train_data) + cfg.batch_size - 1) // cfg.batch_size
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
        if cfg.schedule == "cosine"
        else None
    )
    loss_fn = nn.BCEWithLogitsLoss()
    plain = _plain(cfg)
    if cfg.distill_alpha and train_data.soft is None:
        raise ValueError("distillation needs the teacher's scores in train_data.soft")
    gen = torch.Generator().manual_seed(seed)  # batch order and augmentation, on CPU for portability
    val_metric = average_precision if cfg.select_metric == "auc_pr" else roc_auc
    ema = copy.deepcopy(model) if cfg.ema else None
    flux_views = [k for k in FLUX_VIEWS if k in train_data.inputs]

    history, best_score, best_state, best_epoch = [], -1.0, None, cfg.epochs
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        n = len(train_data)
        perm = torch.randperm(n, generator=gen).to(device)
        flips = (torch.rand(n, generator=gen) < cfg.reverse_prob).to(device)
        shifts = {}
        if cfg.shift_global:
            shifts["global"] = torch.randint(-cfg.shift_global, cfg.shift_global + 1, (n,), generator=gen).to(device)
        if cfg.shift_local:
            shifts["local"] = torch.randint(-cfg.shift_local, cfg.shift_local + 1, (n,), generator=gen).to(device)
        total = torch.zeros((), device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            batch = _reverse(train_data.batch(idx), flips[i : i + cfg.batch_size])
            if shifts:
                batch = _shift(batch, {k: v[i : i + cfg.batch_size] for k, v in shifts.items()})
            if cfg.noise:
                for k in flux_views:
                    batch[k] = batch[k] + cfg.noise * torch.randn_like(batch[k])
            logit = model(**batch)
            if plain:
                loss = loss_fn(logit, train_data.label[idx])
            else:
                loss = _loss(logit, train_data.label[idx], None if train_data.soft is None else train_data.soft[idx], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                with torch.no_grad():
                    e_params, m_params = list(ema.parameters()), list(model.parameters())
                    torch._foreach_lerp_(e_params, m_params, 1.0 - cfg.ema)
                    for eb, mb in zip(ema.buffers(), model.buffers()):
                        eb.copy_(mb)
            total += loss.detach() * len(idx)
        entry = {"epoch": epoch, "train_loss": float(total) / len(train_data)}
        scored = ema if ema is not None else model
        if val_data is not None:
            y_val, p_val = val_data.label.cpu().numpy(), predict(scored, val_data)
            entry["val_auc"] = roc_auc(y_val, p_val)
            entry["val_auc_pr"] = average_precision(y_val, p_val)
            score = val_metric(y_val, p_val)
            if cfg.select == "best_val" and score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(scored.state_dict())
        history.append(entry)
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"    epoch {epoch:3d}  loss {entry['train_loss']:.4f}  val AUC {entry.get('val_auc', float('nan')):.4f}", flush=True)
    synchronize(device)
    seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    elif ema is not None:
        model.load_state_dict(ema.state_dict())
    return {"train_seconds": seconds, "selected_epoch": best_epoch, "history": history}


# --- Inference timing --------------------------------------------------------


@torch.no_grad()
def time_inference(model: nn.Module, data: Tensors, device: torch.device, batch_size: int, n: int | None = None, repeats: int = 3) -> float:
    """Seconds per light curve: best of ``repeats`` passes over ``n`` examples,
    after a warm-up pass. ``data`` may live on any device; it is moved first."""
    model = model.to(device).eval()
    n = min(n or len(data), len(data))
    inputs = {k: v[:n].to(device) for k, v in data.inputs.items()}

    def run():
        for i in range(0, n, batch_size):
            model(**{k: v[i : i + batch_size] for k, v in inputs.items()})
        synchronize(device)

    run()
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - t)
    return best / n
