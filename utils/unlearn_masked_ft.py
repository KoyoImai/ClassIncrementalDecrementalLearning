"""Masked unlearning finetune utilities (pruning-like).

This module implements the "score -> mask -> finetune only masked weights" pipeline
that was prototyped in notebooks.

Pipeline
--------
1) Estimate Fisher-diagonal-like importance on a loader:
     I(θ) ≈ E[(∂L/∂θ)^2]
2) Unlearning score:
     S(θ) = I_forget(θ) - λ * I_retain(θ)
3) Build element-wise mask by selecting top-ratio of positive scores.
4) Finetune only masked entries (others are hard-fixed to their initial values).

Notes
-----
* "Fisher" here is a common practical surrogate (grad^2 average), not exact Fisher.
* The finetune loss can be customized; default forget loss minimizes within-class
  cosine similarity of features (collapse-avoid / spread-out) for forget samples.
* This module is framework-agnostic: the caller provides feature extractors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class MaskedFTConfig:
    # scoring
    max_batches_forget: int = 50
    max_batches_retain: int = 50
    lambda_retain: float = 1.0
    only_ndim_ge2: bool = True
    prune_ratio: float = 0.01

    # data subsampling (per class)
    n_per_class_score: Optional[int] = None
    n_per_class_ft: Optional[int] = None
    sample_seed: int = 0

    # finetune
    ft_steps: int = 350
    ft_lr: float = 1e-3
    ft_weight_decay: float = 0.0
    ft_momentum: float = 0.9
    ft_batch_size: int = 128

    # loss weights
    w_forget: float = 1.0
    w_retain_ce: float = 1.0
    w_retain_kd: float = 1.0
    kd_temp: float = 2.0

    # cosine-loss settings
    clamp_positive: bool = True


def subset_per_class(dataset: Dataset, classes: List[int], n_per_class: Optional[int], seed: int) -> Dataset:
    """Return a Subset with up to n_per_class samples per class.

    Assumes dataset has attribute `labels` (DummyDataset) or returns labels as 3rd item.
    """
    if n_per_class is None or n_per_class <= 0:
        return dataset

    # Try DummyDataset.labels first (this repo's datasets use it).
    labels = getattr(dataset, "labels", None)
    if labels is None:
        # Fallback: scan once (slow but safe for small N)
        ys = []
        for i in range(len(dataset)):
            item = dataset[i]
            # expected: (idx, img, label) or (idx, img1, img2, ..., label)
            y = item[-1]
            if torch.is_tensor(y):
                y = int(y.item())
            ys.append(y)
        labels = np.asarray(ys)
    else:
        labels = np.asarray(labels)

    rng = np.random.RandomState(seed)
    idx_all: List[int] = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) > n_per_class:
            idx = rng.choice(idx, n_per_class, replace=False)
        idx_all.extend(idx.tolist())

    if len(idx_all) == 0:
        return Subset(dataset, [])
    return Subset(dataset, idx_all)


def fisher_diag_on_loader(
    model: nn.Module,
    module: nn.Module,
    loader: DataLoader,
    get_logits_fn,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, torch.Tensor]:
    """Estimate grad^2 average for parameters inside `module`.

    Returns dict[name] -> CPU tensor with same shape as parameter.
    """
    model.eval()

    fisher: Dict[str, torch.Tensor] = {}
    for n, p in module.named_parameters():
        if p.requires_grad:
            fisher[n] = torch.zeros_like(p, device="cpu")

    nb = 0
    for batch in loader:
        # batch: (idx, x, y) or (idx, x1, x2, ..., y) when aug>1
        x = batch[1]
        y = batch[-1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        out = model(x)
        logits = get_logits_fn(out)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        for n, p in module.named_parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue
            fisher[n] += (p.grad.detach().cpu() ** 2)

        nb += 1
        if max_batches is not None and nb >= max_batches:
            break

    for n in fisher:
        fisher[n] /= max(nb, 1)
    return fisher


def compute_unlearn_score(
    fisher_forget: Dict[str, torch.Tensor],
    fisher_retain: Dict[str, torch.Tensor],
    lambda_retain: float = 1.0,
) -> Dict[str, torch.Tensor]:
    score: Dict[str, torch.Tensor] = {}
    for n in fisher_forget:
        if n in fisher_retain:
            score[n] = fisher_forget[n] - lambda_retain * fisher_retain[n]
        else:
            score[n] = fisher_forget[n].clone()
    return score


def build_mask_by_score(
    module: nn.Module,
    score: Dict[str, torch.Tensor],
    prune_ratio: float = 0.01,
    only_ndim_ge2: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Build element-wise mask (CPU bool tensor) for top-ratio positive scores."""
    all_scores = []
    for n, p in module.named_parameters():
        if n not in score:
            continue
        if only_ndim_ge2 and p.ndim < 2:
            continue
        s_pos = torch.clamp(score[n], min=0.0)
        all_scores.append(s_pos.flatten())

    if len(all_scores) == 0:
        raise RuntimeError("No parameters selected for masking (check module / only_ndim_ge2).")

    all_scores_cat = torch.cat(all_scores)
    k = int(prune_ratio * all_scores_cat.numel())
    if k <= 0:
        raise RuntimeError(f"prune_ratio too small: {prune_ratio} (k=0)")

    topk = torch.topk(all_scores_cat, k=k, largest=True)
    thr = float(topk.values.min().item())

    mask_dict: Dict[str, torch.Tensor] = {}
    total_elems = 0
    masked_elems = 0
    for n, p in module.named_parameters():
        if n not in score:
            continue
        if only_ndim_ge2 and p.ndim < 2:
            continue
        s_pos = torch.clamp(score[n], min=0.0)
        m = (s_pos >= thr)
        mask_dict[n] = m.cpu()
        total_elems += m.numel()
        masked_elems += int(m.sum().item())

    stats = {
        "threshold": thr,
        "total_elems": float(total_elems),
        "masked_elems": float(masked_elems),
        "masked_ratio_actual": float(masked_elems / max(total_elems, 1)),
    }
    return mask_dict, stats


def cosine_min_loss_within_class(
    feats: torch.Tensor,
    labels: torch.Tensor,
    clamp_positive: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Minimize within-class cosine similarity (off-diagonal mean).

    feats: (B, D) (raw features)
    labels: (B,)
    """
    if feats.numel() == 0:
        return feats.new_tensor(0.0)
    f = F.normalize(feats, p=2, dim=1, eps=eps)
    y = labels
    loss_sum = feats.new_tensor(0.0)
    cnt = 0
    for c in torch.unique(y):
        idx = (y == c).nonzero(as_tuple=False).squeeze(1)
        n = idx.numel()
        if n < 2:
            continue
        fc = f[idx]  # (n, D)
        sim = fc @ fc.t()  # (n, n)
        # remove diagonal
        sim = sim - torch.diag_embed(torch.diagonal(sim))
        if clamp_positive:
            sim = torch.clamp(sim, min=0.0)
        denom = float(n * (n - 1))
        loss_sum = loss_sum + sim.sum() / denom
        cnt += 1
    if cnt == 0:
        return feats.new_tensor(0.0)
    return loss_sum / cnt


def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """KL( teacher || student ) with temperature scaling (batchmean)."""
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)


def cycle(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Infinite iterator over (x, y) from a DataLoader."""
    while True:
        for batch in loader:
            x = batch[1]
            y = batch[-1]
            yield x, y


@torch.no_grad()
def _snapshot_params(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot current parameters (device tensors)."""
    snap: Dict[str, torch.Tensor] = {}
    for n, p in module.named_parameters():
        if p.requires_grad:
            snap[n] = p.data.clone()
    return snap


def masked_finetune(
    *,
    model: nn.Module,
    module: nn.Module,
    mask_dict_cpu: Dict[str, torch.Tensor],
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    teacher_model: Optional[nn.Module],
    get_logits_fn,
    get_target_features_fn,
    device: torch.device,
    cfg: MaskedFTConfig,
) -> Dict[str, float]:
    """Finetune only masked entries of `module`.

    - model: full network used for retain CE/KD
    - module: target backbone to update
    - mask_dict_cpu: CPU bool masks per parameter name
    - get_target_features_fn(model, module, x) -> feats (B, D)
    """
    model.eval()  # keep deterministic; BN/dropout frozen unless caller changes
    if teacher_model is not None:
        teacher_model.eval()

    # optimizer only sees module params
    optim_params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        optim_params,
        lr=cfg.ft_lr,
        momentum=cfg.ft_momentum,
        weight_decay=cfg.ft_weight_decay,
    )

    # snapshot initial values to hard-fix outside mask
    p_init = _snapshot_params(module)

    it_f = cycle(forget_loader)
    it_r = cycle(retain_loader)

    loss_f_avg = 0.0
    loss_rce_avg = 0.0
    loss_rkd_avg = 0.0

    for step in range(cfg.ft_steps):
        x_f, y_f = next(it_f)
        x_r, y_r = next(it_r)
        x_f = x_f.to(device, non_blocking=True)
        y_f = y_f.to(device, non_blocking=True)
        x_r = x_r.to(device, non_blocking=True)
        y_r = y_r.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # forget loss: within-class cosine similarity minimization on target backbone features
        feats_f = get_target_features_fn(model, module, x_f)
        loss_forget = cosine_min_loss_within_class(feats_f, y_f, clamp_positive=cfg.clamp_positive)

        # retain loss: CE + KD (optional)
        out_r = model(x_r)
        logits_r = get_logits_fn(out_r)
        loss_rce = F.cross_entropy(logits_r, y_r)
        loss_rkd = logits_r.new_tensor(0.0)
        if teacher_model is not None and cfg.w_retain_kd > 0:
            with torch.no_grad():
                logits_t = get_logits_fn(teacher_model(x_r))
            loss_rkd = kd_kl_loss(logits_r, logits_t, cfg.kd_temp)

        loss = cfg.w_forget * loss_forget + cfg.w_retain_ce * loss_rce + cfg.w_retain_kd * loss_rkd
        loss.backward()
        optimizer.step()

        # hard-fix outside mask (restore to initial values)
        with torch.no_grad():
            for n, p in module.named_parameters():
                if (not p.requires_grad) or (n not in mask_dict_cpu) or (n not in p_init):
                    continue
                m = mask_dict_cpu[n].to(p.device)
                # restore where mask is False
                p.data[~m] = p_init[n][~m]

        loss_f_avg += float(loss_forget.item())
        loss_rce_avg += float(loss_rce.item())
        loss_rkd_avg += float(loss_rkd.item())

    denom = max(cfg.ft_steps, 1)
    return {
        "loss_forget": loss_f_avg / denom,
        "loss_retain_ce": loss_rce_avg / denom,
        "loss_retain_kd": loss_rkd_avg / denom,
    }
