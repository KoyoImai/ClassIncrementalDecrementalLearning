"""Gradient-conflict measurement utilities.

This module is designed for Continual Learning + Machine Unlearning experiments where
training objectives are composed of multiple loss terms (e.g., new-class CE, distillation
for old classes, unlearning loss for forget classes).

Design goals
------------
1) Method-agnostic: works with iCaRL/DER/FOSTER/etc. You just pass a dict of loss tensors.
2) Variable number of losses: any number of loss terms is supported (>=1).

Typical usage (inside the training step, before loss.backward()):

    from utils.grad_conflict import compute_grad_conflicts, select_named_params

    loss_dict = {
        "new": loss_new,
        "kd": loss_kd,
        "forg": loss_forg,
    }

    named_params = select_named_params(model, include_prefixes=["_network"],
                                       exclude_prefixes=["_network.fc"])  # backbone only
    params = [p for _, p in named_params]
    stats = compute_grad_conflicts(loss_dict, params)

    # stats["pairwise_cos"]["new|kd"], stats["grad_norm"]["forg"], ...

You can also use GradConflictLogger to write JSONL/CSV logs.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch


NamedParams = List[Tuple[str, torch.nn.Parameter]]
LossDict = Dict[str, torch.Tensor]


def _ensure_scalar(loss: torch.Tensor) -> torch.Tensor:
    """Ensure the given loss is a scalar tensor."""
    if not torch.is_tensor(loss):
        raise TypeError(f"loss must be a torch.Tensor, got {type(loss)}")
    if loss.ndim == 0:
        return loss
    # Be conservative: sum over batch/terms so gradients match common training usage.
    return loss.sum()


def select_named_params(
    model: torch.nn.Module,
    include_prefixes: Optional[Sequence[str]] = None,
    exclude_prefixes: Optional[Sequence[str]] = None,
    include_fn: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
    exclude_fn: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
    requires_grad: bool = True,
) -> NamedParams:
    """Select (name, parameter) pairs from a model with flexible filtering."""
    include_prefixes = list(include_prefixes or [])
    exclude_prefixes = list(exclude_prefixes or [])

    def _startswith_any(name: str, prefixes: Sequence[str]) -> bool:
        return any(name.startswith(p) for p in prefixes)

    out: NamedParams = []
    for n, p in model.named_parameters():
        if requires_grad and not p.requires_grad:
            continue

        if include_prefixes and not _startswith_any(n, include_prefixes):
            continue
        if exclude_prefixes and _startswith_any(n, exclude_prefixes):
            continue

        if include_fn is not None and not include_fn(n, p):
            continue
        if exclude_fn is not None and exclude_fn(n, p):
            continue

        out.append((n, p))
    return out


def compute_loss_grads(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    retain_graph: bool = True,
    allow_unused: bool = True,
    create_graph: bool = False,
    cast_dtype: torch.dtype = torch.float32,
    detach: bool = True,
) -> List[Optional[torch.Tensor]]:
    """Compute per-parameter gradients for a loss without touching .grad fields."""
    loss = _ensure_scalar(loss)
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
        create_graph=create_graph,
    )

    out: List[Optional[torch.Tensor]] = []
    for g in grads:
        if g is None:
            out.append(None)
            continue
        if detach:
            g = g.detach()
        # cast for stable dot/norm computations (esp. under AMP)
        if cast_dtype is not None and g.dtype != cast_dtype:
            g = g.to(dtype=cast_dtype)
        out.append(g)
    return out


def _grad_norm_sq(
    grads: Sequence[Optional[torch.Tensor]],
    acc_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute squared L2 norm of a gradient list (treat None as zero)."""
    device = None
    for g in grads:
        if g is not None:
            device = g.device
            break
    if device is None:
        return torch.zeros((), dtype=acc_dtype)

    s = torch.zeros((), device=device, dtype=acc_dtype)
    for g in grads:
        if g is None:
            continue
        s = s + (g * g).sum(dtype=acc_dtype)
    return s


def _grad_dot(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
    acc_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Dot product between two gradient lists (treat None as zero)."""
    assert len(grads_a) == len(grads_b)
    device = None
    for ga, gb in zip(grads_a, grads_b):
        if ga is not None:
            device = ga.device
            break
        if gb is not None:
            device = gb.device
            break
    if device is None:
        return torch.zeros((), dtype=acc_dtype)

    s = torch.zeros((), device=device, dtype=acc_dtype)
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            continue
        if ga.device != gb.device:
            gb = gb.to(device=ga.device)
        s = s + (ga * gb).sum(dtype=acc_dtype)
    return s


def compute_grad_conflicts(
    losses: Union[LossDict, Sequence[Tuple[str, torch.Tensor]]],
    params: Sequence[torch.nn.Parameter],
    retain_graph: bool = True,
    allow_unused: bool = True,
    create_graph: bool = False,
    cast_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, float]]:
    """Compute gradient norms and pairwise gradient cosine similarities."""
    if isinstance(losses, dict):
        items = list(losses.items())
    else:
        items = list(losses)

    if len(items) == 0:
        raise ValueError("losses must contain at least one loss")
    if len(params) == 0:
        raise ValueError("params must contain at least one parameter")

    # Stable ordering
    items = sorted(items, key=lambda x: x[0])

    grads_by_name: Dict[str, List[Optional[torch.Tensor]]] = {}
    loss_value: Dict[str, float] = {}

    for name, loss in items:
        loss = _ensure_scalar(loss)
        loss_value[name] = float(loss.detach().item())
        grads_by_name[name] = compute_loss_grads(
            loss,
            params,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
            create_graph=create_graph,
            cast_dtype=cast_dtype,
            detach=True,
        )

    grad_norm: Dict[str, float] = {}
    norm_sq: Dict[str, torch.Tensor] = {}
    for name, grads in grads_by_name.items():
        nsq = _grad_norm_sq(grads, acc_dtype=acc_dtype)
        norm_sq[name] = nsq
        grad_norm[name] = float(
            torch.sqrt(nsq + torch.tensor(eps, device=nsq.device, dtype=acc_dtype)).item()
        )

    names = sorted(grads_by_name.keys())
    pairwise_dot: Dict[str, float] = {}
    pairwise_cos: Dict[str, float] = {}
    eps_t = None

    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            d = _grad_dot(grads_by_name[a], grads_by_name[b], acc_dtype=acc_dtype)
            pair_key = f"{a}|{b}"
            pairwise_dot[pair_key] = float(d.item())

            nsq_a = norm_sq[a]
            nsq_b = norm_sq[b]
            if eps_t is None or eps_t.device != d.device:
                eps_t = torch.tensor(eps, device=d.device, dtype=acc_dtype)
            denom = torch.sqrt(nsq_a.to(d.device) + eps_t) * torch.sqrt(nsq_b.to(d.device) + eps_t)
            c = d / denom
            pairwise_cos[pair_key] = float(c.item())

    return {
        "loss_value": loss_value,
        "grad_norm": grad_norm,
        "pairwise_dot": pairwise_dot,
        "pairwise_cos": pairwise_cos,
    }


def flatten_conflict_stats(stats: Dict[str, Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    """Flatten compute_grad_conflicts() output into a single-level dict."""
    out: Dict[str, float] = {}
    for group, kv in stats.items():
        for k, v in kv.items():
            out[f"{prefix}{group}.{k}"] = v
    return out


class GradConflictLogger:
    """Lightweight logger for gradient-conflict measurements.

    Writes:
      - JSONL always (robust to changing keys)
      - CSV optionally (best when loss set is fixed)
    """

    def __init__(
        self,
        save_dir: str,
        jsonl_name: str = "grad_conflict.jsonl",
        csv_name: Optional[str] = None,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.jsonl_path = os.path.join(save_dir, jsonl_name)
        self.csv_path = os.path.join(save_dir, csv_name) if csv_name else None
        self._csv_header: Optional[List[str]] = None

    def log(
        self,
        losses: LossDict,
        params: Sequence[torch.nn.Parameter],
        meta: Optional[Dict[str, Union[int, float, str]]] = None,
        retain_graph: bool = True,
        allow_unused: bool = True,
        cast_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float64,
        eps: float = 1e-12,
    ) -> Dict[str, Dict[str, float]]:
        """Compute and append one record."""
        stats = compute_grad_conflicts(
            losses,
            params,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
            create_graph=False,
            cast_dtype=cast_dtype,
            acc_dtype=acc_dtype,
            eps=eps,
        )
        record: Dict[str, Union[int, float, str]] = {}
        if meta:
            record.update(meta)
        record.update(flatten_conflict_stats(stats))

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.csv_path is not None:
            self._append_csv(record)

        return stats

    def _append_csv(self, record: Dict[str, Union[int, float, str]]) -> None:
        assert self.csv_path is not None

        keys = sorted(record.keys())
        if self._csv_header is None:
            self._csv_header = keys
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_header)
                writer.writeheader()
                writer.writerow(record)
            return

        # If keys changed, fall back to JSONL only (CSV header cannot change safely).
        if keys != self._csv_header:
            return

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_header)
            writer.writerow(record)
