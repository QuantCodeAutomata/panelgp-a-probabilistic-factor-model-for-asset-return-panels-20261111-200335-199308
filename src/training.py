"""Training loop for the PanelGP model (paper Appendix C.4).

Adam optimization with two learning rates (variational vs. model
parameters), gradient clipping to unit Euclidean norm, minibatching, and a
learning-rate halving schedule at fixed fractions of training.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src import config as cfg
from src.panelgp_model import PanelGPModel


def split_param_groups(model: PanelGPModel):
    """Splits parameters into the variational group (m_omega, L_omega_raw)
    and the model-parameter group (kernel parameters, inducing inputs,
    variances), each with its own learning rate (paper App. C.4)."""
    variational_names = {"m_omega", "L_omega_raw"}
    variational, model_params = [], []
    for name, p in model.named_parameters():
        (variational if name in variational_names else model_params).append(p)
    return variational, model_params


def train_model(model: PanelGPModel, X: torch.Tensor, date_idx: torch.Tensor,
                 r: torch.Tensor, n_epochs: int, batch_size: int, n_mc: int,
                 lr_variational: float, lr_model: float,
                 decay_fractions: tuple[float, ...], grad_clip_norm: float,
                 seed: int, log_every: int = 200) -> dict:
    """Trains the model by minibatch ELBO maximization and returns a log
    dict with the loss trajectory and wall-clock time."""
    torch.manual_seed(seed)
    n_total = X.shape[0]
    variational, model_params = split_param_groups(model)
    optimizer = torch.optim.Adam([
        {"params": variational, "lr": lr_variational},
        {"params": model_params, "lr": lr_model},
    ])
    milestones = sorted({int(n_epochs * f) for f in decay_fractions} - {0})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    n_batches = max(1, int(np.ceil(n_total / batch_size)))
    losses = []
    start = time.time()
    rng = np.random.default_rng(seed)
    for epoch in range(n_epochs):
        perm = rng.permutation(n_total)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if len(idx) == 0:
                continue
            idx_t = torch.as_tensor(idx, dtype=torch.long)
            optimizer.zero_grad()
            elbo = model.elbo(X[idx_t], date_idx[idx_t], r[idx_t], n_total, n_mc)
            loss = -elbo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            losses.append((epoch + 1, epoch_loss / n_batches))
    return {"losses": losses, "wall_time_sec": time.time() - start,
            "n_epochs": n_epochs, "seed": seed}
