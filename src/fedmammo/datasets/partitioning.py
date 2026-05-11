"""Federated data partitioning.

Given an array of training labels and a number of clients, produce a list of
index lists (one per client). Three schemes are supported:

- ``iid``           : shuffle-and-chunk; clients are statistically identical.
- ``dirichlet``     : draw class proportions per client from Dir(alpha).
                      Lower alpha => stronger label skew.
- ``quantity_skew`` : IID label distribution but unequal sample counts per
                      client (log-normal scaling with sigma).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)

Scheme = Literal["iid", "dirichlet", "quantity_skew"]


def partition_indices(
    labels: np.ndarray,
    num_clients: int,
    *,
    scheme: Scheme = "iid",
    alpha: float = 0.5,
    min_per_client: int = 16,
    max_retries: int = 20,
    quantity_skew_sigma: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """Partition ``len(labels)`` indices among ``num_clients`` clients.

    Returns:
        A list of length ``num_clients``; each entry is a list of integer
        indices into ``labels``.

    Raises:
        ValueError: on invalid arguments.
        RuntimeError: if Dirichlet partitioning cannot satisfy
            ``min_per_client`` within ``max_retries`` attempts.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    n = labels.shape[0]
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if num_clients > n:
        raise ValueError(f"num_clients ({num_clients}) exceeds dataset size ({n})")

    rng = np.random.default_rng(seed)

    if scheme == "iid":
        return _iid_partition(n, num_clients, rng)
    if scheme == "dirichlet":
        return _dirichlet_partition(
            labels, num_clients, alpha, min_per_client, max_retries, rng
        )
    if scheme == "quantity_skew":
        return _quantity_skew_partition(n, num_clients, quantity_skew_sigma, rng)
    raise ValueError(f"Unknown partitioning scheme: {scheme!r}")


def _iid_partition(n: int, num_clients: int, rng: np.random.Generator) -> list[list[int]]:
    idx = np.arange(n)
    rng.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    return [p.tolist() for p in parts]


def _dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_per_client: int,
    max_retries: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Standard non-IID partition: per class, sample Dir(alpha) proportions."""
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")
    classes = np.unique(labels)
    last_min = -1
    for attempt in range(max_retries):
        client_idx: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            c_indices = np.where(labels == c)[0]
            rng.shuffle(c_indices)
            proportions = rng.dirichlet([alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
            parts = np.split(c_indices, split_points)
            for ci, p in enumerate(parts):
                client_idx[ci].extend(p.tolist())
        sizes = [len(c) for c in client_idx]
        last_min = min(sizes)
        if last_min >= min_per_client:
            for c in client_idx:
                rng.shuffle(c)
            _logger.info(
                "Dirichlet partition (alpha=%.2f) succeeded on attempt %d; "
                "client sizes=%s",
                alpha,
                attempt + 1,
                sizes,
            )
            return client_idx
        _logger.debug(
            "Dirichlet retry %d/%d: min client size %d < %d",
            attempt + 1,
            max_retries,
            last_min,
            min_per_client,
        )
    raise RuntimeError(
        f"Dirichlet partitioning failed to meet min_per_client={min_per_client} "
        f"within {max_retries} retries (last min size={last_min}). "
        "Try a larger alpha, smaller min_per_client, or fewer clients."
    )


def _quantity_skew_partition(
    n: int, num_clients: int, sigma: float, rng: np.random.Generator
) -> list[list[int]]:
    """IID partition with log-normally distributed client sizes."""
    if sigma < 0:
        raise ValueError(f"quantity_skew_sigma must be >= 0, got {sigma}")
    raw = rng.lognormal(mean=0.0, sigma=sigma, size=num_clients)
    proportions = raw / raw.sum()
    sizes = (proportions * n).astype(int)
    # Make sure totals match n exactly.
    diff = n - int(sizes.sum())
    if diff != 0:
        sizes[0] += diff
    idx = np.arange(n)
    rng.shuffle(idx)
    client_idx: list[list[int]] = []
    cursor = 0
    for s in sizes:
        client_idx.append(idx[cursor : cursor + s].tolist())
        cursor += s
    return client_idx


__all__ = ["partition_indices", "Scheme"]
