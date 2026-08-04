"""Tie-aware ranking statistics for final-hub prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class SimultaneousCMCBand:
    """Observed CMC and one simultaneous bootstrap confidence band."""

    observed: NDArray[np.float64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]


def tie_aware_hit_at_k(
    order: ArrayLike,
    hub_ids: ArrayLike,
) -> NDArray[np.float64]:
    """Return per-episode Hit@k scores with random credit for bottom ties."""

    order_array = np.asarray(order, dtype=np.int64)
    hub_array = np.asarray(hub_ids, dtype=np.int64)
    if order_array.ndim != 2 or hub_array.shape != (order_array.shape[0],):
        raise ValueError("order and hub_ids must align by episode")

    n_episodes, n_agents = order_array.shape
    hit_at_k = np.zeros((n_episodes, n_agents), dtype=np.float64)

    for episode in range(n_episodes):
        episode_order = order_array[episode]
        observed_count = int(np.count_nonzero(episode_order >= 0))
        hub_positions = np.flatnonzero(
            episode_order[:observed_count] == hub_array[episode]
        )
        if hub_positions.size:
            rank = int(hub_positions[0]) + 1
            hit_at_k[episode, rank - 1 :] = 1.0
            continue

        tied_ranks = np.arange(observed_count + 1, n_agents + 1, dtype=np.float64)
        hit_at_k[episode, observed_count:] = (
            np.arange(1, tied_ranks.size + 1, dtype=np.float64) / tied_ranks.size
        )
    return hit_at_k


def simultaneous_cmc_band(
    scores: ArrayLike,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 4000,
    seed: int = 20260804,
) -> SimultaneousCMCBand:
    """Bootstrap episodes and form a max-deviation band over all CMC curves."""

    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 3:
        raise ValueError("scores must have shape (episode, ordering, rank)")

    observed = score_array.mean(axis=0)
    result = stats.bootstrap(
        (score_array,),
        np.mean,
        axis=0,
        vectorized=True,
        n_resamples=n_resamples,
        batch=min(128, n_resamples),
        method="basic",
        rng=np.random.default_rng(seed),
    )
    deviations = np.abs(result.bootstrap_distribution - observed[..., None])
    maximum_deviations = deviations[:, :-1].max(axis=(0, 1))
    width = float(np.quantile(maximum_deviations, confidence_level))
    low = np.clip(observed - width, 0.0, 1.0)
    high = np.clip(observed + width, 0.0, 1.0)
    low[:, -1] = observed[:, -1]
    high[:, -1] = observed[:, -1]
    return SimultaneousCMCBand(observed=observed, low=low, high=high)
