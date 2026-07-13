"""Data preparation and summaries for posterior parameter recovery."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .posterior import PosteriorArtifact, category_probabilities


def observation_from_policy_outputs(
    outputs: dict[str, Any],
    artifact: PosteriorArtifact,
    *,
    n_agents: int,
    n_actors: int,
    input_values: Mapping[str, Any] | None = None,
) -> NDArray[np.float32]:
    values = {"n_agents": n_agents}
    if input_values is not None:
        values.update(input_values)
    return artifact.build_x(
        outputs,
        values,
        n_agents=n_agents,
        n_actors=n_actors,
    )


def parameter_recovery_summary(
    artifact: PosteriorArtifact,
    *,
    true_theta: Any,
    posterior_samples: Any,
) -> list[dict[str, Any]]:
    true_values = np.asarray(true_theta, dtype=np.float32)
    samples = np.asarray(posterior_samples, dtype=np.float32)
    if true_values.shape != (len(artifact.theta_order),):
        raise ValueError("true_theta must have shape (theta,)")
    if samples.ndim != 2 or samples.shape[1] != len(artifact.theta_order):
        raise ValueError("posterior_samples must have shape (sample, theta)")

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(artifact.theta_order):
        if artifact.is_discrete_theta(name):
            categories = artifact.category_values(name)
            probabilities = category_probabilities(samples[:, index], categories)
            true_value = int(round(float(true_values[index])))
            matches = np.flatnonzero(categories == true_value)
            rows.append(
                {
                    "theta": name,
                    "type": "categorical",
                    "true": true_value,
                    "posterior_mode": int(categories[int(np.argmax(probabilities))]),
                    "posterior_p_true": (
                        float(probabilities[int(matches[0])]) if len(matches) else 0.0
                    ),
                }
            )
            continue

        column = samples[:, index]
        q05, median, q95 = np.quantile(column, [0.05, 0.5, 0.95])
        mean = float(np.mean(column))
        rows.append(
            {
                "theta": name,
                "type": "continuous",
                "true": float(true_values[index]),
                "posterior_mean": mean,
                "posterior_median": float(median),
                "q05": float(q05),
                "q95": float(q95),
                "abs_error_mean": abs(mean - float(true_values[index])),
                "covered_by_90_interval": bool(
                    q05 <= true_values[index] <= q95
                ),
            }
        )
    return rows
