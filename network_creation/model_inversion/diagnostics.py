"""Notebook helpers for model-inversion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .posterior import PosteriorArtifact


@dataclass(frozen=True)
class DiagnosticData:
    theta: NDArray[np.float32]
    x: NDArray[np.float32]
    source_sample_ids: NDArray[np.int64]

def load_diagnostic_data(artifact: PosteriorArtifact) -> DiagnosticData:
    filename = artifact.metadata.sbi.diagnostic_data_file
    with np.load(artifact.artifact_dir / filename) as data:
        diagnostic = DiagnosticData(
            theta=np.asarray(data["theta"], dtype=np.float32),
            x=np.asarray(data["x"], dtype=np.float32),
            source_sample_ids=np.asarray(
                data["source_sample_ids"], dtype=np.int64
            ),
        )
        continuous_names = tuple(
            str(name) for name in data["continuous_theta_names"].tolist()
        )
        discrete_names = tuple(
            str(name) for name in data["discrete_theta_names"].tolist()
        )
        x_components = tuple(str(name) for name in data["x_components"].tolist())
    metadata = artifact.metadata.dataset
    if continuous_names != metadata.continuous_theta_names:
        raise ValueError("diagnostic continuous theta names do not match metadata")
    if discrete_names != metadata.discrete_theta_names:
        raise ValueError("diagnostic discrete theta names do not match metadata")
    if x_components != metadata.x_components:
        raise ValueError("diagnostic x components do not match metadata")
    sample_count = diagnostic.theta.shape[0]
    if diagnostic.theta.shape != (sample_count, len(metadata.theta_order)):
        raise ValueError("diagnostic theta shape does not match metadata")
    if diagnostic.x.shape != (sample_count, *metadata.x_event_shape):
        raise ValueError("diagnostic x shape does not match metadata")
    if diagnostic.source_sample_ids.shape != (sample_count,):
        raise ValueError("diagnostic source sample IDs do not match theta rows")
    return diagnostic


def diagnostic_input_parameter_values(
    artifact: PosteriorArtifact,
    diagnostic_data: DiagnosticData,
) -> dict[str, NDArray[np.float32]]:
    return artifact.input_values_from_x(diagnostic_data.x)


def run_expected_coverage(
    artifact: PosteriorArtifact,
    diagnostic_data: DiagnosticData,
    *,
    num_posterior_samples: int = 1000,
    num_workers: int = 1,
    show_progress_bar: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch
    from sbi.diagnostics import run_sbc  # type: ignore[import-untyped]

    torch_device = artifact.to_device(device)
    ranks, dap_samples = run_sbc(
        torch.as_tensor(diagnostic_data.theta, dtype=torch.float32, device=torch_device),
        torch.as_tensor(diagnostic_data.x, dtype=torch.float32, device=torch_device),
        artifact.posterior,
        num_posterior_samples=num_posterior_samples,
        num_workers=num_workers,
        show_progress_bar=show_progress_bar,
    )
    return {
        "ranks": ranks,
        "dap_samples": dap_samples,
        "num_posterior_samples": num_posterior_samples,
        "n_diagnostic_simulations": int(diagnostic_data.theta.shape[0]),
    }


def marginal_sbc_ranks(
    artifact: PosteriorArtifact,
    posterior_samples: Any,
    true_theta: Any,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both.")

    samples = np.asarray(posterior_samples)
    theta = np.asarray(true_theta)
    if samples.ndim != 3:
        raise ValueError(
            "posterior_samples must have shape "
            "(n_cases, n_posterior_samples, n_theta)"
        )
    if theta.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError(
            "true_theta must have shape (n_cases, n_theta) matching posterior_samples"
        )
    theta_order = list(artifact.theta_order)
    if len(theta_order) != samples.shape[2]:
        raise ValueError(
            "artifact.theta_order length must match the posterior sample theta dimension"
        )
    ranks = (samples < theta[:, None, :]).sum(axis=1).astype(np.int64)
    discrete_indices = [
        index for index, name in enumerate(theta_order) if artifact.is_discrete_theta(name)
    ]
    if not discrete_indices:
        return ranks
    rank_rng = np.random.default_rng(seed) if rng is None else rng
    for index in discrete_indices:
        tie_counts = np.isclose(samples[:, :, index], theta[:, None, index]).sum(axis=1)
        ranks[:, index] += np.asarray(
            [rank_rng.integers(0, int(tie_count) + 1) for tie_count in tie_counts],
            dtype=np.int64,
        )
    return ranks


def plot_expected_coverage(
    artifact: PosteriorArtifact,
    coverage: dict[str, Any],
    **plot_kwargs: Any,
):
    from sbi.analysis.plot import sbc_rank_plot  # type: ignore[import-untyped]

    return sbc_rank_plot(
        coverage["ranks"],
        coverage["num_posterior_samples"],
        parameter_labels=list(artifact.theta_order),
        **plot_kwargs,
    )
