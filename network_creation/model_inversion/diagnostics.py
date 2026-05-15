"""Notebook helpers for model-inversion checks."""

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
    source_sample: NDArray[np.int64]


def load_diagnostic_data(artifact: PosteriorArtifact) -> DiagnosticData:
    with np.load(
        artifact.artifact_dir / artifact.metadata["diagnostics"]["file"]
    ) as data:
        return DiagnosticData(
            theta=np.asarray(data["theta"], dtype=np.float32),
            x=np.asarray(data["x"], dtype=np.float32),
            source_sample=np.asarray(data["source_sample"], dtype=np.int64),
        )


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
    from sbi.diagnostics import run_sbc

    torch_device = artifact.to_device(device)

    ranks, dap_samples = run_sbc(
        torch.as_tensor(
            diagnostic_data.theta,
            dtype=torch.float32,
            device=torch_device,
        ),
        torch.as_tensor(
            diagnostic_data.x,
            dtype=torch.float32,
            device=torch_device,
        ),
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


def plot_expected_coverage(artifact: PosteriorArtifact, coverage: dict[str, Any]):
    from sbi.analysis.plot import sbc_rank_plot

    return sbc_rank_plot(
        coverage["ranks"],
        coverage["num_posterior_samples"],
        parameter_labels=list(artifact.theta_order),
    )
