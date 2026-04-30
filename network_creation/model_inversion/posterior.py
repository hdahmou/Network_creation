"""Notebook-friendly helpers for using a saved posterior artifact."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from network_creation.io import read_json


@dataclass(frozen=True)
class PosteriorArtifact:
    artifact_dir: Path
    posterior: Any
    metadata: dict[str, Any]

    @property
    def observation_order(self) -> list[str]:
        return list(self.metadata["dataset"]["x_names"])

    @property
    def theta_order(self) -> list[str]:
        return list(self.metadata["dataset"]["theta_names"])

    @property
    def tensorboard_log_dir(self) -> Path:
        path = Path(self.metadata["sbi"]["tensorboard_log_dir"])
        if path.is_absolute():
            return path
        return self.artifact_dir / path

    def sample(
        self,
        x: NDArray[np.floating[Any]] | list[float],
        *,
        num_samples: int,
        seed: int | None = None,
        device: str = "cpu",
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Draw posterior samples for one artificial observation."""

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        dataset_meta = self.metadata["dataset"]
        x_event_shape = tuple(dataset_meta["x_event_shape"])
        x_array = np.asarray(x, dtype=np.float32).reshape(x_event_shape)
        x_tensor = torch.as_tensor(
            x_array,
            dtype=torch.float32,
            device=torch.device(device),
        )

        samples = self.posterior.sample((num_samples,), x=x_tensor)
        samples_np = np.asarray(samples.detach().cpu().numpy(), dtype=np.float32)
        return samples_np, list(dataset_meta["theta_names"])


def load_posterior_artifact(artifact_dir: str | Path) -> PosteriorArtifact:
    artifact_dir = Path(artifact_dir)
    metadata = read_json(artifact_dir / "metadata.json")
    posterior_file = metadata["sbi"].get("posterior_file", "posterior.pkl")
    with (artifact_dir / posterior_file).open("rb") as handle:
        posterior = pickle.load(handle)
    return PosteriorArtifact(
        artifact_dir=artifact_dir,
        posterior=posterior,
        metadata=metadata,
    )
