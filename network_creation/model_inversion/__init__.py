"""Model inversion utilities built around amortized `sbi` NPE."""

from __future__ import annotations

from .data import (
    ObservationRepresentation,
    filter_inversion_dataset,
    inversion_dataset_metadata,
)
from .posterior import (
    PosteriorArtifact,
    load_posterior_artifact,
)

__all__ = [
    "ObservationRepresentation",
    "PosteriorArtifact",
    "filter_inversion_dataset",
    "inversion_dataset_metadata",
    "load_posterior_artifact",
]
