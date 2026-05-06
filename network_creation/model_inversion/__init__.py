"""Model inversion utilities built around amortized `sbi` NPE/MNPE."""

from __future__ import annotations

from .data import (
    ObservationRepresentation,
    filter_inversion_dataset,
    inversion_dataset_metadata,
    move_discrete_theta_to_end,
    resolve_discrete_theta_parameters,
)
from .posterior import (
    PosteriorArtifact,
    PosteriorWithPriorSamples,
    attach_prior_samples_to_posterior,
    category_probabilities,
    decode_discrete_theta_samples,
    discrete_category_values,
    load_posterior_artifact,
    select_prior_samples,
)

__all__ = [
    "ObservationRepresentation",
    "PosteriorArtifact",
    "PosteriorWithPriorSamples",
    "filter_inversion_dataset",
    "inversion_dataset_metadata",
    "attach_prior_samples_to_posterior",
    "category_probabilities",
    "decode_discrete_theta_samples",
    "discrete_category_values",
    "load_posterior_artifact",
    "move_discrete_theta_to_end",
    "resolve_discrete_theta_parameters",
    "select_prior_samples",
]
