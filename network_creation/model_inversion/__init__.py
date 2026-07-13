"""Public model-inversion training and posterior artifact APIs."""

from .artifacts import TrainingRunSummary
from .pipeline import train_inversion_from_config
from .posterior import PosteriorArtifact, load_posterior_artifact

__all__ = [
    "PosteriorArtifact",
    "TrainingRunSummary",
    "load_posterior_artifact",
    "train_inversion_from_config",
]
