"""Sequential model-inversion training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from .artifacts import (
    SBI_LOG_DIRNAME,
    TrainingRunSummary,
    prepare_artifact_dir,
    write_inversion_artifact,
)
from .config import InversionConfig
from .dataset import ZarrInversionDataset, build_row_index
from .schema import resolve_inversion_schema
from .source import load_inversion_source
from .splits import select_prior_samples, split_inversion_rows
from .training import (
    build_direct_posterior,
    build_posterior_estimator,
    fit_posterior_estimator,
    make_training_loaders,
)


LOGGER = logging.getLogger(__name__)


def train_inversion_from_config(config_path: str | Path) -> TrainingRunSummary:
    source = load_inversion_source(config_path)
    schema = resolve_inversion_schema(source)
    artifact_dir = prepare_artifact_dir(source.config, schema)
    config = source.config

    LOGGER.info(
        "[data] Opened %s with %s successful rows; method=%s x_event_shape=%s",
        source.zarr_path,
        int(source.input_index.shape[0]),
        schema.sbi_method,
        schema.x_event_shape,
    )
    row_index = build_row_index(source, schema)
    dataset = ZarrInversionDataset(
        source=source,
        row_index=row_index,
        schema=schema,
    )
    split = split_inversion_rows(row_index, dataset, config)
    prior_samples = select_prior_samples(dataset, split, config)
    estimator = build_posterior_estimator(dataset, split, schema, config)
    loaders = make_training_loaders(dataset, split, config)
    training = fit_posterior_estimator(
        estimator,
        loaders,
        config,
        tensorboard_dir=artifact_dir / SBI_LOG_DIRNAME,
    )
    posterior = build_direct_posterior(training, schema, config)
    summary = write_inversion_artifact(
        posterior=posterior,
        source=source,
        schema=schema,
        row_index=row_index,
        split=split,
        prior_samples=prior_samples,
        training=training,
        config=config,
        dataset=dataset,
        artifact_dir=artifact_dir,
    )
    LOGGER.info("[complete] Training run complete: artifact_dir=%s", summary.artifact_dir)
    return summary
