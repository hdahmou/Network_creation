from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from network_creation.model_inversion.diagnostics import (
    DiagnosticData,
    load_diagnostic_data,
    marginal_sbc_ranks,
)
from network_creation.model_inversion.posterior import PosteriorArtifact
from tests.model_inversion.support import artifact_metadata


def artifact(metadata_extra=None, posterior=None) -> PosteriorArtifact:
    metadata = artifact_metadata(dataset_overrides=metadata_extra)
    theta_count = len(metadata.dataset.theta_order)
    return PosteriorArtifact(
        artifact_dir=Path("."),
        posterior=object() if posterior is None else posterior,
        metadata=metadata,
        prior_samples=np.empty((0, theta_count), dtype=np.float32),
        prior_sample_indices=np.empty((0,), dtype=np.int64),
    )


class DiagnosticTests(unittest.TestCase):
    def test_load_diagnostic_data_reads_source_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            item = artifact(
                metadata_extra={
                    "discrete_theta_names": [],
                    "discrete_theta_values": [],
                }
            )
            np.savez_compressed(
                artifact_dir / "diagnostic_data.npz",
                theta=np.asarray([[0.1], [0.2]], dtype=np.float32),
                x=np.asarray(
                    [[1.0, 1.0, 0.0], [2.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                source_sample_ids=np.asarray([3, 5], dtype=np.int64),
                continuous_theta_names=np.asarray(["alpha"]),
                discrete_theta_names=np.asarray([], dtype=str),
                x_components=np.asarray(item.x_components),
            )
            item = PosteriorArtifact(
                artifact_dir,
                item.posterior,
                item.metadata,
                item.prior_samples,
                item.prior_sample_indices,
            )

            data = load_diagnostic_data(item)

        np.testing.assert_allclose(data.theta, [[0.1], [0.2]])
        np.testing.assert_allclose(
            data.x,
            [[1.0, 1.0, 0.0], [2.0, 0.0, 1.0]],
        )
        np.testing.assert_array_equal(data.source_sample_ids, [3, 5])

    def test_marginal_sbc_ranks_randomize_discrete_ties(self) -> None:
        class MaxTieRng:
            def integers(self, low, high):  # noqa: ANN001
                return high - 1

        samples = np.asarray(
            [
                [[0.1, 10.0], [0.5, 20.0], [0.8, 20.0]],
                [[0.3, 10.0], [0.1, 10.0], [0.2, 20.0]],
            ],
            dtype=np.float32,
        )
        theta = np.asarray([[0.5, 20.0], [0.2, 10.0]], dtype=np.float32)

        ranks = marginal_sbc_ranks(artifact(), samples, theta, rng=MaxTieRng())

        np.testing.assert_array_equal(ranks, [[1, 3], [1, 2]])


if __name__ == "__main__":
    unittest.main()
