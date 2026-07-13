from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from network_creation.model_inversion.artifact_metadata import ArtifactMetadata
from tasks.execution.schema import OutputArraySpec, OutputSpec
from tasks.execution.zarr_store import ZarrResultStore


def artifact_metadata(
    *,
    dataset_overrides: dict[str, Any] | None = None,
    prior_sample_count: int = 0,
) -> ArtifactMetadata:
    dataset: dict[str, Any] = {
        "result_store_path": "/tmp/run.zarr",
        "training_config_path": "/tmp/train.json",
        "simulation_config_path": "/tmp/simulation.json",
        "observed_outputs": ["density.mean"],
        "x_components": ["density.mean", "n_agents=10", "n_agents=20"],
        "x_event_shape": [3],
        "x_representation": "final",
        "normalize_time": False,
        "time_axis": "raw_simulator_steps",
        "raw_step_labels": [0, 1],
        "step_labels": None,
        "final_step_label": 1,
        "n_actors": None,
        "continuous_theta_names": ["alpha"],
        "discrete_theta_names": ["n_agents"],
        "continuous_input_names": [],
        "discrete_input_names": ["n_agents"],
        "discrete_input_values": [[10, 20]],
        "discrete_theta_values": [[10, 20]],
        "constant_theta": {},
        "train_count": 1,
        "diagnostic_count": 1,
        "n_samples": 2,
    }
    if dataset_overrides:
        dataset.update(dataset_overrides)
    theta_count = len(dataset["continuous_theta_names"]) + len(
        dataset["discrete_theta_names"]
    )
    dataset.setdefault("prior_low", [0.0] * theta_count)
    dataset.setdefault("prior_high", [1.0] * theta_count)
    mixed_parameters = bool(dataset["discrete_theta_names"])
    payload = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "dataset": dataset,
        "sbi": {
            "method": "MNPE" if mixed_parameters else "NPE",
            "amortized": True,
            "mixed_parameters": mixed_parameters,
            "posterior_file": "posterior.pkl",
            "training_metrics_file": "training_metrics.json",
            "diagnostic_data_file": "diagnostic_data.npz",
            "prior_samples_file": "prior_samples.npz",
            "tensorboard_dir": "sbi_logs",
            "prior_samples": {
                "source": "training_theta_rows",
                "count": prior_sample_count,
                "selection_seed": 0,
            },
        },
        "training": {},
        "model": {},
    }
    return ArtifactMetadata.model_validate_json(json.dumps(payload))


def write_result_store(path: Path, *, include_nan: bool = False) -> Path:
    n_agents = np.asarray([10, 20], dtype=np.int64)
    visibility = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.float32)
    outside = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    exploration = np.asarray([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float32)
    constant_beta = np.full(6, 5.0, dtype=np.float32)
    parameter_names = [
        "best_response_proportion",
        "visibility_distance",
        "outside_optimism",
        "exploration",
        "constant_beta",
    ]
    parameters = np.empty((2, 1, 6, len(parameter_names)), dtype=np.float32)
    for agent_index in range(2):
        parameters[agent_index, 0, :, 0] = 1.0
        parameters[agent_index, 0, :, 1] = visibility
        parameters[agent_index, 0, :, 2] = outside
        parameters[agent_index, 0, :, 3] = exploration
        parameters[agent_index, 0, :, 4] = constant_beta
    input_coordinates = np.asarray(
        [[agent_index, 0, row] for agent_index in range(2) for row in range(6)],
        dtype=np.int64,
    )
    policy_config: dict[str, Any] = {
        "simulator": {
            "max_steps": 2,
            "actor_scheduler": {"n_actors": 1},
        },
        "population": {
            "n_agents": [10, 20],
            "policy_proportions": {"policy": "best_response", "values": [1.0]},
            "policies": [{"id": "best_response", "kind": "best-response"}],
        },
        "observers": {
            "defaults": {
                "sampling": {"kind": "periodic", "every": 1},
                "aggregations": ["mean"],
            },
            "outputs": [
                {"name": "density"},
                {"name": "distance"},
            ],
        },
    }
    config_path = path / "run_config.json"
    path.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    store = ZarrResultStore(
        str(path),
        output_schema=OutputSpec(
            fields=(
                OutputArraySpec(name="density.mean", shape=(3,), dtype=np.float32),
                OutputArraySpec(name="distance.mean", shape=(3,), dtype=np.float32),
            )
        ),
        batch_size=4,
        total_inputs=int(input_coordinates.shape[0]),
        metadata_attrs={
            "parameter_names": parameter_names,
            "policy_config": policy_config,
            "config_path": str(config_path),
        },
        metadata_arrays={
            "n_agents": n_agents,
            "parameters": parameters,
            "parameter_low": np.asarray([1.0, 1.0, 0.0, 0.0, 5.0], dtype=np.float32),
            "parameter_high": np.asarray([1.0, 2.0, 1.0, 0.1, 5.0], dtype=np.float32),
            "input_coordinates": input_coordinates,
        },
    )
    for input_index in range(input_coordinates.shape[0]):
        density = np.asarray(
            [float(input_index), float(input_index) + 1.0, float(input_index) + 2.0],
            dtype=np.float32,
        )
        distance = np.asarray(
            [float(input_index) + 10.0, float(input_index) + 11.0, float(input_index) + 12.0],
            dtype=np.float32,
        )
        if include_nan and input_index == 0:
            density[-1] = np.nan
        store.add_success(
            int(input_index),
            {
                "density.mean": density,
                "distance.mean": distance,
            },
        )
    store.flush()
    return path


def write_training_config(
    path: Path,
    *,
    zarr_path: Path,
    output_root: Path | None = None,
    artifact_name: str = "run",
    discrete_theta: bool = True,
    x_representation: str = "final",
    normalize_time: bool = False,
) -> Path:
    payload = {
        "data": {
            "zarr_path": str(zarr_path),
            "observed_outputs": ["density.mean"],
            "continuous_theta_names": ["outside_optimism", "exploration", "constant_beta"],
            "discrete_theta_names": ["visibility_distance"] if discrete_theta else [],
            "continuous_input_names": [],
            "discrete_input_names": ["n_agents"],
            "x_representation": x_representation,
            "normalize_time": normalize_time,
        },
        "model": {
            "density_estimator": {"flow_model": "nsf"},
            "embedding": {
                "kind": "rnn" if x_representation == "timeseries" else "identity"
            },
        },
        "training": {
            "seed": 2,
            "device": "cpu",
            "batch_size": 4,
            "learning_rate": 0.0,
            "validation_fraction": 0.5,
            "stop_after_epochs": 1,
            "max_num_epochs": 1,
            "show_progress_bars": False,
            "calibration_sample_count": 4,
        },
        "output": {
            "output_root": str(output_root or path.parent / "artifacts"),
            "artifact_name": artifact_name,
            "save_training_data": True,
            "prior_sample_count": 3,
        },
        "diagnostic_holdout_count_per_group": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
