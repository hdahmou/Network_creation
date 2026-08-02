"""Per-agent heterogeneity around simulation-level parameter centers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, cast

import numpy as np


type HeterogeneityKind = Literal["normal", "three_point"]
_THREE_POINT_OFFSETS = (-1, 0, 1)


@dataclass(frozen=True)
class Heterogeneity:
    kind: HeterogeneityKind
    std: float | None = None
    probabilities: tuple[float, float, float] | None = None
    minimum: float | None = None
    maximum: float | None = None


def parse_heterogeneity(
    parameter_name: str,
    parameter_spec: Mapping[str, object],
) -> Heterogeneity | None:
    if "std" in parameter_spec:
        raise ValueError(
            f"{parameter_name}.std is unsupported; use "
            f"{parameter_name}.heterogeneity"
        )

    raw = parameter_spec.get("heterogeneity")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{parameter_name}.heterogeneity must be an object")

    kind = str(raw.get("kind", ""))
    if kind == "normal":
        if str(parameter_spec.get("kind")) != "continuous":
            raise ValueError(
                f"{parameter_name} normal heterogeneity requires kind 'continuous'"
            )
        std = _finite_real(raw.get("std"), name=f"{parameter_name}.heterogeneity.std")
        if std < 0.0:
            raise ValueError(
                f"{parameter_name}.heterogeneity.std must be non-negative"
            )
        minimum = _finite_real(parameter_spec.get("min"), name=f"{parameter_name}.min")
        maximum = _finite_real(parameter_spec.get("max"), name=f"{parameter_name}.max")
        return Heterogeneity(
            kind="normal",
            std=std,
            minimum=minimum,
            maximum=maximum,
        )

    if kind == "three_point":
        probabilities = _probability_triplet(
            raw.get("probabilities"),
            name=f"{parameter_name}.heterogeneity.probabilities",
        )
        minimum_center = _validate_integer_centers(parameter_name, parameter_spec)
        _validate_visibility_support(parameter_name, minimum_center, probabilities)
        return Heterogeneity(kind="three_point", probabilities=probabilities)

    raise ValueError(
        f"{parameter_name}.heterogeneity.kind must be 'normal' or "
        "'three_point'"
    )


def sample_heterogeneous_value(
    center: float,
    heterogeneity: Heterogeneity,
    *,
    seed: int,
    n_agents: int,
    parameter_row_index: int,
    node_id: int,
    parameter_name: str,
) -> np.float32:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            (
                int(seed),
                int(n_agents),
                int(parameter_row_index),
                int(node_id),
                *parameter_name.encode("utf-8"),
            )
        )
    )
    if heterogeneity.kind == "normal":
        assert heterogeneity.std is not None
        assert heterogeneity.minimum is not None
        assert heterogeneity.maximum is not None
        return np.float32(
            np.clip(
                rng.normal(loc=center, scale=heterogeneity.std),
                heterogeneity.minimum,
                heterogeneity.maximum,
            )
        )
    assert heterogeneity.probabilities is not None
    offset = rng.choice(_THREE_POINT_OFFSETS, p=heterogeneity.probabilities)
    return np.float32(center + offset)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _probability_triplet(
    value: object,
    *,
    name: str,
) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly three probabilities")
    raw_probabilities = tuple(value)
    if len(raw_probabilities) != 3:
        raise ValueError(f"{name} must contain exactly three probabilities")
    if any(
        isinstance(probability, bool) or not isinstance(probability, Real)
        for probability in raw_probabilities
    ):
        raise ValueError(f"{name} must contain finite, non-negative numbers")

    probabilities = tuple(float(probability) for probability in raw_probabilities)
    if any(
        not np.isfinite(probability) or probability < 0.0
        for probability in probabilities
    ):
        raise ValueError(f"{name} must contain finite, non-negative numbers")
    total = sum(probabilities)
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} must sum to 1")

    lower, center, upper = probabilities
    return lower / total, center / total, upper / total


def _validate_integer_centers(
    parameter_name: str,
    parameter_spec: Mapping[str, object],
) -> int:
    kind = str(parameter_spec.get("kind"))
    values: tuple[object, ...]
    centers: tuple[object, ...]
    if kind == "fixed":
        values = (parameter_spec.get("value"),)
        centers = values
    elif kind == "sampled_choice":
        raw_values = parameter_spec.get("values")
        if not isinstance(raw_values, Sequence) or isinstance(
            raw_values, (str, bytes)
        ):
            raise ValueError(f"{parameter_name}.values must be a sequence")
        values = tuple(raw_values)
        centers = values
    elif kind == "sampled_discrete_range":
        values = (
            parameter_spec.get("min"),
            parameter_spec.get("max"),
            parameter_spec.get("step", 1),
        )
        centers = values[:1]
    else:
        raise ValueError(
            f"{parameter_name} three-point heterogeneity requires an "
            "integer-valued parameter kind"
        )
    if not centers or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError(f"{parameter_name} centers must be integers")
    return min(int(cast(Integral, center)) for center in centers)


def _validate_visibility_support(
    parameter_name: str,
    minimum_center: int,
    probabilities: tuple[float, float, float],
) -> None:
    if parameter_name.rsplit(".", maxsplit=1)[-1] != "visibility_distance":
        return

    supported_offsets = tuple(
        offset
        for offset, probability in zip(
            _THREE_POINT_OFFSETS, probabilities, strict=True
        )
        if probability > 0.0
    )
    if minimum_center + min(supported_offsets) < 1:
        raise ValueError(
            f"{parameter_name} three-point distribution must produce positive values"
        )


__all__ = [
    "Heterogeneity",
    "parse_heterogeneity",
    "sample_heterogeneous_value",
]
