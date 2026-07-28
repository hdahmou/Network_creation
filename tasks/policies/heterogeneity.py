"""Per-agent heterogeneity around simulation-level parameter centers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np


type HeterogeneityKind = Literal["normal", "discrete_offsets"]


@dataclass(frozen=True)
class Heterogeneity:
    kind: HeterogeneityKind
    std: float | None = None
    offsets: tuple[int, ...] = ()
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

    if kind == "discrete_offsets":
        offsets = _integer_offsets(
            raw.get("offsets"),
            name=f"{parameter_name}.heterogeneity.offsets",
        )
        _validate_integer_centers(parameter_name, parameter_spec)
        return Heterogeneity(kind="discrete_offsets", offsets=offsets)

    raise ValueError(
        f"{parameter_name}.heterogeneity.kind must be 'normal' or "
        "'discrete_offsets'"
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
    return np.float32(center + rng.choice(heterogeneity.offsets))


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _integer_offsets(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    offsets = tuple(value)
    if not offsets or any(
        isinstance(offset, bool) or not isinstance(offset, Integral)
        for offset in offsets
    ):
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    integer_offsets = tuple(int(offset) for offset in offsets)
    if len(set(integer_offsets)) != len(integer_offsets):
        raise ValueError(f"{name} must contain unique values")
    if sum(integer_offsets) != 0:
        raise ValueError(f"{name} must have mean zero")
    return integer_offsets


def _validate_integer_centers(
    parameter_name: str,
    parameter_spec: Mapping[str, object],
) -> None:
    kind = str(parameter_spec.get("kind"))
    values: tuple[object, ...]
    if kind == "fixed":
        values = (parameter_spec.get("value"),)
    elif kind == "sampled_choice":
        raw_values = parameter_spec.get("values")
        if not isinstance(raw_values, Sequence) or isinstance(
            raw_values, (str, bytes)
        ):
            raise ValueError(f"{parameter_name}.values must be a sequence")
        values = tuple(raw_values)
    elif kind == "sampled_discrete_range":
        values = (
            parameter_spec.get("min"),
            parameter_spec.get("max"),
            parameter_spec.get("step", 1),
        )
    else:
        raise ValueError(
            f"{parameter_name} discrete-offset heterogeneity requires an "
            "integer-valued parameter kind"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError(f"{parameter_name} centers must be integers")


__all__ = [
    "Heterogeneity",
    "parse_heterogeneity",
    "sample_heterogeneous_value",
]
