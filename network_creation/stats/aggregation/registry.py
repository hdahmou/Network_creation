"""Factory helpers for named batch aggregations."""

from __future__ import annotations

from collections.abc import Iterable

from .base import StatisticAccumulator
from .moments import MomentsAccumulator
from .runs import RunsCollector
from typing import TypeAlias

AccumulatorFactory: TypeAlias = type[StatisticAccumulator]

_ACCUMULATOR_FAMILIES: tuple[tuple[frozenset[str], AccumulatorFactory], ...] = (
    (frozenset({"mean", "std"}), MomentsAccumulator),
    (frozenset({"runs"}), RunsCollector),
)

AVAILABLE_AGGREGATIONS = frozenset().union(
    *(output_names for output_names, _ in _ACCUMULATOR_FAMILIES)
)


def build_accumulators(
    requested_outputs: Iterable[str],
) -> tuple[StatisticAccumulator, ...]:
    """Build the accumulator set for the requested output names."""

    requested = frozenset(requested_outputs)
    unknown_outputs = requested.difference(AVAILABLE_AGGREGATIONS)
    if unknown_outputs:
        unknown = ", ".join(sorted(unknown_outputs))
        raise ValueError(f"unsupported aggregation outputs: {unknown}")

    accumulators: list[StatisticAccumulator] = []
    for output_names, accumulator_factory in _ACCUMULATOR_FAMILIES:
        selected_outputs = requested.intersection(output_names)
        if selected_outputs:
            accumulators.append(accumulator_factory(selected_outputs))

    return tuple(accumulators)
