"""Dense local/global node-index mapping."""

from __future__ import annotations

from collections.abc import Iterator, Mapping


class LocalGlobalMap(Mapping[int, int]):
    """Dense bidirectional node-index map for contiguous global node ids."""

    __slots__ = ("_local_to_global", "inverse")

    def __init__(self, local_to_global: list[int]) -> None:
        n_entries = len(local_to_global)
        inverse = [-1] * n_entries
        for local_idx, global_idx in enumerate(local_to_global):
            if not 0 <= global_idx < n_entries:
                raise ValueError("global node ids must be in range 0..n_entries-1")
            if inverse[global_idx] != -1:
                raise ValueError("global node ids must be unique")
            inverse[global_idx] = local_idx

        self._local_to_global = local_to_global
        self.inverse = inverse

    def __getitem__(self, local_idx: int) -> int:
        return self._local_to_global[local_idx]

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self._local_to_global)))

    def __len__(self) -> int:
        return len(self._local_to_global)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(
            other.get(local_idx) == global_idx
            for local_idx, global_idx in enumerate(self._local_to_global)
        )

    def __repr__(self) -> str:
        return repr(dict(enumerate(self._local_to_global)))
