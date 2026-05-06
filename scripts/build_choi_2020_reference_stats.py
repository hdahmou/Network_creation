#!/usr/bin/env python3
"""Build cached Choi 2020 reference stats for result-loader plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network_creation.data.choi_2020_references import (  # noqa: E402
    DEFAULT_REFERENCE_CACHE_PATH,
    DEFAULT_REFERENCE_PARAMETERS_PATH,
    SIGMOID_FIT_OBSERVER_NAMES,
    available_reference_observer_names,
    build_reference_cache,
    load_reference_parameters,
    reference_requests,
    save_reference_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute Choi 2020 terminal and full-resolution sigmoid reference "
            "stats for plot colorbars."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "choi_2020",
        help="Choi 2020 data root. A nested raw/ directory is accepted.",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_REFERENCE_PARAMETERS_PATH,
        help="Choi 2020 parameter JSON. Group sizes are read from this file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_REFERENCE_CACHE_PATH,
        help="Output pickle path. Defaults to artifacts/data/choi_2020_reference_stats.pkl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log(f"reading Choi parameters from {args.parameters}")
    parameters = load_reference_parameters(args.parameters)
    group_sizes = sorted({row.group_size for row in parameters})
    observer_names = available_reference_observer_names()
    requests = reference_requests(parameters)

    _log(f"using group sizes={group_sizes}")
    _log(f"using observers={observer_names}")
    _log(f"building cache for {len(requests)} reference requests")
    _log(
        "sigmoid fits use full-resolution experimental sampling every=1 "
        f"for observers={sorted(SIGMOID_FIT_OBSERVER_NAMES)}"
    )
    cache = build_reference_cache(
        data_root=args.data_root,
        requests=requests,
        progress=_log,
    )

    _log(f"saving cache to {args.output}")
    output_path = save_reference_cache(cache, args.output)
    _log("done")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "n_requests": len(requests),
                "n_values": len(cache["values"]),
                "n_sigmoid_values": len(cache["sigmoid_values"]),
                "missing_group_sizes": cache["missing_group_sizes"],
                "observers": observer_names,
                "group_sizes": group_sizes,
            },
            indent=2,
        )
    )


def _log(message: str) -> None:
    print(f"[choi-reference] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
