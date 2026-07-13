#!/usr/bin/env python3
"""Train and save a zarr-backed SBI NPE/MNPE posterior from one JSON config."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network_creation.model_inversion import train_inversion_from_config  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one amortized SBI NPE/MNPE posterior from a JSON config."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the model-inversion training config JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for progress messages. Defaults to INFO.",
    )
    return parser.parse_args(argv)


def _configure_logging(log_level: str) -> None:
    level_name = str(log_level).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"unknown log level: {log_level!r}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.log_level)
    summary = train_inversion_from_config(args.config_path.expanduser().resolve())
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
