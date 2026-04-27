"""Repo-owned dispatcher for running tasks on a Dask cluster."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dask.distributed import Client, as_completed

from network_creation.io import read_json, write_json

from tasks.catalog import load_task
from .result_store import ResultStore
from .runtime import PreparedTask, TaskOutput


@dataclass(frozen=True)
class DispatchPlan:
    config: dict[str, Any]
    scheduler_address: str
    task_name: str
    batch_size: int
    prepared: PreparedTask[Any]
    task_inputs: tuple[Any, ...]
    started_slug: str
    output_zarr: Path
    config_copy_path: Path


def configure_logging(logger_name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)

    logging.getLogger("distributed").setLevel(logging.WARNING)
    logging.getLogger("tornado").setLevel(logging.WARNING)
    return logger


def _load_config(config_path: Path) -> dict[str, Any]:
    data = read_json(config_path)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")

    config = dict(data)
    config_dir = config_path.parent
    for key in ("inputs_json", "output_dir"):
        if key not in config:
            continue
        path = Path(str(config[key]))
        if not path.is_absolute():
            path = (config_dir / path).resolve()
        config[key] = str(path)
    return config


def _build_dispatch_plan(
    config_path: Path,
    *,
    logger: logging.Logger,
) -> DispatchPlan:
    config = _load_config(config_path)

    task_key = str(config["task"])
    task = load_task(task_key)
    task_name = str(task.name)
    logger.info("Preparing task %s", task_name)
    prepared = task.prepare(config)
    task_inputs = tuple(prepared.inputs)
    if not task_inputs:
        raise ValueError("No tasks found")

    started_slug = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    output_zarr = (
        Path(str(config["output_dir"])) / f"{task_name}-{started_slug}.zarr"
    ).resolve()
    config_copy_path = output_zarr.parent / f"{output_zarr.stem}.dispatch_config.json"

    return DispatchPlan(
        config=config,
        scheduler_address=str(config["scheduler_address"]),
        task_name=task_name,
        batch_size=int(config.get("write_batch_size", 16)),
        prepared=prepared,
        task_inputs=task_inputs,
        started_slug=started_slug,
        output_zarr=output_zarr,
        config_copy_path=config_copy_path,
    )


def run_tasks(
    scheduler_address: str,
    task_name: str,
    task_fn: Callable[[Any], TaskOutput],
    task_inputs: Sequence[Any],
    on_result: Callable[[int, Any, TaskOutput, float], None],
    on_failure: Callable[[int, Any, float, BaseException], None],
    *,
    logger: logging.Logger,
) -> dict[str, int]:
    task_inputs_list = list(task_inputs)
    if not task_inputs_list:
        raise ValueError("No tasks found")

    with Client(scheduler_address) as client:
        logger.info("Connected to scheduler %s", scheduler_address)

        futures = client.map(task_fn, task_inputs_list, key=task_name, pure=False)
        submitted_time = time.perf_counter()
        future_context = {
            future.key: (input_index, task_input)
            for input_index, (future, task_input) in enumerate(
                zip(futures, task_inputs_list, strict=True)
            )
        }

        success_count = 0
        failure_count = 0
        for future in as_completed(futures):
            input_index, task_input = future_context.pop(future.key)
            duration_ms = (time.perf_counter() - submitted_time) * 1000.0
            try:
                task_output = future.result()
                on_result(input_index, task_input, task_output, duration_ms)
            except Exception as exc:
                failure_count += 1
                on_failure(input_index, task_input, duration_ms, exc)
            else:
                success_count += 1
            finally:
                future.release()

    return {
        "success_count": success_count,
        "failure_count": failure_count,
    }


def _write_runtime_config_copy(
    config: Mapping[str, Any], config_copy_path: Path
) -> Path:
    config_copy_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_copy_path, dict(config))
    return config_copy_path


def dispatch_from_config(config_path: str, log_level: str = "INFO") -> dict[str, Any]:
    logger = configure_logging("dispatcher", log_level)
    plan = _build_dispatch_plan(Path(config_path).resolve(), logger=logger)
    config_copy_path = _write_runtime_config_copy(plan.config, plan.config_copy_path)

    writer = ResultStore(
        str(plan.output_zarr),
        task_name=plan.task_name,
        output_schema=plan.prepared.output_schema,
        batch_size=plan.batch_size,
        total_inputs=len(plan.task_inputs),
        metadata_attrs={
            **dict(plan.prepared.result_attrs),
            "config_path": str(config_copy_path),
        },
        metadata_arrays=plan.prepared.result_arrays,
    )
    logger.info("Initialized output %s", plan.output_zarr)

    def handle_result(
        input_index: int,
        _task_input: Any,
        task_output: TaskOutput,
        duration_ms: float,
    ) -> None:
        writer.add_success(
            input_index=input_index,
            output=task_output,
            duration_ms=duration_ms,
        )

    def handle_failure(
        input_index: int,
        task_input: Any,
        duration_ms: float,
        error: BaseException,
    ) -> None:
        error_text = "".join(
            traceback.format_exception_only(type(error), error)
        ).strip()
        writer.add_failure()
        logger.error(
            "Task failed for input %s at index %s after %.3f ms: %s",
            task_input,
            input_index,
            duration_ms,
            error_text,
        )

    try:
        stats = run_tasks(
            scheduler_address=plan.scheduler_address,
            task_name=plan.task_name,
            task_fn=plan.prepared.run,
            task_inputs=plan.task_inputs,
            on_result=handle_result,
            on_failure=handle_failure,
            logger=logger,
        )
    finally:
        writer.flush()

    summary = {
        **stats,
        "task_name": plan.task_name,
        "started_at": plan.started_slug,
        "success_rows_written": writer.success_count,
        "has_failures": writer.has_failures,
        "output_zarr": str(plan.output_zarr),
        "config_copy_path": str(config_copy_path),
    }
    logger.info(
        "Run finished for %s: %s successes, %s failures",
        plan.task_name,
        summary["success_count"],
        summary["failure_count"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Network creation task dispatcher")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()
    dispatch_from_config(args.config, args.log_level)


if __name__ == "__main__":
    main()
