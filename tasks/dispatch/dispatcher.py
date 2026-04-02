"""Repo-owned dispatcher for running tasks on a Dask cluster."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dask.distributed import Client, as_completed

from .registry import load_task
from .result_store import ResultStore, task_class_path
from .runtime import TaskOutput


_LOG_RECORD_SKIP_FIELDS = {
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _LOG_RECORD_SKIP_FIELDS:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(logger_name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    logging.getLogger("distributed").setLevel(logging.WARNING)
    logging.getLogger("tornado").setLevel(logging.WARNING)
    return logger


def _describe(value: Any, describe_fn: Callable[[Any], str]) -> str:
    try:
        return str(describe_fn(value))
    except Exception:
        return repr(value)


def _load_config(config_path: Path) -> dict[str, Any]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
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


def _run_started_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def run_tasks(
    scheduler_address: str,
    task_name: str,
    task_fn: Any,
    task_inputs: Sequence[Any],
    on_result: Any,
    on_failure: Any,
    *,
    logger: logging.Logger,
) -> dict[str, int]:
    task_inputs_list = list(task_inputs)
    if not task_inputs_list:
        raise ValueError("No tasks found")

    with Client(scheduler_address) as client:
        logger.info(
            "Connected to scheduler",
            extra={
                "scheduler_address": scheduler_address,
                "dashboard": client.dashboard_link,
            },
        )

        success_count = 0
        failure_count = 0
        futures = client.map(task_fn, task_inputs_list, key=task_name, pure=False)
        submitted_time = time.perf_counter()
        future_context = {
            future.key: (input_index, task_input)
            for input_index, (future, task_input) in enumerate(
                zip(futures, task_inputs_list, strict=True)
            )
        }
        task_count = len(futures)
        completed_futures = as_completed(futures)
        del futures
        del task_inputs_list

        logger.debug(
            "Submitted tasks",
            extra={"task_name": task_name, "task_count": task_count},
        )

        for future in completed_futures:
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


def _write_runtime_config_copy(config: Mapping[str, Any], output_zarr: Path) -> Path:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    config_copy_path = output_zarr.parent / f"{output_zarr.stem}.dispatch_config.json"
    config_copy_path.write_text(
        json.dumps(dict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    return config_copy_path


def dispatch_from_config(config_path: str, log_level: str = "INFO") -> dict[str, Any]:
    logger = configure_logging("dispatcher", log_level)
    config_path_obj = Path(config_path).resolve()
    config = _load_config(config_path_obj)

    missing = [
        key
        for key in ("scheduler_address", "task", "output_dir")
        if key not in config
    ]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    task_key = str(config["task"])
    task = load_task(task_key)
    task_class_name = task_class_path(task)
    task_name = str(task.name)
    started_slug = _run_started_slug()
    output_zarr = (
        Path(str(config["output_dir"])) / f"{task_name}-{started_slug}.zarr"
    ).resolve()
    batch_size = int(config.get("write_batch_size", 16))
    scheduler_address = str(config["scheduler_address"])
    inputs_json = config.get("inputs_json")

    if batch_size <= 0:
        raise ValueError("write_batch_size must be > 0")

    prepare_extra = {"task": task_key, "task_class": task_class_name}
    if inputs_json is not None:
        prepare_extra["inputs_json"] = str(inputs_json)
    logger.info("Preparing task", extra=prepare_extra)
    prepared = task.prepare(config)
    task_inputs = list(prepared.inputs)
    if not task_inputs:
        raise ValueError("No tasks found")

    config_copy_path = _write_runtime_config_copy(config, output_zarr)
    writer = ResultStore(
        str(output_zarr),
        task_name=task_name,
        task_class=task_class_name,
        output_schema=prepared.output_schema,
        batch_size=batch_size,
        total_inputs=len(task_inputs),
    )
    logger.info(
        "Initialized output",
        extra={
            "task_name": task_name,
            "output_zarr": str(output_zarr),
            "batch_size": batch_size,
        },
    )

    def handle_result(
        input_index: int,
        task_input: Any,
        task_output: TaskOutput,
        duration_ms: float,
    ) -> None:
        input_label = _describe(task_input, prepared.describe_input)
        output_summary = _describe(task_output, prepared.describe_output)
        writer.add_success(
            input_index=input_index,
            input_label=input_label,
            output=task_output,
            duration_ms=duration_ms,
        )
        logger.debug(
            "Task completed",
            extra={
                "task_name": task_name,
                "input_index": input_index,
                "input_label": input_label,
                "task_output": output_summary,
                "duration_ms": round(duration_ms, 3),
            },
        )

    def handle_failure(
        input_index: int,
        task_input: Any,
        duration_ms: float,
        error: BaseException,
    ) -> None:
        input_label = _describe(task_input, prepared.describe_input)
        error_text = "".join(traceback.format_exception_only(type(error), error)).strip()
        writer.add_failure(
            error=error_text,
        )
        logger.error(
            "Recorded task failure",
            extra={
                "task_name": task_name,
                "input_index": input_index,
                "input_label": input_label,
                "duration_ms": round(duration_ms, 3),
                "error": error_text,
            },
        )

    try:
        stats = run_tasks(
            scheduler_address=scheduler_address,
            task_name=task_name,
            task_fn=prepared.run,
            task_inputs=task_inputs,
            on_result=handle_result,
            on_failure=handle_failure,
            logger=logger,
        )
    finally:
        writer.flush()

    summary = {
        **stats,
        "task_name": task_name,
        "started_at": started_slug,
        "task_class": task_class_name,
        "success_rows_written": writer.success_count,
        "failure_rows_written": writer.failure_count,
        "output_zarr": str(output_zarr),
        "config_copy_path": str(config_copy_path),
    }
    logger.info("Run finished", extra=summary)
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
