"""Run policy simulations on a Dask cluster."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from dask.distributed import Client, as_completed
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from network_creation.io import read_json, write_json

from tasks.policies import prepare_policy_run
from .schema import PreparedRun, RunOutput
from .zarr_store import ZarrResultStore


_IN_FLIGHT_TASKS_PER_THREAD = 2


@dataclass(frozen=True)
class RunPlan:
    config: dict[str, Any]
    scheduler_address: str
    batch_size: int
    prepared: PreparedRun[Any]
    run_inputs: tuple[Any, ...]
    started_slug: str
    output_zarr: Path
    config_copy_path: Path


def configure_run_logging(logger_name: str, level: str = "INFO") -> logging.Logger:
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


def _resolve_config_path(raw_path: Any, config_dir: Path) -> str:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return str(path)


def _load_config(config_path: Path) -> dict[str, Any]:
    data = read_json(config_path)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")

    config = dict(data)
    config_dir = config_path.parent
    for key in ("inputs_json",):
        if key in config:
            config[key] = _resolve_config_path(config[key], config_dir)
    execution = config.get("execution")
    if isinstance(execution, Mapping) and "output_dir" in execution:
        normalized_execution = dict(execution)
        normalized_execution["output_dir"] = _resolve_config_path(
            execution["output_dir"],
            config_dir,
        )
        config["execution"] = normalized_execution
    return config


def _build_run_plan(
    config_path: Path,
    *,
    logger: logging.Logger,
) -> RunPlan:
    config = _load_config(config_path)

    policy_run = prepare_policy_run(config)
    logger.info("Preparing run")
    prepared = getattr(policy_run, "prepared", policy_run)
    run_inputs = tuple(prepared.inputs)
    if not run_inputs:
        raise ValueError("No run inputs found")

    started_slug = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    output_zarr = (
        Path(str(config["execution"]["output_dir"])) / f"run-{started_slug}.zarr"
    ).resolve()
    config_copy_path = output_zarr / "run_config.json"

    return RunPlan(
        config=config,
        scheduler_address=str(config["execution"]["scheduler_address"]),
        batch_size=int(config["execution"].get("write_batch_size", 16)),
        prepared=prepared,
        run_inputs=run_inputs,
        started_slug=started_slug,
        output_zarr=output_zarr,
        config_copy_path=config_copy_path,
    )


def _materialize_run_fn(
    run_fn: Callable[[Any], RunOutput],
) -> Callable[[Any], RunOutput]:
    return run_fn


def _execute_run_input(
    run_input: Any,
    *,
    run_fn: Callable[[Any], RunOutput],
) -> RunOutput:
    return run_fn(run_input)


def execute_inputs(
    scheduler_address: str,
    run_key: str,
    run_fn: Callable[[Any], RunOutput],
    run_inputs: Sequence[Any],
    on_result: Callable[[int, Any, RunOutput], None],
    on_failure: Callable[[int, Any, BaseException], None],
    *,
    logger: logging.Logger,
) -> dict[str, int]:
    total_inputs = len(run_inputs)
    if not total_inputs:
        raise ValueError("No run inputs found")

    with Client(scheduler_address) as client:
        logger.info("Connected to scheduler %s", scheduler_address)
        run_token = uuid4().hex
        run_fn_future = client.submit(
            _materialize_run_fn,
            run_fn,
            key=f"{run_key}-runtime-{run_token}",
            pure=False,
        )
        completed = as_completed()
        future_context: dict[str, tuple[int, Any]] = {}
        next_input_index = 0
        success_count = 0
        failure_count = 0

        try:
            with tqdm(
                total=total_inputs,
                desc="Simulations",
                unit="run",
                dynamic_ncols=True,
                disable=None,
            ) as progress:
                while next_input_index < total_inputs or future_context:
                    worker_threads = client.nthreads()
                    worker_count = len(worker_threads)
                    in_flight_target = max(
                        1,
                        _IN_FLIGHT_TASKS_PER_THREAD * sum(worker_threads.values()),
                    )

                    submit_count = min(
                        in_flight_target - len(future_context),
                        total_inputs - next_input_index,
                    )
                    if submit_count > 0:
                        stop_input_index = next_input_index + submit_count
                        indexed_inputs = [
                            (input_index, run_inputs[input_index])
                            for input_index in range(
                                next_input_index,
                                stop_input_index,
                            )
                        ]
                        futures = client.map(
                            _execute_run_input,
                            [run_input for _, run_input in indexed_inputs],
                            key=[
                                f"{run_key}-{run_token}-{input_index}"
                                for input_index, _ in indexed_inputs
                            ],
                            pure=False,
                            run_fn=run_fn_future,
                        )
                        for future, context in zip(
                            futures,
                            indexed_inputs,
                            strict=True,
                        ):
                            future_context[future.key] = context
                        completed.update(futures)
                        next_input_index = stop_input_index

                    completed_batch = completed.next_batch()
                    for future in completed_batch:
                        input_index, run_input = future_context.pop(future.key)
                        try:
                            run_output = future.result()
                            on_result(input_index, run_input, run_output)
                        except Exception as exc:
                            failure_count += 1
                            on_failure(input_index, run_input, exc)
                        else:
                            success_count += 1
                        finally:
                            future.release()

                    progress.set_postfix(
                        successes=success_count,
                        failures=failure_count,
                        workers=worker_count,
                        refresh=False,
                    )
                    progress.update(len(completed_batch))
        finally:
            run_fn_future.release()

    return {
        "success_count": success_count,
        "failure_count": failure_count,
    }


def _write_config_copy(
    config: Mapping[str, Any], config_copy_path: Path
) -> Path:
    config_copy_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_copy_path, dict(config))
    return config_copy_path


def run_from_config(config_path: str, log_level: str = "INFO") -> dict[str, Any]:
    logger = configure_run_logging("runner", log_level)
    plan = _build_run_plan(Path(config_path).resolve(), logger=logger)
    config_copy_path = plan.config_copy_path

    writer = ZarrResultStore(
        str(plan.output_zarr),
        output_schema=plan.prepared.output_schema,
        batch_size=plan.batch_size,
        total_inputs=len(plan.run_inputs),
        metadata_attrs={
            **dict(plan.prepared.result_attrs),
            "config_path": str(config_copy_path),
        },
        metadata_arrays=plan.prepared.result_arrays,
    )
    config_copy_path = _write_config_copy(plan.config, config_copy_path)
    logger.info("Initialized output %s", plan.output_zarr)

    def handle_result(
        input_index: int,
        _run_input: Any,
        run_output: RunOutput,
    ) -> None:
        writer.add_success(
            input_index=input_index,
            output=run_output,
        )

    def handle_failure(
        input_index: int,
        run_input: Any,
        error: BaseException,
    ) -> None:
        error_text = "".join(
            traceback.format_exception_only(type(error), error)
        ).strip()
        writer.add_failure()
        logger.error(
            "Run failed for input %s at index %s: %s",
            run_input,
            input_index,
            error_text,
        )

    try:
        stats = execute_inputs(
            scheduler_address=plan.scheduler_address,
            run_key="run",
            run_fn=plan.prepared.run,
            run_inputs=plan.run_inputs,
            on_result=handle_result,
            on_failure=handle_failure,
            logger=logger,
        )
    finally:
        writer.flush()

    (plan.output_zarr / "done").touch()

    summary = {
        **stats,
        "started_at": plan.started_slug,
        "success_rows_written": writer.success_count,
        "has_failures": writer.has_failures,
        "output_zarr": str(plan.output_zarr),
        "config_copy_path": str(config_copy_path),
    }
    logger.info(
        "Run finished: %s successes, %s failures",
        summary["success_count"],
        summary["failure_count"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Network creation run executor")
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
    run_from_config(args.config, args.log_level)


if __name__ == "__main__":
    main()
