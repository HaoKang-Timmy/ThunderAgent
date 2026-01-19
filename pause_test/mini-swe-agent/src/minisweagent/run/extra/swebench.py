#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import os
import random
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()
_ROLLOUTS_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        checkpoint_path: Path | None = None,
        checkpoint_interval_steps: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id
        self._checkpoint_path = checkpoint_path
        self._checkpoint_interval_steps = max(0, int(checkpoint_interval_steps))
        self._last_checkpoint_step = 0

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        response = super().step()

        if self._checkpoint_path is not None and self._checkpoint_interval_steps > 0:
            step_idx = int(self.model.n_calls)
            if step_idx - self._last_checkpoint_step >= self._checkpoint_interval_steps:
                save_traj(
                    self,
                    self._checkpoint_path,
                    print_path=False,
                    exit_status="running",
                    result=None,
                    instance_id=self.instance_id,
                )
                self._last_checkpoint_step = step_idx

        return response


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def _load_rollouts_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def update_rollouts_file(
    output_path: Path,
    instance_id: str,
    rollout_idx: int,
    model_name: str,
    result: str,
    *,
    task_id: str,
    exit_status: str,
) -> None:
    """Append/overwrite a rollout result for an instance."""
    with _ROLLOUTS_FILE_LOCK:
        output_data: dict = _load_rollouts_file(output_path)
        entries = output_data.get(instance_id, [])
        if not isinstance(entries, list):
            entries = []

        new_entry = {
            "rollout_idx": int(rollout_idx),
            "task_id": task_id,
            "model_name_or_path": model_name,
            "model_patch": result,
            "exit_status": exit_status,
        }

        replaced = False
        for i, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("rollout_idx") == rollout_idx:
                entries[i] = new_entry
                replaced = True
                break
        if not replaced:
            entries.append(new_entry)

        entries.sort(key=lambda e: int(e.get("rollout_idx", 0)) if isinstance(e, dict) else 0)
        output_data[instance_id] = entries
        output_path.write_text(json.dumps(output_data, indent=2))


def get_completed_rollouts(output_path: Path, instance_id: str) -> set[int]:
    output_data = _load_rollouts_file(output_path)
    entries = output_data.get(instance_id, [])
    completed: set[int] = set()
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                completed.add(int(entry.get("rollout_idx")))
            except Exception:
                continue
    return completed


def release_router_program(job_id: str) -> None:
    """Best-effort: ask router to drop any per-program state for this job_id.

    Enabled only when MSWEA_ROUTER_ADMIN_URL is set (e.g., "http://localhost:8300").
    """
    router_admin_url = os.getenv("MSWEA_ROUTER_ADMIN_URL", "").strip()
    if not router_admin_url:
        return

    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover - best-effort dependency
        logger.warning(f"MSWEA_ROUTER_ADMIN_URL is set but 'requests' is unavailable; skipping release: {exc}")
        return

    url = f"{router_admin_url.rstrip('/')}/programs/release"
    try:
        resp = requests.post(url, json={"job_id": str(job_id), "program_id": str(job_id)}, timeout=2.0)
        if resp.status_code >= 400:  # pragma: no cover - best-effort
            body = (resp.text or "").strip().replace("\n", " ")
            logger.warning(f"Router release failed for job_id={job_id}: HTTP {resp.status_code} {body[:200]}")
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning(f"Router release request failed for job_id={job_id}: {exc}")


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    instance_number: int = 0,
    rollout_idx: int = 1,
    rollouts_per_instance: int = 1,
    checkpoint_interval_steps: int = 0,
    cleanup_images: bool = False,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    task_id = instance_id if rollouts_per_instance <= 1 else f"{instance_id}#r{rollout_idx:02d}"
    instance_dir = output_dir / instance_id
    rollout_dir = (
        instance_dir
        if rollouts_per_instance <= 1
        else (instance_dir / "rollouts" / f"r{rollout_idx:02d}")
    )
    # avoid inconsistent state if something here fails and there's leftover previous files
    if rollouts_per_instance <= 1:
        remove_from_preds_file(output_dir / "preds.json", instance_id)
        (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    else:
        (rollout_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    # For vLLM models, pass job_id and step_limit
    model_config = config.get("model", {}).copy()
    is_vllm_model = model_config.get("model_class") == "vllm" or "vllm" in model_config.get("model_class", "").lower()
    if is_vllm_model:
        model_config["job_id"] = instance_number
        model_config["step_limit"] = config.get("agent", {}).get("step_limit", 0)

    model = get_model(config=model_config)
    task = instance["problem_statement"]

    progress_manager.on_instance_start(task_id)
    progress_manager.update_instance_status(task_id, "Pulling/starting docker")

    agent = None
    extra_info = None
    env = None
    env_prepare_timing: dict | None = None

    try:
        env_prepare_start_ts = time.time()
        try:
            env = get_sb_environment(config, instance)
        finally:
            env_prepare_end_ts = time.time()
            env_prepare_timing = {
                "start_ts": env_prepare_start_ts,
                "end_ts": env_prepare_end_ts,
                "total_s": max(0.0, env_prepare_end_ts - env_prepare_start_ts),
                "environment_class": config.get("environment", {}).get("environment_class", "docker"),
                "image": get_swebench_docker_image_name(instance),
            }
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=task_id,
            checkpoint_path=rollout_dir / f"{instance_id}.traj.json",
            checkpoint_interval_steps=checkpoint_interval_steps,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(task)
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Persist timing metrics for this instance (best-effort).
        try:
            rollout_dir.mkdir(parents=True, exist_ok=True)
            timing_path = rollout_dir / f"{instance_id}.timings.json"
            timing_payload = {
                "instance_id": instance_id,
                "task_id": task_id,
                "rollout_idx": rollout_idx,
                "rollouts_per_instance": rollouts_per_instance,
                "env_prepare": env_prepare_timing,
                "steps": getattr(agent, "step_timings", []) if agent is not None else [],
            }
            timing_path.write_text(json.dumps(timing_payload, indent=2))
        except Exception as timing_error:  # pragma: no cover - best-effort
            logger.warning(f"Failed to write timings for {instance_id}: {timing_error}")

        save_traj(
            agent,
            rollout_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        if rollouts_per_instance <= 1:
            update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        else:
            update_rollouts_file(
                output_dir / "preds_rollouts.json",
                instance_id,
                rollout_idx,
                model.config.model_name,
                result,
                task_id=task_id,
                exit_status=exit_status,
            )
        if env and hasattr(env, "cleanup"):
            try:
                env.cleanup()
            except Exception as cleanup_error:  # pragma: no cover - best-effort cleanup
                logger.warning(f"Error during environment cleanup for {instance_id}: {cleanup_error}")
        if cleanup_images:
            image_name = get_swebench_docker_image_name(instance)
            docker_executable = getattr(getattr(env, "config", None), "executable", "docker")
            try:
                cleanup_cmds = [
                    [docker_executable, "image", "rm", "-f", image_name],
                ]
                for cmd in cleanup_cmds:
                    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if result.returncode != 0:  # pragma: no cover - best-effort cleanup
                        logger.warning(f"Cleanup command failed (code {result.returncode}): {' '.join(cmd)}")
                # Explicitly delete dangling (<none>) layers.
                dangling = subprocess.run(
                    [docker_executable, "images", "-q", "--filter", "dangling=true"],
                    text=True,
                    capture_output=True,
                )
                dangling_ids = [line.strip() for line in dangling.stdout.splitlines() if line.strip()]
                for image_id in dangling_ids:
                    try:
                        containers = subprocess.run(
                            [docker_executable, "ps", "-a", "-q", "--filter", f"ancestor={image_id}"],
                            text=True,
                            capture_output=True,
                        )
                        container_ids = [line.strip() for line in containers.stdout.splitlines() if line.strip()]
                        if container_ids:
                            subprocess.run(
                                [docker_executable, "rm", "-f", *container_ids],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                    except Exception:  # pragma: no cover - best-effort cleanup
                        pass
                    result = subprocess.run(
                        [docker_executable, "image", "rm", "-f", image_id],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if result.returncode != 0:  # pragma: no cover - best-effort cleanup
                        logger.warning(f"Failed to remove dangling image {image_id} (code {result.returncode})")
            except Exception as cleanup_error:  # pragma: no cover - best-effort cleanup
                logger.warning(f"Error removing image {image_name}: {cleanup_error}")
        progress_manager.on_instance_end(task_id, exit_status)
        if is_vllm_model:
            release_router_program(str(instance_number))


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    cleanup_images: bool = typer.Option( False, "--cleanup-images", help="Remove SWEBench docker image after each instance", rich_help_panel="Advanced"),
    rollouts_per_instance: int = typer.Option(1, "--rollouts-per-instance", help="Repeat each instance up to N times (rollouts)", rich_help_panel="Data selection"),
    checkpoint_interval_steps: int = typer.Option(0, "--checkpoint-interval-steps", help="Save trajectory every N steps (0 disables)", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    rollouts_path = output_path / "preds_rollouts.json"
    single_preds_path = output_path / "preds.json"
    if not redo_existing and rollouts_per_instance <= 1 and single_preds_path.exists():
        existing_instances = list(json.loads(single_preds_path.read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]

    tasks: list[tuple[dict, int]] = []
    tasks_by_round: list[list[tuple[dict, int]]] = []
    if rollouts_per_instance <= 1:
        tasks = [(instance, 1) for instance in instances]
    else:
        max_rollouts = max(1, rollouts_per_instance)
        remaining_by_instance: dict[str, set[int]] = {}
        for instance in instances:
            instance_id = instance["instance_id"]
            completed = set() if redo_existing else get_completed_rollouts(rollouts_path, instance_id)
            remaining = {r for r in range(1, max_rollouts + 1) if redo_existing or r not in completed}
            if remaining:
                remaining_by_instance[instance_id] = remaining

        # Ordered pool scheduling: enqueue rollout 1 for all instances first, then
        # rollout 2, etc. (no barrier between rounds).
        for rollout_idx in range(1, max_rollouts + 1):
            round_tasks: list[tuple[dict, int]] = []
            for instance in instances:
                instance_id = instance["instance_id"]
                if rollout_idx in remaining_by_instance.get(instance_id, set()):
                    round_tasks.append((instance, rollout_idx))
            if round_tasks:
                tasks_by_round.append(round_tasks)

        tasks = [t for round_tasks in tasks_by_round for t in round_tasks]

    logger.info(
        f"Running {len(tasks)} tasks ({len(instances)} instances, up to {rollouts_per_instance} rollouts/instance)..."
    )

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(tasks), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            task_counter = 0

            def submit_task(instance: dict, rollout_idx: int) -> tuple[concurrent.futures.Future, str]:
                nonlocal task_counter
                task_counter += 1
                instance_id = instance["instance_id"]
                task_id = instance_id if rollouts_per_instance <= 1 else f"{instance_id}#r{rollout_idx:02d}"
                future = executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                    task_counter,
                    rollout_idx,
                    rollouts_per_instance,
                    checkpoint_interval_steps,
                    cleanup_images,
                )
                return future, task_id

            if rollouts_per_instance <= 1:
                futures = dict(submit_task(instance, rollout_idx) for instance, rollout_idx in tasks)
                try:
                    process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    process_futures(futures)
            else:
                futures = dict(submit_task(instance, rollout_idx) for instance, rollout_idx in tasks)
                try:
                    process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    process_futures(futures)


if __name__ == "__main__":
    app()
