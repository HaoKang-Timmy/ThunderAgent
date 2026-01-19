#!/usr/bin/env python3

"""Prefetch SWE-bench docker images to speed up evaluation runs."""

from __future__ import annotations

import concurrent.futures
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterable

import typer
from datasets import load_dataset

app = typer.Typer(add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def get_swebench_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def filter_instances(
    instances: list[dict],
    *,
    filter_spec: str,
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[dict]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    if filter_spec:
        instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
    return instances


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def pull_image(docker_exe: str, image: str, *, timeout_s: float | None) -> tuple[str, int, str]:
    cmd = [docker_exe, "pull", image]
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        return image, completed.returncode, completed.stdout or ""
    except subprocess.TimeoutExpired:
        return image, 124, f"Timed out after {timeout_s}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return image, 127, f"docker executable not found: {docker_exe}"


@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset name or dataset path"),
    split: str = typer.Option("test", "--split", help="Dataset split (e.g., test/dev)"),
    count: int = typer.Option(300, "--count", help="How many instances to prefetch"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex (re.match)"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle before taking the first N instances"),
    docker: str = typer.Option("docker", "--docker", help="Docker executable"),
    workers: int = typer.Option(8, "--workers", help="Parallel docker pulls"),
    timeout_s: float | None = typer.Option(0.0, "--timeout-s", help="Per-image pull timeout (0 disables)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned docker pull commands only"),
) -> None:
    dataset_path = DATASET_MAPPING.get(subset, subset)
    typer.echo(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    slice_spec = f"0:{max(0, count)}" if count > 0 else ""
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    images = unique_in_order(get_swebench_docker_image_name(i) for i in instances)

    typer.echo(f"Instances selected: {len(instances)}")
    typer.echo(f"Unique images: {len(images)}")

    if dry_run:
        for image in images:
            typer.echo(f"{docker} pull {image}")
        return

    timeout_value = None if (timeout_s is None or timeout_s == 0) else timeout_s
    started = time.time()
    failures: list[tuple[str, int]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(pull_image, docker, image, timeout_s=timeout_value) for image in images]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            image, code, out = fut.result()
            status = "OK" if code == 0 else f"FAIL({code})"
            typer.echo(f"[{idx}/{len(images)}] {status} {image}")
            if code != 0:
                failures.append((image, code))
                if out:
                    sys.stdout.write(out.rstrip() + "\n")

    elapsed = time.time() - started
    typer.echo(f"Done in {elapsed:.1f}s")
    if failures:
        typer.echo(f"Failures: {len(failures)}")
        for image, code in failures:
            typer.echo(f"- {code} {image}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

