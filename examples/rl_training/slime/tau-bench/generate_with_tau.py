"""
Tau-Bench Integration for slime Training

This module provides the main interface for training agents in tau-bench environments
using the slime framework. It handles agent-environment interactions and converts
results to the format expected by slime's training pipeline.
"""

import logging
import json
import os
import fcntl
import uuid
from datetime import datetime, timezone
from typing import Any

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from slime.utils.http_utils import post
from slime.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)
PROGRAM_RELEASE_SUPPORTED = True
TAU_RELEASE_MAX_RETRIES = max(1, int(os.getenv("TAU_RELEASE_MAX_RETRIES", "10")))

# Tau-bench configuration
TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"]
    "user_model": "gemini-2.5-flash-lite",  # Cheap Model for user simulator
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router",  # Unused, required
    "model": "qwen3-4b",  # Unused, required
    "user_model_provider": "gemini",
}
# Read API key from environment to avoid hardcoding secrets in source.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is required for tau-bench user simulation. "
        "Please export GEMINI_API_KEY before running examples/tau-bench/run_qwen3_4B.sh"
    )
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
tau_config = RunConfig(**TAU_CONFIGS)


def _ensure_sample_metadata(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    return metadata


def get_sample_program_id(sample: Sample) -> str:
    """
    Build a unique program id per (prompt, sample).
    """
    metadata = _ensure_sample_metadata(sample)
    existing_id = metadata.get("program_id")
    if existing_id:
        return str(existing_id)

    prompt_part = sample.prompt if sample.prompt is not None else "na"
    group_part = sample.group_index if sample.group_index is not None else "na"
    sample_part = sample.index if sample.index is not None else "na"
    unique_part = uuid.uuid4().hex[:8]
    program_id = f"prompt-{prompt_part}-group-{group_part}-sample-{sample_part}-{unique_part}"
    metadata["program_id"] = program_id
    return program_id


async def release_program(args: dict[str, Any], sample: Sample, reason: str) -> None:
    """
    Explicitly release program state from ThunderAgent when a sample ends.
    """
    global PROGRAM_RELEASE_SUPPORTED
    if not PROGRAM_RELEASE_SUPPORTED:
        return

    if not getattr(args, "use_slime_router", False):
        # /programs/release is expected in the ThunderAgent path.
        return

    metadata = _ensure_sample_metadata(sample)
    program_id = metadata.get("program_id")
    if not program_id:
        return

    if metadata.get("program_released"):
        return

    release_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/programs/release"
    try:
        resp = await post(release_url, {"program_id": str(program_id)}, max_retries=TAU_RELEASE_MAX_RETRIES)
        metadata["program_released"] = True
        metadata["program_release_resp"] = resp
        metadata["program_release_reason"] = reason
    except Exception as exc:
        if "404" in str(exc):
            PROGRAM_RELEASE_SUPPORTED = False
        metadata["program_release_error"] = str(exc)
        metadata["program_release_reason"] = reason


def _resolve_traj_json_dir(args: dict[str, Any]) -> str | None:
    path = os.getenv("TAU_TRAJ_JSON_DIR", "").strip()
    if path:
        return path

    path_template = getattr(args, "save_debug_rollout_data", None)
    if path_template:
        try:
            sample_path = path_template.format(rollout_id=0)
        except Exception:
            sample_path = path_template
        return os.path.join(os.path.dirname(sample_path), "samples_json")
    return None


def _resolve_step_timing_jsonl_path(args: dict[str, Any]) -> str | None:
    path = os.getenv("TAU_STEP_TIMING_PATH", "").strip()
    if path:
        return path

    traj_dir = _resolve_traj_json_dir(args)
    if traj_dir:
        return os.path.join(traj_dir, "step_timing.jsonl")

    path_template = getattr(args, "save_debug_rollout_data", None)
    if path_template:
        try:
            sample_path = path_template.format(rollout_id=0)
        except Exception:
            sample_path = path_template
        return os.path.join(os.path.dirname(sample_path), "step_timing.jsonl")
    return None


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(row, ensure_ascii=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line + "\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _dump_step_timing_jsonl(args: dict[str, Any], sample: Sample, res: InteractionResult, task_index: int) -> None:
    path = _resolve_step_timing_jsonl_path(args)
    if path is None:
        return

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_index": task_index,
        "sample_index": sample.index,
        "sample_group_index": sample.group_index,
        "sample_prompt": sample.prompt,
        "status": res.status.value if hasattr(res.status, "value") else str(res.status),
        "reward": float(res.reward),
        "step_timing": res.info.get("tau_step_timing", []),
        "timing_summary": res.info.get("tau_timing_summary", {}),
    }

    try:
        _append_jsonl(path, row)
    except Exception as e:
        logger.warning(f"Failed to dump step timing to jsonl at {path}: {e}")


def _dump_sample_traj_json(
    args: dict[str, Any], sample: Sample, res: InteractionResult, result_sample: Sample, task_index: int
) -> None:
    dump_dir = _resolve_traj_json_dir(args)
    if dump_dir is None:
        return

    os.makedirs(dump_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    group_idx = sample.group_index if sample.group_index is not None else "na"
    sample_idx = sample.index if sample.index is not None else "na"
    file_name = f"task_{task_index}_group_{group_idx}_sample_{sample_idx}_{ts}_pid{os.getpid()}.json"
    file_path = os.path.join(dump_dir, file_name)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_index": task_index,
        "sample_index": sample.index,
        "sample_group_index": sample.group_index,
        "sample_prompt": sample.prompt,
        "status": res.status.value if hasattr(res.status, "value") else str(res.status),
        "reward": float(res.reward),
        "response_length": getattr(result_sample, "response_length", None),
        "token_count": len(res.tokens) if isinstance(res.tokens, list) else 0,
        "loss_mask_length": len(res.loss_mask) if isinstance(res.loss_mask, list) else 0,
        "prompt_text": res.prompt,
        "response_text": res.response,
        "messages": res.messages,
        "step_timing": res.info.get("tau_step_timing", []),
        "timing_summary": res.info.get("tau_timing_summary", {}),
        "info": res.info,
    }

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception as e:
        logger.warning(f"Failed to dump sample traj json at {file_path}: {e}")


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    This function transforms the tau-bench interaction result into the format
    expected by slime's training pipeline, handling status mapping and response
    length calculation.

    Args:
        res: InteractionResult from tau-bench agent
        task_index: Index of the task being processed

    Returns:
        Sample object for slime training
    """
    # Map tau-bench status to slime status
    status_mapping = {
        Status.COMPLETED: Sample.Status.COMPLETED,
        Status.TRUNCATED: Sample.Status.TRUNCATED,
        Status.ABORTED: Sample.Status.ABORTED,
    }
    status = status_mapping.get(res.status, Sample.Status.PENDING)

    # Debug logging for response tracking
    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    # Create sample with basic information
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
    )

    # Ensure response_length is set correctly
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        # Fallback: calculate from loss_mask if available
        if res.loss_mask:
            # loss_mask only contains response part, so length equals response_length
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            # If no loss_mask available, use total tokens as fallback
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for tau-bench.

    This is the main entry point for slime training. It creates a tau-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)
    program_id = get_sample_program_id(sample)
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    # Initialize tau-bench environment
    env = get_env(
        env_name=tau_config.env,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
        task_index=task_index,
    )

    # Create trainable agent
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute one independent sample-program and explicitly release it on finish.
    interaction_result = None
    release_reason = "sample_exception"
    try:
        # Note: sample.prompt contains the task index for repeatability.
        interaction_result = await agent.asolve(
            env,
            agent.rollout_args,
            agent.sampling_params,
            task_index,
            program_id=program_id,
        )
        status_value = interaction_result.status.value if hasattr(interaction_result.status, "value") else "completed"
        release_reason = f"sample_{status_value}"
    finally:
        await release_program(args, sample, reason=release_reason)

    if interaction_result is None:
        raise RuntimeError("Agent interaction failed before producing InteractionResult.")

    # Persist program/release info for debugging and later analysis.
    interaction_info = dict(interaction_result.info or {})
    interaction_info["program_id"] = program_id
    metadata = _ensure_sample_metadata(sample)
    for key in (
        "program_released",
        "program_release_resp",
        "program_release_reason",
        "program_release_error",
    ):
        if key in metadata:
            interaction_info[key] = metadata[key]
    interaction_result.info = interaction_info

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, task_index)
    _dump_step_timing_jsonl(args, sample, interaction_result, task_index)
    _dump_sample_traj_json(args, sample, interaction_result, result_sample, task_index)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result_sample
