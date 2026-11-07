# SPDX-License-Identifier: Apache-2.0
"""MedQA final-hidden MSE runner using generation plus prefill replay."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import common

TESTS_DIR = common.ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import heal_policy  # noqa: E402


@dataclass(frozen=True)
class ModeSpec:
    name: str
    version: str
    dtype: str
    tf32: bool = False
    significand_bits: int | None = None
    heal_mode: str | None = None
    setting: str | None = None
    attention_backend: str | None = None


@dataclass(frozen=True)
class GenerationArtifact:
    directory: Path
    generated_path: Path
    replay_manifest_path: Path
    replay_positions_path: Path
    result_path: Path
    hidden_dir: Path


GENERATION_SPEC = ModeSpec(
    name="reference_layercast_fp32_generate",
    version="VLLM_LAYERCAST",
    dtype="float32",
    tf32=False,
    setting="layercast_fp32_generate",
    attention_backend="TRITON_ATTN",
)

GENERATION_BATCH_SIZE_CAPS_BY_MODEL = {
    # Qwen3-32B FP32 weights leave too little headroom for vLLM's profile
    # dummy run when all 200 prompts are profiled together.
    "Qwen/Qwen3-32B": 16,
}

PREFILL_GPU_MEMORY_UTILIZATION_OVERRIDES_BY_MODEL_VERSION = {
    # Significand boundary rounding adds enough transient memory pressure on
    # Qwen3-32B that the default replay utilization leaves no KV cache blocks.
    ("Qwen/Qwen3-32B", "VLLM_SIGNIFICAND_BITS"): 0.91,
}

PREFILL_BATCH_SIZE_OVERRIDES_BY_MODEL_MODE = {
    ("Qwen/Qwen3-32B", "heal_full_prefill"): 1,
    ("Qwen/Qwen3-32B", "heal_full_plus_prefill"): 1,
}

PREFILL_TOKEN_BUDGET_OVERRIDES_BY_MODEL_MODE = {
    ("Qwen/Qwen3-32B", "heal_full_prefill"): 1024,
    ("Qwen/Qwen3-32B", "heal_full_plus_prefill"): 1024,
}

PREFILL_MAX_MODEL_LEN_OVERRIDES_BY_MODEL_MODE = {
    ("Qwen/Qwen3-32B", "heal_full_prefill"): "auto",
    ("Qwen/Qwen3-32B", "heal_full_plus_prefill"): "auto",
}

HEAL_DYNAMIC_INT8_GEMM_WEIGHT_BY_MODEL_MODE: dict[tuple[str, str], bool] = {}
HEAL_FP16_GEMM_COMPENSATION_BY_MODEL_MODE: dict[tuple[str, str], bool] = {}

PREFILL_SPECS: list[ModeSpec] = [
    ModeSpec(
        name="layercast_fp32_prefill_ref",
        version="VLLM_LAYERCAST",
        dtype="float32",
        tf32=False,
        setting="layercast_fp32_ref",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="original_fp16_prefill",
        version="VLLM_ORIGINAL",
        dtype="float16",
        setting="native_fp16",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="original_bf16_prefill",
        version="VLLM_ORIGINAL",
        dtype="bfloat16",
        setting="native_bf16",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="layercast_fp32_prefill",
        version="VLLM_LAYERCAST",
        dtype="float32",
        tf32=False,
        setting="layercast_fp32",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="layercast_tf32_prefill",
        version="VLLM_LAYERCAST",
        dtype="float32",
        tf32=True,
        setting="layercast_tf32",
        attention_backend="TRITON_ATTN",
    ),
] + [
    ModeSpec(
        name=f"significand_k{k:02d}_prefill",
        version="VLLM_SIGNIFICAND_BITS",
        dtype="float32",
        tf32=False,
        significand_bits=k,
        setting=f"mantissa_k{k}",
        attention_backend="TRITON_ATTN",
    )
    for k in range(3, 24)
] + [
    ModeSpec(
        name=f"heal_{mode}_prefill",
        version="VLLM_HEAL",
        dtype="float16",
        tf32=False,
        heal_mode=mode,
        setting=f"heal_{mode}",
        attention_backend="TRITON_ATTN",
    )
    for mode in ("base", "plus", "full")
] + [
    ModeSpec(
        name="heal_plus_plus_prefill",
        version="VLLM_HEAL",
        dtype="float16",
        tf32=False,
        heal_mode="plus",
        setting="heal_plus_plus",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="heal_full_plus_prefill",
        version="VLLM_HEAL",
        dtype="float16",
        tf32=False,
        heal_mode="full",
        setting="heal_full_plus",
        attention_backend="TRITON_ATTN",
    ),
]

PREFILL_SPECS_BY_NAME = {spec.name: spec for spec in PREFILL_SPECS}
PATCH_ORDER = (
    "VLLM_LAYERCAST",
    "VLLM_ORIGINAL",
    "VLLM_SIGNIFICAND_BITS",
    "VLLM_HEAL",
)

METRIC_FIELDS = [
    "model",
    "model_arg",
    "mode",
    "version",
    "dtype",
    "attention_backend",
    "tf32",
    "significand_bits",
    "setting",
    "status",
    "wall_time_s",
    "worker_elapsed_s",
    "mse_compute_elapsed_s",
    "generated_tokens",
    "replay_generated_tokens",
    "truncated_generated_tokens",
    "limit_truncated_generated_tokens",
    "self_circulation_docs",
    "self_circulation_dropped_tokens",
    "replay_tokens",
    "docs",
    "batches",
    "compared_hidden_rows",
    "hidden_size",
    "mse",
    "mae",
    "max_abs",
    "token_top1_rows",
    "top1_flip_count",
    "top1_flip_rate",
    "top1_expected_mismatch_count",
    "top1_expected_mismatch_rate",
    "ref_top1_flip_count",
    "ref_top1_flip_rate",
    "token_top1_file",
    "top1_flip_file",
    "ref_top1_flip_file",
    "std_hidden_mse",
    "median_hidden_mse",
    "p90_hidden_mse",
    "p95_hidden_mse",
    "p99_hidden_mse",
    "max_hidden_mse",
    "token_mse_file",
    "sample_mse_file",
    "output_dir",
    "data_dir",
    "patch_sha256_before",
    "patch_sha256_after",
    "env_settings_json",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ModeSpec):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def git_output(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def vllm_git_sha() -> str:
    return git_output(["git", "-C", str(common.VLLM_DIR), "rev-parse", "HEAD"])


def vllm_git_branch() -> str:
    return git_output(
        ["git", "-C", str(common.VLLM_DIR), "rev-parse", "--abbrev-ref", "HEAD"]
    )


def active_patch_name() -> str | None:
    status = common.active_patch_status()
    first = status.splitlines()[0] if status.splitlines() else ""
    if first.strip() == "Active patch: none":
        return None
    prefix = "Active patch:"
    if first.startswith(prefix):
        return first[len(prefix) :].strip()
    return first.strip() or None


def ensure_no_active_patch() -> None:
    active = active_patch_name()
    if active is not None:
        raise RuntimeError(f"vLLM already has an active patch: {active}")


def ensure_active_patch(version: str) -> None:
    active = active_patch_name()
    if active != version:
        raise RuntimeError(f"expected active vLLM patch {version}, found {active}")


def revert_active_patch_best_effort() -> None:
    try:
        if active_patch_name() is not None:
            log("reverting active vLLM patch")
            common.run_patch_command("revert-version")
    except Exception as exc:  # pragma: no cover - recovery path
        log(f"WARNING: failed to revert active patch automatically: {exc}")


def model_args_from_cli(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(common.MODEL_ORDER)
    if args.models:
        return args.models
    return [args.model]


def path_from_cli(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return common.ROOT / path


def _subset_selection_from_file(path: Path | None) -> dict[str, set[Any]] | None:
    if path is None:
        return None
    rows = common.read_jsonl(path_from_cli(path))
    selection: dict[str, set[Any]] = {"id": set(), "subset_pos": set(), "original_index": set()}
    for index, row in enumerate(rows):
        if row.get("id") is not None:
            selection["id"].add(str(row["id"]))
        if row.get("subset_pos") is not None:
            try:
                selection["subset_pos"].add(int(row["subset_pos"]))
            except (TypeError, ValueError):
                pass
        if row.get("error_contribution_source_subset_pos") is not None:
            try:
                selection["subset_pos"].add(int(row["error_contribution_source_subset_pos"]))
            except (TypeError, ValueError):
                pass
        if row.get("original_index") is not None:
            try:
                selection["original_index"].add(int(row["original_index"]))
            except (TypeError, ValueError):
                pass
        if row.get("id") is None and row.get("subset_pos") is None and row.get("original_index") is None:
            selection["subset_pos"].add(index)
    return selection


def _row_matches_subset_selection(row: dict[str, Any], selection: dict[str, set[Any]] | None) -> bool:
    if selection is None:
        return True
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    for container in (row, source):
        if container.get("id") is not None and str(container["id"]) in selection["id"]:
            return True
        if container.get("subset_pos") is not None:
            try:
                if int(container["subset_pos"]) in selection["subset_pos"]:
                    return True
            except (TypeError, ValueError):
                pass
        if container.get("original_index") is not None:
            try:
                if int(container["original_index"]) in selection["original_index"]:
                    return True
            except (TypeError, ValueError):
                pass
    if row.get("subset_pos") is None and source.get("subset_pos") is None and row.get("doc_id") is not None:
        try:
            return int(row["doc_id"]) in selection["subset_pos"]
        except (TypeError, ValueError):
            return False
    return False


def filter_rows_by_subset_file(rows: list[dict[str, Any]], subset_file: Path | None) -> list[dict[str, Any]]:
    selection = _subset_selection_from_file(subset_file)
    if selection is None:
        return rows
    filtered = [row for row in rows if _row_matches_subset_selection(row, selection)]
    if not filtered:
        raise RuntimeError(f"subset filter {path_from_cli(subset_file)} matched no generation rows")
    return filtered


def multiple_models_requested(args: argparse.Namespace) -> bool:
    return bool(args.all_models or (args.models is not None and len(args.models) > 1))


def generation_artifact_for_model(
    args: argparse.Namespace,
    *,
    run_root: Path,
    model_label: str,
) -> GenerationArtifact:
    if args.generation_dir is None:
        directory = run_root.parent / "generation_artifacts" / model_label
    else:
        base_dir = path_from_cli(args.generation_dir)
        if multiple_models_requested(args):
            directory = base_dir / model_label
        elif (
            args.stage == "replay"
            and not (base_dir / "replay_manifest.jsonl").exists()
            and not (base_dir / "generated_tokens.jsonl").exists()
            and (base_dir / model_label).exists()
        ):
            directory = base_dir / model_label
        else:
            directory = base_dir
    return GenerationArtifact(
        directory=directory,
        generated_path=directory / "generated_tokens.jsonl",
        replay_manifest_path=directory / "replay_manifest.jsonl",
        replay_positions_path=directory / "replay_positions.csv",
        result_path=directory / "worker_result.json",
        hidden_dir=directory / "hidden_states",
    )


def generation_batch_size_for_model(args: argparse.Namespace, model_id: str) -> int:
    batch_size = int(args.generation_batch_size)
    cap = GENERATION_BATCH_SIZE_CAPS_BY_MODEL.get(model_id)
    if cap is None:
        return batch_size
    return min(batch_size, cap)


def replay_gpu_memory_utilization_for_mode(
    args: argparse.Namespace,
    model_id: str,
    spec: ModeSpec,
) -> float:
    override = PREFILL_GPU_MEMORY_UTILIZATION_OVERRIDES_BY_MODEL_VERSION.get(
        (model_id, spec.version)
    )
    if override is not None:
        return override
    return float(args.replay_gpu_memory_utilization)


def prefill_batch_size_for_mode(
    args: argparse.Namespace,
    model_id: str,
    spec: ModeSpec,
) -> int:
    return int(
        PREFILL_BATCH_SIZE_OVERRIDES_BY_MODEL_MODE.get(
            (model_id, spec.name), args.prefill_batch_size
        )
    )


def prefill_token_budget_for_mode(
    args: argparse.Namespace,
    model_id: str,
    spec: ModeSpec,
) -> int:
    return int(
        PREFILL_TOKEN_BUDGET_OVERRIDES_BY_MODEL_MODE.get(
            (model_id, spec.name), args.prefill_token_budget
        )
    )


def prefill_max_model_len_for_mode(
    args: argparse.Namespace,
    model_id: str,
    spec: ModeSpec,
) -> str:
    return str(
        PREFILL_MAX_MODEL_LEN_OVERRIDES_BY_MODEL_MODE.get(
            (model_id, spec.name), args.max_model_len
        )
    )


def heal_dynamic_int8_gemm_weight_for_mode(
    model_id: str,
    spec: ModeSpec,
) -> bool:
    return bool(
        HEAL_DYNAMIC_INT8_GEMM_WEIGHT_BY_MODEL_MODE.get((model_id, spec.name), False)
    )


def heal_fp16_gemm_compensation_for_mode(
    model_id: str,
    spec: ModeSpec,
) -> bool:
    return spec.heal_mode == "full" or bool(
        HEAL_FP16_GEMM_COMPENSATION_BY_MODEL_MODE.get((model_id, spec.name), False)
    )


def spec_mode_dict(spec: ModeSpec, mode_dir: Path) -> dict[str, Any]:
    payload = asdict(spec)
    payload["mode_dir"] = str(mode_dir)
    return payload


def mode_env_settings(mode: dict[str, Any]) -> dict[str, str]:
    env = common.configure_precision_env({}, mode)
    keep = [
        "VLLM_VERSION_NAME",
        "VLLM_EXPERIMENT_OUTPUT_DIR",
        "VLLM_DISABLE_CUDA_GRAPH",
        "VLLM_USE_V1",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "VLLM_FLOAT32_MATMUL_PRECISION",
        "VLLM_LAYERCAST_FORCE_FP32",
        "VLLM_DISABLE_COMPILE_CACHE",
        "VLLM_SIGNIFICAND_BITS",
        "VLLM_SIGNIFICAND_BOUNDARY_SET",
        "HEAL_MODE",
        "HEAL_SKIP_QKV_QUANT_LAYERS",
        "HEAL_SKIP_INT8_COMP_LAYERS",
        "HEAL_SKIP_FP16_COMP_LAYERS",
        "HEAL_ALLOW_TRITON_FP32_GEMM",
        "HEAL_FP16_GEMM_COMPENSATION_CHUNK_ROWS",
        "HEAL_INT8_GEMM_COMPENSATION",
        "HEAL_DYNAMIC_INT8_GEMM_WEIGHT",
        "HEAL_DYNAMIC_INT8_GEMM_WEIGHT_CHUNK_ROWS",
        "HEAL_FP16_GEMM_COMPENSATION",
        "HEAL_INT8_GEMM_STATS_FILE",
    ]
    return {key: env[key] for key in keep if key in env}


def resolve_max_model_len(setting: str, required: int) -> int:
    if setting == "auto":
        return required
    value = int(setting)
    if value < required:
        raise ValueError(f"--max-model-len={value} is smaller than required {required}")
    return value


def llm_kwargs(
    *,
    model_id: str,
    mode: dict[str, Any],
    seed: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    trust_remote_code: bool,
    max_logprobs: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": mode["dtype"],
        "seed": seed,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "enforce_eager": enforce_eager,
        "trust_remote_code": trust_remote_code,
        "disable_log_stats": True,
    }
    if mode.get("attention_backend"):
        kwargs["attention_backend"] = mode["attention_backend"]
    if max_logprobs is not None:
        kwargs["max_logprobs"] = int(max_logprobs)
    return kwargs


SELF_CIRCULATION_WINDOWS = (128, 256, 512, 1024, 2048, 4096)
SELF_CIRCULATION_MIN_REPEATS = 3
SELF_CIRCULATION_MIN_DROP_TOKENS = 32


def kmp_failure_array(values: list[int]) -> list[int]:
    failure = [0] * len(values)
    matched = 0
    for idx in range(1, len(values)):
        while matched > 0 and values[idx] != values[matched]:
            matched = failure[matched - 1]
        if values[idx] == values[matched]:
            matched += 1
            failure[idx] = matched
    return failure


def detect_self_circulation_tail(token_ids: list[int]) -> dict[str, int | bool]:
    """Find an exact repeated suffix and keep only two periods.

    We run KMP on suffix windows ending at the truncation point. For a suffix of
    length X, KMP gives a failure value F, so X - F is the candidate period.
    Once a period is found, walk backward while the same period still matches.
    """

    token_count = len(token_ids)
    best: dict[str, int | bool] | None = None
    for window in SELF_CIRCULATION_WINDOWS:
        if window > token_count:
            continue
        suffix = token_ids[token_count - window :]
        failure = kmp_failure_array(suffix)
        failure_value = failure[-1] if failure else 0
        period = window - failure_value
        if period <= 0 or period >= window:
            continue
        if window < period * 2:
            continue
        if any(suffix[idx] != suffix[idx - period] for idx in range(period, window)):
            continue

        cycle_start = token_count - window
        while cycle_start > 0 and token_ids[cycle_start - 1] == token_ids[
            cycle_start - 1 + period
        ]:
            cycle_start -= 1
        cycle_len = token_count - cycle_start
        trim_to = cycle_start + 2 * period
        dropped = token_count - trim_to
        if cycle_len < period * SELF_CIRCULATION_MIN_REPEATS:
            continue
        if dropped < SELF_CIRCULATION_MIN_DROP_TOKENS:
            continue

        candidate: dict[str, int | bool] = {
            "trimmed": True,
            "window": window,
            "failure_value": failure_value,
            "period": period,
            "cycle_start": cycle_start,
            "cycle_length": cycle_len,
            "trimmed_token_count": trim_to,
            "dropped_token_count": dropped,
        }
        if best is None or int(candidate["dropped_token_count"]) > int(
            best["dropped_token_count"]
        ):
            best = candidate

    if best is not None:
        return best
    return {
        "trimmed": False,
        "window": 0,
        "failure_value": 0,
        "period": 0,
        "cycle_start": 0,
        "cycle_length": 0,
        "trimmed_token_count": token_count,
        "dropped_token_count": 0,
    }


def replay_row_summary(replay_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "generated_tokens": sum(
            int(row.get("full_generated_token_count", row["generated_token_count"]))
            for row in replay_rows
        ),
        "replay_generated_tokens": sum(
            int(row["generated_token_count"]) for row in replay_rows
        ),
        "truncated_generated_tokens": sum(
            int(row.get("truncated_generated_token_count", 0)) for row in replay_rows
        ),
        "limit_truncated_generated_tokens": sum(
            int(row.get("limit_truncated_generated_token_count", 0))
            for row in replay_rows
        ),
        "self_circulation_docs": sum(
            1 for row in replay_rows if row.get("self_circulation_trimmed")
        ),
        "self_circulation_dropped_tokens": sum(
            int(row.get("self_circulation_dropped_token_count", 0))
            for row in replay_rows
        ),
        "prompt_tokens": sum(int(row["prompt_token_count"]) for row in replay_rows),
        "replay_tokens": sum(int(row["replay_token_count"]) for row in replay_rows),
    }


def _logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def _logprob_decoded_token(value: Any) -> str | None:
    if hasattr(value, "decoded_token"):
        decoded = value.decoded_token
        return None if decoded is None else str(decoded)
    if isinstance(value, dict) and "decoded_token" in value:
        decoded = value["decoded_token"]
        return None if decoded is None else str(decoded)
    return None


def _topk_from_prompt_logprob(entry: Any, top_k: int) -> list[tuple[int, float, float, str | None]]:
    if not entry:
        return []
    values: list[tuple[int, float, float, str | None]] = []
    for token_id, value in entry.items():
        logprob = _logprob_value(value)
        values.append(
            (
                int(token_id),
                logprob,
                math.exp(logprob),
                _logprob_decoded_token(value),
            )
        )
    values.sort(key=lambda item: item[1], reverse=True)
    return values[:top_k]


def record_replay_token_topk(
    *,
    batch_rows: list[dict[str, Any]],
    outputs: list[Any],
    mode_name: str,
    batch_id: int,
    top_k: int,
    token_ids_tensor: Any,
    probs_tensor: Any,
    doc_ids_tensor: Any,
    token_pos_tensor: Any,
    expected_tensor: Any,
    cursor: int,
) -> tuple[int, list[dict[str, Any]]]:
    top1_rows: list[dict[str, Any]] = []
    for row, output in zip(batch_rows, outputs, strict=True):
        prompt_logprobs = output.prompt_logprobs or []
        prompt_tokens = int(row["prompt_token_count"])
        generated_ids = [int(token_id) for token_id in row["generated_token_ids"]]
        missing = 0
        for token_pos, expected_token_id in enumerate(generated_ids):
            logprob_index = prompt_tokens + token_pos
            entry = (
                prompt_logprobs[logprob_index]
                if logprob_index < len(prompt_logprobs)
                else None
            )
            topk = _topk_from_prompt_logprob(entry, top_k)
            if not topk:
                missing += 1
            for rank, (token_id, _logprob, prob, _decoded) in enumerate(topk):
                token_ids_tensor[cursor, rank] = token_id
                probs_tensor[cursor, rank] = prob
            top1_token_id = topk[0][0] if topk else None
            top1_logprob = topk[0][1] if topk else None
            top1_decoded = topk[0][3] if topk else None
            doc_ids_tensor[cursor] = int(row["doc_id"])
            token_pos_tensor[cursor] = token_pos
            expected_tensor[cursor] = expected_token_id
            top1_rows.append(
                {
                    "doc_id": int(row["doc_id"]),
                    "subset_pos": int(row["subset_pos"]),
                    "original_index": int(row["original_index"]),
                    "mode": mode_name,
                    "batch_id": batch_id,
                    "token_pos": token_pos,
                    "absolute_token_pos": prompt_tokens - 1 + token_pos,
                    "prompt_logprob_index": logprob_index,
                    "expected_token_id": expected_token_id,
                    "top1_token_id": top1_token_id,
                    "top1_logprob": top1_logprob,
                    "top1_prob": topk[0][2] if topk else None,
                    "top1_decoded_token": top1_decoded,
                    "top1_matches_expected": top1_token_id == expected_token_id,
                    "generated_token_count": len(generated_ids),
                    "replay_token_count": int(row["replay_token_count"]),
                }
            )
            cursor += 1
        if missing:
            log(
                f"WARNING: mode={mode_name} batch={batch_id} doc={int(row['doc_id'])} "
                f"missing prompt top-k rows={missing}"
            )
    return cursor, top1_rows


def summarize_token_top1(path: Path, *, flip_path: Path | None = None) -> dict[str, Any]:
    rows = 0
    expected_mismatch_count = 0
    flip_handle = None
    if flip_path is not None:
        flip_path.parent.mkdir(parents=True, exist_ok=True)
        flip_handle = flip_path.open("w", encoding="utf-8")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                row = json.loads(line)
                if row.get("top1_token_id") != row.get("expected_token_id"):
                    expected_mismatch_count += 1
                    if flip_handle is not None:
                        flip_handle.write(line)
    finally:
        if flip_handle is not None:
            flip_handle.close()
    rate = expected_mismatch_count / rows if rows else None
    return {
        "token_top1_rows": rows,
        "top1_flip_count": expected_mismatch_count,
        "top1_flip_rate": rate,
        "top1_expected_mismatch_count": expected_mismatch_count,
        "top1_expected_mismatch_rate": rate,
        "token_top1_file": str(path),
        "top1_flip_file": str(flip_path) if flip_path is not None else "",
    }


def compare_token_top1(
    *,
    reference_path: Path,
    candidate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    compared = 0
    flip_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with reference_path.open("r", encoding="utf-8") as ref_handle, (
        candidate_path.open("r", encoding="utf-8")
    ) as cand_handle, output_path.open("w", encoding="utf-8") as out_handle:
        for ref_line, cand_line in zip(ref_handle, cand_handle, strict=True):
            if not ref_line.strip() and not cand_line.strip():
                continue
            ref = json.loads(ref_line)
            cand = json.loads(cand_line)
            ref_key = (int(ref["doc_id"]), int(ref["token_pos"]))
            cand_key = (int(cand["doc_id"]), int(cand["token_pos"]))
            if ref_key != cand_key:
                raise RuntimeError(
                    "token top1 alignment mismatch: "
                    f"reference={ref_key} candidate={cand_key}"
                )
            compared += 1
            if ref.get("top1_token_id") == cand.get("top1_token_id"):
                continue
            flip_count += 1
            out_handle.write(
                json.dumps(
                    {
                        "doc_id": cand_key[0],
                        "token_pos": cand_key[1],
                        "reference_top1_token_id": ref.get("top1_token_id"),
                        "candidate_top1_token_id": cand.get("top1_token_id"),
                        "expected_token_id": cand.get("expected_token_id"),
                        "reference_top1_decoded_token": ref.get(
                            "top1_decoded_token"
                        ),
                        "candidate_top1_decoded_token": cand.get(
                            "top1_decoded_token"
                        ),
                        "reference_top1_logprob": ref.get("top1_logprob"),
                        "candidate_top1_logprob": cand.get("top1_logprob"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "ref_top1_flip_count": flip_count,
        "ref_top1_flip_rate": flip_count / compared if compared else None,
        "ref_top1_flip_file": str(output_path),
    }


def shutdown_llm_and_distributed(llm: Any) -> None:
    try:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
    except Exception as exc:
        log(f"WARNING: vLLM engine shutdown failed: {exc}")
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception as exc:
        log(f"WARNING: vLLM distributed cleanup failed: {exc}")
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as exc:
        log(f"WARNING: torch distributed cleanup failed: {exc}")


def build_replay_rows_from_generated(
    generated_rows: list[dict[str, Any]],
    *,
    replay_generated_token_limit: int | None,
    trim_self_circulation: bool = True,
) -> list[dict[str, Any]]:
    replay_rows = []
    for fallback_doc_id, row in enumerate(generated_rows):
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        doc_id = int(row.get("doc_id", source.get("doc_id", fallback_doc_id)))
        prompt_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
        full_generated_ids = [int(token_id) for token_id in row["generated_token_ids"]]
        if replay_generated_token_limit is not None and replay_generated_token_limit > 0:
            limited_generated_ids = full_generated_ids[:replay_generated_token_limit]
        else:
            limited_generated_ids = full_generated_ids

        circulation = (
            detect_self_circulation_tail(limited_generated_ids)
            if trim_self_circulation
            else {
                "trimmed": False,
                "window": 0,
                "failure_value": 0,
                "period": 0,
                "cycle_start": 0,
                "cycle_length": 0,
                "trimmed_token_count": len(limited_generated_ids),
                "dropped_token_count": 0,
            }
        )
        generated_ids = limited_generated_ids[
            : int(circulation["trimmed_token_count"])
        ]
        limit_truncated = len(full_generated_ids) - len(limited_generated_ids)
        circulation_dropped = int(circulation["dropped_token_count"])
        original_finish_reason = row.get("finish_reason")
        finish_reason = original_finish_reason
        if bool(circulation["trimmed"]):
            finish_reason = "self_circulation_trimmed"
        elif limit_truncated > 0:
            finish_reason = "replay_truncated"
        replay_rows.append(
            {
                "doc_id": doc_id,
                "subset_pos": int(row.get("subset_pos", source.get("subset_pos", doc_id))),
                "original_index": int(
                    row.get("original_index", source.get("original_index", doc_id))
                ),
                "id": row.get("id", source.get("id")),
                "request_id": (
                    str(row["request_id"]) if row.get("request_id") is not None else None
                ),
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "replay_token_ids": prompt_ids + generated_ids,
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(generated_ids),
                "full_generated_token_count": len(full_generated_ids),
                "limit_truncated_generated_token_count": limit_truncated,
                "self_circulation_dropped_token_count": circulation_dropped,
                "truncated_generated_token_count": limit_truncated
                + circulation_dropped,
                "replay_generated_token_limit": replay_generated_token_limit or 0,
                "replay_token_count": len(prompt_ids) + len(generated_ids),
                "finish_reason": finish_reason,
                "original_finish_reason": original_finish_reason,
                "stop_reason": row.get("stop_reason"),
                "self_circulation_trimmed": bool(circulation["trimmed"]),
                "self_circulation_window": int(circulation["window"]),
                "self_circulation_failure_value": int(circulation["failure_value"]),
                "self_circulation_period": int(circulation["period"]),
                "self_circulation_cycle_start": int(circulation["cycle_start"]),
                "self_circulation_cycle_length": int(circulation["cycle_length"]),
            }
        )
    return replay_rows


def materialize_generation_from_source(
    *,
    args: argparse.Namespace,
    model_label: str,
    gen_dir: Path,
    replay_manifest: Path,
    generated_path: Path,
    result_path: Path,
    started: float,
) -> dict[str, Any]:
    source_path = Path(args.generated_source_jsonl)
    generated_rows = common.read_jsonl(source_path)
    if args.limit is not None:
        generated_rows = generated_rows[: args.limit]
    replay_rows = build_replay_rows_from_generated(
        generated_rows,
        replay_generated_token_limit=args.replay_generated_token_limit,
        trim_self_circulation=args.self_circulation_trim,
    )
    summary = replay_row_summary(replay_rows)
    common.write_jsonl(generated_path, generated_rows)
    common.write_jsonl(replay_manifest, replay_rows)
    common.write_jsonl(gen_dir / "generation_batches.jsonl", [])
    common.write_replay_positions(gen_dir / "replay_positions.csv", replay_rows)
    result = {
        "kind": "generate",
        "mode": GENERATION_SPEC.name,
        "model": model_label,
        "docs": len(replay_rows),
        "batches": 0,
        **summary,
        "replay_generated_token_limit": int(args.replay_generated_token_limit),
        "self_circulation_trim": bool(args.self_circulation_trim),
        "elapsed_s": time.perf_counter() - started,
        "generated_source_jsonl": str(source_path),
        "generated_path": str(generated_path),
        "replay_manifest_path": str(replay_manifest),
        "replay_positions_path": str(gen_dir / "replay_positions.csv"),
        "output_dir": str(gen_dir),
    }
    common.write_json(result_path, result)
    log(
        f"materialized generation from source={source_path} "
        f"docs={result['docs']} replay_generated={result['replay_generated_tokens']}"
    )
    return result


def worker_generate(config: dict[str, Any]) -> None:
    started = time.perf_counter()
    mode = config["mode"]
    os.environ.update(common.configure_precision_env(dict(os.environ), mode))
    common.apply_torch_precision(bool(mode.get("tf32")))
    common.add_local_vllm_to_path()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    dataset_kind = str(config.get("dataset_kind", "medqa"))
    subset_rows = common.load_subset(
        Path(config["subset_file"]),
        limit=config["limit"],
    )
    prompt_rows: list[dict[str, Any]] = []
    max_prompt_tokens = 0
    for doc_id, row in enumerate(subset_rows):
        messages = common.messages_for_row(
            row,
            dataset_kind=dataset_kind,
            max_source_chars=int(config.get("mcr_max_source_chars", 12000)),
        )
        prompt, prompt_ids = common.chat_prompt_and_ids(tokenizer, messages)
        source_metadata = (
            common.mcr_bench_source_metadata(row)
            if dataset_kind == "mcr-bench"
            else {}
        )
        max_prompt_tokens = max(max_prompt_tokens, len(prompt_ids))
        prompt_rows.append(
            {
                "doc_id": doc_id,
                "subset_pos": int(row.get("subset_pos", doc_id)),
                "original_index": int(row.get("original_index", doc_id)),
                "id": row.get("id"),
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "dataset_kind": dataset_kind,
                "source_row": row,
                "task": source_metadata.get("task"),
                "target": source_metadata.get("target"),
                "target_kind": source_metadata.get("target_kind"),
                "source_metadata": source_metadata,
            }
        )

    max_new_tokens = int(config["max_new_tokens"])
    generation_batch_size = int(config["generation_batch_size"])
    max_batch_prompt_tokens = max(
        sum(
            int(item["prompt_token_count"])
            for item in prompt_rows[start : start + generation_batch_size]
        )
        for start in range(0, len(prompt_rows), generation_batch_size)
    )
    max_model_len = resolve_max_model_len(
        str(config["max_model_len"]),
        max_prompt_tokens + max_new_tokens,
    )
    max_num_batched_tokens = max(
        int(config["generation_token_budget"]),
        max_prompt_tokens + 1,
        max_model_len,
        max_batch_prompt_tokens if bool(config["save_generation_hidden"]) else 0,
    )
    log(
        "worker generate loading "
        f"model={model_id} dtype={mode['dtype']} max_model_len={max_model_len} "
        f"max_num_batched_tokens={max_num_batched_tokens}"
    )
    load_start = time.perf_counter()
    llm = LLM(
        **llm_kwargs(
            model_id=model_id,
            mode=mode,
            seed=int(config["seed"]),
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=int(config["generation_batch_size"]),
            gpu_memory_utilization=float(config["gpu_memory_utilization"]),
            enforce_eager=bool(config["generation_enforce_eager"]),
            trust_remote_code=bool(config["trust_remote_code"]),
        )
    )
    common.clear_tensor_trace()
    load_elapsed = time.perf_counter() - load_start
    log(f"worker generate model loaded in {load_elapsed:.1f}s")

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    batch_size = generation_batch_size
    generated_rows: list[dict[str, Any]] = []
    trace_dir = Path(config["output_dir"]) / "trace_batches"
    hidden_dir = Path(config["output_dir"]) / "hidden_states"
    save_generation_hidden = bool(config["save_generation_hidden"])
    use_token_prompts = bool(config["generation_use_token_prompts"])
    keep_trace_batches = bool(config.get("keep_generation_trace_batches", False))
    batch_stats: list[dict[str, Any]] = []
    split_stats: list[dict[str, Any]] = []

    try:
        for start in range(0, len(prompt_rows), batch_size):
            batch_id = start // batch_size
            batch = prompt_rows[start : start + batch_size]
            if use_token_prompts:
                prompts = [
                    TokensPrompt(prompt_token_ids=item["prompt_token_ids"])
                    for item in batch
                ]
            else:
                prompts = [str(item["prompt"]) for item in batch]
            trace_path = trace_dir / f"batch_{batch_id:04d}.safetensors"
            batch_started = time.perf_counter()
            if save_generation_hidden:
                common.enable_tensor_trace(trace_path)
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            elapsed = time.perf_counter() - batch_started
            batch_generated = 0
            batch_generated_rows: list[dict[str, Any]] = []
            for item, output in zip(batch, outputs, strict=True):
                completion = output.outputs[0]
                token_ids = [int(token_id) for token_id in completion.token_ids]
                batch_generated += len(token_ids)
                row = {
                    "doc_id": int(item["doc_id"]),
                    "subset_pos": int(item["subset_pos"]),
                    "original_index": int(item["original_index"]),
                    "id": item["id"],
                    "request_id": str(output.request_id),
                    "prompt_token_ids": item["prompt_token_ids"],
                    "prompt_token_count": int(item["prompt_token_count"]),
                    "dataset_kind": item.get("dataset_kind"),
                    "source_row": item.get("source_row"),
                    "source_metadata": item.get("source_metadata"),
                    "task": item.get("task"),
                    "target": item.get("target"),
                    "target_kind": item.get("target_kind"),
                    "generated_token_ids": token_ids,
                    "generated_token_count": len(token_ids),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                }
                batch_generated_rows.append(row)
                generated_rows.append(row)
            if save_generation_hidden:
                common.flush_tensor_trace()
                if not trace_path.exists():
                    raise RuntimeError(
                        "generation tensor trace was not written; rerun with "
                        "--generation-enforce-eager so generation hidden can be "
                        "used as the MSE reference"
                    )
                batch_replay_rows = build_replay_rows_from_generated(
                    batch_generated_rows,
                    replay_generated_token_limit=int(
                        config["replay_generated_token_limit"]
                    ),
                    trim_self_circulation=bool(
                        config.get("self_circulation_trim", True)
                    ),
                )
                stats = common.split_generation_trace_to_doc_hidden(
                    trace_path=trace_path,
                    batch_rows=batch_replay_rows,
                    hidden_dir=hidden_dir,
                )
                stats.update(
                    {
                        "batch_id": batch_id,
                        "docs": len(batch),
                        "elapsed_s": elapsed,
                    }
                )
                split_stats.append(stats)
                common.clear_tensor_trace()
                if not keep_trace_batches:
                    trace_path.unlink(missing_ok=True)
                    trace_path.with_suffix(trace_path.suffix + ".meta.json").unlink(
                        missing_ok=True
                    )
            batch_stats.append(
                {
                    "batch_id": batch_id,
                    "docs": len(batch),
                    "elapsed_s": elapsed,
                    "generated_tokens": batch_generated,
                    "trace_path": str(trace_path) if save_generation_hidden else "",
                }
            )
            log(
                f"worker generate batch={batch_id} docs={len(batch)} "
                f"generated={batch_generated} elapsed={elapsed:.1f}s"
            )
    finally:
        shutdown_llm_and_distributed(llm)
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    replay_rows = build_replay_rows_from_generated(
        generated_rows,
        replay_generated_token_limit=int(config["replay_generated_token_limit"]),
        trim_self_circulation=bool(config.get("self_circulation_trim", True)),
    )
    summary = replay_row_summary(replay_rows)

    replay_manifest = Path(config["replay_manifest_path"])
    common.write_jsonl(Path(config["generated_path"]), generated_rows)
    common.write_jsonl(replay_manifest, replay_rows)
    common.write_replay_positions(Path(config["replay_positions_path"]), replay_rows)
    common.write_jsonl(Path(config["output_dir"]) / "generation_batches.jsonl", batch_stats)
    common.write_jsonl(Path(config["output_dir"]) / "generation_split_manifest.jsonl", split_stats)

    elapsed = time.perf_counter() - started
    result = {
        "kind": "generate",
        "mode": mode["name"],
        "model": config["model_label"],
        "model_id": model_id,
        "docs": len(replay_rows),
        "batches": len(batch_stats),
        **summary,
        "replay_generated_token_limit": int(config["replay_generated_token_limit"]),
        "self_circulation_trim": bool(config.get("self_circulation_trim", True)),
        "max_prompt_tokens": max_prompt_tokens,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_batch_prompt_tokens": max_batch_prompt_tokens,
        "elapsed_s": elapsed,
        "model_load_elapsed_s": load_elapsed,
        "generated_path": config["generated_path"],
        "replay_manifest_path": str(replay_manifest),
        "replay_positions_path": str(config["replay_positions_path"]),
        "hidden_dir": str(hidden_dir) if save_generation_hidden else "",
        "output_dir": config["output_dir"],
    }
    common.write_json(Path(config["result_path"]), result)


def worker_prefill(config: dict[str, Any]) -> None:
    started = time.perf_counter()
    mode = config["mode"]
    os.environ.update(common.configure_precision_env(dict(os.environ), mode))
    common.apply_torch_precision(bool(mode.get("tf32")))
    common.add_local_vllm_to_path()

    from vllm import LLM, SamplingParams, TokensPrompt

    replay_rows = common.read_jsonl(Path(config["replay_manifest_path"]))
    batch_manifest = common.read_jsonl(Path(config["batch_manifest_path"]))
    rows_by_doc = {int(row["doc_id"]): row for row in replay_rows}
    max_replay_tokens = max(int(row["replay_token_count"]) for row in replay_rows)
    max_batch_tokens = max(int(batch["total_replay_tokens"]) for batch in batch_manifest)
    max_model_len = resolve_max_model_len(
        str(config["max_model_len"]),
        max_replay_tokens + 1,
    )
    max_num_batched_tokens = max(
        int(config["prefill_token_budget"]),
        max_replay_tokens + 1,
        max_batch_tokens,
    )

    log(
        "worker prefill loading "
        f"mode={mode['name']} model={config['model_id']} dtype={mode['dtype']} "
        f"max_model_len={max_model_len} max_num_batched_tokens={max_num_batched_tokens}"
    )
    load_start = time.perf_counter()
    llm = LLM(
        **llm_kwargs(
            model_id=config["model_id"],
            mode=mode,
            seed=int(config["seed"]),
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=int(config["prefill_batch_size"]),
            gpu_memory_utilization=float(config["gpu_memory_utilization"]),
            enforce_eager=True,
            trust_remote_code=bool(config["trust_remote_code"]),
            max_logprobs=int(config.get("replay_top_k", 100)),
        )
    )
    common.clear_tensor_trace()
    load_elapsed = time.perf_counter() - load_start
    log(f"worker prefill mode={mode['name']} model loaded in {load_elapsed:.1f}s")

    mode_dir = Path(config["mode_dir"])
    hidden_dir = mode_dir / "hidden_states"
    trace_dir = mode_dir / "trace_batches"
    split_stats: list[dict[str, Any]] = []
    max_token_rows: list[dict[str, Any]] = []
    top_k = int(config.get("replay_top_k", 100))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=None,
        prompt_logprobs=top_k,
    )
    rows_written = 0
    token_top1_path = mode_dir / "replay_token_top1.jsonl"
    token_top1_path.unlink(missing_ok=True)
    token_topk_path = mode_dir / f"replay_token_top{top_k}.pt"
    token_topk_meta_path = mode_dir / f"replay_token_top{top_k}.meta.json"
    token_topk_rows_expected = sum(
        int(row["generated_token_count"]) for row in replay_rows
    )

    import torch

    token_topk_ids = torch.full(
        (token_topk_rows_expected, top_k),
        -1,
        dtype=torch.int32,
    )
    token_topk_probs = torch.zeros(
        (token_topk_rows_expected, top_k),
        dtype=torch.float32,
    )
    token_topk_doc_ids = torch.empty(token_topk_rows_expected, dtype=torch.int32)
    token_topk_token_pos = torch.empty(token_topk_rows_expected, dtype=torch.int32)
    token_topk_expected = torch.empty(token_topk_rows_expected, dtype=torch.int32)
    token_topk_cursor = 0

    try:
        for batch in batch_manifest:
            batch_id = int(batch["batch_id"])
            batch_rows = [rows_by_doc[int(doc_id)] for doc_id in batch["doc_indices"]]
            prompts = [
                TokensPrompt(prompt_token_ids=row["replay_token_ids"])
                for row in batch_rows
            ]
            trace_path = trace_dir / f"batch_{batch_id:04d}.safetensors"
            batch_started = time.perf_counter()
            common.enable_tensor_trace(trace_path)
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            token_topk_cursor, top1_rows = record_replay_token_topk(
                batch_rows=batch_rows,
                outputs=outputs,
                mode_name=mode["name"],
                batch_id=batch_id,
                top_k=top_k,
                token_ids_tensor=token_topk_ids,
                probs_tensor=token_topk_probs,
                doc_ids_tensor=token_topk_doc_ids,
                token_pos_tensor=token_topk_token_pos,
                expected_tensor=token_topk_expected,
                cursor=token_topk_cursor,
            )
            common.append_jsonl(token_top1_path, top1_rows)
            for row, output in zip(batch_rows, outputs, strict=True):
                completion = output.outputs[0]
                sampled_ids = [int(token_id) for token_id in completion.token_ids]
                max_token_rows.append(
                    {
                        "doc_id": int(row["doc_id"]),
                        "subset_pos": int(row["subset_pos"]),
                        "original_index": int(row["original_index"]),
                        "mode": mode["name"],
                        "batch_id": batch_id,
                        "replay_token_count": int(row["replay_token_count"]),
                        "generated_token_count": int(row["generated_token_count"]),
                        "max_token_id": sampled_ids[0] if sampled_ids else None,
                        "top1_token_id": sampled_ids[0] if sampled_ids else None,
                        "sampled_token_ids": sampled_ids,
                        "sampled_text": completion.text,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                    }
                )
            common.flush_tensor_trace()
            stats = common.split_replay_trace_to_doc_hidden(
                trace_path=trace_path,
                batch_rows=batch_rows,
                hidden_dir=hidden_dir,
            )
            common.clear_tensor_trace()
            if not bool(config["keep_trace_batches"]):
                trace_path.unlink(missing_ok=True)
                trace_path.with_suffix(trace_path.suffix + ".meta.json").unlink(
                    missing_ok=True
                )
            elapsed = time.perf_counter() - batch_started
            batch_hidden_rows = sum(
                int(row["generated_token_count"]) for row in batch_rows
            )
            rows_written += batch_hidden_rows
            stats.update(
                {
                    "batch_id": batch_id,
                    "docs": len(batch_rows),
                    "elapsed_s": elapsed,
                    "generated_hidden_rows": batch_hidden_rows,
                }
            )
            split_stats.append(stats)
            log(
                f"worker prefill mode={mode['name']} batch={batch_id} "
                f"docs={len(batch_rows)} rows={batch_hidden_rows} elapsed={elapsed:.1f}s"
            )
    finally:
        shutdown_llm_and_distributed(llm)
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    common.write_jsonl(mode_dir / "split_manifest.jsonl", split_stats)
    max_token_path = mode_dir / "replay_max_tokens.jsonl"
    common.write_jsonl(max_token_path, max_token_rows)
    if token_topk_cursor != token_topk_rows_expected:
        raise RuntimeError(
            f"token top-k rows mismatch: wrote {token_topk_cursor}, "
            f"expected {token_topk_rows_expected}"
        )
    torch.save(
        {
            "token_ids": token_topk_ids,
            "probs": token_topk_probs,
            "doc_id": token_topk_doc_ids,
            "token_pos": token_topk_token_pos,
            "expected_token_id": token_topk_expected,
            "top_k": top_k,
        },
        token_topk_path,
    )
    common.write_json(
        token_topk_meta_path,
        {
            "mode": mode["name"],
            "rows": token_topk_rows_expected,
            "top_k": top_k,
            "token_ids_shape": list(token_topk_ids.shape),
            "probs_shape": list(token_topk_probs.shape),
            "source": "vllm SamplingParams(prompt_logprobs=top_k)",
            "scope": "generated-token positions only; prompt-only positions omitted",
            "token_top1_jsonl": str(token_top1_path),
        },
    )
    elapsed = time.perf_counter() - started
    summary = replay_row_summary(replay_rows)
    result = {
        "kind": "prefill",
        "mode": mode["name"],
        "model": config["model_label"],
        "model_id": config["model_id"],
        "docs": len(replay_rows),
        "batches": len(batch_manifest),
        **summary,
        "hidden_rows_written": rows_written,
        "max_replay_tokens": max_replay_tokens,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "elapsed_s": elapsed,
        "model_load_elapsed_s": load_elapsed,
        "hidden_dir": str(hidden_dir),
        "max_token_path": str(max_token_path),
        "max_token_rows": len(max_token_rows),
        "token_top1_path": str(token_top1_path),
        "token_top1_rows": token_topk_rows_expected,
        "token_topk_path": str(token_topk_path),
        "token_topk_meta_path": str(token_topk_meta_path),
        "replay_top_k": top_k,
        "output_dir": str(mode_dir),
    }
    common.write_json(Path(config["result_path"]), result)


def mode_outputs_complete(mode_dir: Path, replay_rows: list[dict[str, Any]]) -> bool:
    return common.hidden_outputs_complete(mode_dir / "hidden_states", replay_rows)


def run_worker_process(
    *,
    kind: str,
    config: dict[str, Any],
    mode: dict[str, Any],
    config_dir: Path,
) -> dict[str, Any]:
    config_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(config["result_path"])
    result_path.unlink(missing_ok=True)
    label = mode["name"]
    config_path = config_dir / f"{kind}_{label}.json"
    common.write_json(config_path, jsonable(config))
    env = common.configure_precision_env(dict(os.environ), mode)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        kind,
        "--config",
        str(config_path),
    ]
    log(f"starting worker kind={kind} mode={label}")
    subprocess.run(cmd, cwd=common.ROOT, env=env, check=True)
    return common.read_json(result_path)


def write_metrics(results_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        for key, value in list(csv_row.items()):
            if isinstance(value, (dict, list)):
                csv_row[key] = json.dumps(value, sort_keys=True)
        csv_rows.append(csv_row)
    common.write_csv(results_root / "metrics.csv", csv_rows, METRIC_FIELDS)
    common.write_json(results_root / "metrics.json", {"metrics": rows})


def append_metric_defaults(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    model_arg: str,
    model_label: str,
    spec: ModeSpec,
    mode_dir: Path,
    data_dir: Path | None,
    patch_hashes_before: dict[str, str],
) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "model": model_label,
            "model_arg": model_arg,
            "mode": spec.name,
            "version": spec.version,
            "dtype": spec.dtype,
            "attention_backend": spec.attention_backend,
            "tf32": spec.tf32,
            "significand_bits": spec.significand_bits,
            "setting": spec.setting,
            "output_dir": str(mode_dir),
            "data_dir": str(data_dir) if data_dir is not None else "",
            "patch_sha256_before": patch_hashes_before.get(spec.version, ""),
            "patch_sha256_after": common.patch_hashes().get(spec.version, ""),
            "env_settings_json": json.dumps(
                mode_env_settings(spec_mode_dict(spec, mode_dir)),
                sort_keys=True,
            ),
        }
    )
    for field in METRIC_FIELDS:
        out.setdefault(field, None)
    return out


def dry_run_model(args: argparse.Namespace, run_id: str, model_arg: str) -> None:
    model_id = common.resolve_model(model_arg)
    model_label = common.model_label(model_arg)
    subset_rows = common.load_subset(Path(args.subset_file), limit=args.limit)
    run_root = common.OUTPUT_ROOT / run_id / model_label
    data_root = common.DATA_ROOT / run_id / model_label
    results_root = common.RESULTS_ROOT / run_id / model_label
    generation = generation_artifact_for_model(
        args,
        run_root=run_root,
        model_label=model_label,
    )
    effective_generation_batch_size = generation_batch_size_for_model(args, model_id)
    prefill_specs = selected_prefill_specs(args)
    plan = {
        "dry_run": True,
        "stage": args.stage,
        "run_id": run_id,
        "model_arg": model_arg,
        "model_id": model_id,
        "model_label": model_label,
        "subset_file": str(args.subset_file),
        "dataset_kind": args.dataset_kind,
        "mcr_max_source_chars": args.mcr_max_source_chars,
        "subset_rows": len(subset_rows),
        "output_root": str(run_root),
        "data_root": str(data_root),
        "results_root": str(results_root),
        "generation_artifact_dir": str(generation.directory),
        "generation": asdict(GENERATION_SPEC),
        "prefill_modes": [asdict(spec) for spec in prefill_specs],
        "patch_order": list(PATCH_ORDER),
        "batching": {
            "generation_batch_size": args.generation_batch_size,
            "effective_generation_batch_size": effective_generation_batch_size,
            "generation_token_budget": args.generation_token_budget,
            "generation_enforce_eager": args.generation_enforce_eager,
            "generation_use_token_prompts": args.generation_use_token_prompts,
            "generation_gpu_memory_utilization": (
                args.generation_gpu_memory_utilization
            ),
            "prefill_batch_size": args.prefill_batch_size,
            "prefill_token_budget": args.prefill_token_budget,
            "replay_gpu_memory_utilization": args.replay_gpu_memory_utilization,
            "replay_generated_token_limit": args.replay_generated_token_limit,
            "self_circulation_trim": args.self_circulation_trim,
            "replay_top_k": args.replay_top_k,
            "keep_generation_trace_batches": args.keep_generation_trace_batches,
            "stop_after_generation": args.stop_after_generation,
        },
        "patch_sha256": common.patch_hashes(),
        "active_patch_status": common.active_patch_status(),
        "vllm_git_sha": vllm_git_sha(),
        "vllm_git_branch": vllm_git_branch(),
    }
    common.write_json(results_root / "run_manifest.json", plan)
    log(
        f"dry-run model={model_label} docs={len(subset_rows)} "
        f"output={run_root} modes={len(prefill_specs)}"
    )


def prepare_version(version: str) -> None:
    common.ensure_vllm_clean()
    ensure_no_active_patch()
    log(f"prepare-version {version}")
    common.run_patch_command("prepare-version", version)


def revert_version(version: str) -> None:
    log(f"revert-version {version}")
    common.run_patch_command("revert-version", version)
    common.ensure_vllm_clean()


def selected_prefill_specs(args: argparse.Namespace) -> list[ModeSpec]:
    specs = list(PREFILL_SPECS)
    if args.prefill_modes:
        requested_modes = set(args.prefill_modes)
        specs = [spec for spec in specs if spec.name in requested_modes]
    return specs


def patch_versions_for_run(
    args: argparse.Namespace,
    prefill_specs: list[ModeSpec],
) -> list[str]:
    prefill_versions = {spec.version for spec in prefill_specs}
    versions: list[str] = []
    for version in PATCH_ORDER:
        needs_generation = version == "VLLM_LAYERCAST" and args.stage in {
            "all",
            "generate",
        }
        if needs_generation or version in prefill_versions:
            versions.append(version)
    return versions


def load_generation_replay_rows(
    args: argparse.Namespace,
    generation: GenerationArtifact,
) -> list[dict[str, Any]]:
    if not generation.replay_manifest_path.exists():
        if not generation.generated_path.exists():
            raise RuntimeError(
                "generation artifact is missing replay_manifest.jsonl and "
                f"generated_tokens.jsonl: {generation.directory}"
            )
        generated_rows = common.read_jsonl(generation.generated_path)
        replay_rows = build_replay_rows_from_generated(
            generated_rows,
            replay_generated_token_limit=args.replay_generated_token_limit,
            trim_self_circulation=args.self_circulation_trim,
        )
    else:
        replay_rows = common.read_jsonl(generation.replay_manifest_path)
        if args.replay_generated_token_limit is not None and int(args.replay_generated_token_limit) > 0:
            replay_rows = build_replay_rows_from_generated(
                replay_rows,
                replay_generated_token_limit=int(args.replay_generated_token_limit),
                trim_self_circulation=args.self_circulation_trim,
            )
    replay_rows = filter_rows_by_subset_file(replay_rows, args.subset_file)
    if args.limit is not None and args.limit < len(replay_rows):
        replay_rows = replay_rows[: args.limit]
    return replay_rows


def prepare_replay_inputs(
    *,
    args: argparse.Namespace,
    run_root: Path,
    generation: GenerationArtifact,
) -> tuple[list[dict[str, Any]], Path, Path]:
    replay_rows = load_generation_replay_rows(args, generation)
    if not replay_rows:
        raise RuntimeError(f"generation artifact has no replay rows: {generation.directory}")
    if not common.hidden_outputs_complete(generation.hidden_dir, replay_rows):
        raise RuntimeError(
            "generation artifact is missing complete hidden_states for replay/MSE: "
            f"{generation.hidden_dir}"
        )

    replay_manifest_path = run_root / "replay_manifest.jsonl"
    batch_manifest_path = run_root / "batch_manifest.jsonl"
    replay_positions_path = run_root / "replay_positions.csv"
    common.write_jsonl(replay_manifest_path, replay_rows)
    batches = common.build_batch_manifest(
        replay_rows,
        batch_size=int(args.prefill_batch_size),
        token_budget=int(args.prefill_token_budget),
    )
    common.write_jsonl(batch_manifest_path, batches)
    common.write_replay_positions(replay_positions_path, replay_rows)
    return replay_rows, replay_manifest_path, batch_manifest_path


def load_generation_result(
    *,
    args: argparse.Namespace,
    model_label: str,
    generation: GenerationArtifact,
) -> dict[str, Any]:
    replay_rows = load_generation_replay_rows(args, generation)
    summary = replay_row_summary(replay_rows)
    result = (
        common.read_json(generation.result_path)
        if generation.result_path.exists()
        else {}
    )
    return {
        "kind": "generate",
        "mode": GENERATION_SPEC.name,
        "model": model_label,
        "docs": len(replay_rows),
        "batches": result.get("batches", 0),
        **summary,
        "elapsed_s": result.get("elapsed_s", 0.0),
        "generated_path": str(generation.generated_path),
        "replay_manifest_path": str(generation.replay_manifest_path),
        "replay_positions_path": str(generation.replay_positions_path),
        "hidden_dir": str(generation.hidden_dir),
        "output_dir": str(generation.directory),
        "reused_generation_artifact": True,
    }


def run_generation_if_needed(
    *,
    args: argparse.Namespace,
    model_arg: str,
    model_id: str,
    model_label: str,
    run_root: Path,
    generation: GenerationArtifact,
    config_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    if (
        args.resume
        and not args.rerun_generation
        and generation.replay_manifest_path.exists()
        and generation.generated_path.exists()
    ):
        log(f"resume: using existing generation manifest for {model_label}")
        return load_generation_result(
            args=args,
            model_label=model_label,
            generation=generation,
        )

    if args.generated_source_jsonl is not None:
        return materialize_generation_from_source(
            args=args,
            model_label=model_label,
            gen_dir=generation.directory,
            replay_manifest=generation.replay_manifest_path,
            generated_path=generation.generated_path,
            result_path=generation.result_path,
            started=started,
        )

    mode = spec_mode_dict(GENERATION_SPEC, generation.directory)
    generation_batch_size = generation_batch_size_for_model(args, model_id)
    if generation_batch_size != int(args.generation_batch_size):
        log(
            f"model-specific generation batch cap for {model_label}: "
            f"{args.generation_batch_size} -> {generation_batch_size}"
        )
    config = {
        "model_arg": model_arg,
        "model_id": model_id,
        "model_label": model_label,
        "mode": mode,
        "subset_file": str(args.subset_file),
        "dataset_kind": args.dataset_kind,
        "mcr_max_source_chars": args.mcr_max_source_chars,
        "limit": args.limit,
        "run_root": str(run_root),
        "output_dir": str(generation.directory),
        "generated_path": str(generation.generated_path),
        "replay_manifest_path": str(generation.replay_manifest_path),
        "replay_positions_path": str(generation.replay_positions_path),
        "max_new_tokens": args.max_new_tokens,
        "replay_generated_token_limit": args.replay_generated_token_limit,
        "self_circulation_trim": args.self_circulation_trim,
        "generation_batch_size": generation_batch_size,
        "requested_generation_batch_size": args.generation_batch_size,
        "generation_token_budget": args.generation_token_budget,
        "generation_enforce_eager": args.generation_enforce_eager,
        "generation_use_token_prompts": args.generation_use_token_prompts,
        "prefill_batch_size": args.prefill_batch_size,
        "prefill_token_budget": args.prefill_token_budget,
        "keep_generation_trace_batches": args.keep_generation_trace_batches,
        "gpu_memory_utilization": args.generation_gpu_memory_utilization,
        "generation_gpu_memory_utilization": args.generation_gpu_memory_utilization,
        "replay_gpu_memory_utilization": args.replay_gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "save_generation_hidden": args.save_generation_hidden,
        "result_path": str(generation.result_path),
    }
    return run_worker_process(
        kind="generate",
        config=config,
        mode=mode,
        config_dir=config_dir,
    )


def run_prefill_mode(
    *,
    args: argparse.Namespace,
    model_arg: str,
    model_id: str,
    model_label: str,
    run_root: Path,
    data_root: Path,
    results_root: Path,
    config_dir: Path,
    replay_rows: list[dict[str, Any]],
    replay_manifest_path: Path,
    batch_manifest_path: Path,
    reference_hidden_dir: Path,
    spec: ModeSpec,
    patch_hashes_before: dict[str, str],
    metric_rows: list[dict[str, Any]],
) -> None:
    mode_dir = run_root / spec.name
    data_dir = data_root / spec.name
    result_path = mode_dir / "worker_result.json"
    mode = spec_mode_dict(spec, mode_dir)
    if heal_dynamic_int8_gemm_weight_for_mode(model_id, spec):
        mode["heal_dynamic_int8_gemm_weight"] = True
    if heal_fp16_gemm_compensation_for_mode(model_id, spec):
        mode["heal_fp16_gemm_compensation"] = True
    if spec.version == "VLLM_HEAL":
        if spec.name in {"heal_plus_prefill", "heal_plus_plus_prefill"}:
            qkv_skip_layers = args.heal_skip_qkv_quant_layers
            if qkv_skip_layers is None and spec.name == "heal_plus_plus_prefill":
                qkv_skip_layers = heal_policy.policy_skip_layers(
                    args.heal_policy,
                    model_arg=model_arg,
                    model_id=model_id,
                    model_label=model_label,
                    variant="HEAL++",
                    field="HEAL_SKIP_QKV_QUANT_LAYERS",
                )
            if qkv_skip_layers is not None:
                mode["heal_skip_qkv_quant_layers"] = str(qkv_skip_layers)
        if spec.name in {"heal_full_prefill", "heal_full_plus_prefill"}:
            fp16_skip_layers = (
                args.heal_skip_fp16_comp_layers
                if args.heal_skip_fp16_comp_layers is not None
                else args.heal_skip_int8_comp_layers
            )
            if fp16_skip_layers is None and spec.name == "heal_full_plus_prefill":
                fp16_skip_layers = heal_policy.policy_skip_layers(
                    args.heal_policy,
                    model_arg=model_arg,
                    model_id=model_id,
                    model_label=model_label,
                    variant="HEAL-FULL+",
                    field="HEAL_SKIP_FP16_COMP_LAYERS",
                )
            if fp16_skip_layers is not None:
                mode["heal_skip_fp16_comp_layers"] = str(fp16_skip_layers)
        if args.heal_fp16_gemm_compensation is not None:
            mode["heal_fp16_gemm_compensation"] = bool(args.heal_fp16_gemm_compensation)
    test_started = time.perf_counter()
    worker_result: dict[str, Any]

    token_outputs_complete = (
        (mode_dir / "replay_token_top1.jsonl").exists()
        and (mode_dir / f"replay_token_top{args.replay_top_k}.pt").exists()
    )
    if args.resume and mode_outputs_complete(mode_dir, replay_rows) and token_outputs_complete:
        log(f"resume: hidden outputs already complete for mode={spec.name}")
        summary = replay_row_summary(replay_rows)
        worker_result = (
            common.read_json(result_path)
            if result_path.exists()
            else {
                "kind": "prefill",
                "mode": spec.name,
                "docs": len(replay_rows),
                "batches": len(common.read_jsonl(batch_manifest_path)),
                **summary,
                "elapsed_s": 0.0,
                "hidden_dir": str(mode_dir / "hidden_states"),
                "output_dir": str(mode_dir),
            }
        )
    else:
        replay_gpu_memory_utilization = replay_gpu_memory_utilization_for_mode(
            args, model_id, spec
        )
        prefill_batch_size = prefill_batch_size_for_mode(args, model_id, spec)
        prefill_token_budget = prefill_token_budget_for_mode(args, model_id, spec)
        prefill_max_model_len = prefill_max_model_len_for_mode(args, model_id, spec)
        effective_batch_manifest_path = batch_manifest_path
        if (
            prefill_batch_size != int(args.prefill_batch_size)
            or prefill_token_budget != int(args.prefill_token_budget)
        ):
            mode_dir.mkdir(parents=True, exist_ok=True)
            effective_batch_manifest_path = mode_dir / "batch_manifest.jsonl"
            common.write_jsonl(
                effective_batch_manifest_path,
                common.build_batch_manifest(
                    replay_rows,
                    batch_size=prefill_batch_size,
                    token_budget=prefill_token_budget,
                ),
            )
            log(
                "model/mode-specific replay batching for "
                f"{model_label} {spec.name}: "
                f"batch_size={args.prefill_batch_size} -> {prefill_batch_size}, "
                f"token_budget={args.prefill_token_budget} -> {prefill_token_budget}"
            )
        if prefill_max_model_len != str(args.max_model_len):
            log(
                "model/mode-specific max model len for "
                f"{model_label} {spec.name}: "
                f"{args.max_model_len} -> {prefill_max_model_len}"
            )
        if replay_gpu_memory_utilization != float(args.replay_gpu_memory_utilization):
            log(
                "model/mode-specific replay gpu utilization for "
                f"{model_label} {spec.name}: "
                f"{args.replay_gpu_memory_utilization} -> "
                f"{replay_gpu_memory_utilization}"
            )
        config = {
            "model_arg": model_arg,
            "model_id": model_id,
            "model_label": model_label,
            "mode": mode,
            "mode_dir": str(mode_dir),
            "replay_manifest_path": str(replay_manifest_path),
            "batch_manifest_path": str(effective_batch_manifest_path),
            "max_model_len": prefill_max_model_len,
            "prefill_batch_size": prefill_batch_size,
            "prefill_token_budget": prefill_token_budget,
            "gpu_memory_utilization": replay_gpu_memory_utilization,
            "generation_gpu_memory_utilization": (
                args.generation_gpu_memory_utilization
            ),
            "replay_gpu_memory_utilization": replay_gpu_memory_utilization,
            "trust_remote_code": args.trust_remote_code,
            "seed": args.seed,
            "keep_trace_batches": args.keep_trace_batches,
            "replay_top_k": args.replay_top_k,
            "result_path": str(result_path),
        }
        worker_result = run_worker_process(
            kind="prefill",
            config=config,
            mode=mode,
            config_dir=config_dir,
        )

    mse_started = time.perf_counter()
    candidate_hidden_dir = mode_dir / "hidden_states"
    mse_metrics = common.compute_hidden_mse(
        reference_hidden_dir=reference_hidden_dir,
        candidate_hidden_dir=candidate_hidden_dir,
        replay_rows=replay_rows,
        data_dir=data_dir,
        mode_name=spec.name,
    )
    token_top1_path_value = str(worker_result.get("token_top1_path") or "")
    token_top1_path = Path(token_top1_path_value) if token_top1_path_value else None
    token_top1_metrics: dict[str, Any] = {}
    if token_top1_path is not None and token_top1_path.exists():
        token_top1_metrics.update(
            summarize_token_top1(
                token_top1_path,
                flip_path=data_dir / "token_top1_flips_vs_generated.jsonl",
            )
        )
        ref_top1_path = run_root / "layercast_fp32_prefill_ref" / "replay_token_top1.jsonl"
        if spec.name == "layercast_fp32_prefill_ref":
            token_top1_metrics.update(
                {
                    "ref_top1_flip_count": 0,
                    "ref_top1_flip_rate": 0.0,
                    "ref_top1_flip_file": "",
                }
            )
        elif ref_top1_path.exists():
            token_top1_metrics.update(
                compare_token_top1(
                    reference_path=ref_top1_path,
                    candidate_path=token_top1_path,
                    output_path=data_dir / "token_top1_flips_vs_ref.jsonl",
                )
            )
    mse_elapsed = time.perf_counter() - mse_started
    total_elapsed = time.perf_counter() - test_started
    metric = append_metric_defaults(
        row={
            "status": "ok",
            "wall_time_s": total_elapsed,
            "worker_elapsed_s": worker_result.get("elapsed_s"),
            "mse_compute_elapsed_s": mse_elapsed,
            "generated_tokens": worker_result.get("generated_tokens"),
            "replay_generated_tokens": worker_result.get("replay_generated_tokens"),
            "truncated_generated_tokens": worker_result.get(
                "truncated_generated_tokens"
            ),
            "limit_truncated_generated_tokens": worker_result.get(
                "limit_truncated_generated_tokens"
            ),
            "self_circulation_docs": worker_result.get("self_circulation_docs"),
            "self_circulation_dropped_tokens": worker_result.get(
                "self_circulation_dropped_tokens"
            ),
            "replay_tokens": worker_result.get("replay_tokens"),
            "docs": worker_result.get("docs"),
            "batches": worker_result.get("batches"),
            **token_top1_metrics,
            **mse_metrics,
        },
        args=args,
        model_arg=model_arg,
        model_label=model_label,
        spec=spec,
        mode_dir=mode_dir,
        data_dir=data_dir,
        patch_hashes_before=patch_hashes_before,
    )
    metric_rows.append(metric)
    write_metrics(results_root, metric_rows)
    log(
        f"mode done {spec.name}: rows={metric['compared_hidden_rows']} "
        f"mse={metric['mse']:.6g} wall={total_elapsed:.1f}s"
    )


def gate_qwen3_8b(
    *,
    args: argparse.Namespace,
    model_label: str,
    results_root: Path,
    metric_rows: list[dict[str, Any]],
) -> bool:
    if args.skip_gate:
        return True
    if args.stage == "generate":
        return True
    if model_label != "Qwen-Qwen3-8B" or args.limit != 200:
        return True
    if not args.strict_qwen_gate:
        report = {
            "model": model_label,
            "passed": True,
            "strict_qwen_gate": False,
            "note": (
                "Strict H100 row-count/baseline checks are disabled by "
                "default because the current implementation intentionally may "
                "produce different generated-token counts."
            ),
            "checked_at": common.utc_timestamp(),
        }
        common.write_json(results_root / "gate_report.json", report)
        log("Qwen3-8B strict H100 gate skipped")
        return True

    by_mode = {row["mode"]: row for row in metric_rows if row.get("status") == "ok"}
    baseline = common.load_h100_qwen3_gate_baseline(args.qwen_gate_baseline_file)
    failures: list[str] = []
    warnings: list[str] = []
    expected_rows = 305734

    for spec in PREFILL_SPECS:
        row = by_mode.get(spec.name)
        if row is None:
            failures.append(f"missing metrics for {spec.name}")
            continue
        rows = int(row.get("compared_hidden_rows") or 0)
        if rows != expected_rows:
            failures.append(f"{spec.name} rows={rows}, expected {expected_rows}")

    lc_same = by_mode.get("layercast_fp32_prefill", {}).get("mse")
    if lc_same is None or float(lc_same) > 1e-8:
        failures.append(
            "layercast_fp32_prefill vs layercast_fp32_prefill_ref "
            f"MSE={lc_same}, expected <=1e-8"
        )

    for mode_name, setting in [
        ("original_bf16_prefill", "native_bf16"),
        ("original_fp16_prefill", "native_fp16"),
        ("layercast_tf32_prefill", "layercast_tf32"),
        ("significand_k03_prefill", "mantissa_k3"),
        ("significand_k07_prefill", "mantissa_k7"),
        ("significand_k10_prefill", "mantissa_k10"),
        ("significand_k16_prefill", "mantissa_k16"),
    ]:
        value = by_mode.get(mode_name, {}).get("mse")
        ref = baseline.get(setting)
        if value is None:
            failures.append(f"missing MSE for {mode_name}")
            continue
        if ref is None:
            warnings.append(f"missing H100 baseline for {setting}")
            continue
        ratio = float(value) / ref if ref else float("inf")
        if ratio < 0.1 or ratio > 10.0:
            failures.append(
                f"{mode_name} MSE={value:.6g} baseline={ref:.6g} ratio={ratio:.3g}"
            )

    k_values: dict[int, float] = {}
    for k in range(3, 24):
        value = by_mode.get(f"significand_k{k:02d}_prefill", {}).get("mse")
        if value is not None:
            k_values[k] = float(value)
    for left, right in [(3, 7), (7, 10), (10, 16)]:
        if left in k_values and right in k_values and not k_values[left] > k_values[right]:
            failures.append(
                f"mantissa trend violation: k{left}={k_values[left]:.6g} "
                f"not greater than k{right}={k_values[right]:.6g}"
            )
    if 16 in k_values:
        tail_limit = max(10.0 * k_values[16], 1e-5)
        for k in range(17, 24):
            if k in k_values and k_values[k] > tail_limit:
                failures.append(
                    f"mantissa tail k{k}={k_values[k]:.6g} exceeds tail limit "
                    f"{tail_limit:.6g}"
                )

    report = {
        "model": model_label,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "expected_rows": expected_rows,
        "baseline_file": str(args.qwen_gate_baseline_file) if args.qwen_gate_baseline_file else None,
        "checked_at": common.utc_timestamp(),
    }
    common.write_json(results_root / "gate_report.json", report)
    if failures:
        log("Qwen3-8B gate FAILED")
        for failure in failures:
            log(f"  gate failure: {failure}")
        return False
    log("Qwen3-8B gate passed")
    return True


def gate_qwen3_8b_generation_rows(
    *,
    args: argparse.Namespace,
    model_label: str,
    results_root: Path,
    gen_result: dict[str, Any],
) -> None:
    if args.skip_gate:
        return
    if model_label != "Qwen-Qwen3-8B" or args.limit != 200:
        return
    if not args.strict_qwen_gate:
        return
    expected_rows = 305734
    replay_rows = int(
        gen_result.get("replay_generated_tokens")
        or gen_result.get("generated_tokens")
        or 0
    )
    if replay_rows == expected_rows:
        return
    report = {
        "model": model_label,
        "passed": False,
        "stage": "generation_replay_row_count",
        "expected_rows": expected_rows,
        "replay_generated_tokens": replay_rows,
        "generated_tokens": gen_result.get("generated_tokens"),
        "truncated_generated_tokens": gen_result.get("truncated_generated_tokens"),
        "max_new_tokens": args.max_new_tokens,
        "generation_batch_size": args.generation_batch_size,
        "replay_generated_token_limit": args.replay_generated_token_limit,
        "checked_at": common.utc_timestamp(),
    }
    common.write_json(results_root / "gate_report.json", report)
    raise RuntimeError(
        "Qwen3-8B generation replay rows do not match H100 gate: "
        f"{replay_rows} vs expected {expected_rows}"
    )


def run_model(args: argparse.Namespace, run_id: str, model_arg: str) -> bool:
    model_id = common.resolve_model(model_arg)
    model_label = common.model_label(model_arg)
    run_root = common.OUTPUT_ROOT / run_id / model_label
    data_root = common.DATA_ROOT / run_id / model_label
    results_root = common.RESULTS_ROOT / run_id / model_label
    generation = generation_artifact_for_model(
        args,
        run_root=run_root,
        model_label=model_label,
    )
    config_dir = results_root / "worker_configs"
    patch_hashes_before = common.patch_hashes()
    metric_rows: list[dict[str, Any]] = []

    if run_root.exists() and any(run_root.iterdir()) and not args.resume:
        raise RuntimeError(f"output directory already exists; pass --resume: {run_root}")
    if (
        args.stage in {"all", "generate"}
        and generation.directory.exists()
        and any(generation.directory.iterdir())
        and not args.resume
    ):
        raise RuntimeError(
            "generation artifact directory already exists; pass --resume: "
            f"{generation.directory}"
        )
    if args.stage == "replay" and not (
        generation.replay_manifest_path.exists() or generation.generated_path.exists()
    ):
        raise RuntimeError(
            "replay stage needs an existing generation artifact directory; "
            f"missing replay_manifest.jsonl/generated_tokens.jsonl under {generation.directory}"
        )
    if args.stage == "replay":
        replay_rows_for_check = load_generation_replay_rows(args, generation)
        if not common.hidden_outputs_complete(generation.hidden_dir, replay_rows_for_check):
            raise RuntimeError(
                "replay stage needs complete generation hidden_states under "
                f"{generation.hidden_dir}"
            )

    if args.prepared_vllm_version:
        ensure_active_patch(args.prepared_vllm_version)
    else:
        common.ensure_vllm_clean()
        ensure_no_active_patch()
    manifest = {
        "run_id": run_id,
        "model_arg": model_arg,
        "model_id": model_id,
        "model_label": model_label,
        "subset_file": str(args.subset_file),
        "dataset_kind": args.dataset_kind,
        "mcr_max_source_chars": args.mcr_max_source_chars,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "stage": args.stage,
        "output_root": str(run_root),
        "data_root": str(data_root),
        "results_root": str(results_root),
        "generation_artifact_dir": str(generation.directory),
        "vllm_git_sha": vllm_git_sha(),
        "vllm_git_branch": vllm_git_branch(),
        "patch_sha256_before": patch_hashes_before,
        "started_at": common.utc_timestamp(),
        "args": jsonable(vars(args)),
    }
    common.write_json(results_root / "run_manifest.json", manifest)

    generation_ready = False
    replay_rows: list[dict[str, Any]] = []
    replay_manifest_path = run_root / "replay_manifest.jsonl"
    batch_manifest_path = run_root / "batch_manifest.jsonl"
    prefill_specs_to_run = selected_prefill_specs(args)
    prefill_specs_by_version = {
        version: [spec for spec in prefill_specs_to_run if spec.version == version]
        for version in PATCH_ORDER
    }
    versions_to_run = patch_versions_for_run(args, prefill_specs_to_run)
    try:
        for version in versions_to_run:
            version_prepared_by_caller = args.prepared_vllm_version == version
            if version_prepared_by_caller:
                ensure_active_patch(version)
                log(f"using prepared-version {version}")
            else:
                prepare_version(version)
            try:
                if version == "VLLM_LAYERCAST":
                    if args.stage in {"all", "generate"}:
                        gen_result = run_generation_if_needed(
                            args=args,
                            model_arg=model_arg,
                            model_id=model_id,
                            model_label=model_label,
                            run_root=run_root,
                            generation=generation,
                            config_dir=config_dir,
                        )
                    else:
                        gen_result = load_generation_result(
                            args=args,
                            model_label=model_label,
                            generation=generation,
                        )

                    replay_rows = load_generation_replay_rows(args, generation)
                    gen_mse_metrics: dict[str, Any] = {}
                    if common.hidden_outputs_complete(generation.hidden_dir, replay_rows):
                        gen_mse_metrics = common.compute_hidden_mse(
                            reference_hidden_dir=generation.hidden_dir,
                            candidate_hidden_dir=generation.hidden_dir,
                            replay_rows=replay_rows,
                            data_dir=data_root / GENERATION_SPEC.name,
                            mode_name=GENERATION_SPEC.name,
                        )
                    gen_metric = append_metric_defaults(
                        row={
                            "status": "ok",
                            "wall_time_s": gen_result.get("elapsed_s"),
                            "worker_elapsed_s": gen_result.get("elapsed_s"),
                            "generated_tokens": gen_result.get("generated_tokens"),
                            "replay_generated_tokens": gen_result.get(
                                "replay_generated_tokens"
                            ),
                            "truncated_generated_tokens": gen_result.get(
                                "truncated_generated_tokens"
                            ),
                            "limit_truncated_generated_tokens": gen_result.get(
                                "limit_truncated_generated_tokens"
                            ),
                            "self_circulation_docs": gen_result.get(
                                "self_circulation_docs"
                            ),
                            "self_circulation_dropped_tokens": gen_result.get(
                                "self_circulation_dropped_tokens"
                            ),
                            "replay_tokens": gen_result.get("replay_tokens"),
                            "docs": gen_result.get("docs"),
                            "batches": gen_result.get("batches"),
                            **gen_mse_metrics,
                        },
                        args=args,
                        model_arg=model_arg,
                        model_label=model_label,
                        spec=GENERATION_SPEC,
                        mode_dir=generation.directory,
                        data_dir=(
                            data_root / GENERATION_SPEC.name
                            if gen_mse_metrics
                            else None
                        ),
                        patch_hashes_before=patch_hashes_before,
                    )
                    metric_rows.append(gen_metric)
                    write_metrics(results_root, metric_rows)
                    gate_qwen3_8b_generation_rows(
                        args=args,
                        model_label=model_label,
                        results_root=results_root,
                        gen_result=gen_result,
                    )
                    if args.stage == "generate":
                        log("stopping after generation stage as requested")
                        break

                if not generation_ready:
                    (
                        replay_rows,
                        replay_manifest_path,
                        batch_manifest_path,
                    ) = prepare_replay_inputs(
                        args=args,
                        run_root=run_root,
                        generation=generation,
                    )
                    generation_ready = True
                for spec in prefill_specs_by_version.get(version, []):
                    run_prefill_mode(
                        args=args,
                        model_arg=model_arg,
                        model_id=model_id,
                        model_label=model_label,
                        run_root=run_root,
                        data_root=data_root,
                        results_root=results_root,
                        config_dir=config_dir,
                        replay_rows=replay_rows,
                        replay_manifest_path=replay_manifest_path,
                        batch_manifest_path=batch_manifest_path,
                        reference_hidden_dir=generation.hidden_dir,
                        spec=spec,
                        patch_hashes_before=patch_hashes_before,
                        metric_rows=metric_rows,
                    )
            finally:
                if not version_prepared_by_caller:
                    revert_version(version)
            if args.stage == "generate":
                break
    except Exception:
        if not args.prepared_vllm_version:
            revert_active_patch_best_effort()
        raise

    patch_hashes_after = common.patch_hashes()
    if patch_hashes_after != patch_hashes_before:
        raise RuntimeError(
            "patch files changed during run; before="
            f"{patch_hashes_before} after={patch_hashes_after}"
        )
    manifest.update(
        {
            "ended_at": common.utc_timestamp(),
            "patch_sha256_after": patch_hashes_after,
            "metrics_csv": str(results_root / "metrics.csv"),
            "metrics_json": str(results_root / "metrics.json"),
        }
    )
    common.write_json(results_root / "run_manifest.json", manifest)
    return gate_qwen3_8b(
        args=args,
        model_label=model_label,
        results_root=results_root,
        metric_rows=metric_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedQA final-hidden MSE prefill-replay runner."
    )
    parser.add_argument("--worker", choices=["generate", "prefill"], help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "replay"],
        default="all",
        help=(
            "Run the full pipeline, only materialize generation artifacts, "
            "or replay from an existing generation artifact directory."
        ),
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=None,
        help=(
            "Directory for standalone generation artifacts. Defaults to "
            "output/final-hidden-mse/<run_id>/generation_artifacts/<model>. "
            "When provided with multiple models, the model label is appended "
            "under this directory."
        ),
    )
    parser.add_argument("--subset-file", type=Path, default=common.DEFAULT_SUBSET_FILE)
    parser.add_argument(
        "--dataset-kind",
        choices=["medqa", "mcr-bench"],
        default="medqa",
        help="Prompt/metadata adapter for --subset-file. Defaults to the MedQA adapter.",
    )
    parser.add_argument(
        "--mcr-max-source-chars",
        type=int,
        default=12000,
        help="Maximum source text characters kept in each MCR-Bench prompt.",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--generated-source-jsonl", type=Path)
    parser.add_argument("--generation-batch-size", type=int, default=200)
    parser.add_argument("--generation-token-budget", type=int, default=4096)
    parser.add_argument(
        "--generation-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--generation-use-token-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--replay-generated-token-limit", type=int, default=4096)
    parser.add_argument("--replay-top-k", type=int, default=100)
    parser.add_argument(
        "--prefill-modes",
        nargs="+",
        choices=sorted(PREFILL_SPECS_BY_NAME),
        help="Optional subset of replay/prefill modes to run.",
    )
    parser.add_argument(
        "--prepared-vllm-version",
        help=(
            "Assume this vLLM patch is already active and skip per-run "
            "prepare/revert for that version."
        ),
    )
    parser.add_argument("--heal-skip-qkv-quant-layers")
    parser.add_argument("--heal-skip-fp16-comp-layers")
    parser.add_argument(
        "--heal-skip-int8-comp-layers",
        help="Alias for --heal-skip-fp16-comp-layers.",
    )
    parser.add_argument(
        "--heal-policy-json",
        type=Path,
        help=(
            "Portable HEAL++/HEAL-FULL+ policy JSON exported by "
            "tests/error-contribution/export_heal_policy.py. Manual skip-layer "
            "flags override the policy."
        ),
    )
    parser.add_argument(
        "--heal-fp16-gemm-compensation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--self-circulation-trim",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefill-batch-size", type=int, default=2)
    parser.add_argument("--prefill-token-budget", type=int, default=6000)
    parser.add_argument(
        "--gpu-memory-utilization",
        "--generation-gpu-memory-utilization",
        dest="generation_gpu_memory_utilization",
        type=float,
        default=0.93,
        help=(
            "GPU memory utilization for the LayerCast generation reference "
            "worker. The --gpu-memory-utilization spelling is kept as "
            "an alias for generation."
        ),
    )
    parser.add_argument(
        "--replay-gpu-memory-utilization",
        "--prefill-gpu-memory-utilization",
        dest="replay_gpu_memory_utilization",
        type=float,
        default=0.86,
        help="GPU memory utilization for replay/prefill workers.",
    )
    parser.add_argument("--max-model-len", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-generation", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--strict-qwen-gate", action="store_true")
    parser.add_argument(
        "--qwen-gate-baseline-file",
        type=Path,
        default=common.DEFAULT_H100_QWEN3_8B_BASELINE,
        help="Optional CSV baseline used only with --strict-qwen-gate.",
    )
    parser.add_argument("--stop-after-generation", action="store_true")
    parser.add_argument("--continue-on-gate-failure", action="store_true")
    parser.add_argument("--keep-trace-batches", action="store_true")
    parser.add_argument("--keep-generation-trace-batches", action="store_true")
    parser.add_argument(
        "--save-generation-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.heal_policy = heal_policy.load_heal_policy(
        args.heal_policy_json,
        root=common.ROOT,
    )
    if args.stop_after_generation:
        if args.stage == "replay":
            raise SystemExit("--stop-after-generation cannot be combined with --stage replay")
        args.stage = "generate"
    if args.worker:
        if args.config is None:
            raise SystemExit("--config is required for worker mode")
        config = common.read_json(args.config)
        if args.worker == "generate":
            worker_generate(config)
        else:
            worker_prefill(config)
        return 0

    run_id = args.run_id or common.utc_timestamp()
    selected_models = model_args_from_cli(args)
    if args.dry_run:
        for model_arg in selected_models:
            dry_run_model(args, run_id, model_arg)
        return 0

    for index, model_arg in enumerate(selected_models):
        model_label = common.model_label(model_arg)
        log(f"starting model {index + 1}/{len(selected_models)}: {model_label}")
        passed = run_model(args, run_id, model_arg)
        if (
            model_label == "Qwen-Qwen3-8B"
            and not passed
            and not args.continue_on_gate_failure
        ):
            log("stopping because Qwen3-8B gate failed")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
