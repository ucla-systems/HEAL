# SPDX-License-Identifier: Apache-2.0
"""Residual/KV error-contribution runner for MedQA generation replays."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess

import numpy as np
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import safe_open

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
FINAL_HIDDEN_MSE_DIR = ROOT / "tests" / "final-hidden-mse"
if str(FINAL_HIDDEN_MSE_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_HIDDEN_MSE_DIR))
import common as fhm_common  # noqa: E402

VLLM_DIR = ROOT / "vllm"
PATCH_SCRIPT = ROOT / "scripts" / "vllm_incremental_build.sh"
PATCH_NAME = "VLLM_ERROR_CONTRIBUTION"
HEAL_PATCH_NAME = "VLLM_ERROR_CONTRIBUTION_HEAL"
OUTPUT_ROOT = ROOT / "output" / "error-contribution"
RESULTS_ROOT = TEST_DIR / "results"
FIGURES_ROOT = ROOT / "figures"
DEFAULT_REFERENCE_GENERATION_DIR = (
    ROOT
    / "output"
    / "final-hidden-mse"
    / "medqa_ref"
    / "generation_artifacts"
)
MIN_USABLE_FIT_R2 = 0.10
DEFAULT_MSE_OUTLIER_QUANTILE = 0.995
DEFAULT_MSE_OUTLIER_MAD_MULTIPLIER = 80.0
OUTLIER_SCORE_FIELDS = (
    "mlp_total_mse",
    "attention_total_mse",
    "kv_mse",
    "final_hidden_mse",
)


STAGE_REGEX = (
    r"^(?:layer_\d+\.(?:input_layernorm|post_attention_layernorm)\.output\.residual|"
    r"attention\.(?:q|k_full|v_full|core_output)\.\d+|"
    r"block\.mlp\.(?:input|output)\.\d+|"
    r"oracle\.block\.mlp\.output\.\d+|"
    r"forward\.norm\.(?:input|hidden))$"
)


@dataclass(frozen=True)
class ModeSpec:
    name: str
    dtype: str
    trace_mode: str
    layercast_force_fp32: bool = False
    tf32: bool = False
    attention_backend: str | None = "TRITON_ATTN"
    patch_name: str = PATCH_NAME
    heal_mode: str | None = None
    heal_fp16_gemm_compensation: bool = False


REFERENCE_MODE = ModeSpec(
    name="layercast_fp32_ref",
    dtype="float32",
    trace_mode="reference",
    layercast_force_fp32=True,
)
CANDIDATE_MODES = {
    "bf16": ModeSpec(name="original_bf16", dtype="bfloat16", trace_mode="candidate"),
    "fp16": ModeSpec(name="original_fp16", dtype="float16", trace_mode="candidate"),
    "heal": ModeSpec(
        name="heal_base",
        dtype="float16",
        trace_mode="candidate",
        patch_name=HEAL_PATCH_NAME,
        heal_mode="base",
    ),
    "heal_base": ModeSpec(
        name="heal_base",
        dtype="float16",
        trace_mode="candidate",
        patch_name=HEAL_PATCH_NAME,
        heal_mode="base",
    ),
    "heal_plus": ModeSpec(
        name="heal_plus",
        dtype="float16",
        trace_mode="candidate",
        patch_name=HEAL_PATCH_NAME,
        heal_mode="plus",
    ),
    "heal_full": ModeSpec(
        name="heal_full",
        dtype="float16",
        trace_mode="candidate",
        patch_name=HEAL_PATCH_NAME,
        heal_mode="full",
        heal_fp16_gemm_compensation=True,
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ModeSpec):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def path_from_cli(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _subset_selection_from_file(path: Path | None) -> dict[str, set[Any]] | None:
    if path is None:
        return None
    subset_path = path_from_cli(path)
    rows = read_jsonl(subset_path)
    selection: dict[str, set[Any]] = {"id": set(), "subset_pos": set(), "original_index": set()}
    for index, row in enumerate(rows):
        if row.get("id") is not None:
            selection["id"].add(str(row["id"]))
        for field in ("subset_pos", "original_index", "error_contribution_source_subset_pos"):
            if row.get(field) is not None:
                try:
                    selection["subset_pos" if field == "error_contribution_source_subset_pos" else field].add(int(row[field]))
                except (TypeError, ValueError):
                    pass
        if not any(selection[field] for field in selection):
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


def model_args_from_cli(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(fhm_common.MODEL_ORDER)
    if args.models:
        return args.models
    return [args.model]


def generation_dir_for_model(base: Path, model_arg: str, multiple_models: bool) -> Path:
    model_label = fhm_common.model_label(model_arg)
    if multiple_models:
        return base / model_label
    if (base / "generated_tokens.jsonl").exists() or (base / "replay_manifest.jsonl").exists():
        return base
    return base / model_label


def detect_self_circulation_tail(token_ids: list[int]) -> dict[str, int | bool]:
    return {
        "trimmed": False,
        "window": 0,
        "failure_value": 0,
        "period": 0,
        "cycle_start": 0,
        "cycle_length": 0,
        "trimmed_token_count": len(token_ids),
        "dropped_token_count": 0,
    }


def build_replay_rows_from_generated(
    generated_rows: list[dict[str, Any]],
    *,
    replay_generated_token_limit: int,
) -> list[dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    for fallback_doc_id, row in enumerate(generated_rows):
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        doc_id = int(row.get("doc_id", source.get("doc_id", fallback_doc_id)))
        prompt_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
        full_generated_ids = [int(token_id) for token_id in row["generated_token_ids"]]
        limited_generated_ids = full_generated_ids[:replay_generated_token_limit]
        circulation = detect_self_circulation_tail(limited_generated_ids)
        generated_ids = limited_generated_ids[: int(circulation["trimmed_token_count"])]
        replay_rows.append(
            {
                "doc_id": doc_id,
                "subset_pos": int(row.get("subset_pos", source.get("subset_pos", doc_id))),
                "original_index": int(row.get("original_index", source.get("original_index", doc_id))),
                "id": row.get("id", source.get("id")),
                "request_id": str(row["request_id"]) if row.get("request_id") is not None else None,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "replay_token_ids": prompt_ids + generated_ids,
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(generated_ids),
                "full_generated_token_count": len(full_generated_ids),
                "replay_generated_token_limit": replay_generated_token_limit,
                "replay_token_count": len(prompt_ids) + len(generated_ids),
                "finish_reason": row.get("finish_reason"),
                "stop_reason": row.get("stop_reason"),
            }
        )
    return replay_rows


def trim_replay_rows(rows: list[dict[str, Any]], max_gen_tokens: int) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for row in rows:
        prompt_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
        generated_ids = [int(token_id) for token_id in row["generated_token_ids"]][:max_gen_tokens]
        out = dict(row)
        out.update(
            {
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "replay_token_ids": prompt_ids + generated_ids,
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(generated_ids),
                "replay_generated_token_limit": max_gen_tokens,
                "replay_token_count": len(prompt_ids) + len(generated_ids),
            }
        )
        trimmed.append(out)
    return trimmed


def load_generation_replay_rows(
    generation_dir: Path,
    *,
    max_gen_tokens: int,
    limit: int | None,
    subset_file: Path | None = None,
) -> list[dict[str, Any]]:
    generated_path = generation_dir / "generated_tokens.jsonl"
    replay_path = generation_dir / "replay_manifest.jsonl"
    if generated_path.exists():
        rows = build_replay_rows_from_generated(
            read_jsonl(generated_path),
            replay_generated_token_limit=max_gen_tokens,
        )
    elif replay_path.exists():
        rows = trim_replay_rows(read_jsonl(replay_path), max_gen_tokens)
    else:
        raise FileNotFoundError(f"missing generated_tokens.jsonl/replay_manifest.jsonl under {generation_dir}")
    rows = filter_rows_by_subset_file(rows, subset_file)
    if limit is not None:
        rows = rows[:limit]
    rows = [row for row in rows if int(row.get("generated_token_count", 0)) > 0]
    if not rows:
        raise RuntimeError(f"no replayable generated rows in {generation_dir}")
    return rows


def all_candidate_positions(replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for row in replay_rows:
        prompt_count = int(row["prompt_token_count"])
        if prompt_count <= 0:
            continue
        for token_pos, token_id in enumerate(row["generated_token_ids"]):
            positions.append(
                {
                    "doc_id": int(row["doc_id"]),
                    "subset_pos": int(row.get("subset_pos", row["doc_id"])),
                    "original_index": int(row.get("original_index", row["doc_id"])),
                    "token_pos": int(token_pos),
                    "row_index": int(prompt_count - 1 + token_pos),
                    "expected_token_id": int(token_id),
                    "prompt_token_count": prompt_count,
                    "generated_token_count": int(row["generated_token_count"]),
                    "replay_token_count": int(row["replay_token_count"]),
                }
            )
    for sample_index, row in enumerate(positions):
        row["sample_index"] = sample_index
    return positions


def select_positions(
    positions: list[dict[str, Any]],
    *,
    sample_probability: float,
    sample_positions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if sample_positions > 0:
        selected = rng.sample(positions, min(sample_positions, len(positions)))
    else:
        selected = [row for row in positions if rng.random() < sample_probability]
        if not selected and positions:
            selected = [rng.choice(positions)]
    selected.sort(key=lambda row: (int(row["doc_id"]), int(row["row_index"]), int(row["token_pos"])))
    for idx, row in enumerate(selected):
        row["sample_index"] = idx
    return selected


def selected_doc_rows(replay_rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_docs = {int(row["doc_id"]) for row in selected}
    return [row for row in replay_rows if int(row["doc_id"]) in selected_docs]


def truncate_replay_rows_to_max_generated_tokens(
    rows: list[dict[str, Any]],
    *,
    max_generated_tokens_per_doc: int,
) -> list[dict[str, Any]]:
    if max_generated_tokens_per_doc <= 0:
        return rows
    trimmed: list[dict[str, Any]] = []
    for row in rows:
        prompt_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
        generated_ids = [int(token_id) for token_id in row["generated_token_ids"]][:max_generated_tokens_per_doc]
        if not generated_ids:
            continue
        out = dict(row)
        out.update(
            {
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "replay_token_ids": prompt_ids + generated_ids,
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(generated_ids),
                "replay_generated_token_limit": max_generated_tokens_per_doc,
                "replay_token_count": len(prompt_ids) + len(generated_ids),
                "full_generated_token_count": int(
                    row.get("full_generated_token_count", row.get("generated_token_count", len(generated_ids)))
                ),
            }
        )
        trimmed.append(out)
    return trimmed


def build_trace_batches(
    replay_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    batch_size: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    rows_by_doc = {int(row["doc_id"]): row for row in replay_rows}
    positions_by_doc: dict[int, list[dict[str, Any]]] = {}
    for item in selected:
        positions_by_doc.setdefault(int(item["doc_id"]), []).append(item)
    batches = fhm_common.build_batch_manifest(replay_rows, batch_size=batch_size, token_budget=token_budget)
    trace_batches: list[dict[str, Any]] = []
    for batch in batches:
        offset = 0
        selected_rows: list[dict[str, Any]] = []
        doc_offsets: dict[str, int] = {}
        doc_lengths: dict[str, int] = {}
        batch_rows = [rows_by_doc[int(doc_id)] for doc_id in batch["doc_indices"]]
        for row in batch_rows:
            doc_id = int(row["doc_id"])
            replay_token_count = int(row["replay_token_count"])
            doc_offsets[str(doc_id)] = offset
            doc_lengths[str(doc_id)] = replay_token_count
            for item in positions_by_doc.get(doc_id, []):
                local_index = int(item["row_index"])
                if 0 <= local_index < replay_token_count:
                    out = dict(item)
                    out["flat_index"] = offset + local_index
                    selected_rows.append(out)
            offset += replay_token_count
        selected_rows.sort(key=lambda row: int(row["flat_index"]))
        if not selected_rows:
            continue
        trace_batches.append(
            {
                **batch,
                "expected_rows": int(batch["total_replay_tokens"]),
                "selected_count": len(selected_rows),
                "token_indices": [int(row["flat_index"]) for row in selected_rows],
                "selected_rows": selected_rows,
                "doc_offsets": doc_offsets,
                "doc_lengths": doc_lengths,
            }
        )
    return trace_batches


def ensure_vllm_clean() -> None:
    status = subprocess.check_output(["git", "-C", str(VLLM_DIR), "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError(f"vLLM worktree is not clean:\n{status}")


def active_patch_name() -> str | None:
    status = subprocess.check_output([str(PATCH_SCRIPT), "status"], cwd=ROOT, text=True)
    first = status.splitlines()[0] if status.splitlines() else ""
    if first.strip() == "Active patch: none":
        return None
    prefix = "Active patch:"
    if first.startswith(prefix):
        return first[len(prefix) :].strip()
    return first.strip() or None


def selected_patch_name(args: argparse.Namespace) -> str:
    patch_names: set[str] = set()
    if not args.skip_reference:
        patch_names.add(REFERENCE_MODE.patch_name)
    if not args.skip_candidates:
        patch_names.update(CANDIDATE_MODES[name].patch_name for name in args.candidate_modes)
    if not patch_names:
        return PATCH_NAME
    if len(patch_names) != 1:
        modes = ", ".join(args.candidate_modes)
        raise RuntimeError(
            "selected modes require multiple vLLM patches; run them separately "
            f"(candidate_modes={modes}, skip_reference={args.skip_reference})"
        )
    return next(iter(patch_names))


def prepare_patch(patch_name: str) -> None:
    ensure_vllm_clean()
    active = active_patch_name()
    if active is not None:
        raise RuntimeError(f"vLLM already has active patch: {active}")
    subprocess.run([str(PATCH_SCRIPT), "prepare-version", patch_name], cwd=ROOT, check=True)
    importlib.invalidate_caches()


def revert_patch_best_effort() -> None:
    try:
        if active_patch_name() is not None:
            subprocess.run([str(PATCH_SCRIPT), "revert-version"], cwd=ROOT, check=True)
    finally:
        ensure_vllm_clean()


def configure_mode_env(mode: ModeSpec, mode_dir: Path) -> None:
    for key in list(os.environ):
        if key.startswith("VLLM_ERROR_CONTRIBUTION"):
            os.environ.pop(key, None)
    for key in (
        "HEAL_MODE",
        "HEAL_DYNAMIC_INT8_GEMM_WEIGHT",
        "HEAL_DYNAMIC_INT8_GEMM_WEIGHT_CHUNK_ROWS",
        "HEAL_INT8_GEMM_COMPENSATION",
        "HEAL_SKIP_FP16_COMP_LAYERS",
        "HEAL_SKIP_INT8_COMP_LAYERS",
        "HEAL_FP16_GEMM_COMPENSATION",
        "HEAL_FP16_GEMM_COMPENSATION_CHUNK_ROWS",
    ):
        os.environ.pop(key, None)
    os.environ["PYTHONPATH"] = (
        str(VLLM_DIR)
        if not os.environ.get("PYTHONPATH")
        else str(VLLM_DIR) + os.pathsep + os.environ["PYTHONPATH"]
    )
    os.environ["VLLM_VERSION_NAME"] = mode.patch_name
    os.environ["VLLM_EXPERIMENT_OUTPUT_DIR"] = str(mode_dir)
    os.environ["VLLM_DISABLE_CUDA_GRAPH"] = "1"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1" if mode.tf32 else "0"
    os.environ["VLLM_FLOAT32_MATMUL_PRECISION"] = "high" if mode.tf32 else "highest"
    if mode.layercast_force_fp32:
        os.environ["VLLM_LAYERCAST_FORCE_FP32"] = "1"
    else:
        os.environ.pop("VLLM_LAYERCAST_FORCE_FP32", None)
    if mode.heal_mode is not None:
        os.environ["HEAL_MODE"] = mode.heal_mode
    if mode.heal_fp16_gemm_compensation:
        os.environ["HEAL_FP16_GEMM_COMPENSATION"] = "1"
    if mode.layercast_force_fp32 or mode.heal_mode is not None:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    else:
        os.environ.pop("VLLM_DISABLE_COMPILE_CACHE", None)


def apply_torch_precision(tf32: bool) -> None:
    fhm_common.apply_torch_precision(tf32)


def add_local_vllm_to_path() -> None:
    vllm_path = str(VLLM_DIR)
    sys.path[:] = [path for path in sys.path if path != vllm_path]
    sys.path.insert(0, vllm_path)
    for module_name in list(sys.modules):
        if module_name == "vllm" or module_name.startswith("vllm."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def llm_kwargs(
    *,
    model_id: str,
    mode: ModeSpec,
    seed: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    cpu_offload_gb: float,
    trust_remote_code: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": mode.dtype,
        "seed": seed,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "trust_remote_code": trust_remote_code,
        "disable_log_stats": True,
    }
    if mode.attention_backend:
        kwargs["attention_backend"] = mode.attention_backend
    if cpu_offload_gb > 0:
        kwargs["cpu_offload_gb"] = cpu_offload_gb
    return kwargs


def shutdown_llm(llm: Any) -> None:
    for path in (("llm_engine", "engine_core", "shutdown"), ("llm_engine", "shutdown")):
        obj = llm
        try:
            for attr in path:
                obj = getattr(obj, attr)
            obj()
            break
        except Exception:
            pass
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass


def set_trace_env(
    *,
    mode: ModeSpec,
    output_file: Path,
    token_indices: list[int],
    expected_rows: int,
    reference_file: Path | None = None,
) -> None:
    os.environ["VLLM_ERROR_CONTRIBUTION_MODE"] = mode.trace_mode
    os.environ["VLLM_ERROR_CONTRIBUTION_OUTPUT_FILE"] = str(output_file)
    os.environ["VLLM_ERROR_CONTRIBUTION_TOKEN_INDICES"] = ",".join(str(idx) for idx in token_indices)
    os.environ["VLLM_ERROR_CONTRIBUTION_EXPECTED_ROWS"] = str(expected_rows)
    os.environ["VLLM_ERROR_CONTRIBUTION_STAGE_REGEX"] = STAGE_REGEX
    if reference_file is None:
        os.environ.pop("VLLM_ERROR_CONTRIBUTION_REFERENCE_FILE", None)
    else:
        os.environ["VLLM_ERROR_CONTRIBUTION_REFERENCE_FILE"] = str(reference_file)


def run_trace_mode(
    *,
    args: argparse.Namespace,
    model_id: str,
    mode: ModeSpec,
    mode_dir: Path,
    replay_rows: list[dict[str, Any]],
    trace_batches: list[dict[str, Any]],
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    add_local_vllm_to_path()
    configure_mode_env(mode, mode_dir)
    apply_torch_precision(mode.tf32)

    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.model_executor.models.error_contribution_trace import (
        clear_error_contribution_trace,
        flush_error_contribution_trace,
        set_error_contribution_token_filter,
    )

    rows_by_doc = {int(row["doc_id"]): row for row in replay_rows}
    max_replay_tokens = max(int(row["replay_token_count"]) for row in replay_rows)
    max_batch_tokens = max(int(batch["total_replay_tokens"]) for batch in trace_batches)
    max_model_len = max(int(args.max_model_len), max_replay_tokens + 1)
    max_num_batched_tokens = max(int(args.prefill_token_budget), max_batch_tokens, max_replay_tokens + 1)

    log(
        f"loading {model_id} mode={mode.name} dtype={mode.dtype} "
        f"batches={len(trace_batches)}"
    )
    load_started = time.perf_counter()
    llm = LLM(
        **llm_kwargs(
            model_id=model_id,
            mode=mode,
            seed=int(args.seed),
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=int(args.prefill_batch_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            cpu_offload_gb=float(args.cpu_offload_gb),
            trust_remote_code=bool(args.trust_remote_code),
        )
    )
    load_elapsed = time.perf_counter() - load_started
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    batch_rows_out: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for trace_batch in trace_batches:
            batch_id = int(trace_batch["batch_id"])
            output_file = (
                mode_dir
                / ("reference_batches" if mode.trace_mode == "reference" else "mse_batches")
                / (f"batch_{batch_id:04d}.safetensors" if mode.trace_mode == "reference" else f"batch_{batch_id:04d}.pt")
            )
            reference_file = None
            if mode.trace_mode == "candidate":
                assert reference_dir is not None
                reference_file = reference_dir / "reference_batches" / f"batch_{batch_id:04d}.safetensors"
                if not reference_file.exists():
                    raise FileNotFoundError(f"missing reference batch: {reference_file}")
            set_trace_env(
                mode=mode,
                output_file=output_file,
                token_indices=[int(idx) for idx in trace_batch["token_indices"]],
                expected_rows=int(trace_batch["expected_rows"]),
                reference_file=reference_file,
            )
            set_error_contribution_token_filter(
                [int(idx) for idx in trace_batch["token_indices"]],
                expected_rows=int(trace_batch["expected_rows"]),
            )
            clear_error_contribution_trace()
            batch_rows = [rows_by_doc[int(doc_id)] for doc_id in trace_batch["doc_indices"]]
            prompts = [TokensPrompt(prompt_token_ids=row["replay_token_ids"]) for row in batch_rows]
            batch_started = time.perf_counter()
            llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            flush_error_contribution_trace(output_file)
            elapsed = time.perf_counter() - batch_started
            clear_error_contribution_trace()
            batch_row = {
                "mode": mode.name,
                "batch_id": batch_id,
                "docs": len(batch_rows),
                "expected_rows": int(trace_batch["expected_rows"]),
                "selected_rows": int(trace_batch["selected_count"]),
                "output_file": str(output_file),
                "reference_file": str(reference_file) if reference_file else "",
                "elapsed_s": elapsed,
            }
            batch_rows_out.append(batch_row)
            log(
                f"mode={mode.name} batch={batch_id} docs={len(batch_rows)} "
                f"selected={trace_batch['selected_count']} elapsed={elapsed:.1f}s"
            )
    finally:
        shutdown_llm(llm)
        del llm
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    result = {
        "mode": asdict(mode),
        "model_id": model_id,
        "batches": len(batch_rows_out),
        "selected_rows": sum(int(row["selected_rows"]) for row in batch_rows_out),
        "elapsed_s": time.perf_counter() - started,
        "load_elapsed_s": load_elapsed,
        "mode_dir": str(mode_dir),
        "batch_manifest": str(mode_dir / "trace_batches.jsonl"),
    }
    write_jsonl(mode_dir / "trace_batches.jsonl", batch_rows_out)
    write_json(mode_dir / "worker_result.json", result)
    return result


def _section_float_tensor(payload: dict[str, Any], section: str, key: str) -> torch.Tensor | None:
    tensors = payload.get(section)
    if not isinstance(tensors, dict) or key not in tensors:
        return None
    value = tensors[key]
    if not torch.is_tensor(value):
        return None
    return value.to(torch.float32).reshape(-1)


def _float_tensor(payload: dict[str, Any], stage: str) -> torch.Tensor | None:
    return _section_float_tensor(payload, "stages", stage)


def _pair_tensor(
    payload: dict[str, Any],
    kind: str,
    metric: str,
    layer: int | None = None,
) -> torch.Tensor | None:
    key = f"{kind}.{metric}" if layer is None else f"{kind}.{metric}.{layer}"
    return _section_float_tensor(payload, "pairs", key)


PAIR_METRIC_FIELDS = {
    "delta_mse": "delta_mse",
    "covariance": "covariance",
    "cross": "cross",
    "mean_product": "mean_product",
    "before_mse": "before_mse",
    "before_delta_rho": "before_delta_rho",
    "centered_covariance": "centered_covariance",
}


def _layer_pair_tensors(
    payload: dict[str, Any],
    kind: str,
    layer: int,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for metric, suffix in PAIR_METRIC_FIELDS.items():
        tensor = _pair_tensor(payload, kind, metric, layer)
        if tensor is not None:
            tensors[f"{kind}_{suffix}"] = tensor
    return tensors


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (denom_x * denom_y)


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0, 0.0, 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    predicted = [slope * x + intercept for x in xs]
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return slope, intercept, r2


def _weighted_linear_fit(xs: list[float], ys: list[float], weights: list[float]) -> tuple[float, float, float]:
    if len(xs) < 2 or len(xs) != len(ys) or len(xs) != len(weights):
        return 0.0, 0.0, 0.0
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return _linear_fit(xs, ys)
    mean_x = sum(w * x for x, w in zip(xs, weights)) / weight_sum
    mean_y = sum(w * y for y, w in zip(ys, weights)) / weight_sum
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, weights))
    if sxx <= 0.0:
        return 0.0, mean_y, 0.0
    slope = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in zip(xs, ys, weights)) / sxx
    intercept = mean_y - slope * mean_x
    predicted = [slope * x + intercept for x in xs]
    ss_res = sum(w * (y - y_hat) ** 2 for y, y_hat, w in zip(ys, predicted, weights))
    ss_tot = sum(w * (y - mean_y) ** 2 for y, w in zip(ys, weights))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return slope, intercept, r2


def _density_inliers(
    points: list[tuple[float, float]],
    *,
    keep_fraction: float,
    k: int = 20,
) -> list[tuple[float, float]]:
    if len(points) < max(k + 2, 16):
        return list(points)
    keep_fraction = max(0.0, min(1.0, keep_fraction))
    if keep_fraction >= 1.0:
        return list(points)
    logs = np.array([[math.log10(x), math.log10(y)] for x, y in points], dtype=np.float64)
    center = np.median(logs, axis=0)
    q75 = np.quantile(logs, 0.75, axis=0)
    q25 = np.quantile(logs, 0.25, axis=0)
    scale = q75 - q25
    scale[scale <= 1e-12] = np.std(logs[:, scale <= 1e-12], axis=0) + 1e-12
    normalized = (logs - center) / scale
    delta = normalized[:, None, :] - normalized[None, :, :]
    distance_sq = np.sum(delta * delta, axis=2)
    kth = min(k, len(points) - 1)
    kth_distance = np.sqrt(np.partition(distance_sq, kth, axis=1)[:, kth])
    threshold = float(np.quantile(kth_distance, keep_fraction))
    return [point for point, distance in zip(points, kth_distance) if float(distance) <= threshold]


def summarize_layer_wls(
    rows: list[dict[str, Any]],
    *,
    trim_input_tail: float,
    robust_mad_threshold: float,
    density_keep_fraction: float,
) -> dict[str, float]:
    xs = [float(row["input_residual_mse"]) for row in rows]
    ys = [float(row["kv_mse"]) for row in rows]
    positive = [(x, y) for x, y in zip(xs, ys) if x > 0.0 and y > 0.0]
    log_pearson = (
        _pearson([math.log10(x) for x, _ in positive], [math.log10(y) for _, y in positive])
        if len(positive) >= 2
        else 0.0
    )
    ratios = [float(row["kv_over_input_residual_mse"]) for row in rows if float(row["kv_over_input_residual_mse"]) > 0.0]
    slope, intercept, r2 = _linear_fit(xs, ys)

    trim_threshold = _quantile(xs, max(0.0, min(1.0, 1.0 - trim_input_tail)))
    retained = [
        (x, y)
        for x, y in zip(xs, ys)
        if x > 0.0 and y >= 0.0 and x <= trim_threshold
    ]
    positive_retained = [(x, y) for x, y in retained if y > 0.0]
    density_retained = _density_inliers(
        positive_retained,
        keep_fraction=density_keep_fraction,
    )
    if len(density_retained) >= 2:
        log_xs = [math.log10(x) for x, _ in density_retained]
        log_ys = [math.log10(y) for _, y in density_retained]
        log_slope, log_intercept, _ = _linear_fit(log_xs, log_ys)
        residuals = [
            log_y - (log_slope * log_x + log_intercept)
            for log_x, log_y in zip(log_xs, log_ys)
        ]
        median_residual = _quantile(residuals, 0.5)
        abs_deviation = [abs(value - median_residual) for value in residuals]
        mad = _quantile(abs_deviation, 0.5)
        sigma = 1.4826 * mad
        if sigma > 0.0 and robust_mad_threshold > 0.0:
            residual_limit = robust_mad_threshold * sigma
            retained = [
                (x, y)
                for (x, y), residual in zip(density_retained, residuals)
                if abs(residual - median_residual) <= residual_limit
            ]
        else:
            residual_limit = float("inf")
            retained = list(density_retained)
    else:
        log_slope = 0.0
        log_intercept = 0.0
        mad = 0.0
        residual_limit = float("inf")

    retained_xs = [x for x, _ in retained]
    retained_ys = [y for _, y in retained]
    weight_floor = _quantile(retained_xs, 0.1)
    if weight_floor <= 0.0:
        weight_floor = min(retained_xs) if retained_xs else 0.0
    weights = [
        1.0 / max(x, weight_floor) if max(x, weight_floor) > 0.0 else 0.0
        for x in retained_xs
    ]
    wls_slope, wls_intercept, wls_r2 = _weighted_linear_fit(retained_xs, retained_ys, weights)
    retained_mean_kv = _mean(retained_ys)
    kv_mean = _mean(ys)
    direct_fraction_retained = max(wls_intercept, 0.0) / retained_mean_kv if retained_mean_kv > 0.0 else 0.0
    direct_fraction_retained = max(0.0, min(1.0, direct_fraction_retained))
    direct_fraction_all = max(wls_intercept, 0.0) / kv_mean if kv_mean > 0.0 else 0.0
    direct_fraction_all = max(0.0, min(1.0, direct_fraction_all))
    return {
        "points": float(len(rows)),
        "wls_points": float(len(retained)),
        "wls_density_points": float(len(density_retained)),
        "wls_density_keep_fraction": density_keep_fraction,
        "wls_trim_input_tail": trim_input_tail,
        "wls_input_trim_threshold": trim_threshold,
        "wls_log_prefit_slope": log_slope,
        "wls_log_prefit_intercept": log_intercept,
        "wls_log_residual_mad": mad,
        "wls_log_residual_limit": residual_limit,
        "wls_robust_mad_threshold": robust_mad_threshold,
        "wls_weight_floor_input_mse": weight_floor,
        "wls_fit_x_min": min(retained_xs) if retained_xs else 0.0,
        "wls_fit_x_max": max(retained_xs) if retained_xs else 0.0,
        "input_residual_mse_mean": _mean(xs),
        "input_residual_mse_p10": _quantile(xs, 0.1),
        "input_residual_mse_p50": _quantile(xs, 0.5),
        "input_residual_mse_p90": _quantile(xs, 0.9),
        "post_attention_residual_mse_mean": _mean([float(row["post_attention_residual_mse"]) for row in rows]),
        "post_attention_residual_mse_p10": _quantile([float(row["post_attention_residual_mse"]) for row in rows], 0.1),
        "post_attention_residual_mse_p50": _quantile([float(row["post_attention_residual_mse"]) for row in rows], 0.5),
        "post_attention_residual_mse_p90": _quantile([float(row["post_attention_residual_mse"]) for row in rows], 0.9),
        "k_mse_mean": _mean([float(row["k_mse"]) for row in rows]),
        "v_mse_mean": _mean([float(row["v_mse"]) for row in rows]),
        "kv_mse_mean": kv_mean,
        "final_norm_input_mse_mean": _mean([float(row["final_norm_input_mse"]) for row in rows]),
        "final_norm_input_mse_p10": _quantile([float(row["final_norm_input_mse"]) for row in rows], 0.1),
        "final_norm_input_mse_p50": _quantile([float(row["final_norm_input_mse"]) for row in rows], 0.5),
        "final_norm_input_mse_p90": _quantile([float(row["final_norm_input_mse"]) for row in rows], 0.9),
        "final_hidden_mse_mean": _mean([float(row["final_hidden_mse"]) for row in rows]),
        "final_hidden_mse_p10": _quantile([float(row["final_hidden_mse"]) for row in rows], 0.1),
        "final_hidden_mse_p50": _quantile([float(row["final_hidden_mse"]) for row in rows], 0.5),
        "final_hidden_mse_p90": _quantile([float(row["final_hidden_mse"]) for row in rows], 0.9),
        "pearson": _pearson(xs, ys),
        "log_pearson": log_pearson,
        "linear_slope": slope,
        "linear_intercept": intercept,
        "linear_r2": r2,
        "linear_intercept_over_mean_kv_mse": intercept / kv_mean if kv_mean > 0.0 else 0.0,
        "wls_slope": wls_slope,
        "wls_intercept": wls_intercept,
        "wls_r2": wls_r2,
        "wls_intercept_over_retained_mean_kv_mse": direct_fraction_retained,
        "wls_intercept_over_mean_kv_mse": direct_fraction_all,
        "kv_wls_direct_mse_mean": kv_mean * direct_fraction_all,
        "ratio_p10": _quantile(ratios, 0.1),
        "ratio_p50": _quantile(ratios, 0.5),
        "ratio_p90": _quantile(ratios, 0.9),
    }


def load_mse_batch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_reference_tensors(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if not path.exists():
        return tensors
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).to(torch.float32)
    return tensors


def _stage_tensor_from_mapping(mapping: dict[str, torch.Tensor], stage: str) -> torch.Tensor | None:
    value = mapping.get(stage)
    if torch.is_tensor(value):
        return value.to(torch.float32)
    prefix = f"{stage}__chunk_"
    chunks = [mapping[key].to(torch.float32) for key in sorted(mapping) if key.startswith(prefix)]
    if chunks:
        return torch.cat(chunks, dim=0)
    return None


def _candidate_stage_tensor(payload: dict[str, Any], stage: str) -> torch.Tensor | None:
    tensors = payload.get("candidate_tensors")
    if not isinstance(tensors, dict):
        return None
    value = tensors.get(stage)
    if torch.is_tensor(value):
        return value.to(torch.float32)
    return None


def _row_mse_tensor(delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim == 0:
        return delta.square().reshape(1).to(torch.float32)
    return delta.to(torch.float32).reshape(delta.shape[0], -1).square().mean(dim=1)


def _candidate_attention_value_dtype(candidate: str) -> torch.dtype:
    name = candidate.lower()
    if "bf16" in name:
        return torch.bfloat16
    if "fp16" in name or "heal" in name:
        return torch.float16
    return torch.float32


def _triton_value_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype in (torch.float16, torch.bfloat16):
        return tensor.to(dtype).to(torch.float32)
    return tensor.to(torch.float32)


def _doc_offset(doc_offsets: dict[str, Any], doc_id: int) -> int:
    value = doc_offsets.get(str(doc_id), doc_offsets.get(doc_id, 0))
    return int(value)


def _selected_or_full_value(tensor: torch.Tensor | None, idx: int, sample: dict[str, Any], selected_count: int) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    if tensor.numel() == selected_count:
        index = idx
    else:
        index = int(sample.get("flat_index", idx))
        if index >= tensor.numel():
            index = idx
    if index < 0 or index >= tensor.numel():
        return 0.0
    return float(tensor[index].item())


def _dense_prefix_attention(
    q_rows: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    selected_rows: list[dict[str, Any]],
    doc_offsets: dict[str, Any],
    *,
    value_dtype: torch.dtype,
) -> torch.Tensor:
    if q_rows.ndim != 3 or k_full.ndim != 3 or v_full.ndim != 3:
        raise ValueError("attention tensors must have shape [rows, heads, head_dim]")
    num_rows, num_heads, head_dim = q_rows.shape
    outputs = torch.zeros((num_rows, num_heads, head_dim), dtype=torch.float32)
    if num_rows == 0:
        return outputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = 1.0 / math.sqrt(float(head_dim))
    q_values = _triton_value_tensor(q_rows, value_dtype)
    k_values = _triton_value_tensor(k_full, value_dtype)
    v_values = _triton_value_tensor(v_full, value_dtype)
    by_doc: dict[int, list[int]] = {}
    for idx, sample in enumerate(selected_rows[:num_rows]):
        by_doc.setdefault(int(sample["doc_id"]), []).append(idx)
    for doc_id, indices in by_doc.items():
        row_indices = [int(selected_rows[idx]["row_index"]) for idx in indices]
        start = _doc_offset(doc_offsets, doc_id)
        stop = min(start + max(row_indices) + 1, int(k_values.shape[0]))
        if start < 0 or stop <= start:
            continue
        q = q_values[indices].to(device=device, dtype=torch.float32)
        k = k_values[start:stop].to(device=device, dtype=torch.float32)
        v = v_values[start:stop].to(device=device, dtype=torch.float32)
        if k.shape[1] != num_heads:
            repeat = max(1, math.ceil(num_heads / max(1, int(k.shape[1]))))
            k = k.repeat_interleave(repeat, dim=1)[:, :num_heads, :]
            v = v.repeat_interleave(repeat, dim=1)[:, :num_heads, :]
        scores = torch.einsum("shd,lhd->shl", q, k) * scale
        arange = torch.arange(k.shape[0], device=device).view(1, 1, -1)
        rows = torch.tensor(row_indices, device=device).view(-1, 1, 1)
        scores = scores.masked_fill(arange > rows, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("shl,lhd->shd", probs, v)
        outputs[indices] = out.cpu()
    return outputs


def _block_mlp_mse_components(
    ref_tensors: dict[str, torch.Tensor],
    payload: dict[str, Any],
    layer: int,
) -> dict[str, torch.Tensor]:
    ref_input = _stage_tensor_from_mapping(ref_tensors, f"block.mlp.input.{layer}")
    cand_input = _candidate_stage_tensor(payload, f"block.mlp.input.{layer}")
    ref_output = _stage_tensor_from_mapping(ref_tensors, f"block.mlp.output.{layer}")
    cand_output = _candidate_stage_tensor(payload, f"block.mlp.output.{layer}")
    oracle_output = _candidate_stage_tensor(payload, f"oracle.block.mlp.output.{layer}")
    if ref_output is None or cand_output is None or oracle_output is None:
        return {}
    n = min(ref_output.shape[0], cand_output.shape[0], oracle_output.shape[0])
    if n <= 0:
        return {}
    ref_output = ref_output[:n]
    cand_output = cand_output[:n]
    oracle_output = oracle_output[:n]
    components = {
        "mlp_total_mse": _row_mse_tensor(cand_output - ref_output),
        "mlp_input_effect_mse": _row_mse_tensor(oracle_output - ref_output),
        "mlp_compute_mse": _row_mse_tensor(cand_output - oracle_output),
    }
    components["mlp_cross_mse"] = (
        components["mlp_total_mse"]
        - components["mlp_input_effect_mse"]
        - components["mlp_compute_mse"]
    )
    if ref_input is not None and cand_input is not None:
        m = min(n, ref_input.shape[0], cand_input.shape[0])
        components["mlp_input_tensor_mse"] = _row_mse_tensor(cand_input[:m] - ref_input[:m])
    return components


def _attention_mse_components(
    ref_tensors: dict[str, torch.Tensor],
    payload: dict[str, Any],
    trace_batch: dict[str, Any],
    layer: int,
    *,
    candidate: str,
) -> dict[str, torch.Tensor]:
    ref_q = _stage_tensor_from_mapping(ref_tensors, f"attention.q.{layer}")
    ref_k = _stage_tensor_from_mapping(ref_tensors, f"attention.k_full.{layer}")
    ref_v = _stage_tensor_from_mapping(ref_tensors, f"attention.v_full.{layer}")
    ref_core = _stage_tensor_from_mapping(ref_tensors, f"attention.core_output.{layer}")
    cand_q = _candidate_stage_tensor(payload, f"attention.q.{layer}")
    cand_k = _candidate_stage_tensor(payload, f"attention.k_full.{layer}")
    cand_v = _candidate_stage_tensor(payload, f"attention.v_full.{layer}")
    cand_core = _candidate_stage_tensor(payload, f"attention.core_output.{layer}")
    required = [ref_q, ref_k, ref_v, ref_core, cand_q, cand_k, cand_v, cand_core]
    if any(value is None for value in required):
        return {}
    selected_rows = trace_batch["selected_rows"]
    n = min(len(selected_rows), ref_q.shape[0], ref_core.shape[0], cand_q.shape[0], cand_core.shape[0])
    if n <= 0:
        return {}
    selected_rows = selected_rows[:n]
    doc_offsets = trace_batch.get("doc_offsets", {})
    candidate_dtype = _candidate_attention_value_dtype(candidate)
    ref_dense = _dense_prefix_attention(
        ref_q[:n], ref_k, ref_v, selected_rows, doc_offsets, value_dtype=torch.float32
    )
    ref_kv_dense = _dense_prefix_attention(
        cand_q[:n], ref_k, ref_v, selected_rows, doc_offsets, value_dtype=candidate_dtype
    )
    cand_dense = _dense_prefix_attention(
        cand_q[:n], cand_k, cand_v, selected_rows, doc_offsets, value_dtype=candidate_dtype
    )
    ref_core = ref_core[:n]
    cand_core = cand_core[:n]
    total_delta = cand_core - ref_core
    q_delta = ref_kv_dense - ref_dense
    kv_delta = cand_dense - ref_kv_dense
    compute_delta = cand_core - cand_dense
    reference_backend_delta = ref_dense - ref_core
    q_effect_mse = _row_mse_tensor(q_delta)
    return {
        "attention_total_mse": _row_mse_tensor(total_delta),
        "attention_q_effect_mse": q_effect_mse,
        "attention_input_effect_mse": q_effect_mse,
        "attention_kv_effect_mse": _row_mse_tensor(kv_delta),
        "attention_compute_mse": _row_mse_tensor(compute_delta),
        "attention_reference_backend_mse": _row_mse_tensor(reference_backend_delta),
        "attention_cross_mse": (
            _row_mse_tensor(total_delta)
            - _row_mse_tensor(q_delta)
            - _row_mse_tensor(kv_delta)
            - _row_mse_tensor(compute_delta)
            - _row_mse_tensor(reference_backend_delta)
        ),
        "attention_q_tensor_mse": _row_mse_tensor(cand_q[:n] - ref_q[:n]),
    }


def _fit_kv_theil_sen(rows: list[dict[str, Any]], *, seed: int) -> dict[str, float]:
    xs = np.asarray([_safe_row_float(row, "input_residual_mse") for row in rows], dtype=np.float64)
    ys = np.asarray([_safe_row_float(row, "kv_mse") for row in rows], dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0.0) & (ys >= 0.0)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size == 0:
        return {"wls_slope": 0.0, "wls_intercept": 0.0, "wls_r2": 0.0, "wls_points": 0.0}
    slope = _sampled_theil_sen_slope(xs, ys, seed=seed)
    intercept = max(float(np.median(ys - slope * xs)), 0.0)
    predicted = slope * xs + intercept
    mean_y = float(np.mean(ys))
    ss_res = float(np.sum((ys - predicted) ** 2))
    ss_tot = float(np.sum((ys - mean_y) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {
        "wls_slope": float(slope),
        "wls_intercept": float(intercept),
        "wls_r2": float(r2),
        "wls_points": float(xs.size),
        "kv_fit_slope": float(slope),
        "kv_fit_intercept": float(intercept),
        "kv_fit_r2": float(r2),
    }


def _mse_token_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("batch_id", "")),
        str(row.get("sample_index", "")),
        str(row.get("doc_id", "")),
        str(row.get("row_index", "")),
    )


def mark_extreme_mse_token_outliers(
    point_rows: list[dict[str, Any]],
    *,
    quantile: float = DEFAULT_MSE_OUTLIER_QUANTILE,
    mad_multiplier: float = DEFAULT_MSE_OUTLIER_MAD_MULTIPLIER,
    enabled: bool = True,
) -> dict[str, float]:
    token_scores: dict[tuple[str, str, str, str], float] = {}
    for row in point_rows:
        values = [_safe_row_float(row, field, 0.0) for field in OUTLIER_SCORE_FIELDS]
        score = max(values) if values else 0.0
        key = _mse_token_key(row)
        token_scores[key] = max(token_scores.get(key, 0.0), score)
    finite_scores = [score for score in token_scores.values() if math.isfinite(score)]
    if not finite_scores:
        for row in point_rows:
            row["mse_outlier"] = 0
            row["mse_outlier_score"] = 0.0
            row["mse_outlier_threshold"] = 0.0
        return {
            "mse_outlier_tokens": 0.0,
            "mse_outlier_fraction": 0.0,
            "mse_outlier_threshold": 0.0,
            "mse_outlier_score_p995": 0.0,
            "mse_outlier_score_median": 0.0,
            "mse_outlier_score_mad": 0.0,
        }
    quantile = max(0.0, min(1.0, quantile))
    median = _quantile(finite_scores, 0.5)
    mad = _quantile([abs(score - median) for score in finite_scores], 0.5)
    quantile_threshold = _quantile(finite_scores, quantile)
    mad_threshold = median + mad_multiplier * 1.4826 * mad
    threshold = max(quantile_threshold, mad_threshold) if enabled else float("inf")
    outlier_tokens = {key for key, score in token_scores.items() if score > threshold}
    for row in point_rows:
        key = _mse_token_key(row)
        row["mse_outlier"] = 1 if key in outlier_tokens else 0
        row["mse_outlier_score"] = token_scores.get(key, 0.0)
        row["mse_outlier_threshold"] = threshold if math.isfinite(threshold) else ""
    total = float(len(token_scores))
    return {
        "mse_outlier_tokens": float(len(outlier_tokens)),
        "mse_outlier_fraction": float(len(outlier_tokens)) / total if total > 0.0 else 0.0,
        "mse_outlier_threshold": threshold if math.isfinite(threshold) else 0.0,
        "mse_outlier_score_p995": _quantile(finite_scores, 0.995),
        "mse_outlier_score_median": median,
        "mse_outlier_score_mad": mad,
    }


def _non_outlier_rows(point_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in point_rows if int(row.get("mse_outlier", 0) or 0) == 0]


def summarize_block_io_layers(
    candidate: str,
    by_layer: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for layer, rows in sorted(by_layer.items()):
        kv_fit = _fit_kv_theil_sen(rows, seed=1729 + 8191 * int(layer))
        kv_mean = _mean([_safe_row_float(row, "kv_mse") for row in rows])
        direct_fraction = kv_fit["wls_intercept"] / kv_mean if kv_mean > 0.0 else 0.0
        direct_fraction = max(0.0, min(1.0, direct_fraction))
        summary = {
            "candidate": candidate,
            "layer": layer,
            "points": float(len(rows)),
            "input_residual_mse_mean": _mean([_safe_row_float(row, "input_residual_mse") for row in rows]),
            "post_attention_residual_mse_mean": _mean([_safe_row_float(row, "post_attention_residual_mse") for row in rows]),
            "k_mse_mean": _mean([_safe_row_float(row, "k_mse") for row in rows]),
            "v_mse_mean": _mean([_safe_row_float(row, "v_mse") for row in rows]),
            "kv_mse_mean": kv_mean,
            "kv_wls_direct_mse_mean": kv_mean * direct_fraction,
            "wls_intercept_over_mean_kv_mse": direct_fraction,
            "mlp_total_mse_mean": _mean([_safe_row_float(row, "mlp_total_mse") for row in rows]),
            "mlp_input_effect_mse_mean": _mean([_safe_row_float(row, "mlp_input_effect_mse") for row in rows]),
            "mlp_compute_mse_mean": _mean([_safe_row_float(row, "mlp_compute_mse") for row in rows]),
            "mlp_cross_mse_mean": _mean([_safe_row_float(row, "mlp_cross_mse") for row in rows]),
            "attention_total_mse_mean": _mean([_safe_row_float(row, "attention_total_mse") for row in rows]),
            "attention_q_effect_mse_mean": _mean([_safe_row_float(row, "attention_q_effect_mse") for row in rows]),
            "attention_input_effect_mse_mean": _mean([_safe_row_float(row, "attention_input_effect_mse") for row in rows]),
            "attention_kv_effect_mse_mean": _mean([_safe_row_float(row, "attention_kv_effect_mse") for row in rows]),
            "attention_compute_mse_mean": _mean([_safe_row_float(row, "attention_compute_mse") for row in rows]),
            "attention_reference_backend_mse_mean": _mean([_safe_row_float(row, "attention_reference_backend_mse") for row in rows]),
            "attention_cross_mse_mean": _mean([_safe_row_float(row, "attention_cross_mse") for row in rows]),
            "final_norm_input_mse_mean": _mean([_safe_row_float(row, "final_norm_input_mse") for row in rows]),
            "final_hidden_mse_mean": _mean([_safe_row_float(row, "final_hidden_mse") for row in rows]),
        }
        summary.update(kv_fit)
        summary_rows.append(summary)
    return summary_rows


def _clip_share(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def build_block_io_contributions(
    *,
    candidate: str,
    summary_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    incoming = 1.0
    contribution_rows: list[dict[str, Any]] = []
    for summary in sorted(summary_rows, key=lambda row: int(row["layer"]), reverse=True):
        layer = int(summary["layer"])
        mlp_total = float(summary.get("mlp_total_mse_mean", 0.0))
        attn_total = float(summary.get("attention_total_mse_mean", 0.0))
        mlp_local = _clip_share(float(summary.get("mlp_compute_mse_mean", 0.0)) / mlp_total) if mlp_total > 0.0 else 0.0
        mlp_final = incoming * mlp_local
        incoming_to_attention = incoming * (1.0 - mlp_local)

        attn_local = _clip_share(float(summary.get("attention_compute_mse_mean", 0.0)) / attn_total) if attn_total > 0.0 else 0.0
        kv_fraction = _clip_share(float(summary.get("wls_intercept_over_mean_kv_mse", 0.0)))
        kv_local_mse = float(summary.get("attention_kv_effect_mse_mean", 0.0)) * kv_fraction
        kv_local = _clip_share(kv_local_mse / attn_total) if attn_total > 0.0 else 0.0
        if attn_local + kv_local > 1.0:
            kv_local = max(0.0, 1.0 - attn_local)
        attn_final = incoming_to_attention * attn_local
        kv_final = incoming_to_attention * kv_local
        incoming = incoming_to_attention * max(0.0, 1.0 - attn_local - kv_local)
        contribution_rows.append(
            {
                "candidate": candidate,
                "layer": layer,
                "mlp_compute_local_share": mlp_local,
                "attention_compute_local_share": attn_local,
                "kv_local_share": kv_local,
                "mlp_compute_final_share": mlp_final,
                "attention_compute_final_share": attn_final,
                "kv_final_share": kv_final,
                "remaining_input_share_after_layer": incoming,
                "layer_ending_tensor_mse_mean": mlp_total,
                "attention_total_mse_mean": attn_total,
                "kv_fit_intercept_over_mean_kv_mse": kv_fraction,
            }
        )
    contribution_rows.sort(key=lambda row: int(row["layer"]))
    summary = {
        "total_attention_compute_share": sum(float(row["attention_compute_final_share"]) for row in contribution_rows),
        "total_mlp_compute_share": sum(float(row["mlp_compute_final_share"]) for row in contribution_rows),
        "total_kv_share": sum(float(row["kv_final_share"]) for row in contribution_rows),
        "unattributed_input_share": incoming,
    }
    summary["total_modeled_share"] = (
        summary["total_attention_compute_share"]
        + summary["total_mlp_compute_share"]
        + summary["total_kv_share"]
    )
    return contribution_rows, summary


def plot_kv_fit_panels(
    *,
    model_label: str,
    candidate: str,
    point_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, str]:
    if not point_rows or not summary_rows:
        return {}
    configure_matplotlib()
    import matplotlib.pyplot as plt

    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in point_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    summary_by_layer = {int(row["layer"]): row for row in summary_rows}
    layers = sorted(by_layer)
    display_candidate = display_candidate_name(candidate)
    model_title = model_label.replace("-", " ")
    paths: dict[str, str] = {}
    for chunk_start in range(0, len(layers), 16):
        chunk = layers[chunk_start:chunk_start + 16]
        cols = 4
        rows = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 2.45), squeeze=False)
        for ax, layer in zip(axes.ravel(), chunk):
            values = by_layer[layer]
            xs = np.asarray([_safe_row_float(row, "input_residual_mse") for row in values], dtype=np.float64)
            ys = np.asarray([_safe_row_float(row, "kv_mse") for row in values], dtype=np.float64)
            valid = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
            ax.scatter(xs[valid], ys[valid], s=4, alpha=0.38, linewidths=0, color="#4c78a8")
            summary = summary_by_layer.get(layer, {})
            slope = float(summary.get("wls_slope", 0.0))
            intercept = float(summary.get("wls_intercept", 0.0))
            if bool(valid.any()):
                xmin = float(np.nanmin(xs[valid]))
                xmax = float(np.nanmax(xs[valid]))
                if xmin > 0.0 and xmax > xmin:
                    line_x = np.geomspace(xmin, xmax, 96)
                    ax.plot(line_x, slope * line_x + intercept, color="#d62728", linewidth=1.1)
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_title(f"L{layer}  k={slope:.2g} b={intercept:.2g} R2={float(summary.get('wls_r2', 0.0)):.2f}", fontsize=7.8)
            ax.grid(True, alpha=0.22, linewidth=0.5)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.suptitle(f"{model_title} {display_candidate_short_label(candidate)} KV Theil-Sen fit", fontsize=11.0)
        fig.supxlabel("Input residual MSE", fontsize=9.0)
        fig.supylabel("KV tensor MSE", fontsize=9.0)
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
        output_path = figure_dir / f"{display_candidate}_kv_theil_sen_fit_layers_{chunk[0]}_{chunk[-1]}.png"
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths[f"kv_fit_panel_{chunk[0]}_{chunk[-1]}"] = str(output_path)
    return paths


def safe_log10_array(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(arr > 0.0, arr, np.nan)
    return np.log10(arr)


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return value.strip("_") or "unknown"


def detect_gpu_name() -> str:
    if torch.cuda.is_available():
        try:
            return sanitize_name(torch.cuda.get_device_name(0))
        except Exception:
            pass
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
        return sanitize_name(raw)
    except Exception:
        return "unknown_gpu"


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )



def residual_boundary_series(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(summary_rows, key=lambda row: int(row["layer"]))
    series: list[dict[str, Any]] = []
    for row in rows:
        layer = int(row["layer"])
        series.append(
            {
                "index": 2 * layer,
                "label": f"L{layer} in",
                "kind": "input residual",
                "mean": float(row["input_residual_mse_mean"]),
                "p10": float(row["input_residual_mse_p10"]),
                "p50": float(row["input_residual_mse_p50"]),
                "p90": float(row["input_residual_mse_p90"]),
            }
        )
        series.append(
            {
                "index": 2 * layer + 1,
                "label": f"L{layer} post",
                "kind": "post-attn residual",
                "mean": float(row["post_attention_residual_mse_mean"]),
                "p10": float(row["post_attention_residual_mse_p10"]),
                "p50": float(row["post_attention_residual_mse_p50"]),
                "p90": float(row["post_attention_residual_mse_p90"]),
            }
        )
    if rows:
        last = rows[-1]
        final_input_index = 2 * len(rows)
        series.append(
            {
                "index": final_input_index,
                "label": "final in",
                "kind": "final norm input",
                "mean": float(last["final_norm_input_mse_mean"]),
                "p10": float(last["final_norm_input_mse_p10"]),
                "p50": float(last["final_norm_input_mse_p50"]),
                "p90": float(last["final_norm_input_mse_p90"]),
            }
        )
        series.append(
            {
                "index": final_input_index + 1,
                "label": "final hidden",
                "kind": "final hidden",
                "mean": float(last["final_hidden_mse_mean"]),
                "p10": float(last["final_hidden_mse_p10"]),
                "p50": float(last["final_hidden_mse_p50"]),
                "p90": float(last["final_hidden_mse_p90"]),
            }
        )
    return series


def adjacent_residual_pair_rows(point_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[int, dict[int, dict[str, Any]]] = {}
    for row in point_rows:
        by_layer.setdefault(int(row["layer"]), {})[int(row["sample_index"])] = row
    layers = sorted(by_layer)
    pairs: list[dict[str, Any]] = []
    for layer in layers:
        current = by_layer[layer]
        for sample_index, row in current.items():
            pairs.append(
                {
                    "pair_index": 2 * layer,
                    "pair": f"L{layer} input -> L{layer} post",
                    "pair_kind": "attention block",
                    "x": float(row["input_residual_mse"]),
                    "y": float(row["post_attention_residual_mse"]),
                }
            )
        next_layer = layer + 1
        if next_layer in by_layer:
            nxt = by_layer[next_layer]
            for sample_index, row in current.items():
                next_row = nxt.get(sample_index)
                if next_row is None:
                    continue
                pairs.append(
                    {
                        "pair_index": 2 * layer + 1,
                        "pair": f"L{layer} post -> L{next_layer} input",
                        "pair_kind": "mlp block",
                        "x": float(row["post_attention_residual_mse"]),
                        "y": float(next_row["input_residual_mse"]),
                    }
                )
        else:
            for sample_index, row in current.items():
                pairs.append(
                    {
                        "pair_index": 2 * layer + 1,
                        "pair": f"L{layer} post -> final input",
                        "pair_kind": "final mlp block",
                        "x": float(row["post_attention_residual_mse"]),
                        "y": float(row["final_norm_input_mse"]),
                    }
                )
                pairs.append(
                    {
                        "pair_index": 2 * layer + 2,
                        "pair": "final input -> final hidden",
                        "pair_kind": "final rmsnorm",
                        "x": float(row["final_norm_input_mse"]),
                        "y": float(row["final_hidden_mse"]),
                    }
                )
    return pairs


def _finite_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(float(row["x"]), float(row["y"])) for row in rows]
    valid = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if not valid:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    xs, ys = zip(*valid)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _clip_for_log(values: np.ndarray) -> tuple[np.ndarray, float, int]:
    positive = values[values > 0.0]
    if positive.size:
        floor = float(np.min(positive)) / 10.0
    else:
        floor = 1.0e-30
    clipped = np.where(values > 0.0, values, floor)
    return clipped, floor, int(np.count_nonzero(values <= 0.0))


def _padded_log_limits(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    logx = np.log10(xs)
    logy = np.log10(ys)
    low_x, high_x = float(np.min(logx)), float(np.max(logx))
    low_y, high_y = float(np.min(logy)), float(np.max(logy))
    if high_x <= low_x:
        low_x -= 0.5
        high_x += 0.5
    if high_y <= low_y:
        low_y -= 0.5
        high_y += 0.5
    pad_x = 0.08 * (high_x - low_x)
    pad_y = 0.08 * (high_y - low_y)
    return (10 ** (low_x - pad_x), 10 ** (high_x + pad_x), 10 ** (low_y - pad_y), 10 ** (high_y + pad_y))


def _plot_pair_scatter(
    ax: Any,
    rows_for_pair: list[dict[str, Any]],
    pair_name: str,
    pair_kind: str,
) -> None:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    xs_raw, ys_raw = _finite_xy(rows_for_pair)
    xs, _x_floor, x_clipped = _clip_for_log(xs_raw)
    ys, _y_floor, y_clipped = _clip_for_log(ys_raw)
    if xs.size:
        logx = np.log10(xs)
        logy = np.log10(ys)
        low_x, high_x = np.quantile(logx, [0.01, 0.99])
        low_y, high_y = np.quantile(logy, [0.01, 0.99])
        inlier = (logx >= low_x) & (logx <= high_x) & (logy >= low_y) & (logy <= high_y)
        outlier = ~inlier
        ax.scatter(xs[inlier], ys[inlier], s=1.0, alpha=0.16, color="#277da1", linewidths=0)
        if inlier.any():
            xmin, xmax, ymin, ymax = _padded_log_limits(xs[inlier], ys[inlier])
        else:
            xmin, xmax, ymin, ymax = _padded_log_limits(xs, ys)
        if outlier.any():
            inset = inset_axes(ax, width="34%", height="31%", loc="lower right", borderpad=0.75)
            inset.scatter(xs[outlier], ys[outlier], s=1.0, alpha=0.28, color="#f94144", linewidths=0)
            inset.set_xscale("log")
            inset.set_yscale("log")
            inset.set_title(f"out {int(outlier.sum())}", fontsize=5.8)
            inset.tick_params(labelsize=5.2, pad=1)
            inset.grid(True, alpha=0.16)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        diag_min = max(min(xmin, ymin), np.finfo(np.float64).tiny)
        diag_max = max(xmax, ymax)
        ax.plot([diag_min, diag_max], [diag_min, diag_max], color="#111111", linewidth=0.55, alpha=0.45)
        clip_label = ""
        if x_clipped or y_clipped:
            clip_label = f"\nclip x={x_clipped:,} y={y_clipped:,}"
        ax.text(
            0.03,
            0.97,
            f"n={xs.size:,}{clip_label}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.6,
            bbox={"facecolor": "white", "alpha": 0.68, "edgecolor": "none", "pad": 1.5},
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{pair_name}\n{pair_kind}", fontsize=7.3)
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=6.2, pad=1)


def plot_residual_mse_diagnostics(
    *,
    model_label: str,
    candidate: str,
    point_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, str]:
    if not point_rows or not summary_rows:
        return {}
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figure_dir.mkdir(parents=True, exist_ok=True)
    display_candidate = {
        "original_bf16": "native_bf16",
        "original_fp16": "native_fp16",
        "heal_base": "heal_base",
        "heal_plus": "heal_plus",
        "heal_full": "heal_full",
    }.get(candidate, candidate)
    output_pdf = figure_dir / f"{display_candidate}_residual_mse_diagnostics.pdf"
    series = residual_boundary_series(summary_rows)
    pair_rows = adjacent_residual_pair_rows(point_rows)

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(12.5, 4.8))
        x = np.array([row["index"] for row in series], dtype=np.float64)
        mean = np.array([row["mean"] for row in series], dtype=np.float64)
        p10 = np.array([row["p10"] for row in series], dtype=np.float64)
        p50 = np.array([row["p50"] for row in series], dtype=np.float64)
        p90 = np.array([row["p90"] for row in series], dtype=np.float64)
        ax.fill_between(x, p10, p90, color="#8ecae6", alpha=0.24, label="p10-p90")
        ax.plot(x, p50, color="#277da1", linewidth=1.3, label="median")
        ax.plot(x, mean, color="#111111", linewidth=1.1, marker="o", markersize=2.8, label="mean")
        marker_styles = {
            "input residual": ("#f94144", "input residual", 20),
            "post-attn residual": ("#43aa8b", "post-attn residual", 20),
            "final norm input": ("#577590", "final norm input", 26),
            "final hidden": ("#f8961e", "final hidden", 34),
        }
        for kind, (color, label, size) in marker_styles.items():
            xs = [row["index"] for row in series if row["kind"] == kind]
            if xs:
                ax.scatter(xs, [mean[int(i)] for i in xs], s=size, color=color, label=label, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("residual boundary order")
        ax.set_ylabel("MSE vs FP32 LayerCast")
        ax.set_title(f"{model_label} {display_candidate}: residual stream MSE flow")
        ax.grid(True, axis="y", alpha=0.22)
        tick_stride = max(1, len(series) // 18)
        ax.set_xticks(x[::tick_stride])
        ax.set_xticklabels([series[int(i)]["label"] for i in x[::tick_stride]], rotation=45, ha="right")
        ax.legend(frameon=False, ncol=4, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        grouped_pairs: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
        for row in pair_rows:
            key = (int(row["pair_index"]), str(row["pair"]), str(row["pair_kind"]))
            grouped_pairs.setdefault(key, []).append(row)
        pair_items = sorted(grouped_pairs.items(), key=lambda item: item[0][0])
        plots_per_page = 12
        for page_start in range(0, len(pair_items), plots_per_page):
            page_items = pair_items[page_start : page_start + plots_per_page]
            fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.4), sharex=False, sharey=False)
            flat_axes = list(axes.flat)
            for ax, ((_, pair_name, pair_kind), rows_for_pair) in zip(flat_axes, page_items):
                _plot_pair_scatter(ax, rows_for_pair, pair_name, pair_kind)
            for ax in flat_axes[len(page_items) :]:
                ax.axis("off")
            fig.suptitle(
                f"{model_label} {display_candidate}: adjacent residual-boundary scatter",
                fontsize=12.0,
                y=0.995,
            )
            fig.supxlabel("previous residual boundary MSE", fontsize=10.0)
            fig.supylabel("next residual boundary MSE", fontsize=10.0)
            fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.97])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return {"diagnostics_pdf": str(output_pdf)}


def plot_residual_delta_panels(
    *,
    model_label: str,
    candidate: str,
    point_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, str]:
    if not point_rows:
        return {}
    configure_matplotlib()
    import matplotlib.pyplot as plt

    display_candidate = {
        "original_bf16": "native_bf16",
        "original_fp16": "native_fp16",
        "heal_base": "heal_base",
        "heal_plus": "heal_plus",
        "heal_full": "heal_full",
    }.get(candidate, candidate)
    output_dir = figure_dir / "delta-panels"
    output_dir.mkdir(parents=True, exist_ok=True)

    add_prefix_kv_and_cross_fields(point_rows)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in point_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)

    modules = [
        (
            "attention",
            "input_residual_mse",
            "attention_delta_mse",
            "attention_covariance",
            "Attention",
            "#2878b5",
        ),
        (
            "mlp",
            "post_attention_residual_mse",
            "mlp_delta_mse",
            "mlp_covariance",
            "MLP",
            "#6f4aa1",
        ),
    ]
    figure_paths: dict[str, str] = {}
    max_points_per_layer = 900
    for module, x_field, y_field, cov_field, title, color in modules:
        layers = sorted(
            layer for layer, rows in by_layer.items() if any(y_field in row for row in rows)
        )
        for start in range(0, len(layers), 16):
            chunk = layers[start : start + 16]
            if not chunk:
                continue
            fig, axes = plt.subplots(4, 4, figsize=(13.4, 10.2), sharex=False, sharey=False)
            flat_axes = list(axes.flat)
            for ax, layer in zip(flat_axes, chunk):
                triples = [
                    (
                        float(row[x_field]),
                        float(row[y_field]),
                        float(row[cov_field]) if cov_field in row and row[cov_field] != "" else float("nan"),
                    )
                    for row in by_layer[layer]
                    if x_field in row and y_field in row
                ]
                valid = [
                    (x, y, cov)
                    for x, y, cov in triples
                    if math.isfinite(x) and math.isfinite(y)
                ]
                if valid:
                    x_values = np.asarray([x for x, _, _ in valid], dtype=np.float64)
                    y_values = np.asarray([y for _, y, _ in valid], dtype=np.float64)
                    if x_values.size > max_points_per_layer:
                        keep = np.unique(
                            np.linspace(0, x_values.size - 1, max_points_per_layer, dtype=np.int64)
                        )
                        x_plot_raw = x_values[keep]
                        y_plot_raw = y_values[keep]
                    else:
                        x_plot_raw = x_values
                        y_plot_raw = y_values
                    x_plot, _xfloor, x_clipped = _clip_for_log(x_plot_raw)
                    y_plot, _yfloor, y_clipped = _clip_for_log(y_plot_raw)
                    ax.scatter(x_plot, y_plot, s=2.2, alpha=0.22, color=color, linewidths=0)
                    xmin, xmax, ymin, ymax = _padded_log_limits(x_plot, y_plot)
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    cov_values = [cov for _, _, cov in valid if math.isfinite(cov)]
                    annotation = f"n={len(valid):,}"
                    if cov_values:
                        annotation += f"\ncov={_mean(cov_values):.2e}"
                    if x_clipped or y_clipped:
                        annotation += f"\nclip x={x_clipped:,} y={y_clipped:,}"
                    ax.text(
                        0.03,
                        0.97,
                        annotation,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=6.3,
                        bbox={"facecolor": "white", "alpha": 0.74, "edgecolor": "none", "pad": 1.4},
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"L{layer}", fontsize=8.0, pad=2.0)
                ax.grid(True, alpha=0.18)
                ax.tick_params(labelsize=6.2, pad=1)
            for ax in flat_axes[len(chunk) :]:
                ax.axis("off")
            fig.suptitle(
                f"{model_label} {display_candidate}: {title} MSE[after - before]",
                fontsize=11.0,
                y=0.995,
            )
            fig.supxlabel("MSE[before]", fontsize=9.5)
            fig.supylabel("MSE[after - before]", fontsize=9.5)
            fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.965])
            output_path = output_dir / (
                f"{display_candidate}_{module}_delta_mse_layers_{chunk[0]:02d}_{chunk[-1]:02d}.png"
            )
            fig.savefig(output_path, dpi=190, bbox_inches="tight")
            plt.close(fig)
            figure_paths[f"{module}_delta_panel_{chunk[0]}_{chunk[-1]}"] = str(output_path)
    return figure_paths


def _safe_row_float(row: dict[str, Any], field: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(field, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def display_candidate_name(candidate: str) -> str:
    return {
        "original_bf16": "native_bf16",
        "original_fp16": "native_fp16",
        "heal_base": "heal_base",
        "heal_plus": "heal_plus",
        "heal_full": "heal_full",
    }.get(candidate, candidate)


def display_candidate_short_label(candidate: str) -> str:
    return {
        "original_bf16": "BF16",
        "original_fp16": "FP16",
        "heal_base": "HEAL",
        "heal_plus": "HEAL+",
        "heal_full": "HEAL-Full",
    }.get(candidate, candidate)


def add_prefix_kv_and_cross_fields(point_rows: list[dict[str, Any]]) -> None:
    by_layer_doc: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in point_rows:
        if "attention_mean_product" in row:
            row["attention_cross"] = _safe_row_float(row, "attention_mean_product") - _safe_row_float(
                row, "input_residual_mse"
            )
        if "mlp_mean_product" in row:
            row["mlp_cross"] = _safe_row_float(row, "mlp_mean_product") - _safe_row_float(
                row, "post_attention_residual_mse"
            )
        by_layer_doc.setdefault((int(row["layer"]), int(row["doc_id"])), []).append(row)
    for rows in by_layer_doc.values():
        rows.sort(key=lambda row: int(row["row_index"]))
        running = 0.0
        count = 0
        for row in rows:
            running += _safe_row_float(row, "kv_mse")
            count += 1
            row["prefix_kv_mse"] = running / count if count else 0.0


def _log_space_wls_design(
    features: list[list[float]],
    targets: list[float],
    scales: list[float],
    *,
    iterations: int = 4,
    target_floor_quantile: float = 0.01,
    density_keep_fraction: float = 1.0,
    robust_mad_threshold: float = 0.0,
) -> tuple[list[list[float]], list[float], list[float], dict[str, float]]:
    clean_features: list[list[float]] = []
    clean_targets: list[float] = []
    clean_scales: list[float] = []
    for xs, y, scale in zip(features, targets, scales):
        if not math.isfinite(y) or y < 0.0 or not math.isfinite(scale):
            continue
        if not all(math.isfinite(x) for x in xs):
            continue
        clean_features.append([float(x) for x in xs])
        clean_targets.append(float(y))
        clean_scales.append(float(scale))
    if not clean_features:
        return [], [], [], {"fit_points": 0.0, "fit_total_points": 0.0}

    positive_targets = [value for value in clean_targets if value > 0.0 and math.isfinite(value)]
    y_floor = _quantile(positive_targets, target_floor_quantile) if positive_targets else 0.0
    if y_floor <= 0.0:
        y_floor = min(positive_targets) if positive_targets else 1.0e-30
    y_floor = max(y_floor, 1.0e-30)

    base_weights = [1.0 / (max(value, y_floor) ** 2) for value in clean_targets]
    coef, intercept, _r2 = _weighted_nonnegative_multi_fit(clean_features, clean_targets, base_weights)
    predictions = [
        max(sum(c * x for c, x in zip(coef, xs)) + intercept, y_floor)
        for xs in clean_features
    ]
    residuals = [
        math.log(max(y, y_floor)) - math.log(pred)
        for y, pred in zip(clean_targets, predictions)
    ]
    if residuals:
        median_residual = _quantile(residuals, 0.5)
        mad = _quantile([abs(value - median_residual) for value in residuals], 0.5)
    else:
        median_residual = 0.0
        mad = 0.0
    sigma = max(1.4826 * mad, 1.0e-12)
    huber_limit = max(robust_mad_threshold, 0.0) * sigma if robust_mad_threshold > 0.0 else float("inf")
    robust_weights = [
        1.0 if abs(value - median_residual) <= huber_limit else huber_limit / max(abs(value - median_residual), 1.0e-12)
        for value in residuals
    ]
    weights = [base * robust for base, robust in zip(base_weights, robust_weights)]

    diagnostics = {
        "fit_points": float(len(clean_features)),
        "fit_total_points": float(len(clean_features)),
        "fit_drop_share": 0.0,
        "fit_density_points": float(len(clean_features)),
        "fit_weight_floor": y_floor,
        "fit_log_prefit_slope": 0.0,
        "fit_log_prefit_intercept": 0.0,
        "fit_log_residual_mad": mad,
        "fit_log_residual_limit": huber_limit,
    }
    return clean_features, clean_targets, weights, diagnostics


def _target_log_tail_inlier_mask(
    targets: list[float] | np.ndarray,
    *,
    target_floor_quantile: float = 0.01,
    sigma_threshold: float = 5.0,
    min_keep_fraction: float = 0.90,
) -> tuple[list[bool], dict[str, float]]:
    values = [float(value) for value in targets if math.isfinite(float(value)) and float(value) >= 0.0]
    total = len(targets)
    if total == 0 or len(values) != total:
        mask = [math.isfinite(float(value)) and float(value) >= 0.0 for value in targets]
        return mask, {"target_tail_total_points": float(total), "target_tail_excluded_points": float(total - sum(mask))}
    positive = [value for value in values if value > 0.0]
    y_floor = _quantile(positive, target_floor_quantile) if positive else 0.0
    if y_floor <= 0.0:
        y_floor = min(positive) if positive else 1.0e-30
    y_floor = max(y_floor, 1.0e-30)
    if sigma_threshold <= 0.0:
        return [True for _ in values], {
            "target_tail_total_points": float(total),
            "target_tail_excluded_points": 0.0,
            "target_tail_excluded_share": 0.0,
            "target_tail_floor": y_floor,
            "target_tail_log_median": 0.0,
            "target_tail_log_sigma": 0.0,
            "target_tail_log_limit": float("inf"),
        }
    log_values = np.log(np.maximum(np.asarray(values, dtype=np.float64), y_floor))
    median = float(np.median(log_values))
    mad = float(np.median(np.abs(log_values - median)))
    sigma = 1.4826 * mad
    if sigma <= 1.0e-12 or not math.isfinite(sigma):
        return [True for _ in values], {
            "target_tail_total_points": float(total),
            "target_tail_excluded_points": 0.0,
            "target_tail_excluded_share": 0.0,
            "target_tail_floor": y_floor,
            "target_tail_log_median": median,
            "target_tail_log_sigma": sigma,
            "target_tail_log_limit": float("inf"),
        }
    limit = median + float(sigma_threshold) * sigma
    if min_keep_fraction > 0.0:
        keep_fraction = min(max(float(min_keep_fraction), 0.0), 1.0)
        if keep_fraction < 1.0:
            limit = max(limit, float(np.quantile(log_values, keep_fraction)))
    mask_array = log_values <= limit
    if not bool(mask_array.any()):
        mask_array = np.ones_like(log_values, dtype=bool)
    excluded = int(mask_array.size - int(np.sum(mask_array)))
    return [bool(value) for value in mask_array], {
        "target_tail_total_points": float(total),
        "target_tail_excluded_points": float(excluded),
        "target_tail_excluded_share": float(excluded / total) if total else 0.0,
        "target_tail_floor": y_floor,
        "target_tail_log_median": median,
        "target_tail_log_sigma": sigma,
        "target_tail_log_limit": float(limit),
        "target_tail_sigma_threshold": float(sigma_threshold),
        "target_tail_min_keep_fraction": float(min_keep_fraction),
    }


def _log_loss_weights(residuals: np.ndarray, *, loss: str, f_scale: float) -> np.ndarray:
    if residuals.size == 0:
        return residuals.astype(np.float64)
    safe_scale = max(float(f_scale), 1.0e-12)
    z = (residuals / safe_scale) ** 2
    if loss == "linear":
        return np.ones_like(residuals, dtype=np.float64)
    if loss == "soft_l1":
        return 1.0 / np.sqrt(1.0 + z)
    if loss == "huber":
        abs_scaled = np.abs(residuals) / safe_scale
        return np.where(abs_scaled <= 1.0, 1.0, 1.0 / np.maximum(abs_scaled, 1.0e-12))
    if loss == "cauchy":
        return 1.0 / (1.0 + z)
    return np.ones_like(residuals, dtype=np.float64)


def _fit_normal_space_nonnegative_multi(
    features: list[list[float]],
    targets: list[float],
    *,
    fit_intercept: bool = True,
    target_floor_quantile: float = 0.01,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 512,
    exclude_target_log_tail: bool = True,
    target_log_tail_sigma: float = 6.0,
    target_log_tail_min_keep_fraction: float = 0.98,
) -> tuple[list[float], float, float, list[float], dict[str, float]]:
    clean_features: list[list[float]] = []
    clean_targets: list[float] = []
    width = len(features[0]) if features else 0
    for xs, y in zip(features, targets):
        if len(xs) != width or not math.isfinite(y) or y < 0.0:
            continue
        if not all(math.isfinite(x) for x in xs):
            continue
        clean_features.append([float(x) for x in xs])
        clean_targets.append(float(y))
    if not clean_features:
        return [0.0 for _ in range(width)], 0.0, 0.0, [], {"fit_points": 0.0, "fit_total_points": 0.0}

    positive_targets = [value for value in clean_targets if value > 0.0 and math.isfinite(value)]
    y_floor = _quantile(positive_targets, target_floor_quantile) if positive_targets else 0.0
    if y_floor <= 0.0:
        y_floor = min(positive_targets) if positive_targets else 1.0e-30
    y_floor = max(y_floor, 1.0e-30)

    if exclude_target_log_tail:
        target_mask, tail_diag = _target_log_tail_inlier_mask(
            clean_targets,
            target_floor_quantile=target_floor_quantile,
            sigma_threshold=target_log_tail_sigma,
            min_keep_fraction=target_log_tail_min_keep_fraction,
        )
        fit_features = [row for row, keep in zip(clean_features, target_mask) if keep]
        fit_targets = [value for value, keep in zip(clean_targets, target_mask) if keep]
        if not fit_features:
            fit_features = clean_features
            fit_targets = clean_targets
    else:
        tail_diag = {
            "target_tail_total_points": float(len(clean_targets)),
            "target_tail_excluded_points": 0.0,
            "target_tail_excluded_share": 0.0,
            "target_tail_floor": y_floor,
            "target_tail_log_limit": float("inf"),
            "target_tail_sigma_threshold": 0.0,
            "target_tail_min_keep_fraction": 1.0,
        }
        fit_features = clean_features
        fit_targets = clean_targets

    feature_scales: list[float] = []
    for feature_idx in range(width):
        values = [row[feature_idx] for row in fit_features]
        scale = _tail_filtered_mean(values)
        if scale <= 0.0 or not math.isfinite(scale):
            positive = [value for value in values if value > 0.0 and math.isfinite(value)]
            scale = _quantile(positive, 0.5) if positive else 1.0
        feature_scales.append(max(scale, 1.0e-30))
    target_scale = _tail_filtered_mean(fit_targets)
    if target_scale <= 0.0 or not math.isfinite(target_scale):
        target_scale = _quantile(positive_targets, 0.5) if positive_targets else 1.0
    target_scale = max(target_scale, 1.0e-30)

    scaled_fit_features = [
        [value / scale for value, scale in zip(row, feature_scales)]
        for row in fit_features
    ]
    scaled_fit_targets = [value / target_scale for value in fit_targets]

    # Theil-Sen estimates the slope from pairwise median trends in normalized
    # space. Then fit b separately from the weighted median residual.
    base_weights = _output_inverse_weights(scaled_fit_targets, exponent=0.25, min_weight=0.3, max_weight=3.0)
    positive_weights = [w for w in base_weights if w > 0.0 and math.isfinite(w)]
    if positive_weights:
        mean_weight = sum(positive_weights) / len(positive_weights)
        if mean_weight > 0.0:
            base_weights = [w / mean_weight for w in base_weights]
    else:
        base_weights = [1.0 for _ in scaled_fit_targets]

    x = np.asarray(scaled_fit_features, dtype=np.float64)
    y = np.asarray(scaled_fit_targets, dtype=np.float64)
    w = np.asarray(base_weights, dtype=np.float64)
    coef_array = np.asarray(
        _theil_sen_nonnegative_multi(
            scaled_fit_features,
            scaled_fit_targets,
            fit_intercept=False,
            seed=0,
            iterations=4,
        )[0],
        dtype=np.float64,
    )
    if coef_array.size != width:
        coef_array = np.zeros(width, dtype=np.float64)
    if fit_intercept:
        residual_for_intercept = y - x @ coef_array
        intercept = max(_weighted_median(residual_for_intercept, w), 0.0)
    else:
        intercept = 0.0

    pred = np.maximum(x @ coef_array + intercept, 0.0)
    sqrt_w = np.sqrt(np.maximum(w, 0.0))
    weighted_residual = sqrt_w * (y - pred)
    residual_abs = np.abs(weighted_residual[np.isfinite(weighted_residual)])
    residual_scale = float(np.median(residual_abs)) if residual_abs.size else 0.0
    if residual_scale <= 0.0 or not math.isfinite(residual_scale):
        centered = sqrt_w * (y - float(np.median(y)))
        centered_abs = np.abs(centered[np.isfinite(centered)])
        residual_scale = float(np.median(centered_abs)) if centered_abs.size else 0.0
    residual_scale = max(residual_scale * max(float(f_scale), 1.0e-12), 1.0e-30)
    robust_weights = _log_loss_weights(weighted_residual, loss=loss, f_scale=residual_scale)
    effective_weights = w * robust_weights
    weighted_sse_values = effective_weights * (y - pred) ** 2
    total_sse = float(np.sum(weighted_sse_values))
    if total_sse > 0.0:
        ordered = np.sort(weighted_sse_values)
        top1 = float(ordered[-1] / total_sse)
        top10 = float(np.sum(ordered[-min(10, ordered.size):]) / total_sse)
    else:
        top1 = 0.0
        top10 = 0.0
    weight_sum = float(np.sum(effective_weights))
    if weight_sum > 0.0:
        mean_y = float(np.sum(effective_weights * y) / weight_sum)
        ss_tot = float(np.sum(effective_weights * (y - mean_y) ** 2))
        r2 = 1.0 - total_sse / ss_tot if ss_tot > 0.0 else 0.0
    else:
        r2 = 0.0

    coef_original = [
        float(target_scale * coef_value / feature_scales[idx])
        for idx, coef_value in enumerate(coef_array)
    ]
    intercept_original = float(target_scale * intercept)

    diagnostics = {
        "fit_points": float(len(fit_features)),
        "fit_total_points": float(len(clean_features)),
        "fit_drop_share": float(1.0 - len(fit_features) / len(clean_features)) if clean_features else 0.0,
        "fit_density_points": float(len(fit_features)),
        "fit_weight_floor": y_floor,
        "fit_target_scale": target_scale,
        **{f"fit_feature_scale_{idx}": scale for idx, scale in enumerate(feature_scales)},
        **tail_diag,
        "fit_objective": "normalized_theil_sen_then_weighted_median_b",
        "fit_loss": loss,
        "fit_f_scale": residual_scale,
        "fit_success": 1.0,
        "fit_nfev": 0.0,
        "fit_message": "theil_sen",
        "fit_r2": r2,
        "fit_log_r2": r2,
        "fit_r2_raw": r2,
        "fit_min_usable_r2": MIN_USABLE_FIT_R2,
        "fit_usable": 1.0 if math.isfinite(r2) and r2 >= MIN_USABLE_FIT_R2 else 0.0,
        "fit_top1_sse_share": top1,
        "fit_top10_sse_share": top10,
        "fit_top1_log_loss_share": top1,
        "fit_top10_log_loss_share": top10,
        "fit_normalized_intercept": float(intercept),
    }
    for idx, value in enumerate(coef_array):
        diagnostics[f"fit_normalized_coef_{idx}"] = float(value)
    return coef_original, intercept_original, float(r2), [float(value) for value in effective_weights], diagnostics

def _fit_streaming_wls_line(
    rows: list[dict[str, Any]],
    *,
    x_fields: list[str],
    y_field: str,
    density_keep_fraction: float = 0.90,
    robust_mad_threshold: float = 3.0,
) -> tuple[list[float], float, float, dict[str, float]]:
    features: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        if y_field not in row or any(field not in row for field in x_fields):
            continue
        xs = [_safe_row_float(row, field, float("nan")) for field in x_fields]
        y = _safe_row_float(row, y_field, float("nan"))
        if all(math.isfinite(x) for x in xs) and math.isfinite(y):
            features.append(xs)
            targets.append(y)
    if not features:
        return [0.0 for _ in x_fields], 0.0, 0.0, {"fit_points": 0.0, "fit_total_points": 0.0}
    coef, intercept, r2, _weights, diagnostics = _fit_normal_space_nonnegative_multi(
        features,
        targets,
        fit_intercept=True,
        loss="soft_l1",
        f_scale=1.0,
    )
    return _gate_low_quality_fit(coef, intercept, r2, diagnostics, targets)

def _streaming_prediction(row: dict[str, Any], fit_fields: list[str], coef: list[float], intercept: float) -> float:
    value = float(intercept)
    for idx, field in enumerate(fit_fields):
        if idx < len(coef):
            value += float(coef[idx]) * _safe_row_float(row, field)
    return value


def summarize_streaming_relation_fits(
    *,
    candidate: str,
    point_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not point_rows:
        return []
    add_prefix_kv_and_cross_fields(point_rows)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in point_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    specs = [
        ("mlp", "post_attention_residual_mse", "mlp_delta_mse", ["post_attention_residual_mse"]),
        ("attention_input", "input_residual_mse", "attention_delta_mse", ["input_residual_mse", "prefix_kv_mse"]),
        ("attention_prefix_kv", "prefix_kv_mse", "attention_delta_mse", ["input_residual_mse", "prefix_kv_mse"]),
        ("kv", "input_residual_mse", "kv_mse", ["input_residual_mse"]),
    ]
    rows: list[dict[str, Any]] = []
    fit_cache: dict[tuple[tuple[str, ...], str, int], tuple[list[float], float, float, dict[str, float]]] = {}
    for layer, layer_rows in sorted(by_layer.items()):
        for relation, x_field, y_field, fit_fields in specs:
            if not any(y_field in row and x_field in row for row in layer_rows):
                continue
            cache_key = (tuple(fit_fields), y_field, layer)
            if cache_key not in fit_cache:
                fit_cache[cache_key] = _fit_streaming_wls_line(
                    layer_rows,
                    x_fields=fit_fields,
                    y_field=y_field,
                )
            coef, intercept, r2, diag = fit_cache[cache_key]
            slope_index = fit_fields.index(x_field) if x_field in fit_fields else 0
            row: dict[str, Any] = {
                "candidate": candidate,
                "relation": relation,
                "layer": layer,
                "x_field": x_field,
                "y_field": y_field,
                "fit_fields": ";".join(fit_fields),
                "fit_objective": diag.get("fit_objective", ""),
                "slope_for_x": coef[slope_index] if slope_index < len(coef) else 0.0,
                "intercept": intercept,
                "r2": r2,
                "r2_raw": diag.get("fit_r2_raw", r2),
                "fit_usable": diag.get("fit_usable", 1.0),
                "fit_points": diag.get("fit_points", 0.0),
                "fit_total_points": diag.get("fit_total_points", 0.0),
                "fit_drop_share": diag.get("fit_drop_share", 0.0),
                "target_tail_excluded_points": diag.get("target_tail_excluded_points", 0.0),
                "target_tail_excluded_share": diag.get("target_tail_excluded_share", 0.0),
                "fit_top10_sse_share": diag.get("fit_top10_sse_share", 0.0),
                "fit_target_scale": diag.get("fit_target_scale", 0.0),
                "fit_normalized_intercept": diag.get("fit_normalized_intercept", 0.0),
            }
            for idx, value in enumerate(coef):
                row[f"coef_{idx}"] = value
                row[f"fit_field_{idx}"] = fit_fields[idx]
                row[f"normalized_coef_{idx}"] = diag.get(f"fit_normalized_coef_{idx}", 0.0)
                row[f"feature_scale_{idx}"] = diag.get(f"fit_feature_scale_{idx}", 0.0)
            rows.append(row)
    return rows


def _binned_prediction_curve(x_values: np.ndarray, yhat_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x_values) & np.isfinite(yhat_values) & (x_values > 0.0) & (yhat_values > 0.0)
    x = x_values[valid]
    yhat = yhat_values[valid]
    if x.size < 2:
        return x, yhat
    order = np.argsort(x)
    x = x[order]
    yhat = yhat[order]
    bins = min(28, max(2, int(math.sqrt(float(x.size)))))
    edges = np.linspace(0, x.size, bins + 1, dtype=np.int64)
    line_x: list[float] = []
    line_y: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        line_x.append(float(np.median(x[lo:hi])))
        line_y.append(float(np.median(yhat[lo:hi])))
    return np.asarray(line_x, dtype=np.float64), np.asarray(line_y, dtype=np.float64)


def plot_streaming_relation_panels(
    *,
    model_label: str,
    candidate: str,
    point_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, str]:
    if not point_rows:
        return {}
    configure_matplotlib()
    import matplotlib.pyplot as plt

    display_candidate = display_candidate_name(candidate)
    output_dir = figure_dir / "streaming-panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in point_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)

    specs = [
        ("mlp", "post_attention_residual_mse", "mlp_delta_mse", "mlp_cross", "MLP", "#0072b2", ["post_attention_residual_mse"], "MSE[MLP before]", "MSE[MLP delta]"),
        ("attention_input", "input_residual_mse", "attention_delta_mse", "attention_cross", "Attention input", "#bdbdbd", ["input_residual_mse", "prefix_kv_mse"], "MSE[attn input]", "MSE[attn delta]"),
        ("attention_prefix_kv", "prefix_kv_mse", "attention_delta_mse", "attention_cross", "Attention prefix KV", "#009e73", ["input_residual_mse", "prefix_kv_mse"], "prefix KV MSE", "MSE[attn delta]"),
        ("kv", "input_residual_mse", "kv_mse", "", "KV", "#d55e00", ["input_residual_mse"], "KV input MSE", "KV MSE"),
    ]
    figure_paths: dict[str, str] = {}
    max_points_per_layer = 900
    fit_by_layer: dict[tuple[str, int], tuple[list[float], float, float, dict[str, float]]] = {}
    for layer, rows in by_layer.items():
        fit_by_layer[("mlp", layer)] = _fit_streaming_wls_line(rows, x_fields=["post_attention_residual_mse"], y_field="mlp_delta_mse")
        fit_by_layer[("attention_input", layer)] = _fit_streaming_wls_line(rows, x_fields=["input_residual_mse", "prefix_kv_mse"], y_field="attention_delta_mse")
        fit_by_layer[("attention_prefix_kv", layer)] = fit_by_layer[("attention_input", layer)]
        fit_by_layer[("kv", layer)] = _fit_streaming_wls_line(rows, x_fields=["input_residual_mse"], y_field="kv_mse")

    for name, x_field, y_field, cross_field, title, color, fit_fields, xlabel, ylabel in specs:
        layers = sorted(layer for layer, rows in by_layer.items() if any(y_field in row and x_field in row for row in rows))
        for start in range(0, len(layers), 16):
            chunk = layers[start : start + 16]
            if not chunk:
                continue
            fig, axes = plt.subplots(4, 4, figsize=(13.4, 10.2), sharex=False, sharey=False)
            flat_axes = list(axes.flat)
            for ax, layer in zip(flat_axes, chunk):
                valid: list[tuple[dict[str, Any], float, float, float]] = []
                for row in by_layer[layer]:
                    if x_field not in row or y_field not in row:
                        continue
                    x = _safe_row_float(row, x_field, float("nan"))
                    y = _safe_row_float(row, y_field, float("nan"))
                    cross = _safe_row_float(row, cross_field, float("nan")) if cross_field else float("nan")
                    if math.isfinite(x) and math.isfinite(y):
                        valid.append((row, x, y, cross))
                if valid:
                    tail_mask, _plot_tail_diag = _target_log_tail_inlier_mask(
                        [y for _, _, y, _ in valid],
                        sigma_threshold=6.0,
                        min_keep_fraction=0.98,
                    )
                    plot_valid = [item for item, keep in zip(valid, tail_mask) if keep]
                    excluded_valid = [item for item, keep in zip(valid, tail_mask) if not keep]
                    if not plot_valid:
                        plot_valid = valid
                        excluded_valid = []
                    excluded_points = len(excluded_valid)
                    x_values = np.asarray([x for _, x, _, _ in plot_valid], dtype=np.float64)
                    y_values = np.asarray([y for _, _, y, _ in plot_valid], dtype=np.float64)
                    if x_values.size > max_points_per_layer:
                        keep = np.unique(np.linspace(0, x_values.size - 1, max_points_per_layer, dtype=np.int64))
                        x_plot_raw = x_values[keep]
                        y_plot_raw = y_values[keep]
                    else:
                        x_plot_raw = x_values
                        y_plot_raw = y_values
                    x_plot, _xfloor, x_clipped = _clip_for_log(x_plot_raw)
                    y_plot, _yfloor, y_clipped = _clip_for_log(y_plot_raw)
                    ax.scatter(x_plot, y_plot, s=2.2, alpha=0.22, color=color, linewidths=0)
                    xmin, xmax, ymin, ymax = _padded_log_limits(x_plot, y_plot)
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    if excluded_valid:
                        inset = ax.inset_axes([0.61, 0.07, 0.35, 0.28])
                        tail_x_raw = np.asarray([x for _, x, _, _ in excluded_valid], dtype=np.float64)
                        tail_y_raw = np.asarray([y for _, _, y, _ in excluded_valid], dtype=np.float64)
                        tail_x, _tail_xfloor, _tail_xclipped = _clip_for_log(tail_x_raw)
                        tail_y, _tail_yfloor, _tail_yclipped = _clip_for_log(tail_y_raw)
                        inset.scatter(tail_x, tail_y, s=3.0, alpha=0.40, color="#c44e52", linewidths=0)
                        txmin, txmax, tymin, tymax = _padded_log_limits(tail_x, tail_y)
                        inset.set_xlim(txmin, txmax)
                        inset.set_ylim(tymin, tymax)
                        inset.set_xscale("log")
                        inset.set_yscale("log")
                        inset.set_title(f"jump n={len(excluded_valid):,}", fontsize=5.4, pad=1.0)
                        inset.tick_params(labelsize=4.2, pad=0.5, length=1.6, width=0.45)
                        inset.grid(True, alpha=0.16, linewidth=0.35)
                        inset.set_facecolor((1.0, 1.0, 1.0, 0.88))
                        for spine in inset.spines.values():
                            spine.set_linewidth(0.45)
                            spine.set_alpha(0.65)

                    coef, intercept, r2, fit_diag = fit_by_layer.get((name, layer), ([0.0], 0.0, 0.0, {}))
                    fit_usable = float(fit_diag.get("fit_usable", 1.0)) >= 1.0
                    slope_index = fit_fields.index(x_field) if x_field in fit_fields else 0
                    slope = coef[slope_index] if slope_index < len(coef) else 0.0
                    yhat_values = np.asarray(
                        [_streaming_prediction(row, fit_fields, coef, intercept) for row, _, _, _ in plot_valid],
                        dtype=np.float64,
                    )
                    if fit_usable and len(fit_fields) == 1:
                        positive_x = x_values[np.isfinite(x_values) & (x_values > 0.0)]
                        if positive_x.size and math.isfinite(slope) and math.isfinite(intercept):
                            line_min = float(np.min(positive_x))
                            line_max = float(np.max(positive_x))
                            line_x_raw = (
                                np.geomspace(line_min, line_max, 96)
                                if line_max > line_min > 0.0
                                else np.asarray([line_min, line_min * 1.001], dtype=np.float64)
                            )
                            line_y_raw = slope * line_x_raw + intercept
                        else:
                            line_x_raw = np.asarray([], dtype=np.float64)
                            line_y_raw = np.asarray([], dtype=np.float64)
                    elif fit_usable:
                        line_x_raw, line_y_raw = _binned_prediction_curve(x_values, yhat_values)
                    else:
                        line_x_raw = np.asarray([], dtype=np.float64)
                        line_y_raw = np.asarray([], dtype=np.float64)
                    keep_line = np.isfinite(line_x_raw) & np.isfinite(line_y_raw) & (line_x_raw > 0.0) & (line_y_raw > 0.0)
                    if bool(keep_line.any()):
                        line_x, _line_xfloor, _line_xclipped = _clip_for_log(line_x_raw[keep_line])
                        line_y, _line_yfloor, _line_yclipped = _clip_for_log(line_y_raw[keep_line])
                        ax.plot(line_x, line_y, color="#111111", linewidth=1.0, alpha=0.9)

                    base_annotation = f"n={len(plot_valid):,}/{len(valid):,} used={int(fit_diag.get('fit_points', 0.0)):,}"
                    if not fit_usable:
                        annotation = f"{base_annotation}\nTheil-Sen skipped\nR2={r2:.2f}<0.10"
                    elif name.startswith("attention"):
                        kin = coef[0] if len(coef) > 0 else 0.0
                        kkv = coef[1] if len(coef) > 1 else 0.0
                        annotation = f"{base_annotation}\nTheil-Sen kin={kin:.2e}\nkkv={kkv:.2e}\nb={intercept:.2e}\nR2={r2:.2f}"
                    else:
                        annotation = f"{base_annotation}\nTheil-Sen k={slope:.2e}\nb={intercept:.2e}\nR2={r2:.2f}"
                    top10 = fit_diag.get("fit_top10_sse_share", 0.0)
                    if top10 > 0.0:
                        annotation += f"\ntop10={top10:.0%}"
                    if excluded_points:
                        annotation += f"\nexclude={excluded_points:,}"
                    cross_values = [cross for _, _, _, cross in plot_valid if math.isfinite(cross)]
                    if cross_values:
                        annotation += f"\nE[bD]={_mean(cross_values):.2e}"
                    if x_clipped or y_clipped:
                        annotation += f"\nclip x={x_clipped:,} y={y_clipped:,}"
                    ax.text(
                        0.03,
                        0.97,
                        annotation,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=6.3,
                        bbox={"facecolor": "white", "alpha": 0.74, "edgecolor": "none", "pad": 1.4},
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"L{layer}", fontsize=8.0, pad=2.0)
                ax.grid(True, alpha=0.18)
                ax.tick_params(labelsize=6.2, pad=1)
            for ax in flat_axes[len(chunk) :]:
                ax.axis("off")
            fig.suptitle(f"{model_label} {display_candidate}: {title}", fontsize=11.0, y=0.995)
            fig.supxlabel(xlabel, fontsize=9.5)
            fig.supylabel(ylabel, fontsize=9.5)
            fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.965])
            output_path = output_dir / f"{display_candidate}_{name}_layers_{chunk[0]:02d}_{chunk[-1]:02d}.png"
            fig.savefig(output_path, dpi=190, bbox_inches="tight")
            plt.close(fig)
            figure_paths[f"{name}_{chunk[0]}_{chunk[-1]}"] = str(output_path)
    return figure_paths


def _share(value: float, total: float) -> float:
    if total <= 0.0 or not math.isfinite(total):
        return 0.0
    return value / total


def _clamp_share(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


def _tail_inlier_mask_for_mean(values: list[float]) -> list[bool]:
    if not values:
        return []
    mask, _diag = _target_log_tail_inlier_mask(
        values,
        sigma_threshold=6.0,
        min_keep_fraction=0.98,
    )
    if not any(mask):
        return [True for _ in values]
    return mask


def _masked_mean(values: list[float], mask: list[bool]) -> float:
    filtered = [value for value, keep in zip(values, mask) if keep and math.isfinite(value)]
    if filtered:
        return _mean(filtered)
    return _mean([value for value in values if math.isfinite(value)])


def _tail_filtered_mean(values: list[float]) -> float:
    return _masked_mean(values, _tail_inlier_mask_for_mean(values))


def _gate_low_quality_fit(
    coef: list[float],
    intercept: float,
    r2: float,
    diagnostics: dict[str, float],
    targets: list[float],
    *,
    min_r2: float = MIN_USABLE_FIT_R2,
) -> tuple[list[float], float, float, dict[str, float]]:
    diagnostics = dict(diagnostics)
    diagnostics["fit_r2_raw"] = float(r2)
    diagnostics["fit_min_usable_r2"] = float(min_r2)
    if math.isfinite(r2) and r2 >= min_r2:
        diagnostics["fit_usable"] = 1.0
        return coef, intercept, r2, diagnostics
    diagnostics["fit_usable"] = 0.0
    diagnostics["fit_fallback_intercept"] = _tail_filtered_mean(targets)
    return [0.0 for _ in coef], diagnostics["fit_fallback_intercept"], r2, diagnostics


def _output_inverse_weights(
    values: list[float],
    *,
    exponent: float = 0.5,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
) -> list[float]:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return [1.0 for _ in values]
    floor = _quantile(positive, 0.1)
    if floor <= 0.0:
        floor = min(positive)
    raw = [
        1.0 / (max(value, floor) ** exponent) if value > 0.0 and math.isfinite(value) else 0.0
        for value in values
    ]
    positive_weights = [value for value in raw if value > 0.0 and math.isfinite(value)]
    if not positive_weights:
        return [1.0 for _ in values]
    mean_weight = sum(positive_weights) / len(positive_weights)
    if mean_weight <= 0.0:
        return [1.0 for _ in values]
    return [
        min(max(value / mean_weight, min_weight), max_weight) if value > 0.0 and math.isfinite(value) else 0.0
        for value in raw
    ]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(valid.any()):
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else 0.0
    v = values[valid].astype(np.float64, copy=False)
    w = weights[valid].astype(np.float64, copy=False)
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cumulative = np.cumsum(w)
    cutoff = 0.5 * float(cumulative[-1])
    return float(v[min(int(np.searchsorted(cumulative, cutoff, side="left")), v.size - 1)])


def _sampled_theil_sen_slope(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    seed: int,
    max_pairs: int = 200_000,
) -> float:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[valid].astype(np.float64, copy=False)
    y = y_values[valid].astype(np.float64, copy=False)
    n = int(x.size)
    if n < 2:
        return 0.0
    if float(np.max(x) - np.min(x)) <= 1.0e-12:
        return 0.0
    rng = np.random.default_rng(seed)
    pair_count = min(max_pairs, max(4096, 24 * n))
    left = rng.integers(0, n, size=pair_count, dtype=np.int64)
    right = rng.integers(0, n, size=pair_count, dtype=np.int64)
    keep = left != right
    if not bool(keep.any()):
        return 0.0
    left = left[keep]
    right = right[keep]
    dx = x[right] - x[left]
    dy = y[right] - y[left]
    dx_abs = np.abs(dx)
    scale = max(float(np.quantile(dx_abs[dx_abs > 0.0], 0.1)) if bool(np.any(dx_abs > 0.0)) else 0.0, 1.0e-12)
    keep = dx_abs >= scale
    if not bool(keep.any()):
        keep = dx_abs > 1.0e-12
    if not bool(keep.any()):
        return 0.0
    slopes = dy[keep] / dx[keep]
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return 0.0
    return max(float(np.median(slopes)), 0.0)


def _theil_sen_nonnegative_multi(
    features: list[list[float]],
    targets: list[float],
    *,
    fit_intercept: bool = True,
    seed: int = 0,
    iterations: int = 4,
) -> tuple[list[float], float]:
    if not features or len(features) != len(targets):
        return [], 0.0
    width = len(features[0])
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[valid]
    y = y[valid]
    if y.size == 0:
        return [0.0 for _ in range(width)], 0.0
    coef = np.zeros(width, dtype=np.float64)
    intercept = float(np.median(y)) if fit_intercept else 0.0
    for iteration in range(max(1, iterations)):
        for feature_idx in range(width):
            residual = y - intercept
            if width > 1:
                others = np.arange(width) != feature_idx
                residual = residual - x[:, others] @ coef[others]
            slope = _sampled_theil_sen_slope(
                x[:, feature_idx],
                residual,
                seed=seed + 104729 * (iteration + 1) + 1009 * (feature_idx + 1),
            )
            coef[feature_idx] = slope
        if fit_intercept:
            intercept = max(float(np.median(y - x @ coef)), 0.0)
        else:
            intercept = 0.0
    return [float(value) for value in coef], float(intercept)


def _weighted_multi_fit(
    features: list[list[float]],
    targets: list[float],
    weights: list[float],
) -> tuple[list[float], float, float]:
    if not features or len(features) != len(targets) or len(features) != len(weights):
        return [], 0.0, 0.0
    width = len(features[0])
    rows: list[list[float]] = []
    ys: list[float] = []
    ws: list[float] = []
    for xs, y, w in zip(features, targets, weights):
        if len(xs) != width or not math.isfinite(y) or not math.isfinite(w) or w <= 0.0:
            continue
        if not all(math.isfinite(x) for x in xs):
            continue
        rows.append([float(x) for x in xs] + [1.0])
        ys.append(float(y))
        ws.append(float(w))
    if len(rows) < width + 1:
        return [0.0 for _ in range(width)], _mean(ys), 0.0
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    w = np.asarray(ws, dtype=np.float64)
    sqrt_w = np.sqrt(w)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    try:
        beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return [0.0 for _ in range(width)], _mean(ys), 0.0
    pred = x @ beta
    weight_sum = float(np.sum(w))
    mean_y = float(np.sum(w * y) / weight_sum) if weight_sum > 0.0 else float(np.mean(y))
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - mean_y) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return [float(value) for value in beta[:-1]], float(beta[-1]), r2




def _weighted_origin_fit(xs: list[float], ys: list[float], weights: list[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) != len(weights):
        return 0.0, 0.0
    clean = [
        (float(x), float(y), float(w))
        for x, y, w in zip(xs, ys, weights)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and w > 0.0
    ]
    if not clean:
        return 0.0, 0.0
    denom = sum(w * x * x for x, _, w in clean)
    if denom <= 0.0:
        mean_y = sum(w * y for _, y, w in clean) / sum(w for _, _, w in clean)
        return 0.0, 0.0 if mean_y <= 0.0 else 0.0
    slope = max(sum(w * x * y for x, y, w in clean) / denom, 0.0)
    mean_y = sum(w * y for _, y, w in clean) / sum(w for _, _, w in clean)
    ss_res = sum(w * (y - slope * x) ** 2 for x, y, w in clean)
    ss_tot = sum(w * (y - mean_y) ** 2 for _, y, w in clean)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return slope, r2


def _weighted_nonnegative_multi_fit(
    features: list[list[float]],
    targets: list[float],
    weights: list[float],
) -> tuple[list[float], float, float]:
    if not features or len(features) != len(targets) or len(features) != len(weights):
        return [], 0.0, 0.0
    width = len(features[0])
    rows: list[list[float]] = []
    ys: list[float] = []
    ws: list[float] = []
    for xs, y, w in zip(features, targets, weights):
        if len(xs) != width or not math.isfinite(y) or not math.isfinite(w) or w <= 0.0:
            continue
        if not all(math.isfinite(x) for x in xs):
            continue
        rows.append([float(x) for x in xs] + [1.0])
        ys.append(float(y))
        ws.append(float(w))
    if len(rows) < 1:
        return [0.0 for _ in range(width)], 0.0, 0.0
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    w = np.asarray(ws, dtype=np.float64)
    sqrt_w = np.sqrt(w)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    nvars = width + 1
    best_beta: np.ndarray | None = None
    best_sse = float("inf")
    for mask in range(1, 1 << nvars):
        active = [idx for idx in range(nvars) if mask & (1 << idx)]
        try:
            beta_active, *_ = np.linalg.lstsq(xw[:, active], yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if np.any(beta_active < -1.0e-18):
            continue
        beta = np.zeros(nvars, dtype=np.float64)
        beta[active] = np.maximum(beta_active, 0.0)
        pred = x @ beta
        sse = float(np.sum(w * (y - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_beta = beta
    if best_beta is None:
        best_beta = np.zeros(nvars, dtype=np.float64)
        best_beta[-1] = max(_mean(ys), 0.0)
        pred = x @ best_beta
        best_sse = float(np.sum(w * (y - pred) ** 2))
    weight_sum = float(np.sum(w))
    mean_y = float(np.sum(w * y) / weight_sum) if weight_sum > 0.0 else float(np.mean(y))
    ss_tot = float(np.sum(w * (y - mean_y) ** 2))
    r2 = 1.0 - best_sse / ss_tot if ss_tot > 0.0 else 0.0
    return [float(value) for value in best_beta[:-1]], float(best_beta[-1]), r2


def _fit_local_propagation_models(
    layer_rows: list[dict[str, Any]],
    next_by_sample: dict[int, dict[str, Any]] | None,
    *,
    is_first_layer: bool = False,
) -> dict[str, float]:
    paired: list[tuple[dict[str, Any], float]] = []
    for row in layer_rows:
        sample_index = int(row["sample_index"])
        if next_by_sample is None:
            next_value = float(row["final_norm_input_mse"])
        else:
            next_row = next_by_sample.get(sample_index)
            if next_row is None:
                continue
            next_value = float(next_row["input_residual_mse"])
        paired.append((row, next_value))
    if not paired:
        return {}

    input_values = [float(row["input_residual_mse"]) for row, _ in paired]
    post_values = [float(row["post_attention_residual_mse"]) for row, _ in paired]
    kv_values = [float(row["kv_mse"]) for row, _ in paired]
    prefix_kv_values = [_safe_row_float(row, "prefix_kv_mse", float(row["kv_mse"])) for row, _ in paired]
    next_values = [float(next_value) for _, next_value in paired]
    attention_delta_values = [
        _safe_row_float(row, "attention_delta_mse", max(float(row["post_attention_residual_mse"]) - float(row["input_residual_mse"]), 0.0))
        for row, _ in paired
    ]
    mlp_delta_values = [
        _safe_row_float(row, "mlp_delta_mse", max(float(next_value) - float(row["post_attention_residual_mse"]), 0.0))
        for row, next_value in paired
    ]

    kv_coef, kv_local_intercept, kv_fit_r2, _kv_weights, kv_diag = _fit_normal_space_nonnegative_multi(
        [[x] for x in input_values],
        kv_values,
        fit_intercept=True,
        loss="soft_l1",
        f_scale=1.0,
    )
    kv_coef, kv_local_intercept, kv_fit_r2, kv_diag = _gate_low_quality_fit(
        kv_coef, kv_local_intercept, kv_fit_r2, kv_diag, kv_values
    )
    (kv_input_slope,) = kv_coef

    attention_features_all = [[x, prefix_kv] for x, prefix_kv in zip(input_values, prefix_kv_values)]
    attention_coef, attention_local_intercept, attention_fit_r2, _attention_weights, attention_diag = _fit_normal_space_nonnegative_multi(
        attention_features_all,
        attention_delta_values,
        fit_intercept=True,
        loss="soft_l1",
        f_scale=1.0,
    )
    attention_coef, attention_local_intercept, attention_fit_r2, attention_diag = _gate_low_quality_fit(
        attention_coef, attention_local_intercept, attention_fit_r2, attention_diag, attention_delta_values
    )
    first_layer_attention_boundary_fallback = (
        is_first_layer and float(attention_diag.get("fit_usable", 1.0)) < 1.0
    )
    if first_layer_attention_boundary_fallback:
        # Layer 0 has no previous residual source in this attribution model.
        # A degenerate fit with input_mse ~= 0 would otherwise turn the
        # fallback intercept into a spurious "all of L0 attention" contribution.
        # Keep direct local compute/KV components below, and leave the
        # q/input-induced remainder as the initial boundary term.
        attention_diag = dict(attention_diag)
        attention_diag["fit_first_layer_boundary_fallback"] = 1.0
        attention_coef = [0.0 for _ in attention_coef]
        attention_local_intercept = 0.0
    attention_input_slope = attention_coef[0] if len(attention_coef) > 0 else 0.0
    attention_kv_slope = attention_coef[1] if len(attention_coef) > 1 else 0.0

    mlp_coef, mlp_local_intercept, mlp_fit_r2, _mlp_weights, mlp_diag = _fit_normal_space_nonnegative_multi(
        [[post] for post in post_values],
        mlp_delta_values,
        fit_intercept=True,
        loss="soft_l1",
        f_scale=1.0,
    )
    mlp_coef, mlp_local_intercept, mlp_fit_r2, mlp_diag = _gate_low_quality_fit(
        mlp_coef, mlp_local_intercept, mlp_fit_r2, mlp_diag, mlp_delta_values
    )
    mlp_input_slope = mlp_coef[0] if mlp_coef else 0.0

    kv_mean_mask = _tail_inlier_mask_for_mean(kv_values)
    post_mean_mask = _tail_inlier_mask_for_mean(post_values)
    next_mean_mask = _tail_inlier_mask_for_mean(next_values)
    attention_delta_mean_mask = _tail_inlier_mask_for_mean(attention_delta_values)
    mlp_delta_mean_mask = _tail_inlier_mask_for_mean(mlp_delta_values)
    attention_compute_direct_values = [
        _safe_row_float(row, "attention_compute_mse", 0.0) for row, _ in paired
    ]
    attention_kv_direct_values = [
        _safe_row_float(row, "attention_kv_effect_mse", 0.0) for row, _ in paired
    ]

    input_mean = _masked_mean(input_values, post_mean_mask)
    post_mean = _masked_mean(post_values, post_mean_mask)
    kv_mean = _masked_mean(kv_values, kv_mean_mask)
    next_mean = _masked_mean(next_values, next_mean_mask)
    attention_delta_mean = _masked_mean(attention_delta_values, attention_delta_mean_mask)
    mlp_delta_mean = _masked_mean(mlp_delta_values, mlp_delta_mean_mask)

    prefix_kv_mean = _masked_mean(prefix_kv_values, post_mean_mask)
    kv_local_tensor_mse = max(kv_local_intercept, 0.0)
    kv_from_input_mse = max(kv_input_slope, 0.0) * input_mean
    attention_kv_effect_mse = max(attention_kv_slope, 0.0) * kv_local_tensor_mse
    attention_prefix_kv_effect_mse = max(attention_kv_slope, 0.0) * prefix_kv_mean
    attention_input_effect_mse = max(attention_input_slope, 0.0) * input_mean
    attention_compute_mse = max(attention_local_intercept, 0.0)
    if first_layer_attention_boundary_fallback:
        attention_compute_mse = _masked_mean(
            attention_compute_direct_values,
            attention_delta_mean_mask,
        )
        attention_kv_effect_mse = _masked_mean(
            attention_kv_direct_values,
            attention_delta_mean_mask,
        )
    mlp_compute_mse = max(mlp_local_intercept, 0.0)

    attention_compute_share_uncapped = _share(attention_compute_mse, post_mean)
    kv_share_uncapped = _share(attention_kv_effect_mse, post_mean)
    attention_compute_share = _clamp_share(attention_compute_share_uncapped)
    kv_share = min(_clamp_share(kv_share_uncapped), max(0.0, 1.0 - attention_compute_share))
    attention_previous_share = max(0.0, 1.0 - attention_compute_share - kv_share)

    mlp_compute_share_uncapped = _share(mlp_compute_mse, next_mean)
    mlp_compute_share = _clamp_share(mlp_compute_share_uncapped)
    mlp_previous_share = max(0.0, 1.0 - mlp_compute_share)

    return {
        "points": float(len(paired)),
        "input_residual_mse_mean": input_mean,
        "post_attention_residual_mse_mean": post_mean,
        "kv_mse_mean": kv_mean,
        "prefix_kv_mse_mean": prefix_kv_mean,
        "attention_delta_mse_mean": attention_delta_mean,
        "mlp_delta_mse_mean": mlp_delta_mean,
        "next_input_residual_mse_mean": next_mean,
        "kv_input_slope": kv_input_slope,
        "kv_local_intercept": kv_local_tensor_mse,
        "kv_fit_r2": kv_fit_r2,
        "kv_fit_r2_raw": kv_diag.get("fit_r2_raw", kv_fit_r2),
        "kv_fit_usable": kv_diag.get("fit_usable", 1.0),
        "kv_fit_points": kv_diag.get("fit_points", 0.0),
        "kv_fit_top10_sse_share": kv_diag.get("fit_top10_sse_share", 0.0),
        "kv_local_tensor_mse": kv_local_tensor_mse,
        "kv_local_mse_mean": kv_local_tensor_mse,
        "kv_from_input_mse": kv_from_input_mse,
        "attention_input_slope": attention_input_slope,
        "attention_kv_slope": attention_kv_slope,
        "attention_local_intercept": attention_local_intercept,
        "attention_fit_r2": attention_fit_r2,
        "attention_fit_r2_raw": attention_diag.get("fit_r2_raw", attention_fit_r2),
        "attention_fit_usable": attention_diag.get("fit_usable", 1.0),
        "attention_fit_points": attention_diag.get("fit_points", 0.0),
        "attention_fit_top10_sse_share": attention_diag.get("fit_top10_sse_share", 0.0),
        "attention_input_effect_mse": attention_input_effect_mse,
        "attention_prefix_kv_effect_mse": attention_prefix_kv_effect_mse,
        "attention_compute_mse": attention_compute_mse,
        "attention_kv_local_effect_mse": attention_kv_effect_mse,
        "attention_compute_local_share_uncapped": attention_compute_share_uncapped,
        "kv_local_share_uncapped": kv_share_uncapped,
        "attention_compute_local_share": attention_compute_share,
        "kv_local_share": kv_share,
        "attention_previous_share": attention_previous_share,
        "mlp_input_slope": mlp_input_slope,
        "mlp_local_intercept": mlp_local_intercept,
        "mlp_fit_r2": mlp_fit_r2,
        "mlp_fit_r2_raw": mlp_diag.get("fit_r2_raw", mlp_fit_r2),
        "mlp_fit_usable": mlp_diag.get("fit_usable", 1.0),
        "mlp_fit_points": mlp_diag.get("fit_points", 0.0),
        "mlp_fit_top10_sse_share": mlp_diag.get("fit_top10_sse_share", 0.0),
        "mlp_compute_mse": mlp_compute_mse,
        "mlp_compute_local_share_uncapped": mlp_compute_share_uncapped,
        "mlp_compute_local_share": mlp_compute_share,
        "mlp_previous_share": mlp_previous_share,
    }


def build_local_propagation_contributions(
    *,
    candidate: str,
    point_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    by_layer: dict[int, dict[int, dict[str, Any]]] = {}
    for row in point_rows:
        by_layer.setdefault(int(row["layer"]), {})[int(row["sample_index"])] = row
    layers = sorted(by_layer)
    local_by_layer: dict[int, dict[str, float]] = {}
    for layer in layers:
        next_by_sample = by_layer.get(layer + 1)
        local = _fit_local_propagation_models(
            list(by_layer[layer].values()),
            next_by_sample,
            is_first_layer=(layer == min(layers)),
        )
        if local:
            local_by_layer[layer] = local

    incoming_final_share = 1.0
    rows_by_layer: dict[int, dict[str, Any]] = {}
    for layer in reversed(sorted(local_by_layer)):
        local = local_by_layer[layer]
        mlp_local_share = _clamp_share(local["mlp_compute_local_share"])
        mlp_final_share = incoming_final_share * mlp_local_share
        attention_incoming_share = incoming_final_share * max(0.0, 1.0 - mlp_local_share)

        attention_local_share = _clamp_share(local["attention_compute_local_share"])
        kv_local_share = min(
            _clamp_share(local["kv_local_share"]),
            max(0.0, 1.0 - attention_local_share),
        )
        attention_final_share = attention_incoming_share * attention_local_share
        kv_final_share = attention_incoming_share * kv_local_share
        previous_incoming_share = attention_incoming_share * max(
            0.0, 1.0 - attention_local_share - kv_local_share
        )

        rows_by_layer[layer] = {
            "candidate": candidate,
            "layer": layer,
            "points": local["points"],
            "layer_ending_tensor": f"layer_{layer}.mlp.output_residual" if layer != max(local_by_layer) else "forward.norm.input",
            "layer_ending_tensor_mse_mean": local["next_input_residual_mse_mean"],
            "input_residual_mse_mean": local["input_residual_mse_mean"],
            "post_attention_residual_mse_mean": local["post_attention_residual_mse_mean"],
            "kv_mse_mean": local["kv_mse_mean"],
            "prefix_kv_mse_mean": local["prefix_kv_mse_mean"],
            "kv_input_slope": local["kv_input_slope"],
            "kv_local_intercept": local["kv_local_intercept"],
            "kv_fit_r2": local["kv_fit_r2"],
            "kv_fit_r2_raw": local.get("kv_fit_r2_raw", local["kv_fit_r2"]),
            "kv_fit_usable": local.get("kv_fit_usable", 1.0),
            "attention_input_slope": local["attention_input_slope"],
            "attention_kv_slope": local["attention_kv_slope"],
            "attention_local_intercept": local["attention_local_intercept"],
            "attention_fit_r2": local["attention_fit_r2"],
            "attention_fit_r2_raw": local.get("attention_fit_r2_raw", local["attention_fit_r2"]),
            "attention_fit_usable": local.get("attention_fit_usable", 1.0),
            "mlp_input_slope": local["mlp_input_slope"],
            "mlp_local_intercept": local["mlp_local_intercept"],
            "mlp_fit_r2": local["mlp_fit_r2"],
            "mlp_fit_r2_raw": local.get("mlp_fit_r2_raw", local["mlp_fit_r2"]),
            "mlp_fit_usable": local.get("mlp_fit_usable", 1.0),
            "incoming_final_share": incoming_final_share,
            "attention_incoming_share": attention_incoming_share,
            "attention_compute_local_share": attention_local_share,
            "mlp_compute_local_share": mlp_local_share,
            "kv_local_share": kv_local_share,
            "attention_compute_local_share_uncapped": local["attention_compute_local_share_uncapped"],
            "mlp_compute_local_share_uncapped": local["mlp_compute_local_share_uncapped"],
            "kv_local_share_uncapped": local["kv_local_share_uncapped"],
            "attention_compute_final_share": attention_final_share,
            "mlp_compute_final_share": mlp_final_share,
            "kv_final_share": kv_final_share,
            "compute_plus_kv_final_share": attention_final_share + mlp_final_share + kv_final_share,
            "previous_incoming_share": previous_incoming_share,
        }
        incoming_final_share = previous_incoming_share

    rows = [rows_by_layer[layer] for layer in sorted(rows_by_layer)]
    summary = {
        "initial_unattributed_share": incoming_final_share,
        "attention_compute_sum": sum(float(row["attention_compute_final_share"]) for row in rows),
        "mlp_compute_sum": sum(float(row["mlp_compute_final_share"]) for row in rows),
        "kv_sum": sum(float(row["kv_final_share"]) for row in rows),
    }
    summary["accounted_sum"] = (
        summary["initial_unattributed_share"]
        + summary["attention_compute_sum"]
        + summary["mlp_compute_sum"]
        + summary["kv_sum"]
    )
    return rows, summary




def _plot_sample_indices(count: int, max_points: int = 1800) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, max_points, dtype=np.int64))


def _plot_log_regression_panel(
    ax: Any,
    *,
    x_values: list[float],
    y_values: list[float],
    line_x: list[float],
    line_y: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    annotation: str,
) -> None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size:
        keep = _plot_sample_indices(int(x.size))
        xs, _xfloor, x_clipped = _clip_for_log(x[keep])
        ys, _yfloor, y_clipped = _clip_for_log(y[keep])
        ax.scatter(xs, ys, s=1.2, alpha=0.18, color="#277da1", linewidths=0)
        lx = np.asarray(line_x, dtype=np.float64)
        ly = np.asarray(line_y, dtype=np.float64)
        line_valid = np.isfinite(lx) & np.isfinite(ly) & (lx > 0.0) & (ly > 0.0)
        if np.any(line_valid):
            ax.plot(lx[line_valid], ly[line_valid], color="#d62728", linewidth=0.95)
            all_x = np.concatenate([xs, lx[line_valid]])
            all_y = np.concatenate([ys, ly[line_valid]])
        else:
            all_x = xs
            all_y = ys
        xmin, xmax, ymin, ymax = _padded_log_limits(all_x, all_y)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if x_clipped or y_clipped:
            annotation = f"{annotation}\nclip x={x_clipped:,} y={y_clipped:,}"
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=7.2, pad=2.0)
    ax.set_xlabel(xlabel, fontsize=6.5, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=6.5, labelpad=1.5)
    ax.tick_params(labelsize=5.7, pad=1)
    ax.grid(True, alpha=0.18)
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=5.9,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
    )


def _paired_layer_regression_data(
    layer_rows: list[dict[str, Any]],
    next_by_sample: dict[int, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[float]]:
    paired: list[dict[str, Any]] = []
    next_values: list[float] = []
    for row in layer_rows:
        sample_index = int(row["sample_index"])
        if next_by_sample is None:
            next_value = float(row["final_norm_input_mse"])
        else:
            next_row = next_by_sample.get(sample_index)
            if next_row is None:
                continue
            next_value = float(next_row["input_residual_mse"])
        paired.append(row)
        next_values.append(next_value)
    return paired, next_values


def plot_local_propagation_regression_diagnostics(
    *,
    model_label: str,
    candidate: str,
    point_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, str]:
    if not point_rows:
        return {}
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    display_candidate = {
        "original_bf16": "native_bf16",
        "original_fp16": "native_fp16",
        "heal_base": "heal_base",
        "heal_plus": "heal_plus",
        "heal_full": "heal_full",
    }.get(candidate, candidate)
    output_pdf = figure_dir / f"{display_candidate}_local_propagation_regression_diagnostics.pdf"
    figure_dir.mkdir(parents=True, exist_ok=True)

    by_layer: dict[int, dict[int, dict[str, Any]]] = {}
    for row in point_rows:
        by_layer.setdefault(int(row["layer"]), {})[int(row["sample_index"])] = row
    layers = sorted(by_layer)
    layers_per_page = 5
    with PdfPages(output_pdf) as pdf:
        for start in range(0, len(layers), layers_per_page):
            page_layers = layers[start : start + layers_per_page]
            fig, axes = plt.subplots(len(page_layers), 3, figsize=(11.2, 2.25 * len(page_layers)), squeeze=False)
            for row_idx, layer in enumerate(page_layers):
                paired, next_values = _paired_layer_regression_data(
                    list(by_layer[layer].values()),
                    by_layer.get(layer + 1),
                )
                local = _fit_local_propagation_models(
                    paired,
                    by_layer.get(layer + 1),
                    is_first_layer=(layer == min(layers)),
                )
                if not paired or not local:
                    for ax in axes[row_idx]:
                        ax.axis("off")
                    continue
                input_values = [float(row["input_residual_mse"]) for row in paired]
                kv_values = [float(row["kv_mse"]) for row in paired]
                prefix_kv_values = [_safe_row_float(row, "prefix_kv_mse", float(row["kv_mse"])) for row in paired]
                post_values = [float(row["post_attention_residual_mse"]) for row in paired]
                attention_delta_values = [
                    _safe_row_float(row, "attention_delta_mse", max(float(row["post_attention_residual_mse"]) - float(row["input_residual_mse"]), 0.0))
                    for row in paired
                ]
                mlp_delta_values = [
                    _safe_row_float(row, "mlp_delta_mse", max(float(next_value) - float(row["post_attention_residual_mse"]), 0.0))
                    for row, next_value in zip(paired, next_values)
                ]
                attention_partial = [
                    delta - float(local["attention_input_slope"]) * x
                    for delta, x in zip(attention_delta_values, input_values)
                ]
                x_positive = [value for value in input_values if value > 0.0 and math.isfinite(value)]
                if x_positive:
                    xmin, xmax = min(x_positive), max(x_positive)
                    kv_line_x = np.geomspace(xmin, xmax, 64).tolist() if xmax > xmin else [xmin, xmax * 1.01]
                else:
                    kv_line_x = [1.0e-30, 1.0e-29]
                kv_line_y = [float(local["kv_input_slope"]) * x + float(local["kv_local_intercept"]) for x in kv_line_x]
                _plot_log_regression_panel(
                    axes[row_idx, 0],
                    x_values=input_values,
                    y_values=kv_values,
                    line_x=kv_line_x,
                    line_y=kv_line_y,
                    title=f"L{layer}: KV = a * input + local",
                    xlabel="input residual MSE",
                    ylabel="KV MSE",
                    annotation=(
                        f"a={float(local['kv_input_slope']):.2g}\n"
                        f"b_KV={float(local['kv_local_intercept']):.2g}\n"
                        f"R2={float(local['kv_fit_r2']):.2f}"
                    ),
                )

                prefix_kv_positive = [value for value in prefix_kv_values if value > 0.0 and math.isfinite(value)]
                if prefix_kv_positive:
                    xmin, xmax = min(prefix_kv_positive), max(prefix_kv_positive)
                    attn_line_x = np.geomspace(xmin, xmax, 64).tolist() if xmax > xmin else [xmin, xmax * 1.01]
                else:
                    attn_line_x = [1.0e-30, 1.0e-29]
                attn_line_y = [
                    float(local["attention_kv_slope"]) * x + float(local["attention_local_intercept"])
                    for x in attn_line_x
                ]
                _plot_log_regression_panel(
                    axes[row_idx, 1],
                    x_values=prefix_kv_values,
                    y_values=attention_partial,
                    line_x=attn_line_x,
                    line_y=attn_line_y,
                    title=f"L{layer}: attn delta - input term = b * prefixKV + attn",
                    xlabel="prefix KV MSE",
                    ylabel="attention delta MSE after input term",
                    annotation=(
                        f"b={float(local['attention_kv_slope']):.2g}\n"
                        f"attn={float(local['attention_local_intercept']):.2g}\n"
                        f"R2={float(local['attention_fit_r2']):.2f}"
                    ),
                )

                post_positive = [value for value in post_values if value > 0.0 and math.isfinite(value)]
                if post_positive:
                    xmin, xmax = min(post_positive), max(post_positive)
                    mlp_line_x = np.geomspace(xmin, xmax, 64).tolist() if xmax > xmin else [xmin, xmax * 1.01]
                else:
                    mlp_line_x = [1.0e-30, 1.0e-29]
                mlp_line_y = [
                    float(local["mlp_input_slope"]) * x + float(local["mlp_local_intercept"])
                    for x in mlp_line_x
                ]
                _plot_log_regression_panel(
                    axes[row_idx, 2],
                    x_values=post_values,
                    y_values=mlp_delta_values,
                    line_x=mlp_line_x,
                    line_y=mlp_line_y,
                    title=f"L{layer}: MLP delta = c * post + MLP",
                    xlabel="post-attn residual MSE",
                    ylabel="MLP delta MSE",
                    annotation=(
                        f"c={float(local['mlp_input_slope']):.2g}\n"
                        f"MLP={float(local['mlp_local_intercept']):.2g}\n"
                        f"R2={float(local['mlp_fit_r2']):.2f}"
                    ),
                )
            fig.suptitle(
                f"{model_label} {display_candidate}: local propagation regression diagnostics",
                fontsize=11.0,
                y=0.995,
            )
            fig.tight_layout(rect=[0.01, 0.01, 1.0, 0.975])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return {"regression_diagnostics_pdf": str(output_pdf)}


def plot_local_propagation_contributions(
    *,
    model_label: str,
    candidate: str,
    rows: list[dict[str, Any]],
    summary: dict[str, float],
    figure_dir: Path,
    write_pdf: bool = False,
) -> dict[str, str]:
    if not rows:
        return {}
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    display_candidate = display_candidate_name(candidate)
    display_candidate_label = display_candidate_short_label(candidate)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / f"{display_candidate}_recursive_truncation_contribution_ratios.pdf"
    png_path = figure_dir / f"{display_candidate}_recursive_truncation_contribution_ratios.png"

    layers = [int(row["layer"]) for row in rows]
    attention = [float(row["attention_compute_final_share"]) for row in rows]
    mlp = [float(row["mlp_compute_final_share"]) for row in rows]
    kv = [float(row["kv_final_share"]) for row in rows]
    layer_mse = [float(row["layer_ending_tensor_mse_mean"]) for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(5.15, 2.75))
    width = 0.72
    attn_bar = ax.bar(layers, attention, width=width, color="#bdbdbd", label="Attention")
    mlp_bar = ax.bar(layers, mlp, width=width, bottom=attention, color="#0072b2", label="MLP")
    kv_bottom = [a + m for a, m in zip(attention, mlp)]
    kv_bar = ax.bar(layers, kv, width=width, bottom=kv_bottom, color="#009e73", label="KV")

    ax.set_xlabel("Layer", fontsize=11.0)
    ax.set_ylabel("Contribution Share", fontsize=11.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    stacked = [a + m + k for a, m, k in zip(attention, mlp, kv)]
    ymax = max(stacked) * 1.18 if stacked else 0.01
    ax.set_ylim(0.0, max(0.01, ymax))
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.7)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    line = ax2.plot(
        layers,
        layer_mse,
        color="#d62728",
        linestyle=(0, (2.0, 2.0)),
        marker="o",
        markersize=2.5,
        linewidth=1.05,
        label="Layer MSE",
    )[0]
    if any(value > 0.0 and math.isfinite(value) for value in layer_mse):
        ax2.set_yscale("log")
    ax2.set_ylabel("Layer MSE", fontsize=11.0)
    ax2.grid(False)

    model_title = {
        "Qwen-Qwen3-8B": "Qwen3 8B",
        "Qwen-Qwen3-14B": "Qwen3 14B",
        "Qwen-Qwen3-32B": "Qwen3 32B",
        "meta-llama-Llama-3.1-8B-Instruct": "Llama 8B",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B": "DS 14B",
    }.get(model_label, model_label.replace("-", " "))
    if not (model_label == "Qwen-Qwen3-14B" and display_candidate == "native_bf16"):
        ax.text(
            -0.13,
            -0.23,
            f"{model_title} {display_candidate_label}",
            transform=ax.transAxes,
            fontsize=11.0,
            fontweight="bold",
            ha="left",
            va="top",
        )
    handles = [attn_bar, mlp_bar, kv_bar, line]
    labels = ["Attention", "MLP", "KV", "Layer MSE"]
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.18),
        columnspacing=1.0,
        handlelength=1.25,
        fontsize=8.8,
    )

    if layers:
        stride = max(1, len(layers) // 8)
        ax.set_xticks(layers[::stride])
    fig.tight_layout(pad=0.8)
    figure_paths = {"contribution_png": str(png_path)}
    if write_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
        figure_paths["contribution_pdf"] = str(pdf_path)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_paths

def aggregate_candidate(
    *,
    candidate: str,
    model_label: str,
    mode_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    trace_batches: list[dict[str, Any]],
    trim_input_tail: float,
    robust_mad_threshold: float,
    density_keep_fraction: float,
    write_pdfs: bool,
    mse_outlier_quantile: float = DEFAULT_MSE_OUTLIER_QUANTILE,
    mse_outlier_mad_multiplier: float = DEFAULT_MSE_OUTLIER_MAD_MULTIPLIER,
    disable_mse_outlier_filter: bool = False,
) -> dict[str, Any]:
    _ = (trim_input_tail, robust_mad_threshold, density_keep_fraction)
    point_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for trace_batch in trace_batches:
        batch_id = int(trace_batch["batch_id"])
        mse_path = mode_dir / "mse_batches" / f"batch_{batch_id:04d}.pt"
        if not mse_path.exists():
            continue
        payload = load_mse_batch(mse_path)
        reference_file = payload.get("reference_file", "")
        ref_tensors = load_reference_tensors(Path(reference_file)) if reference_file else {}
        selected_rows = trace_batch["selected_rows"]
        selected_count = len(selected_rows)

        final_input = _float_tensor(payload, "forward.norm.input")
        final = _float_tensor(payload, "forward.norm.hidden")
        final_norm_delta_mse = _pair_tensor(payload, "final_norm", "delta_mse")
        final_norm_covariance = _pair_tensor(payload, "final_norm", "covariance")
        if final is not None:
            for idx, sample in enumerate(selected_rows):
                if idx >= final.numel():
                    break
                row = {
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "sample_index": int(sample["sample_index"]),
                    "doc_id": int(sample["doc_id"]),
                    "token_pos": int(sample["token_pos"]),
                    "row_index": int(sample["row_index"]),
                    "final_norm_input_mse": (
                        float(final_input[idx].item())
                        if final_input is not None and idx < final_input.numel()
                        else 0.0
                    ),
                    "final_hidden_mse": float(final[idx].item()),
                }
                if final_norm_delta_mse is not None and idx < final_norm_delta_mse.numel():
                    row["final_norm_delta_mse"] = float(final_norm_delta_mse[idx].item())
                if final_norm_covariance is not None and idx < final_norm_covariance.numel():
                    row["final_norm_covariance"] = float(final_norm_covariance[idx].item())
                final_rows.append(row)

        payload_stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
        candidate_tensors = (
            payload.get("candidate_tensors", {})
            if isinstance(payload.get("candidate_tensors"), dict)
            else {}
        )
        keys = set(payload_stages.keys()) | set(candidate_tensors.keys()) | set(ref_tensors.keys())
        layers = sorted(
            {
                int(match.group(1))
                for key in keys
                for match in [
                    re.fullmatch(r"attention\.q\.(\d+)", key),
                    re.fullmatch(r"attention\.k_full\.(\d+)", key),
                    re.fullmatch(r"block\.mlp\.output\.(\d+)", key),
                ]
                if match
            }
        )
        for layer in layers:
            input_mse = _float_tensor(payload, f"layer_{layer}.input_layernorm.output.residual")
            post_mse = _float_tensor(payload, f"layer_{layer}.post_attention_layernorm.output.residual")
            k_mse = _float_tensor(payload, f"attention.k_full.{layer}")
            v_mse = _float_tensor(payload, f"attention.v_full.{layer}")
            if input_mse is None or post_mse is None or k_mse is None or v_mse is None:
                continue
            mlp_components = _block_mlp_mse_components(ref_tensors, payload, layer)
            attention_components = _attention_mse_components(
                ref_tensors,
                payload,
                trace_batch,
                layer,
                candidate=candidate,
            )
            pair_components: dict[str, torch.Tensor] = {}
            pair_components.update(_layer_pair_tensors(payload, "attention", layer))
            pair_components.update(_layer_pair_tensors(payload, "mlp", layer))
            if not mlp_components or not attention_components:
                continue
            component_lengths = [tensor.numel() for tensor in mlp_components.values()]
            component_lengths += [tensor.numel() for tensor in attention_components.values()]
            component_lengths += [tensor.numel() for tensor in pair_components.values()]
            final_input_count = final_input.numel() if final_input is not None else selected_count
            final_count = final.numel() if final is not None else selected_count
            n = min(input_mse.numel(), post_mse.numel(), final_input_count, final_count, selected_count, *component_lengths)
            for idx in range(n):
                sample = selected_rows[idx]
                input_value = float(input_mse[idx].item())
                k_value = _selected_or_full_value(k_mse, idx, sample, selected_count)
                v_value = _selected_or_full_value(v_mse, idx, sample, selected_count)
                kv_value = 0.5 * (k_value + v_value)
                row = {
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "layer": layer,
                    "sample_index": int(sample["sample_index"]),
                    "doc_id": int(sample["doc_id"]),
                    "token_pos": int(sample["token_pos"]),
                    "row_index": int(sample["row_index"]),
                    "flat_index": int(sample.get("flat_index", idx)),
                    "input_residual_mse": input_value,
                    "post_attention_residual_mse": float(post_mse[idx].item()),
                    "k_mse": k_value,
                    "v_mse": v_value,
                    "kv_mse": kv_value,
                    "kv_over_input_residual_mse": kv_value / input_value if input_value > 0.0 else 0.0,
                    "final_norm_input_mse": (
                        float(final_input[idx].item())
                        if final_input is not None and idx < final_input.numel()
                        else 0.0
                    ),
                    "final_hidden_mse": (
                        float(final[idx].item()) if final is not None and idx < final.numel() else 0.0
                    ),
                }
                for field, tensor in mlp_components.items():
                    if idx < tensor.numel():
                        row[field] = float(tensor[idx].item())
                for field, tensor in attention_components.items():
                    if idx < tensor.numel():
                        row[field] = float(tensor[idx].item())
                for field, tensor in pair_components.items():
                    if idx < tensor.numel():
                        row[field] = float(tensor[idx].item())
                point_rows.append(row)

    outlier_summary = mark_extreme_mse_token_outliers(
        point_rows,
        quantile=mse_outlier_quantile,
        mad_multiplier=mse_outlier_mad_multiplier,
        enabled=not disable_mse_outlier_filter,
    )
    usable_point_rows = _non_outlier_rows(point_rows)
    add_prefix_kv_and_cross_fields(usable_point_rows)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in usable_point_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    summary_rows = summarize_block_io_layers(candidate, by_layer)
    for summary in summary_rows:
        summary.update(outlier_summary)
        summary["points_after_mse_outlier_filter"] = float(len(by_layer.get(int(summary["layer"]), [])))
    streaming_summary_rows = summarize_streaming_relation_fits(
        candidate=candidate,
        point_rows=usable_point_rows,
    )

    write_csv(output_dir / f"{candidate}_residual_kv_mse_points.csv", point_rows)
    write_csv(output_dir / f"{candidate}_residual_kv_wls_by_layer.csv", summary_rows)
    write_csv(output_dir / f"{candidate}_streaming_relation_fits.csv", streaming_summary_rows)
    write_csv(output_dir / f"{candidate}_final_hidden_mse_points.csv", final_rows)

    final_values = [float(row["final_hidden_mse"]) for row in final_rows]
    final_input_values = [float(row["final_norm_input_mse"]) for row in final_rows]
    final_summary = {
        "candidate": candidate,
        "points": len(final_values),
        "final_norm_input_mse_mean": _mean(final_input_values),
        "final_norm_input_mse_p50": _quantile(final_input_values, 0.5),
        "final_norm_input_mse_p90": _quantile(final_input_values, 0.9),
        "final_norm_input_mse_max": max(final_input_values) if final_input_values else 0.0,
        "final_hidden_mse_mean": _mean(final_values),
        "final_hidden_mse_p50": _quantile(final_values, 0.5),
        "final_hidden_mse_p90": _quantile(final_values, 0.9),
        "final_hidden_mse_max": max(final_values) if final_values else 0.0,
        **outlier_summary,
    }
    final_delta_values = [float(row["final_norm_delta_mse"]) for row in final_rows if "final_norm_delta_mse" in row]
    final_covariance_values = [
        float(row["final_norm_covariance"]) for row in final_rows if "final_norm_covariance" in row
    ]
    if final_delta_values:
        final_summary["final_norm_delta_mse_mean"] = _mean(final_delta_values)
        final_summary["final_norm_delta_mse_p50"] = _quantile(final_delta_values, 0.5)
    if final_covariance_values:
        final_summary["final_norm_covariance_mean"] = _mean(final_covariance_values)
        final_summary["final_norm_covariance_p50"] = _quantile(final_covariance_values, 0.5)
    write_csv(output_dir / f"{candidate}_final_hidden_mse.csv", [final_summary])

    figure_paths = plot_kv_fit_panels(
        model_label=model_label,
        candidate=candidate,
        point_rows=usable_point_rows,
        summary_rows=summary_rows,
        figure_dir=figure_dir,
    )
    figure_paths.update(
        plot_streaming_relation_panels(
            model_label=model_label,
            candidate=candidate,
            point_rows=usable_point_rows,
            figure_dir=figure_dir,
        )
    )
    contribution_rows, contribution_summary = build_local_propagation_contributions(
        candidate=candidate,
        point_rows=usable_point_rows,
    )
    write_csv(output_dir / f"{candidate}_recursive_truncation_contribution_ratios.csv", contribution_rows)
    figure_paths.update(
        plot_local_propagation_contributions(
            model_label=model_label,
            candidate=candidate,
            rows=contribution_rows,
            summary=contribution_summary,
            figure_dir=figure_dir,
            write_pdf=write_pdfs,
        )
    )

    display_candidate = display_candidate_name(candidate)
    if summary_rows:
        write_csv(figure_dir / f"{display_candidate}_residual_kv_wls_by_layer.csv", summary_rows)
    if streaming_summary_rows:
        write_csv(figure_dir / f"{display_candidate}_streaming_relation_fits.csv", streaming_summary_rows)
    if contribution_rows:
        write_csv(figure_dir / f"{display_candidate}_recursive_truncation_contribution_ratios.csv", contribution_rows)
    return {
        "candidate": candidate,
        "points_csv": str(output_dir / f"{candidate}_residual_kv_mse_points.csv"),
        "wls_csv": str(output_dir / f"{candidate}_residual_kv_wls_by_layer.csv"),
        "kv_fit_csv": str(output_dir / f"{candidate}_residual_kv_wls_by_layer.csv"),
        "final_hidden_mse_csv": str(output_dir / f"{candidate}_final_hidden_mse.csv"),
        "contribution_csv": str(output_dir / f"{candidate}_recursive_truncation_contribution_ratios.csv"),
        "figure_paths": figure_paths,
        **final_summary,
        **{f"contribution_{key}": value for key, value in contribution_summary.items()},
    }


def prepare_model_plan(args: argparse.Namespace, model_arg: str, run_id: str) -> dict[str, Any]:
    model_id = fhm_common.resolve_model(model_arg)
    model_label = fhm_common.model_label(model_arg)
    multiple = bool(args.all_models or (args.models and len(args.models) > 1))
    generation_dir = generation_dir_for_model(path_from_cli(args.reference_generation_dir), model_arg, multiple)
    replay_generated_limit = int(args.max_gen_tokens)
    if bool(args.full_replay_attribution) and int(args.max_replay_tokens_per_doc) > 0:
        replay_generated_limit = max(replay_generated_limit, int(args.max_replay_tokens_per_doc))
    replay_rows_all = load_generation_replay_rows(
        generation_dir,
        max_gen_tokens=replay_generated_limit,
        limit=args.limit,
        subset_file=args.subset_file,
    )
    if bool(args.full_replay_attribution):
        replay_rows = truncate_replay_rows_to_max_generated_tokens(
            replay_rows_all,
            max_generated_tokens_per_doc=int(args.max_replay_tokens_per_doc),
        )
        positions = all_candidate_positions(replay_rows)
        selected = positions
    else:
        positions = all_candidate_positions(replay_rows_all)
        selected = select_positions(
            positions,
            sample_probability=float(args.sample_probability),
            sample_positions=int(args.sample_positions),
            seed=int(args.seed),
        )
        replay_rows = selected_doc_rows(replay_rows_all, selected)
    trace_batches = build_trace_batches(
        replay_rows,
        selected,
        batch_size=int(args.prefill_batch_size),
        token_budget=int(args.prefill_token_budget),
    )
    run_root = OUTPUT_ROOT / run_id / model_label
    results_root = RESULTS_ROOT / run_id / model_label
    figure_dir = path_from_cli(args.figure_root) / detect_gpu_name() / "error-contribution" / model_label
    if args.skip_reference:
        existing_replay_path = run_root / "replay_manifest.jsonl"
        existing_plan_path = run_root / "trace_batch_plan.jsonl"
        if existing_replay_path.exists() and existing_plan_path.exists():
            replay_rows = read_jsonl(existing_replay_path)
            trace_batches = read_jsonl(existing_plan_path)
            sampled_path = run_root / "sampled_positions.json"
            if sampled_path.exists():
                sampled_payload = read_json(sampled_path)
                sampled_positions = sampled_payload.get("positions") if isinstance(sampled_payload, dict) else None
                if isinstance(sampled_positions, list):
                    selected = sampled_positions
            log(
                f"reuse existing trace plan for {model_label}: "
                f"docs={len(replay_rows)} batches={len(trace_batches)}"
            )
    plan = {
        "run_id": run_id,
        "model_arg": model_arg,
        "model_id": model_id,
        "model_label": model_label,
        "generation_dir": str(generation_dir),
        "subset_file": str(path_from_cli(args.subset_file)) if args.subset_file is not None else "",
        "max_gen_tokens": replay_generated_limit,
        "candidate_positions": len(positions),
        "selected_positions": len(selected),
        "selected_docs": len(replay_rows),
        "trace_batches": len(trace_batches),
        "sample_probability": float(args.sample_probability),
        "sample_positions": int(args.sample_positions),
        "full_replay_attribution": bool(args.full_replay_attribution),
        "full_replay_position_scope": "generated_tokens" if bool(args.full_replay_attribution) else "sampled_generated_tokens",
        "max_replay_tokens_per_doc": int(args.max_replay_tokens_per_doc),
        "seed": int(args.seed),
        "run_root": str(run_root),
        "results_root": str(results_root),
        "figure_dir": str(figure_dir),
        "reference_mode": asdict(REFERENCE_MODE),
        "candidate_modes": [asdict(CANDIDATE_MODES[name]) for name in args.candidate_modes],
    }
    run_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_manifest.json", plan)
    write_json(results_root / "run_manifest.json", plan)
    write_jsonl(run_root / "replay_manifest.jsonl", replay_rows)
    write_json(run_root / "sampled_positions.json", {"positions": selected})
    write_jsonl(run_root / "trace_batch_plan.jsonl", trace_batches)
    return {
        "plan": plan,
        "model_id": model_id,
        "model_label": model_label,
        "run_root": run_root,
        "results_root": results_root,
        "figure_dir": figure_dir,
        "replay_rows": replay_rows,
        "trace_batches": trace_batches,
    }


def run_model(args: argparse.Namespace, model_arg: str, run_id: str) -> dict[str, Any]:
    prepared = prepare_model_plan(args, model_arg, run_id)
    plan = prepared["plan"]
    log(
        f"plan model={plan['model_label']} selected={plan['selected_positions']} "
        f"docs={plan['selected_docs']} batches={plan['trace_batches']}"
    )
    if args.dry_run:
        return {"model": plan["model_label"], "dry_run": True, **plan}

    run_root: Path = prepared["run_root"]
    results_root: Path = prepared["results_root"]
    figure_dir: Path = prepared["figure_dir"]
    replay_rows = prepared["replay_rows"]
    trace_batches = prepared["trace_batches"]
    reference_dir = run_root / REFERENCE_MODE.name
    mode_results = []
    if not args.skip_reference:
        mode_results.append(
            run_trace_mode(
                args=args,
                model_id=prepared["model_id"],
                mode=REFERENCE_MODE,
                mode_dir=reference_dir,
                replay_rows=replay_rows,
                trace_batches=trace_batches,
            )
        )
    candidate_summaries = []
    if not args.skip_candidates:
        for candidate_name in args.candidate_modes:
            mode = CANDIDATE_MODES[candidate_name]
            mode_dir = run_root / mode.name
            mode_results.append(
                run_trace_mode(
                    args=args,
                    model_id=prepared["model_id"],
                    mode=mode,
                    mode_dir=mode_dir,
                    replay_rows=replay_rows,
                    trace_batches=trace_batches,
                    reference_dir=reference_dir,
                )
            )
            candidate_summaries.append(
                aggregate_candidate(
                    candidate=mode.name,
                    model_label=plan["model_label"],
                    mode_dir=mode_dir,
                    output_dir=results_root,
                    figure_dir=figure_dir,
                    trace_batches=trace_batches,
                    trim_input_tail=float(args.trim_input_tail),
                    robust_mad_threshold=float(args.robust_mad_threshold),
                    density_keep_fraction=float(args.density_keep_fraction),
                    write_pdfs=bool(args.write_pdfs),
            mse_outlier_quantile=float(args.mse_outlier_quantile),
            mse_outlier_mad_multiplier=float(args.mse_outlier_mad_multiplier),
            disable_mse_outlier_filter=bool(args.disable_mse_outlier_filter),
                )
            )
    summary = {
        "model": plan["model_label"],
        "run_root": str(run_root),
        "results_root": str(results_root),
        "figure_dir": str(figure_dir),
        "mode_results": mode_results,
        "candidate_summaries": candidate_summaries,
    }
    write_json(results_root / "summary.json", summary)
    if bool(args.cleanup_trace_batches):
        shutil.rmtree(reference_dir / "reference_batches", ignore_errors=True)
        for candidate_name in args.candidate_modes:
            mode = CANDIDATE_MODES[candidate_name]
            shutil.rmtree((run_root / mode.name / "mse_batches"), ignore_errors=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--run-id", default=f"error_contribution_{utc_timestamp()}")
    parser.add_argument("--reference-generation-dir", type=Path, default=DEFAULT_REFERENCE_GENERATION_DIR)
    parser.add_argument("--subset-file", type=Path)
    parser.add_argument("--figure-root", type=Path, default=FIGURES_ROOT)
    parser.add_argument("--write-pdfs", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-gen-tokens", type=int, default=512)
    parser.add_argument("--sample-probability", type=float, default=0.10)
    parser.add_argument("--sample-positions", type=int, default=0)
    parser.add_argument("--full-replay-attribution", action="store_true")
    parser.add_argument("--max-replay-tokens-per-doc", type=int, default=1024)
    parser.add_argument("--cleanup-trace-batches", action="store_true")
    parser.add_argument("--trim-input-tail", type=float, default=0.0)
    parser.add_argument("--robust-mad-threshold", type=float, default=3.0)
    parser.add_argument("--density-keep-fraction", type=float, default=0.9)
    parser.add_argument("--mse-outlier-quantile", type=float, default=DEFAULT_MSE_OUTLIER_QUANTILE)
    parser.add_argument("--mse-outlier-mad-multiplier", type=float, default=DEFAULT_MSE_OUTLIER_MAD_MULTIPLIER)
    parser.add_argument("--disable-mse-outlier-filter", action="store_true")
    parser.add_argument("--candidate-modes", nargs="+", choices=sorted(CANDIDATE_MODES), default=["bf16", "fp16"])
    parser.add_argument("--prefill-batch-size", type=int, default=16)
    parser.add_argument("--prefill-token-budget", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_probability < 0.0 or args.sample_probability > 1.0:
        raise SystemExit("--sample-probability must be in [0, 1]")
    models = model_args_from_cli(args)
    summaries = []
    patch_prepared = False
    try:
        if not args.dry_run:
            prepare_patch(selected_patch_name(args))
            patch_prepared = True
        for model_arg in models:
            summaries.append(run_model(args, model_arg, args.run_id))
    finally:
        if patch_prepared:
            revert_patch_best_effort()
    write_json(RESULTS_ROOT / args.run_id / "summary.json", {"models": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
