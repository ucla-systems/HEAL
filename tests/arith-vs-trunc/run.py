# SPDX-License-Identifier: Apache-2.0
"""Operator-level arithmetic-vs-truncation analysis using VLLM_DUMP_TENSORS."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from safetensors.torch import safe_open

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
FINAL_HIDDEN_MSE_DIR = ROOT / "tests" / "final-hidden-mse"
if str(FINAL_HIDDEN_MSE_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_HIDDEN_MSE_DIR))
import common as fhm_common  # noqa: E402

PATCH_NAME = "VLLM_DUMP_TENSORS"
ANALYZE_PATCH_NAME = "VLLM_HEAL"
PATCH_SCRIPT = ROOT / "scripts" / "vllm_incremental_build.sh"
VLLM_DIR = ROOT / "vllm"
OUTPUT_ROOT = ROOT / "output" / "arith-vs-trunc"
RESULTS_ROOT = TEST_DIR / "results"
DEFAULT_SUBSET_FILE = (
    ROOT / "tests" / "error-contribution" / "subsets" / "medqa_calibration32_seed20260528.jsonl"
)
TRACE_STAGE_REGEX = (
    r"^(?:forward\.embed_tokens\.output|forward\.norm\.hidden|"
    r"lm_head\.linear\.logits|forward\.logits|"
    r"layer_\d+\.(?:input_layernorm|post_attention_layernorm)\.output\."
    r"(?:hidden|residual)|"
    r"layer_\d+\.self_attn\.(?:qkv_proj\.output\.(?:q|k|v)|"
    r"q_norm\.output|k_norm\.output|rotary_emb\.output\.(?:q|k)|"
    r"attention\.output|o_proj\.output)|"
    r"layer_\d+\.mlp\.(?:gate_up_proj|silu_and_mul|down_proj)\.output)$"
)

SUMMARY_FIELDS = [
    "run_id",
    "model",
    "model_id",
    "operator",
    "family",
    "layer",
    "points",
    "input_stage",
    "output_stage",
    "fp32_recompute_mse_mean",
    "lower_precision_fp32_mse_mean",
    "ideal_truncation_mse_mean",
    "bf16_impl_mse_mean",
    "lower_precision_fp32_over_bf16_mse",
    "ideal_truncation_over_bf16_mse",
    "arithmetic_mse_mean",
    "arithmetic_over_bf16_mse",
    "arithmetic_residual_mse_mean",
    "arithmetic_residual_over_bf16_mse",
    "bf16_trunc_delta_mse_mean",
    "bf16_trunc_delta_over_bf16_mse",
    "shapley_arithmetic_over_bf16_mse",
    "lower_precision_fp32_mse_p50",
    "lower_precision_fp32_mse_p90",
    "ideal_truncation_mse_p50",
    "ideal_truncation_mse_p90",
    "bf16_impl_mse_p50",
    "bf16_impl_mse_p90",
    "arithmetic_mse_p50",
    "arithmetic_mse_p90",
    "bf16_trunc_delta_mse_p50",
    "bf16_trunc_delta_mse_p90",
]

POINT_FIELDS = [
    "run_id",
    "model",
    "operator",
    "family",
    "layer",
    "row_index",
    "doc_id",
    "generated_token_pos",
    "absolute_token_index",
    "fp32_recompute_mse",
    "lower_precision_fp32_mse",
    "ideal_truncation_mse",
    "bf16_impl_mse",
    "arithmetic_mse",
    "bf16_trunc_delta_mse",
]


@dataclass(frozen=True)
class TraceRunConfig:
    name: str
    dtype: str
    layercast_force_fp32: bool
    tf32: bool = False


FP32_LAYERCAST = TraceRunConfig(
    name="layercast_fp32",
    dtype="float32",
    layercast_force_fp32=True,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, TraceRunConfig):
        return value.__dict__
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def patch_sha256(version: str = PATCH_NAME) -> str:
    return fhm_common.sha256_file(ROOT / "patches" / f"{version}.patch")


def ensure_vllm_clean() -> None:
    status = subprocess.check_output(
        ["git", "-C", str(VLLM_DIR), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError(f"vLLM worktree is not clean:\n{status}")


def active_patch_status() -> str:
    return subprocess.check_output([str(PATCH_SCRIPT), "status"], cwd=ROOT, text=True)


def run_patch_command(action: str, version: str | None = None) -> None:
    cmd = [str(PATCH_SCRIPT), action]
    if version is not None:
        cmd.append(version)
    subprocess.run(cmd, cwd=ROOT, check=True)


def prepare_version(version: str) -> bool:
    status = active_patch_status()
    if f"Active patch: {version}" in status:
        log(f"prepare-version {version}: already active")
        return True
    if "Active patch: none" not in status:
        raise RuntimeError(f"unexpected active vLLM patch before preparing {version}:\n{status}")
    ensure_vllm_clean()
    log(f"prepare-version {version}")
    run_patch_command("prepare-version", version)
    return True


def revert_version(version: str) -> None:
    log(f"revert-version {version}")
    run_patch_command("revert-version", version)
    ensure_vllm_clean()


def configure_env(base_env: dict[str, str], mode: TraceRunConfig, output_dir: Path) -> dict[str, str]:
    env = dict(base_env)
    clear_keys = [
        "VLLM_VERSION_NAME",
        "VLLM_EXPERIMENT_OUTPUT_DIR",
        "VLLM_LAYERCAST_FORCE_FP32",
        "VLLM_DISABLE_COMPILE_CACHE",
        "TENSOR_TRACE_TENSOR_FILE",
        "VLLM_TENSOR_TRACE_TENSOR_FILE",
        "TENSOR_TRACE_STAGE_REGEX",
        "VLLM_TENSOR_TRACE_STAGE_REGEX",
        "VLLM_DUMP_TENSORS_MODE",
        "VLLM_DUMP_TENSORS_TOKEN_INDICES",
        "TENSOR_TRACE_TOKEN_INDICES",
        "VLLM_TENSOR_TRACE_TOKEN_INDICES",
    ]
    for key in clear_keys:
        env.pop(key, None)
    env["PYTHONPATH"] = str(VLLM_DIR) if not env.get("PYTHONPATH") else str(VLLM_DIR) + os.pathsep + env["PYTHONPATH"]
    env["VLLM_VERSION_NAME"] = PATCH_NAME
    env["VLLM_EXPERIMENT_OUTPUT_DIR"] = str(output_dir)
    env["VLLM_DISABLE_CUDA_GRAPH"] = "1"
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1" if mode.tf32 else "0"
    env["VLLM_FLOAT32_MATMUL_PRECISION"] = "high" if mode.tf32 else "highest"
    if mode.layercast_force_fp32:
        env["VLLM_LAYERCAST_FORCE_FP32"] = "1"
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    return env


def apply_torch_precision(tf32: bool) -> None:
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    except Exception:
        pass


def llm_kwargs(
    *,
    model_id: str,
    mode: TraceRunConfig,
    seed: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "dtype": mode.dtype,
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
        "attention_backend": "TRITON_ATTN",
    }


def resolve_max_model_len(setting: str, required: int) -> int:
    if setting == "auto":
        return required
    value = int(setting)
    if value < required:
        raise ValueError(f"--max-model-len={value} is smaller than required {required}")
    return value


def shutdown_llm(llm: Any) -> None:
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
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as exc:
        log(f"WARNING: torch distributed cleanup failed: {exc}")


def build_prompt_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=bool(config["trust_remote_code"])
    )
    subset_rows = fhm_common.load_subset(Path(config["subset_file"]), limit=int(config["limit"]))
    prompt_rows: list[dict[str, Any]] = []
    for doc_id, row in enumerate(subset_rows):
        messages = fhm_common.messages_for_row(row, dataset_kind="medqa")
        prompt, prompt_ids = fhm_common.chat_prompt_and_ids(tokenizer, messages)
        prompt_rows.append(
            {
                "doc_id": doc_id,
                "subset_pos": int(row.get("subset_pos", doc_id)),
                "original_index": int(row.get("original_index", doc_id)),
                "id": row.get("id"),
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "source_row": row,
            }
        )
    return prompt_rows


def worker_generate(config: dict[str, Any]) -> None:
    started = time.perf_counter()
    mode = FP32_LAYERCAST
    os.environ.clear()
    os.environ.update(configure_env(dict(config["base_env"]), mode, Path(config["output_dir"])))
    os.environ["PYTHONUNBUFFERED"] = "1"
    apply_torch_precision(mode.tf32)
    fhm_common.add_local_vllm_to_path()

    from vllm import LLM, SamplingParams, TokensPrompt

    prompt_rows = build_prompt_rows(config)
    max_new_tokens = int(config["max_new_tokens"])
    max_prompt_tokens = max(row["prompt_token_count"] for row in prompt_rows)
    max_model_len = resolve_max_model_len(str(config["max_model_len"]), max_prompt_tokens + max_new_tokens + 1)
    max_num_batched_tokens = max(
        int(config["max_num_batched_tokens"]),
        max_model_len,
        max_prompt_tokens + 1,
    )
    batch_size = int(config["generation_batch_size"])
    log(
        "loading FP32 LayerCast generator "
        f"model={config['model_id']} docs={len(prompt_rows)} max_model_len={max_model_len}"
    )
    llm = LLM(
        **llm_kwargs(
            model_id=config["model_id"],
            mode=mode,
            seed=int(config["seed"]),
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=batch_size,
            gpu_memory_utilization=float(config["gpu_memory_utilization"]),
            enforce_eager=True,
            trust_remote_code=bool(config["trust_remote_code"]),
        )
    )
    try:
        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)
        generated_rows: list[dict[str, Any]] = []
        for start in range(0, len(prompt_rows), batch_size):
            batch = prompt_rows[start : start + batch_size]
            prompts = [TokensPrompt(prompt_token_ids=row["prompt_token_ids"]) for row in batch]
            batch_started = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            elapsed = time.perf_counter() - batch_started
            batch_tokens = 0
            for row, output in zip(batch, outputs, strict=True):
                completion = output.outputs[0]
                generated = [int(token_id) for token_id in completion.token_ids]
                batch_tokens += len(generated)
                generated_rows.append(
                    {
                        "doc_id": int(row["doc_id"]),
                        "subset_pos": int(row["subset_pos"]),
                        "original_index": int(row["original_index"]),
                        "id": row.get("id"),
                        "request_id": str(output.request_id),
                        "prompt_token_ids": row["prompt_token_ids"],
                        "prompt_token_count": int(row["prompt_token_count"]),
                        "generated_token_ids": generated,
                        "generated_token_count": len(generated),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        "source_row": row.get("source_row"),
                    }
                )
            log(f"generated batch={start // batch_size} docs={len(batch)} tokens={batch_tokens} elapsed={elapsed:.1f}s")
    finally:
        shutdown_llm(llm)
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = Path(config["output_dir"])
    generated_path = Path(config["generated_path"])
    replay_rows: list[dict[str, Any]] = []
    for row in generated_rows:
        prompt_ids = [int(x) for x in row["prompt_token_ids"]]
        generated = [int(x) for x in row["generated_token_ids"]]
        replay_rows.append(
            {
                "doc_id": int(row["doc_id"]),
                "subset_pos": int(row["subset_pos"]),
                "original_index": int(row["original_index"]),
                "id": row.get("id"),
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(generated),
                "replay_token_ids": prompt_ids + generated,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated,
            }
        )
    write_jsonl(generated_path, generated_rows)
    write_jsonl(Path(config["replay_manifest_path"]), replay_rows)
    result = {
        "kind": "generate",
        "model_id": config["model_id"],
        "docs": len(generated_rows),
        "generated_tokens": sum(int(row["generated_token_count"]) for row in generated_rows),
        "elapsed_s": time.perf_counter() - started,
        "generated_path": generated_path,
        "replay_manifest_path": config["replay_manifest_path"],
        "output_dir": output_dir,
    }
    write_json(Path(config["result_path"]), result)


def worker_dump(config: dict[str, Any]) -> None:
    started = time.perf_counter()
    mode = FP32_LAYERCAST
    os.environ.clear()
    os.environ.update(configure_env(dict(config["base_env"]), mode, Path(config["output_dir"])))
    os.environ["PYTHONUNBUFFERED"] = "1"
    apply_torch_precision(mode.tf32)
    fhm_common.add_local_vllm_to_path()

    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.model_executor.models.tensor_trace import (
        clear_tensor_trace_tensors,
        clear_tensor_trace_token_index_filter_override,
        flush_tensor_trace_tensors,
        set_tensor_trace_token_index_filter,
    )

    replay_rows = read_jsonl(Path(config["replay_manifest_path"]))
    max_replay_len = max(len(row["replay_token_ids"]) for row in replay_rows)
    max_model_len = resolve_max_model_len(str(config["max_model_len"]), max_replay_len + 1)
    max_num_batched_tokens = max(int(config["max_num_batched_tokens"]), max_model_len)
    trace_path = Path(config["trace_path"])
    trace_path.unlink(missing_ok=True)
    trace_path.with_suffix(trace_path.suffix + ".meta.json").unlink(missing_ok=True)
    trace_segments_dir = Path(config["trace_segments_dir"])
    shutil.rmtree(trace_segments_dir, ignore_errors=True)
    trace_segments_dir.mkdir(parents=True, exist_ok=True)
    trace_segments_path = Path(config["trace_segments_path"])
    trace_segments_path.unlink(missing_ok=True)

    log(
        "loading FP32 LayerCast replay dumper "
        f"model={config['model_id']} docs={len(replay_rows)} max_model_len={max_model_len}"
    )
    llm = LLM(
        **llm_kwargs(
            model_id=config["model_id"],
            mode=mode,
            seed=int(config["seed"]),
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=1,
            gpu_memory_utilization=float(config["gpu_memory_utilization"]),
            enforce_eager=True,
            trust_remote_code=bool(config["trust_remote_code"]),
        )
    )
    os.environ["TENSOR_TRACE_STAGE_REGEX"] = TRACE_STAGE_REGEX
    os.environ["VLLM_TENSOR_TRACE_STAGE_REGEX"] = TRACE_STAGE_REGEX
    os.environ["VLLM_DUMP_TENSORS_MODE"] = "everything"
    clear_tensor_trace_tensors()
    trace_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    try:
        sampling = SamplingParams(temperature=0.0, max_tokens=1)
        for index, row in enumerate(replay_rows):
            prompt_len = int(row["prompt_token_count"])
            generated_len = int(row["generated_token_count"])
            keep = min(generated_len, int(config["analyze_last_generated_tokens"]))
            first_generated = generated_len - keep
            selected = [prompt_len + pos for pos in range(first_generated, generated_len)]
            captured_rows = len(row["replay_token_ids"])
            row_start = len(trace_rows)
            segment_path = trace_segments_dir / f"segment_{index:03d}_doc{int(row['doc_id']):03d}.safetensors"
            segment_path.unlink(missing_ok=True)
            segment_path.with_suffix(segment_path.suffix + ".meta.json").unlink(missing_ok=True)
            os.environ["TENSOR_TRACE_TENSOR_FILE"] = str(segment_path)
            os.environ["VLLM_TENSOR_TRACE_TENSOR_FILE"] = str(segment_path)
            clear_tensor_trace_tensors()
            clear_tensor_trace_token_index_filter_override()
            token_ids = [int(x) for x in row["replay_token_ids"]]
            doc_started = time.perf_counter()
            llm.generate(
                [TokensPrompt(prompt_token_ids=token_ids)],
                sampling_params=sampling,
                use_tqdm=False,
            )
            clear_tensor_trace_token_index_filter_override()
            flush_tensor_trace_tensors(segment_path)
            clear_tensor_trace_tensors()
            if not segment_path.exists():
                raise RuntimeError(f"tensor dump segment was not written: {segment_path}")
            for row_offset, generated_pos in enumerate(range(first_generated, generated_len)):
                trace_rows.append(
                    {
                        "row_index": row_start + row_offset,
                        "segment_index": index,
                        "segment_row_index": prompt_len + generated_pos,
                        "doc_id": int(row["doc_id"]),
                        "subset_pos": int(row["subset_pos"]),
                        "generated_token_pos": generated_pos,
                        "absolute_token_index": prompt_len + generated_pos,
                        "selected_row_offset_in_doc": row_offset,
                    }
                )
            segment_rows.append(
                {
                    "segment_index": index,
                    "doc_id": int(row["doc_id"]),
                    "path": str(segment_path),
                    "row_start": row_start,
                    "rows": keep,
                    "captured_rows": captured_rows,
                    "selected_row_indices": selected,
                    "selected_token_indices": selected,
                    "prompt_token_count": prompt_len,
                    "generated_token_count": generated_len,
                    "first_generated_token_pos": first_generated,
                }
            )
            log(
                f"dump replay doc={index} doc_id={row['doc_id']} rows={keep} captured_rows={captured_rows} "
                f"segment={segment_path.name} elapsed={time.perf_counter() - doc_started:.1f}s"
            )
    finally:
        clear_tensor_trace_token_index_filter_override()
        clear_tensor_trace_tensors()
        shutdown_llm(llm)
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(Path(config["trace_rows_path"]), trace_rows)
    write_jsonl(trace_segments_path, segment_rows)
    result = {
        "kind": "dump",
        "model_id": config["model_id"],
        "docs": len(replay_rows),
        "trace_rows": len(trace_rows),
        "trace_segments": len(segment_rows),
        "elapsed_s": time.perf_counter() - started,
        "trace_segments_dir": trace_segments_dir,
        "trace_segments_path": trace_segments_path,
        "trace_rows_path": config["trace_rows_path"],
        "stage_regex": TRACE_STAGE_REGEX,
    }
    write_json(Path(config["result_path"]), result)


def _trace_keys(path: Path, stage: str) -> list[str]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        metadata = read_json(meta_path)
        rows = metadata.get(stage, [])
        keys = [row["key"] for row in rows if isinstance(row, dict) and "key" in row]
        if keys:
            return keys
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    exact = [key for key in keys if key == stage]
    chunks = sorted(key for key in keys if key.startswith(stage + "__chunk_"))
    return exact or chunks


def load_stage(path: Path, stage: str) -> torch.Tensor | None:
    keys = _trace_keys(path, stage)
    if not keys:
        return None
    tensors = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in keys:
            tensors.append(handle.get_tensor(key).to(torch.float32).contiguous())
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0).contiguous()


def load_stage_rows(path: Path, stage: str, row_indices: list[int] | None) -> torch.Tensor | None:
    tensor = load_stage(path, stage)
    if tensor is None or row_indices is None:
        return tensor
    if not row_indices:
        return tensor[:0].contiguous()
    max_index = max(row_indices)
    if max_index >= tensor.shape[0]:
        return tensor
    index = torch.tensor(row_indices, dtype=torch.long)
    return tensor.index_select(0, index).contiguous()


def stage_exists(path: Path, stage: str) -> bool:
    return bool(_trace_keys(path, stage))


def row_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return torch.empty(0, dtype=torch.float32)
    a = a[:n].to(torch.float32)
    b = b[:n].to(torch.float32)
    return (a - b).square().flatten(start_dim=1).mean(dim=-1)


def summarize_values(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    values = values.detach().to(torch.float32).cpu()
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.5).item()),
        "p90": float(torch.quantile(values, 0.9).item()),
    }


def bf16_boundary(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


if triton is not None:
    @triton.jit
    def _attention_tl_dot_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * k + k_idxs[None, :],
                mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k_idxs[:, None] * n + offs_n[None, :],
                mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n),
                other=0.0,
            )
            acc += tl.dot(a, b, input_precision="tf32")
        tl.store(
            c_ptr + offs_m[:, None] * n + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )


def triton_attention_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for attention tl.dot replay")
    if a.device.type != "cuda":
        raise RuntimeError("attention tl.dot replay requires CUDA tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"triton_attention_dot expects [M,K] @ [K,N], got {tuple(a.shape)} @ {tuple(b.shape)}")
    a = a.contiguous()
    b = b.contiguous()
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[1])
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    block_m = 16
    block_n = 32
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _attention_tl_dot_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out



def linear_fp32(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x_2d, prefix_shape = _reshape_linear_input(x.to(torch.float32))
    weight_t = module.weight.to(device=x_2d.device, dtype=torch.float32).t().contiguous()
    out = torch.mm(x_2d, weight_t)
    bias = getattr(module, "bias", None)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=torch.float32)
    return _reshape_linear_output(out, prefix_shape)


def _reshape_linear_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    return x.reshape(-1, x.shape[-1]).contiguous(), x.shape[:-1]


def _reshape_linear_output(out: torch.Tensor, prefix_shape: torch.Size) -> torch.Tensor:
    return out.reshape(*prefix_shape, out.shape[-1]).contiguous()


def _add_linear_bias(module: torch.nn.Module, out: torch.Tensor) -> torch.Tensor:
    bias = getattr(module, "bias", None)
    if bias is None:
        return out
    return out + bf16_boundary(bias).to(device=out.device, dtype=torch.float32)


_BLASLT_FP32_OUTPUT_GEMM_CHECKED = False
_BLASLT_FP32_OUTPUT_GEMM_AVAILABLE = False


def _blaslt_fp32_output_gemm_op() -> Callable[..., Any] | None:
    global _BLASLT_FP32_OUTPUT_GEMM_CHECKED, _BLASLT_FP32_OUTPUT_GEMM_AVAILABLE
    if str(VLLM_DIR) not in sys.path:
        sys.path.insert(0, str(VLLM_DIR))
    try:
        from vllm import _custom_ops as ops  # type: ignore
    except Exception:
        ops = None
    if ops is not None and hasattr(ops, "blaslt_fp32_output_gemm"):
        _BLASLT_FP32_OUTPUT_GEMM_AVAILABLE = True
        _BLASLT_FP32_OUTPUT_GEMM_CHECKED = True
        return getattr(ops, "blaslt_fp32_output_gemm")

    if not _BLASLT_FP32_OUTPUT_GEMM_CHECKED:
        try:
            import vllm._C  # noqa: F401
        except Exception:
            pass
        _BLASLT_FP32_OUTPUT_GEMM_AVAILABLE = hasattr(torch.ops._C, "blaslt_fp32_output_gemm")
        _BLASLT_FP32_OUTPUT_GEMM_CHECKED = True
    if not _BLASLT_FP32_OUTPUT_GEMM_AVAILABLE:
        return None
    return torch.ops._C.blaslt_fp32_output_gemm


def blaslt_fp32_output_gemm(x: torch.Tensor, weight: torch.Tensor, input_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"blaslt_fp32_output_gemm expects x[M,K] and weight[N,K], got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"blaslt_fp32_output_gemm shape mismatch: x={tuple(x.shape)}, weight={tuple(weight.shape)}")
    x_lp = x.to(device=x.device, dtype=input_dtype).contiguous()
    weight_lp = weight.to(device=x.device, dtype=input_dtype).contiguous()
    if x_lp.device.type != "cuda":
        raise RuntimeError("HEAL cuBLASLt FP32-output GEMM requires CUDA tensors")
    op = _blaslt_fp32_output_gemm_op()
    if op is None:
        raise RuntimeError(
            "HEAL cuBLASLt FP32-output GEMM op is not registered. "
            "Run this analysis with VLLM_HEAL prepared so "
            "vllm._custom_ops.blaslt_fp32_output_gemm is available."
        )
    out = torch.empty((x_lp.shape[0], weight_lp.shape[0]), device=x_lp.device, dtype=torch.float32)
    op(out, x_lp, weight_lp, None)
    return out



def linear_lower_precision_fp32(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x_2d, prefix_shape = _reshape_linear_input(bf16_boundary(x))
    weight_t = bf16_boundary(module.weight).to(device=x_2d.device, dtype=torch.float32).t().contiguous()
    out = torch.mm(x_2d.to(torch.float32), weight_t)
    return _add_linear_bias(module, _reshape_linear_output(out, prefix_shape))


def linear_bf16_compute_fp32(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x_2d, prefix_shape = _reshape_linear_input(x)
    weight = module.weight.to(device=x_2d.device)
    out = blaslt_fp32_output_gemm(x_2d, weight, input_dtype=torch.bfloat16)
    return _add_linear_bias(module, _reshape_linear_output(out, prefix_shape))


def rmsnorm_impl(
    module: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    orig_dtype: torch.dtype,
    *,
    round_inputs_to_bf16: bool = False,
    round_weight_to_bf16: bool = False,
) -> torch.Tensor:
    eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
    weight = getattr(module, "weight", None)
    variance_size_override = getattr(module, "variance_size_override", None)
    hidden_size = int(getattr(module, "hidden_size", x.shape[-1]))
    work = bf16_boundary(x) if round_inputs_to_bf16 else x.to(torch.float32)
    if residual is not None:
        residual_work = bf16_boundary(residual) if round_inputs_to_bf16 else residual.to(torch.float32)
        work = work + residual_work
        # Match RMSNorm.forward_static's residual side effect even for the
        # FP32 replay path; for orig_dtype=float32 this is intentionally a no-op.
        _residual_out = work.to(orig_dtype)
        del _residual_out

    if work.shape[-1] != hidden_size:
        raise ValueError(f"Expected hidden_size to be {hidden_size}, but found: {work.shape[-1]}")

    x_var = work if variance_size_override is None else work[..., : int(variance_size_override)]
    variance = x_var.square().mean(dim=-1, keepdim=True)
    out = (work * torch.rsqrt(variance + eps)).to(orig_dtype)
    if weight is not None:
        weight_work = bf16_boundary(weight) if round_weight_to_bf16 else weight.to(device=out.device, dtype=torch.float32)
        out = out * weight_work.to(device=out.device, dtype=orig_dtype)
    return out.to(torch.float32)


def rmsnorm_fp32(
    module: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    return rmsnorm_impl(module, x, residual, torch.float32)


def rmsnorm_lower_precision_fp32(
    module: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    return rmsnorm_impl(
        module,
        x,
        residual,
        torch.float32,
        round_inputs_to_bf16=True,
        round_weight_to_bf16=True,
    )


def rmsnorm_bf16(
    module: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    return rmsnorm_lower_precision_fp32(module, x, residual)


def silu_and_mul_fp32(x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    x = x.to(torch.float32)
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def silu_and_mul_lower_precision_fp32(x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    x = bf16_boundary(x)
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def silu_and_mul_bf16(x: torch.Tensor) -> torch.Tensor:
    return silu_and_mul_lower_precision_fp32(x)


def _view_attention_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    if x.ndim >= 3 and x.shape[-2] == heads and x.shape[-1] == head_dim:
        return x.to(torch.float32)
    return x.reshape(x.shape[0], heads, head_dim).to(torch.float32)


def _rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    base: float,
    *,
    lower_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = positions.device
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / float(head_dim))
    )
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    if lower_precision:
        cos = bf16_boundary(cos)
        sin = bf16_boundary(sin)
    return cos, sin


def _apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    cos = cos.unsqueeze(-2).to(torch.float32)
    sin = sin.unsqueeze(-2).to(torch.float32)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def rope_fp32(x: torch.Tensor, positions: torch.Tensor, heads: int, head_dim: int, base: float) -> torch.Tensor:
    x_heads = _view_attention_heads(x, heads, head_dim).to(torch.float32)
    cos, sin = _rope_cos_sin(positions, head_dim, base, lower_precision=False)
    return _apply_rope_neox(x_heads, cos, sin).reshape(x.shape[0], heads * head_dim).to(torch.float32)


def rope_lower_precision_fp32(x: torch.Tensor, positions: torch.Tensor, heads: int, head_dim: int, base: float) -> torch.Tensor:
    x_heads = _view_attention_heads(bf16_boundary(x), heads, head_dim).to(torch.float32)
    cos, sin = _rope_cos_sin(positions, head_dim, base, lower_precision=True)
    return _apply_rope_neox(x_heads, cos, sin).reshape(x.shape[0], heads * head_dim).to(torch.float32)


def rope_bf16_compute_fp32(x: torch.Tensor, positions: torch.Tensor, heads: int, head_dim: int, base: float) -> torch.Tensor:
    return rope_lower_precision_fp32(x, positions, heads, head_dim, base)


@dataclass(frozen=True)
class AttentionParams:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scaling: float
    sliding_window: int | None = None


def infer_attention_params(attn: torch.nn.Module) -> AttentionParams:
    config = getattr(attn, "config", None)
    num_heads = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(attn, "num_key_value_heads", None) or getattr(attn, "num_kv_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_heads
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None) if config is not None else None
        if hidden_size is None or num_heads is None:
            raise RuntimeError("could not infer attention head_dim")
        head_dim = int(hidden_size) // int(num_heads)
    scaling = float(getattr(attn, "scaling", float(head_dim) ** -0.5))
    sliding_window = getattr(attn, "sliding_window", None)
    if sliding_window is None and config is not None:
        sliding_window = getattr(config, "sliding_window", None)
    return AttentionParams(int(num_heads), int(num_kv_heads), int(head_dim), scaling, None if sliding_window is None else int(sliding_window))


def infer_rope_base(attn: torch.nn.Module) -> float:
    rotary_emb = getattr(attn, "rotary_emb", None)
    if rotary_emb is not None and hasattr(rotary_emb, "base"):
        return float(getattr(rotary_emb, "base"))
    config = getattr(attn, "config", None)
    if config is not None and hasattr(config, "rope_theta"):
        return float(getattr(config, "rope_theta"))
    return 10000.0


def _attention_mask(query_positions: torch.Tensor, total_tokens: int, params: AttentionParams) -> torch.Tensor:
    key_positions = torch.arange(total_tokens, device=query_positions.device, dtype=torch.long)
    query_positions = query_positions.to(torch.long)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if params.sliding_window is not None:
        delta = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        mask = mask & (delta < int(params.sliding_window))
    return mask


def _dense_causal_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    params: AttentionParams,
    *,
    mode: str,
) -> torch.Tensor:
    total_tokens = q.shape[0]
    if mode == "fp32":
        q_heads = _view_attention_heads(q, params.num_heads, params.head_dim).to(torch.float32)
        k_heads = _view_attention_heads(k, params.num_kv_heads, params.head_dim).to(torch.float32)
        v_heads = _view_attention_heads(v, params.num_kv_heads, params.head_dim).to(torch.float32)
    elif mode == "lower_precision_fp32":
        q_heads = _view_attention_heads(bf16_boundary(q), params.num_heads, params.head_dim).to(torch.float32)
        k_heads = _view_attention_heads(bf16_boundary(k), params.num_kv_heads, params.head_dim).to(torch.float32)
        v_heads = _view_attention_heads(bf16_boundary(v), params.num_kv_heads, params.head_dim).to(torch.float32)
    elif mode == "bf16_compute_fp32":
        q_heads = _view_attention_heads(q, params.num_heads, params.head_dim).to(torch.bfloat16)
        k_heads = _view_attention_heads(k, params.num_kv_heads, params.head_dim).to(torch.bfloat16)
        v_heads = _view_attention_heads(v, params.num_kv_heads, params.head_dim).to(torch.bfloat16)
    else:
        raise ValueError(f"unknown attention mode: {mode}")

    query_positions = query_positions.to(device=q_heads.device, dtype=torch.long)
    q_selected = q_heads.index_select(0, query_positions)
    mask = _attention_mask(query_positions, total_tokens, params)
    repeat = max(1, params.num_heads // params.num_kv_heads)
    outputs: list[torch.Tensor] = []
    for head_idx in range(params.num_heads):
        kv_head_idx = min(params.num_kv_heads - 1, head_idx // repeat)
        q_h = q_selected[:, head_idx, :].contiguous()
        k_t = k_heads[:, kv_head_idx, :].transpose(0, 1).contiguous()
        if mode == "bf16_compute_fp32":
            scores = triton_attention_dot(q_h, k_t)
        else:
            scores = triton_attention_dot(q_h.to(torch.float32), k_t.to(torch.float32))
        scores = scores * params.scaling
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        v_h = v_heads[:, kv_head_idx, :].contiguous()
        if mode == "bf16_compute_fp32":
            out_h = triton_attention_dot(probs.to(torch.bfloat16), v_h)
        else:
            out_h = triton_attention_dot(probs, v_h.to(torch.float32))
        outputs.append(out_h)
    out = torch.stack(outputs, dim=1).reshape(query_positions.numel(), params.num_heads * params.head_dim).to(torch.float32)
    if mode in {"lower_precision_fp32", "bf16_compute_fp32"}:
        return bf16_boundary(out)
    return out


def attention_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    return _dense_causal_attention_impl(q, k, v, query_positions, params, mode="fp32")


def attention_lower_precision_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    return _dense_causal_attention_impl(q, k, v, query_positions, params, mode="lower_precision_fp32")


def attention_bf16_compute_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    return _dense_causal_attention_impl(q, k, v, query_positions, params, mode="bf16_compute_fp32")


def _attention_qk_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    query_positions: torch.Tensor,
    params: AttentionParams,
    *,
    lower_precision: bool,
) -> torch.Tensor:
    total_tokens = q.shape[0]
    if lower_precision:
        q_heads = _view_attention_heads(bf16_boundary(q), params.num_heads, params.head_dim).to(torch.float32)
        k_heads = _view_attention_heads(bf16_boundary(k), params.num_kv_heads, params.head_dim).to(torch.float32)
    else:
        q_heads = _view_attention_heads(q, params.num_heads, params.head_dim).to(torch.float32)
        k_heads = _view_attention_heads(k, params.num_kv_heads, params.head_dim).to(torch.float32)
    query_positions = query_positions.to(device=q_heads.device, dtype=torch.long)
    q_selected = q_heads.index_select(0, query_positions)
    mask = _attention_mask(query_positions, total_tokens, params)
    repeat = max(1, params.num_heads // params.num_kv_heads)
    scores_by_head: list[torch.Tensor] = []
    for head_idx in range(params.num_heads):
        kv_head_idx = min(params.num_kv_heads - 1, head_idx // repeat)
        q_h = q_selected[:, head_idx, :].contiguous()
        k_t = k_heads[:, kv_head_idx, :].transpose(0, 1).contiguous()
        scores = triton_attention_dot(q_h, k_t) * params.scaling
        scores_by_head.append(scores.masked_fill(~mask, 0.0))
    out = torch.stack(scores_by_head, dim=1).flatten(start_dim=1).to(torch.float32)
    if lower_precision:
        return bf16_boundary(out)
    return out


def attention_qk_fp32(q: torch.Tensor, k: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    return _attention_qk_scores(q, k, query_positions, params, lower_precision=False)


def attention_qk_lower_precision_fp32(q: torch.Tensor, k: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    return _attention_qk_scores(q, k, query_positions, params, lower_precision=True)


def attention_qk_bf16(q: torch.Tensor, k: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    total_tokens = q.shape[0]
    q_heads = _view_attention_heads(q, params.num_heads, params.head_dim).to(torch.bfloat16)
    k_heads = _view_attention_heads(k, params.num_kv_heads, params.head_dim).to(torch.bfloat16)
    query_positions = query_positions.to(device=q_heads.device, dtype=torch.long)
    q_selected = q_heads.index_select(0, query_positions)
    mask = _attention_mask(query_positions, total_tokens, params)
    repeat = max(1, params.num_heads // params.num_kv_heads)
    scores_by_head: list[torch.Tensor] = []
    for head_idx in range(params.num_heads):
        kv_head_idx = min(params.num_kv_heads - 1, head_idx // repeat)
        q_h = q_selected[:, head_idx, :].contiguous()
        k_t = k_heads[:, kv_head_idx, :].transpose(0, 1).contiguous()
        scores = triton_attention_dot(q_h, k_t) * params.scaling
        scores_by_head.append(scores.masked_fill(~mask, 0.0))
    out = torch.stack(scores_by_head, dim=1).flatten(start_dim=1).to(torch.float32)
    return bf16_boundary(out)


def _attention_probs_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    query_positions: torch.Tensor,
    params: AttentionParams,
) -> torch.Tensor:
    total_tokens = q.shape[0]
    q_heads = _view_attention_heads(q, params.num_heads, params.head_dim).to(torch.float32)
    k_heads = _view_attention_heads(k, params.num_kv_heads, params.head_dim).to(torch.float32)
    query_positions = query_positions.to(device=q_heads.device, dtype=torch.long)
    q_selected = q_heads.index_select(0, query_positions)
    mask = _attention_mask(query_positions, total_tokens, params)
    repeat = max(1, params.num_heads // params.num_kv_heads)
    probs_by_head: list[torch.Tensor] = []
    for head_idx in range(params.num_heads):
        kv_head_idx = min(params.num_kv_heads - 1, head_idx // repeat)
        scores = triton_attention_dot(
            q_selected[:, head_idx, :].contiguous(),
            k_heads[:, kv_head_idx, :].transpose(0, 1).contiguous(),
        ) * params.scaling
        scores = scores.masked_fill(~mask, float("-inf"))
        probs_by_head.append(torch.softmax(scores, dim=-1))
    return torch.stack(probs_by_head, dim=1).to(torch.float32)


def _attention_pv_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    params: AttentionParams,
    *,
    lower_precision: bool,
) -> torch.Tensor:
    probs = _attention_probs_fp32(q, k, query_positions, params)
    if lower_precision:
        probs = bf16_boundary(probs)
        v_for_dot = bf16_boundary(v)
    else:
        v_for_dot = v
    v_heads = _view_attention_heads(v_for_dot, params.num_kv_heads, params.head_dim).to(torch.float32)
    repeat = max(1, params.num_heads // params.num_kv_heads)
    outputs: list[torch.Tensor] = []
    for head_idx in range(params.num_heads):
        kv_head_idx = min(params.num_kv_heads - 1, head_idx // repeat)
        outputs.append(triton_attention_dot(probs[:, head_idx, :].contiguous(), v_heads[:, kv_head_idx, :].contiguous()))
    out = torch.stack(outputs, dim=1).reshape(query_positions.numel(), params.num_heads * params.head_dim).to(torch.float32)
    if lower_precision:
        return bf16_boundary(out)
    return out


def attention_pv_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    return _attention_pv_output(q, k, v, query_positions, params, lower_precision=False)


def attention_pv_lower_precision_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    return _attention_pv_output(q, k, v, query_positions, params, lower_precision=True)


def attention_pv_bf16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_positions: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    probs = _attention_probs_fp32(q, k, query_positions, params).to(torch.bfloat16)
    v_heads = _view_attention_heads(v, params.num_kv_heads, params.head_dim).to(torch.bfloat16)
    repeat = max(1, params.num_heads // params.num_kv_heads)
    outputs: list[torch.Tensor] = []
    for head_idx in range(params.num_heads):
        kv_head_idx = min(params.num_kv_heads - 1, head_idx // repeat)
        outputs.append(triton_attention_dot(probs[:, head_idx, :].contiguous(), v_heads[:, kv_head_idx, :].contiguous()))
    out = torch.stack(outputs, dim=1).reshape(query_positions.numel(), params.num_heads * params.head_dim).to(torch.float32)
    return bf16_boundary(out)


def tensor_slice_args(args: tuple[torch.Tensor, ...], start: int, stop: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(item[start:stop].to(device=device, dtype=torch.float32, non_blocking=False) for item in args)


@dataclass(frozen=True)
class OperatorSpec:
    operator: str
    family: str
    layer: int | str
    input_stages: tuple[str, ...]
    output_stage: str
    fp32_fn: Callable[..., torch.Tensor]
    bf16_fn: Callable[..., torch.Tensor]
    trunc_fn: Callable[..., torch.Tensor] | None = None
    position_arg: bool = False
    full_context: bool = False
    computed_reference: bool = False

    @property
    def input_stage_label(self) -> str:
        return " + ".join(self.input_stages)


def rmsnorm_heads_fp32(module: torch.nn.Module, x: torch.Tensor, head_dim: int) -> torch.Tensor:
    shape = x.shape
    heads = shape[-1] // head_dim
    return rmsnorm_fp32(module, x.view(*shape[:-1], heads, head_dim))


def rmsnorm_heads_lower_precision_fp32(module: torch.nn.Module, x: torch.Tensor, head_dim: int) -> torch.Tensor:
    shape = x.shape
    heads = shape[-1] // head_dim
    return rmsnorm_lower_precision_fp32(module, x.view(*shape[:-1], heads, head_dim))


def rmsnorm_heads_bf16(module: torch.nn.Module, x: torch.Tensor, head_dim: int) -> torch.Tensor:
    shape = x.shape
    heads = shape[-1] // head_dim
    return rmsnorm_bf16(module, x.view(*shape[:-1], heads, head_dim))




def _call_operator(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    positions: torch.Tensor | None,
    position_arg: bool,
) -> torch.Tensor:
    if position_arg:
        if positions is None:
            raise RuntimeError("operator requires positions")
        return fn(*args, positions)
    return fn(*args)


def evaluate_operator_values(
    *,
    run_id: str,
    model_label: str,
    operator: str,
    family: str,
    layer: int | str,
    inputs: tuple[torch.Tensor, ...],
    reference_output: torch.Tensor,
    fp32_fn: Callable[..., torch.Tensor],
    bf16_fn: Callable[..., torch.Tensor],
    trunc_fn: Callable[..., torch.Tensor] | None,
    trace_rows: list[dict[str, Any]],
    chunk_rows: int,
    device: torch.device,
    write_points: bool,
    positions: torch.Tensor | None = None,
    position_arg: bool = False,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    points = min(reference_output.shape[0], *(item.shape[0] for item in inputs))
    if positions is not None:
        points = min(points, int(positions.numel()))
    fp32_mses: list[torch.Tensor] = []
    trunc_mses: list[torch.Tensor] = []
    bf16_mses: list[torch.Tensor] = []
    bf16_trunc_delta_mses: list[torch.Tensor] = []
    point_rows: list[dict[str, Any]] = []
    chunk_rows = max(1, int(chunk_rows))
    with torch.no_grad():
        for start in range(0, points, chunk_rows):
            stop = min(start + chunk_rows, points)
            chunk_inputs = tensor_slice_args(inputs, start, stop, device)
            chunk_positions = None
            if positions is not None:
                chunk_positions = positions[start:stop].to(device=device, dtype=torch.long, non_blocking=False)
            ref = reference_output[start:stop].to(device=device, dtype=torch.float32)
            fp32_out = _call_operator(fp32_fn, chunk_inputs, chunk_positions, position_arg).to(torch.float32)
            if trunc_fn is None:
                trunc_inputs = tuple(bf16_boundary(item) for item in chunk_inputs)
                trunc_out = _call_operator(fp32_fn, trunc_inputs, chunk_positions, position_arg).to(torch.float32)
            else:
                trunc_out = _call_operator(trunc_fn, chunk_inputs, chunk_positions, position_arg).to(torch.float32)
            bf16_out = _call_operator(bf16_fn, chunk_inputs, chunk_positions, position_arg).to(torch.float32)
            fp32_mse = row_mse(fp32_out, ref).cpu()
            trunc_mse = row_mse(trunc_out, ref).cpu()
            bf16_mse = row_mse(bf16_out, ref).cpu()
            bf16_trunc_delta_mse = row_mse(bf16_out, trunc_out).cpu()
            fp32_mses.append(fp32_mse)
            trunc_mses.append(trunc_mse)
            bf16_mses.append(bf16_mse)
            bf16_trunc_delta_mses.append(bf16_trunc_delta_mse)
            if write_points:
                for local_idx in range(stop - start):
                    local_row_index = start + local_idx
                    meta = trace_rows[local_row_index] if local_row_index < len(trace_rows) else {}
                    point_rows.append(
                        {
                            "run_id": run_id,
                            "model": model_label,
                            "operator": operator,
                            "family": family,
                            "layer": layer,
                            "row_index": meta.get("row_index", local_row_index),
                            "doc_id": meta.get("doc_id"),
                            "generated_token_pos": meta.get("generated_token_pos"),
                            "absolute_token_index": meta.get("absolute_token_index"),
                            "fp32_recompute_mse": float(fp32_mse[local_idx].item()),
                            "lower_precision_fp32_mse": float(trunc_mse[local_idx].item()),
                            "ideal_truncation_mse": float(trunc_mse[local_idx].item()),
                            "bf16_impl_mse": float(bf16_mse[local_idx].item()),
                            "arithmetic_mse": float((bf16_mse[local_idx] - trunc_mse[local_idx]).item()),
                            "bf16_trunc_delta_mse": float(bf16_trunc_delta_mse[local_idx].item()),
                        }
                    )
            del chunk_inputs, chunk_positions, ref, fp32_out, trunc_out, bf16_out
    return {
        "fp32": torch.cat(fp32_mses) if fp32_mses else torch.empty(0),
        "trunc": torch.cat(trunc_mses) if trunc_mses else torch.empty(0),
        "bf16": torch.cat(bf16_mses) if bf16_mses else torch.empty(0),
        "bf16_trunc_delta": torch.cat(bf16_trunc_delta_mses) if bf16_trunc_delta_mses else torch.empty(0),
    }, point_rows


def evaluate_context_operator_values(
    *,
    run_id: str,
    model_label: str,
    operator: str,
    family: str,
    layer: int | str,
    inputs: tuple[torch.Tensor, ...],
    reference_output: torch.Tensor | None,
    fp32_fn: Callable[..., torch.Tensor],
    bf16_fn: Callable[..., torch.Tensor],
    trunc_fn: Callable[..., torch.Tensor] | None,
    query_positions: torch.Tensor,
    trace_rows: list[dict[str, Any]],
    chunk_rows: int,
    device: torch.device,
    write_points: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    points = int(query_positions.numel()) if reference_output is None else min(reference_output.shape[0], int(query_positions.numel()))
    context_inputs = tuple(item.to(device=device, dtype=torch.float32, non_blocking=False) for item in inputs)
    query_positions = query_positions[:points].to(device=device, dtype=torch.long, non_blocking=False)
    fp32_mses: list[torch.Tensor] = []
    trunc_mses: list[torch.Tensor] = []
    bf16_mses: list[torch.Tensor] = []
    bf16_trunc_delta_mses: list[torch.Tensor] = []
    point_rows: list[dict[str, Any]] = []
    chunk_rows = max(1, int(chunk_rows))
    with torch.no_grad():
        for start in range(0, points, chunk_rows):
            stop = min(start + chunk_rows, points)
            qpos = query_positions[start:stop]
            fp32_out = fp32_fn(*context_inputs, qpos).to(torch.float32)
            ref = (
                fp32_out.detach()
                if reference_output is None
                else reference_output[start:stop].to(device=device, dtype=torch.float32)
            )
            if trunc_fn is None:
                trunc_inputs = tuple(bf16_boundary(item) for item in context_inputs)
                trunc_out = fp32_fn(*trunc_inputs, qpos).to(torch.float32)
            else:
                trunc_out = trunc_fn(*context_inputs, qpos).to(torch.float32)
            bf16_out = bf16_fn(*context_inputs, qpos).to(torch.float32)
            fp32_mse = row_mse(fp32_out, ref).cpu()
            trunc_mse = row_mse(trunc_out, ref).cpu()
            bf16_mse = row_mse(bf16_out, ref).cpu()
            bf16_trunc_delta_mse = row_mse(bf16_out, trunc_out).cpu()
            fp32_mses.append(fp32_mse)
            trunc_mses.append(trunc_mse)
            bf16_mses.append(bf16_mse)
            bf16_trunc_delta_mses.append(bf16_trunc_delta_mse)
            if write_points:
                for local_idx in range(stop - start):
                    local_row_index = start + local_idx
                    meta = trace_rows[local_row_index] if local_row_index < len(trace_rows) else {}
                    point_rows.append(
                        {
                            "run_id": run_id,
                            "model": model_label,
                            "operator": operator,
                            "family": family,
                            "layer": layer,
                            "row_index": meta.get("row_index", local_row_index),
                            "doc_id": meta.get("doc_id"),
                            "generated_token_pos": meta.get("generated_token_pos"),
                            "absolute_token_index": meta.get("absolute_token_index"),
                            "fp32_recompute_mse": float(fp32_mse[local_idx].item()),
                            "lower_precision_fp32_mse": float(trunc_mse[local_idx].item()),
                            "ideal_truncation_mse": float(trunc_mse[local_idx].item()),
                            "bf16_impl_mse": float(bf16_mse[local_idx].item()),
                            "arithmetic_mse": float((bf16_mse[local_idx] - trunc_mse[local_idx]).item()),
                            "bf16_trunc_delta_mse": float(bf16_trunc_delta_mse[local_idx].item()),
                        }
                    )
            del qpos, ref, fp32_out, trunc_out, bf16_out
    del context_inputs
    return {
        "fp32": torch.cat(fp32_mses) if fp32_mses else torch.empty(0),
        "trunc": torch.cat(trunc_mses) if trunc_mses else torch.empty(0),
        "bf16": torch.cat(bf16_mses) if bf16_mses else torch.empty(0),
        "bf16_trunc_delta": torch.cat(bf16_trunc_delta_mses) if bf16_trunc_delta_mses else torch.empty(0),
    }, point_rows


def summarize_operator_values(
    *,
    run_id: str,
    model_label: str,
    model_id: str,
    spec: OperatorSpec,
    fp32_values: torch.Tensor,
    trunc_values: torch.Tensor,
    bf16_values: torch.Tensor,
    bf16_trunc_delta_values: torch.Tensor,
) -> dict[str, Any]:
    fp32_stats = summarize_values(fp32_values)
    trunc_stats = summarize_values(trunc_values)
    bf16_stats = summarize_values(bf16_values)
    bf16_trunc_delta_stats = summarize_values(bf16_trunc_delta_values)
    trunc_mean = trunc_stats["mean"]
    bf16_mean = bf16_stats["mean"]
    bf16_trunc_delta_mean = bf16_trunc_delta_stats["mean"]
    trunc_over_all = trunc_mean / bf16_mean if bf16_mean > 0 else 0.0
    arithmetic_mean = bf16_mean - trunc_mean
    arithmetic_over_all = 1.0 - trunc_over_all if bf16_mean > 0 else 0.0
    bf16_trunc_delta_over_all = bf16_trunc_delta_mean / bf16_mean if bf16_mean > 0 else 0.0
    shapley_over_all = 0.5 * (arithmetic_over_all + bf16_trunc_delta_over_all)
    arithmetic_values = bf16_values.to(torch.float32) - trunc_values.to(torch.float32)
    arithmetic_stats = summarize_values(arithmetic_values)
    return {
        "run_id": run_id,
        "model": model_label,
        "model_id": model_id,
        "operator": spec.operator,
        "family": spec.family,
        "layer": spec.layer,
        "points": int(trunc_values.numel()),
        "input_stage": spec.input_stage_label,
        "output_stage": spec.output_stage,
        "fp32_recompute_mse_mean": fp32_stats["mean"],
        "lower_precision_fp32_mse_mean": trunc_mean,
        "ideal_truncation_mse_mean": trunc_mean,
        "bf16_impl_mse_mean": bf16_mean,
        "lower_precision_fp32_over_bf16_mse": trunc_over_all,
        "ideal_truncation_over_bf16_mse": trunc_over_all,
        "arithmetic_mse_mean": arithmetic_mean,
        "arithmetic_over_bf16_mse": arithmetic_over_all,
        "arithmetic_residual_mse_mean": arithmetic_mean,
        "arithmetic_residual_over_bf16_mse": arithmetic_over_all,
        "bf16_trunc_delta_mse_mean": bf16_trunc_delta_mean,
        "bf16_trunc_delta_over_bf16_mse": bf16_trunc_delta_over_all,
        "shapley_arithmetic_over_bf16_mse": shapley_over_all,
        "lower_precision_fp32_mse_p50": trunc_stats["p50"],
        "lower_precision_fp32_mse_p90": trunc_stats["p90"],
        "ideal_truncation_mse_p50": trunc_stats["p50"],
        "ideal_truncation_mse_p90": trunc_stats["p90"],
        "bf16_impl_mse_p50": bf16_stats["p50"],
        "bf16_impl_mse_p90": bf16_stats["p90"],
        "arithmetic_mse_p50": arithmetic_stats["p50"],
        "arithmetic_mse_p90": arithmetic_stats["p90"],
        "bf16_trunc_delta_mse_p50": bf16_trunc_delta_stats["p50"],
        "bf16_trunc_delta_mse_p90": bf16_trunc_delta_stats["p90"],
    }


def build_operator_specs(
    layers: list[torch.nn.Module],
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module | None,
    include_lm_head: bool,
) -> list[OperatorSpec]:
    specs: list[OperatorSpec] = []
    for layer_idx, layer in enumerate(layers):
        prefix = f"layer_{layer_idx}"
        if layer_idx == 0:
            specs.append(
                OperatorSpec(
                    operator="rmsnorm.input_layernorm",
                    family="rmsnorm",
                    layer=layer_idx,
                    input_stages=("forward.embed_tokens.output",),
                    output_stage=f"{prefix}.input_layernorm.output.hidden",
                    fp32_fn=lambda x, norm=layer.input_layernorm: rmsnorm_fp32(norm, x),
                    bf16_fn=lambda x, norm=layer.input_layernorm: rmsnorm_bf16(norm, x),
                    trunc_fn=lambda x, norm=layer.input_layernorm: rmsnorm_lower_precision_fp32(norm, x),
                )
            )
        else:
            specs.append(
                OperatorSpec(
                    operator="rmsnorm.input_layernorm",
                    family="rmsnorm",
                    layer=layer_idx,
                    input_stages=(
                        f"layer_{layer_idx - 1}.mlp.down_proj.output",
                        f"layer_{layer_idx - 1}.post_attention_layernorm.output.residual",
                    ),
                    output_stage=f"{prefix}.input_layernorm.output.hidden",
                    fp32_fn=lambda x, residual, norm=layer.input_layernorm: rmsnorm_fp32(norm, x, residual),
                    bf16_fn=lambda x, residual, norm=layer.input_layernorm: rmsnorm_bf16(norm, x, residual),
                    trunc_fn=lambda x, residual, norm=layer.input_layernorm: rmsnorm_lower_precision_fp32(norm, x, residual),
                )
            )

        attn = layer.self_attn
        params = infer_attention_params(attn)
        rope_base = infer_rope_base(attn)
        for name, module_attr, output_suffix in (
            ("gemm.q_proj", "q_proj", "q"),
            ("gemm.k_proj", "k_proj", "k"),
            ("gemm.v_proj", "v_proj", "v"),
        ):
            module = getattr(attn, module_attr, None)
            if module is not None:
                specs.append(
                    OperatorSpec(
                        operator=name,
                        family="gemm",
                        layer=layer_idx,
                        input_stages=(f"{prefix}.input_layernorm.output.hidden",),
                        output_stage=f"{prefix}.self_attn.qkv_proj.output.{output_suffix}",
                        fp32_fn=lambda x, mod=module: linear_fp32(mod, x),
                        bf16_fn=lambda x, mod=module: linear_bf16_compute_fp32(mod, x),
                        trunc_fn=lambda x, mod=module: linear_lower_precision_fp32(mod, x),
                    )
                )

        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        head_dim = int(params.head_dim)
        q_rope_input_stage = f"{prefix}.self_attn.qkv_proj.output.q"
        k_rope_input_stage = f"{prefix}.self_attn.qkv_proj.output.k"
        if q_norm is not None and head_dim > 0:
            q_rope_input_stage = f"{prefix}.self_attn.q_norm.output"
            specs.append(
                OperatorSpec(
                    operator="rmsnorm.q_norm",
                    family="rmsnorm",
                    layer=layer_idx,
                    input_stages=(f"{prefix}.self_attn.qkv_proj.output.q",),
                    output_stage=q_rope_input_stage,
                    fp32_fn=lambda x, norm=q_norm, dim=head_dim: rmsnorm_heads_fp32(norm, x, dim),
                    bf16_fn=lambda x, norm=q_norm, dim=head_dim: rmsnorm_heads_bf16(norm, x, dim),
                    trunc_fn=lambda x, norm=q_norm, dim=head_dim: rmsnorm_heads_lower_precision_fp32(norm, x, dim),
                )
            )
        if k_norm is not None and head_dim > 0:
            k_rope_input_stage = f"{prefix}.self_attn.k_norm.output"
            specs.append(
                OperatorSpec(
                    operator="rmsnorm.k_norm",
                    family="rmsnorm",
                    layer=layer_idx,
                    input_stages=(f"{prefix}.self_attn.qkv_proj.output.k",),
                    output_stage=k_rope_input_stage,
                    fp32_fn=lambda x, norm=k_norm, dim=head_dim: rmsnorm_heads_fp32(norm, x, dim),
                    bf16_fn=lambda x, norm=k_norm, dim=head_dim: rmsnorm_heads_bf16(norm, x, dim),
                    trunc_fn=lambda x, norm=k_norm, dim=head_dim: rmsnorm_heads_lower_precision_fp32(norm, x, dim),
                )
            )

        specs.append(
            OperatorSpec(
                operator="rope.q",
                family="rope",
                layer=layer_idx,
                input_stages=(q_rope_input_stage,),
                output_stage=f"{prefix}.self_attn.rotary_emb.output.q",
                fp32_fn=lambda x, pos, heads=params.num_heads, dim=head_dim, base=rope_base: rope_fp32(x, pos, heads, dim, base),
                bf16_fn=lambda x, pos, heads=params.num_heads, dim=head_dim, base=rope_base: rope_bf16_compute_fp32(x, pos, heads, dim, base),
                trunc_fn=lambda x, pos, heads=params.num_heads, dim=head_dim, base=rope_base: rope_lower_precision_fp32(x, pos, heads, dim, base),
                position_arg=True,
            )
        )
        specs.append(
            OperatorSpec(
                operator="rope.k",
                family="rope",
                layer=layer_idx,
                input_stages=(k_rope_input_stage,),
                output_stage=f"{prefix}.self_attn.rotary_emb.output.k",
                fp32_fn=lambda x, pos, heads=params.num_kv_heads, dim=head_dim, base=rope_base: rope_fp32(x, pos, heads, dim, base),
                bf16_fn=lambda x, pos, heads=params.num_kv_heads, dim=head_dim, base=rope_base: rope_bf16_compute_fp32(x, pos, heads, dim, base),
                trunc_fn=lambda x, pos, heads=params.num_kv_heads, dim=head_dim, base=rope_base: rope_lower_precision_fp32(x, pos, heads, dim, base),
                position_arg=True,
            )
        )
        specs.append(
            OperatorSpec(
                operator="attention.full",
                family="attention",
                layer=layer_idx,
                input_stages=(
                    f"{prefix}.self_attn.rotary_emb.output.q",
                    f"{prefix}.self_attn.rotary_emb.output.k",
                    f"{prefix}.self_attn.qkv_proj.output.v",
                ),
                output_stage=f"{prefix}.self_attn.attention.output",
                fp32_fn=lambda q, k, v, pos, p=params: attention_fp32(q, k, v, pos, p),
                bf16_fn=lambda q, k, v, pos, p=params: attention_bf16_compute_fp32(q, k, v, pos, p),
                trunc_fn=lambda q, k, v, pos, p=params: attention_lower_precision_fp32(q, k, v, pos, p),
                full_context=True,
            )
        )
        specs.append(
            OperatorSpec(
                operator="gemm.o_proj",
                family="gemm",
                layer=layer_idx,
                input_stages=(f"{prefix}.self_attn.attention.output",),
                output_stage=f"{prefix}.self_attn.o_proj.output",
                fp32_fn=lambda x, mod=attn.o_proj: linear_fp32(mod, x),
                bf16_fn=lambda x, mod=attn.o_proj: linear_bf16_compute_fp32(mod, x),
                trunc_fn=lambda x, mod=attn.o_proj: linear_lower_precision_fp32(mod, x),
            )
        )

        residual_stage = "forward.embed_tokens.output" if layer_idx == 0 else f"{prefix}.input_layernorm.output.residual"
        specs.append(
            OperatorSpec(
                operator="rmsnorm.post_attention_layernorm",
                family="rmsnorm",
                layer=layer_idx,
                input_stages=(f"{prefix}.self_attn.o_proj.output", residual_stage),
                output_stage=f"{prefix}.post_attention_layernorm.output.hidden",
                fp32_fn=lambda x, residual, norm=layer.post_attention_layernorm: rmsnorm_fp32(norm, x, residual),
                bf16_fn=lambda x, residual, norm=layer.post_attention_layernorm: rmsnorm_bf16(norm, x, residual),
                trunc_fn=lambda x, residual, norm=layer.post_attention_layernorm: rmsnorm_lower_precision_fp32(norm, x, residual),
            )
        )

        mlp = layer.mlp
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        if gate_proj is not None and up_proj is not None:
            specs.append(
                OperatorSpec(
                    operator="gemm.gate_up_proj",
                    family="gemm",
                    layer=layer_idx,
                    input_stages=(f"{prefix}.post_attention_layernorm.output.hidden",),
                    output_stage=f"{prefix}.mlp.gate_up_proj.output",
                    fp32_fn=lambda x, gate=gate_proj, up=up_proj: torch.cat((linear_fp32(gate, x), linear_fp32(up, x)), dim=-1),
                    bf16_fn=lambda x, gate=gate_proj, up=up_proj: torch.cat((linear_bf16_compute_fp32(gate, x), linear_bf16_compute_fp32(up, x)), dim=-1),
                    trunc_fn=lambda x, gate=gate_proj, up=up_proj: torch.cat((linear_lower_precision_fp32(gate, x), linear_lower_precision_fp32(up, x)), dim=-1),
                )
            )
        specs.append(
            OperatorSpec(
                operator="silu_and_mul",
                family="silu_and_mul",
                layer=layer_idx,
                input_stages=(f"{prefix}.mlp.gate_up_proj.output",),
                output_stage=f"{prefix}.mlp.silu_and_mul.output",
                fp32_fn=silu_and_mul_fp32,
                bf16_fn=silu_and_mul_bf16,
                trunc_fn=silu_and_mul_lower_precision_fp32,
            )
        )
        specs.append(
            OperatorSpec(
                operator="gemm.down_proj",
                family="gemm",
                layer=layer_idx,
                input_stages=(f"{prefix}.mlp.silu_and_mul.output",),
                output_stage=f"{prefix}.mlp.down_proj.output",
                fp32_fn=lambda x, mod=mlp.down_proj: linear_fp32(mod, x),
                bf16_fn=lambda x, mod=mlp.down_proj: linear_bf16_compute_fp32(mod, x),
                trunc_fn=lambda x, mod=mlp.down_proj: linear_lower_precision_fp32(mod, x),
            )
        )

    last_idx = len(layers) - 1
    specs.append(
        OperatorSpec(
            operator="rmsnorm.final_norm",
            family="rmsnorm",
            layer="global",
            input_stages=(
                f"layer_{last_idx}.mlp.down_proj.output",
                f"layer_{last_idx}.post_attention_layernorm.output.residual",
            ),
            output_stage="forward.norm.hidden",
            fp32_fn=lambda x, residual, norm=final_norm: rmsnorm_fp32(norm, x, residual),
            bf16_fn=lambda x, residual, norm=final_norm: rmsnorm_bf16(norm, x, residual),
            trunc_fn=lambda x, residual, norm=final_norm: rmsnorm_lower_precision_fp32(norm, x, residual),
        )
    )
    if include_lm_head and lm_head is not None:
        specs.append(
            OperatorSpec(
                operator="gemm.lm_head",
                family="gemm",
                layer="global",
                input_stages=("forward.norm.hidden",),
                output_stage="lm_head.linear.logits",
                fp32_fn=lambda x, mod=lm_head: linear_fp32(mod, x),
                bf16_fn=lambda x, mod=lm_head: linear_bf16_compute_fp32(mod, x),
                trunc_fn=lambda x, mod=lm_head: linear_lower_precision_fp32(mod, x),
            )
        )
    return specs

def model_layers(hf_model: torch.nn.Module) -> tuple[torch.nn.Module, list[torch.nn.Module], torch.nn.Module, torch.nn.Module | None]:
    base = getattr(hf_model, "model", None)
    if base is None:
        raise RuntimeError("expected HuggingFace causal LM with .model")
    layers = list(getattr(base, "layers"))
    norm = getattr(base, "norm", None)
    if norm is None:
        raise RuntimeError("model has no final norm")
    lm_head = getattr(hf_model, "lm_head", None)
    return base, layers, norm, lm_head


def available_stage(stages: dict[str, torch.Tensor], name: str) -> torch.Tensor | None:
    return stages.get(name)


def trace_segment_records(config: dict[str, Any], trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments_path_value = config.get("trace_segments_path")
    if segments_path_value:
        segments_path = Path(str(segments_path_value))
        if segments_path.exists():
            return read_jsonl(segments_path)
    trace_path = Path(config["trace_path"])
    if trace_path.exists():
        return [
            {
                "segment_index": 0,
                "doc_id": None,
                "path": str(trace_path),
                "row_start": 0,
                "rows": len(trace_rows),
            }
        ]
    raise RuntimeError(
        "missing trace segments for analysis: "
        f"{segments_path_value or config['trace_path']}"
    )


def analyze_trace(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    from transformers import AutoModelForCausalLM

    trace_rows = read_jsonl(Path(config["trace_rows_path"]))
    segments = trace_segment_records(config, trace_rows)
    model_id = config["model_id"]
    model_label = config["model_label"]
    run_id = config["run_id"]
    device = torch.device(config["analysis_device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--analysis-device=cuda requested but CUDA is not available")

    log(
        f"loading HuggingFace model for streaming operator replay "
        f"model={model_id} device={device} segments={len(segments)}"
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=bool(config["trust_remote_code"]),
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    hf_model.to(device)
    base, layers, final_norm, lm_head = model_layers(hf_model)
    del base

    specs = build_operator_specs(layers, final_norm, lm_head, bool(config["include_lm_head"]))
    summary_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    chunk_rows = int(config["analysis_chunk_rows"])
    write_points = bool(config["write_points"])

    for spec in specs:
        fp32_batches: list[torch.Tensor] = []
        trunc_batches: list[torch.Tensor] = []
        bf16_batches: list[torch.Tensor] = []
        bf16_trunc_delta_batches: list[torch.Tensor] = []
        for segment in segments:
            segment_path = Path(str(segment["path"]))
            row_start = int(segment.get("row_start", 0))
            row_count = int(segment.get("rows", 0))
            segment_trace_rows = trace_rows[row_start : row_start + row_count]
            raw_selected_rows = segment.get("selected_row_indices")
            selected_rows = None if raw_selected_rows is None else [int(idx) for idx in raw_selected_rows]
            position_rows = [int(row.get("absolute_token_index", row.get("segment_row_index", idx))) for idx, row in enumerate(segment_trace_rows)]
            if not position_rows:
                position_rows = selected_rows or []
            positions = torch.tensor(position_rows, dtype=torch.long)

            if spec.full_context:
                loaded_inputs = [load_stage(segment_path, stage) for stage in spec.input_stages]
                reference_output = None if spec.computed_reference else load_stage_rows(segment_path, spec.output_stage, selected_rows)
                if any(item is None for item in loaded_inputs) or (reference_output is None and not spec.computed_reference):
                    del loaded_inputs, reference_output
                    continue
                inputs = tuple(item for item in loaded_inputs if item is not None)
                values, points = evaluate_context_operator_values(
                    run_id=run_id,
                    model_label=model_label,
                    operator=spec.operator,
                    family=spec.family,
                    layer=spec.layer,
                    inputs=inputs,
                    reference_output=reference_output,
                    fp32_fn=spec.fp32_fn,
                    bf16_fn=spec.bf16_fn,
                    trunc_fn=spec.trunc_fn,
                    query_positions=positions,
                    trace_rows=segment_trace_rows,
                    chunk_rows=chunk_rows,
                    device=device,
                    write_points=write_points,
                )
            else:
                loaded_inputs = [load_stage_rows(segment_path, stage, selected_rows) for stage in spec.input_stages]
                reference_output = load_stage_rows(segment_path, spec.output_stage, selected_rows)
                if reference_output is None or any(item is None for item in loaded_inputs):
                    del loaded_inputs, reference_output
                    continue
                inputs = tuple(item for item in loaded_inputs if item is not None)
                values, points = evaluate_operator_values(
                    run_id=run_id,
                    model_label=model_label,
                    operator=spec.operator,
                    family=spec.family,
                    layer=spec.layer,
                    inputs=inputs,
                    reference_output=reference_output,
                    fp32_fn=spec.fp32_fn,
                    bf16_fn=spec.bf16_fn,
                    trunc_fn=spec.trunc_fn,
                    trace_rows=segment_trace_rows,
                    chunk_rows=chunk_rows,
                    device=device,
                    write_points=write_points,
                    positions=positions,
                    position_arg=spec.position_arg,
                )
            if values["trunc"].numel() > 0:
                fp32_batches.append(values["fp32"])
                trunc_batches.append(values["trunc"])
                bf16_batches.append(values["bf16"])
                bf16_trunc_delta_batches.append(values["bf16_trunc_delta"])
                point_rows.extend(points)
            del loaded_inputs, inputs, reference_output, values, points, positions
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not trunc_batches:
            log(f"WARNING: skipped {spec.operator} layer={spec.layer}; missing stages in all segments")
            continue
        fp32_values = torch.cat(fp32_batches)
        trunc_values = torch.cat(trunc_batches)
        bf16_values = torch.cat(bf16_batches)
        bf16_trunc_delta_values = torch.cat(bf16_trunc_delta_batches)
        summary = summarize_operator_values(
            run_id=run_id,
            model_label=model_label,
            model_id=model_id,
            spec=spec,
            fp32_values=fp32_values,
            trunc_values=trunc_values,
            bf16_values=bf16_values,
            bf16_trunc_delta_values=bf16_trunc_delta_values,
        )
        summary_rows.append(summary)
        log(
            f"analyzed {summary['operator']} layer={summary['layer']} "
            f"points={summary['points']} all_mse={summary['bf16_impl_mse_mean']:.4g} "
            f"trunc_mse={summary['ideal_truncation_mse_mean']:.4g} "
            f"arith/all={summary['arithmetic_residual_over_bf16_mse']:.4g}"
        )
        del fp32_batches, trunc_batches, bf16_batches, bf16_trunc_delta_batches, fp32_values, trunc_values, bf16_values, bf16_trunc_delta_values

    out_dir = Path(config["analysis_output_dir"])
    write_csv(out_dir / "arith_vs_trunc_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_json(out_dir / "arith_vs_trunc_summary.json", {"summary": summary_rows})
    if write_points:
        write_csv(out_dir / "arith_vs_trunc_points.csv", point_rows, POINT_FIELDS)
    plot_outputs(out_dir, summary_rows)
    family_interval_outputs = {
        "residual": plot_family_intervals(
            out_dir,
            summary_rows,
            metric_field="arithmetic_residual_over_bf16_mse",
            filename_prefix="arith_vs_trunc_family_intervals",
            ylabel="1 - MSE[Trunc] / MSE[Error]",
        ),
        "bf16_trunc_delta": plot_family_intervals(
            out_dir,
            summary_rows,
            metric_field="bf16_trunc_delta_over_bf16_mse",
            filename_prefix="arith_vs_trunc_bf16_trunc_delta_family_intervals",
            ylabel="MSE[BF16-Trunc] / MSE[Error]",
        ),
        "shapley": plot_family_intervals(
            out_dir,
            summary_rows,
            metric_field="shapley_arithmetic_over_bf16_mse",
            filename_prefix="arith_vs_trunc_shapley_family_intervals",
            ylabel="Shapley MSE[Arith] / MSE[Error]",
        ),
    }
    result = {
        "kind": "analyze",
        "model_id": model_id,
        "operators": len(summary_rows),
        "points": sum(int(row["points"]) for row in summary_rows),
        "segments": len(segments),
        "elapsed_s": time.perf_counter() - started,
        "summary_csv": out_dir / "arith_vs_trunc_summary.csv",
        "summary_json": out_dir / "arith_vs_trunc_summary.json",
        "points_csv": out_dir / "arith_vs_trunc_points.csv" if write_points else None,
        "family_interval_outputs": family_interval_outputs,
    }
    write_json(Path(config["analysis_result_path"]), result)
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

def _operator_order(name: str) -> tuple[int, str]:
    order = {
        "gemm.q_proj": 0,
        "gemm.k_proj": 1,
        "gemm.v_proj": 2,
        "rmsnorm.q_norm": 3,
        "rmsnorm.k_norm": 4,
        "rope.q": 5,
        "rope.k": 6,
        "attention.full": 7,
        "attention.qk_matmul": 8,
        "attention.pv_matmul": 9,
        "gemm.o_proj": 10,
        "rmsnorm.input_layernorm": 11,
        "rmsnorm.post_attention_layernorm": 12,
        "gemm.gate_up_proj": 13,
        "silu_and_mul": 14,
        "gemm.down_proj": 15,
        "rmsnorm.final_norm": 16,
        "gemm.lm_head": 17,
    }
    return order.get(name, 100), name


def _x_position(row: dict[str, Any], offsets: dict[str, float], global_x: int) -> float:
    layer = row["layer"]
    base = global_x if str(layer) == "global" else int(layer)
    return base + offsets.get(str(row["operator"]), 0.0)


FAMILY_INTERVAL_ORDER = ("gemm", "rmsnorm", "rope", "attention", "silu_and_mul")
FAMILY_INTERVAL_LABELS = {
    "gemm": "GEMM",
    "rmsnorm": "RMSNorm",
    "rope": "RoPE",
    "attention": "Attention",
    "silu_and_mul": "SiLUAndMul",
}


def family_interval_key(family: str) -> tuple[int, str]:
    try:
        return FAMILY_INTERVAL_ORDER.index(family), family
    except ValueError:
        return len(FAMILY_INTERVAL_ORDER), family


def quantile_float(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def family_interval_rows(
    rows: list[dict[str, Any]],
    *,
    metric_field: str = "arithmetic_residual_over_bf16_mse",
    cap_gemm_at_zero: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        raw = row.get(metric_field)
        if raw in {None, ""} and metric_field == "arithmetic_residual_over_bf16_mse":
            raw = row.get("arithmetic_over_bf16_mse")
        if raw in {None, ""}:
            continue
        value = float(raw)
        if math.isfinite(value):
            family = str(row.get("family") or "other")
            if cap_gemm_at_zero and family == "gemm":
                value = max(0.0, value)
            grouped.setdefault(family, []).append(value)
    out: list[dict[str, Any]] = []
    for family in sorted(grouped, key=family_interval_key):
        values = grouped[family]
        out.append(
            {
                "family": family,
                "label": FAMILY_INTERVAL_LABELS.get(family, family),
                "operators": len(values),
                "min": min(values),
                "p05": quantile_float(values, 0.05),
                "p25": quantile_float(values, 0.25),
                "p50": quantile_float(values, 0.50),
                "p75": quantile_float(values, 0.75),
                "p95": quantile_float(values, 0.95),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
        )
    return out


INTERVAL_COLORS = {
    "min_max": "#f94144",
    "p05_p95": "#43aa8b",
    "p25_p75": "#277da1",
}


def plot_family_intervals(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    metric_field: str = "arithmetic_residual_over_bf16_mse",
    filename_prefix: str = "arith_vs_trunc_family_intervals",
    ylabel: str = "MSE[Arith] / MSE[Error]",
    cap_gemm_at_zero: bool = True,
) -> dict[str, str] | None:
    interval_rows = family_interval_rows(rows, metric_field=metric_field, cap_gemm_at_zero=cap_gemm_at_zero)
    if not interval_rows:
        return None
    csv_path = out_dir / f"{filename_prefix}.csv"
    write_csv(
        csv_path,
        interval_rows,
        ["family", "label", "operators", "min", "p05", "p25", "p50", "p75", "p95", "max", "mean"],
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        log(f"WARNING: matplotlib unavailable, skipping family interval plot: {exc}")
        return {"csv": str(csv_path)}

    pdf_path = out_dir / f"{filename_prefix}.pdf"
    png_path = out_dir / f"{filename_prefix}.png"
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = list(range(len(interval_rows)))
    plotted: list[float] = []

    def draw_interval(x: int, low: float, high: float, color: str, linewidth: float, capsize: float, zorder: int) -> None:
        center = (low + high) * 0.5
        ax.errorbar(
            [x],
            [center],
            yerr=[[center - low], [high - center]],
            fmt="none",
            ecolor=color,
            elinewidth=linewidth,
            capsize=capsize,
            capthick=linewidth,
            zorder=zorder,
        )

    for x, row in zip(xs, interval_rows):
        minimum = float(row["min"])
        p05 = float(row["p05"])
        p25 = float(row["p25"])
        p75 = float(row["p75"])
        p95 = float(row["p95"])
        maximum = float(row["max"])
        plotted.extend([minimum, p05, p25, p75, p95, maximum])
        draw_interval(x, minimum, maximum, INTERVAL_COLORS["min_max"], linewidth=1.4, capsize=4.5, zorder=1)
        draw_interval(x, p05, p95, INTERVAL_COLORS["p05_p95"], linewidth=1.8, capsize=4.0, zorder=2)
        ax.vlines(x, p25, p75, color=INTERVAL_COLORS["p25_p75"], linewidth=7.0, alpha=0.95, zorder=3)
    ax.set_xticks(xs)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.18)
    ax.set_xticklabels([str(row["label"]) for row in interval_rows], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="both", alpha=0.28)
    ax.legend(
        handles=[
            Line2D([0], [0], color=INTERVAL_COLORS["min_max"], linewidth=1.8, label="min-max"),
            Line2D([0], [0], color=INTERVAL_COLORS["p05_p95"], linewidth=2.2, label="p5-p95"),
            Line2D([0], [0], color=INTERVAL_COLORS["p25_p75"], linewidth=7.0, label="p25-p75"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=9,
    )
    if plotted:
        lo = min(min(plotted), 0.0)
        hi = max(plotted)
        pad = max((hi - lo) * 0.12, 0.01)
        ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return {"csv": str(csv_path), "pdf": str(pdf_path), "png": str(png_path)}


def plot_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        log(f"WARNING: matplotlib unavailable, skipping plots: {exc}")
        return

    operators = sorted({str(row["operator"]) for row in rows}, key=_operator_order)
    if len(operators) > 1:
        offsets = {op: (idx - (len(operators) - 1) / 2.0) * 0.72 / (len(operators) - 1) for idx, op in enumerate(operators)}
    else:
        offsets = {operators[0]: 0.0}
    numeric_layers = [int(row["layer"]) for row in rows if str(row["layer"]).isdigit()]
    global_x = max(numeric_layers) + 1 if numeric_layers else 0
    family_colors = {
        "gemm": "#277da1",
        "rmsnorm": "#f94144",
        "rope": "#984ea3",
        "attention": "#ff7f00",
        "silu_and_mul": "#43aa8b",
    }
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h")
    op_markers = {op: markers[idx % len(markers)] for idx, op in enumerate(operators)}

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for op in operators:
        op_rows = sorted(
            [row for row in rows if row["operator"] == op],
            key=lambda row: (str(row["layer"]) == "global", int(row["layer"]) if str(row["layer"]).isdigit() else 9999),
        )
        xs = [_x_position(row, offsets, global_x) for row in op_rows]
        ys = [float(row.get("lower_precision_fp32_mse_mean") or row["ideal_truncation_mse_mean"]) for row in op_rows]
        family = str(op_rows[0]["family"])
        color = family_colors.get(family, "#555555")
        marker = op_markers[op]
        ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8)
        ax.scatter(xs, ys, color=color, marker=marker, s=24, alpha=0.82, linewidths=0.2, edgecolors="white")
    ax.set_yscale("log")
    xticks = sorted(set(numeric_layers))
    tick_step = max(1, math.ceil(len(xticks) / 12)) if xticks else 1
    shown_ticks = [layer for idx, layer in enumerate(xticks) if idx % tick_step == 0]
    if any(str(row["layer"]) == "global" for row in rows):
        shown_ticks.append(global_x)
    ax.set_xticks(shown_ticks)
    ax.set_xticklabels(["global" if tick == global_x and tick not in xticks else str(tick) for tick in shown_ticks])
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("truncation MSE vs FP32")
    ax.grid(True, axis="y", alpha=0.22)
    family_handles = [
        Line2D([0], [0], color=color, marker="o", linestyle="-", label=family)
        for family, color in family_colors.items()
        if any(str(row["family"]) == family for row in rows)
    ]
    op_handles = [
        Line2D([0], [0], color="#555555", marker=op_markers[op], linestyle="None", label=op.replace("gemm.", "").replace("rmsnorm.", ""))
        for op in operators
    ]
    family_legend = ax.legend(handles=family_handles, frameon=False, loc="upper left")
    ax.add_artist(family_legend)
    ax.legend(handles=op_handles, frameon=False, loc="upper right", fontsize=7, title="operator", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "arith_vs_trunc_mse.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "arith_vs_trunc_mse.pdf", bbox_inches="tight")
    plt.close(fig)


def run_worker(kind: str, config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    config_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(config["result_path"])
    result_path.unlink(missing_ok=True)
    config_path = config_dir / f"{kind}.json"
    write_json(config_path, config)
    env = configure_env(dict(os.environ), FP32_LAYERCAST, Path(config["output_dir"]))
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", kind, "--config", str(config_path)]
    log(f"starting worker {kind}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return read_json(result_path)


def selected_models(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(fhm_common.MODEL_ORDER)
    return list(args.model)


def build_model_config(args: argparse.Namespace, run_id: str, model_arg: str) -> dict[str, Any]:
    model_id = fhm_common.resolve_model(model_arg)
    model_label = fhm_common.model_label(model_arg)
    run_root = OUTPUT_ROOT / run_id / model_label
    results_root = RESULTS_ROOT / run_id / model_label
    config_dir = run_root / "configs"
    return {
        "run_id": run_id,
        "model_arg": model_arg,
        "model_id": model_id,
        "model_label": model_label,
        "subset_file": str(Path(args.subset_file).resolve() if Path(args.subset_file).is_absolute() else ROOT / args.subset_file),
        "limit": int(args.limit),
        "max_new_tokens": int(args.max_new_tokens),
        "analyze_last_generated_tokens": int(args.analyze_last_generated_tokens),
        "seed": int(args.seed),
        "trust_remote_code": bool(args.trust_remote_code),
        "generation_batch_size": int(args.generation_batch_size),
        "max_model_len": str(args.max_model_len),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "analysis_device": str(args.analysis_device),
        "analysis_chunk_rows": int(args.analysis_chunk_rows),
        "write_points": bool(args.write_points),
        "include_lm_head": bool(args.include_lm_head),
        "output_dir": str(run_root),
        "results_dir": str(results_root),
        "config_dir": str(config_dir),
        "generated_path": str(run_root / "generated_tokens.jsonl"),
        "replay_manifest_path": str(run_root / "replay_manifest.jsonl"),
        "trace_path": str(run_root / "replay_trace.safetensors"),
        "trace_segments_dir": str(run_root / "trace_segments"),
        "trace_segments_path": str(run_root / "trace_segments.jsonl"),
        "trace_rows_path": str(run_root / "trace_rows.jsonl"),
        "analysis_output_dir": str(results_root),
        "analysis_result_path": str(results_root / "analysis_result.json"),
        "base_env": dict(os.environ),
        "patch_name": PATCH_NAME,
        "patch_sha256": patch_sha256(),
    }


def dry_run(args: argparse.Namespace, run_id: str) -> None:
    rows = fhm_common.load_subset(Path(args.subset_file), limit=int(args.limit))
    manifest = {
        "dry_run": True,
        "run_id": run_id,
        "task": "arith-vs-trunc",
        "patch_name": PATCH_NAME,
        "patch_sha256": patch_sha256(),
        "active_patch_status": active_patch_status(),
        "subset_file": str(args.subset_file),
        "docs": len(rows),
        "models": selected_models(args),
        "max_new_tokens": args.max_new_tokens,
        "analyze_last_generated_tokens": args.analyze_last_generated_tokens,
        "stage_regex": TRACE_STAGE_REGEX,
    }
    out = RESULTS_ROOT / run_id / "run_manifest.json"
    write_json(out, manifest)
    log(f"wrote dry-run manifest {out}")


def run(args: argparse.Namespace) -> None:
    run_id = args.run_id or f"arith_vs_trunc_{utc_timestamp()}"
    if args.dry_run:
        dry_run(args, run_id)
        return
    if args.analysis_device == "auto":
        args.analysis_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args = selected_models(args)
    run_manifest = {
        "run_id": run_id,
        "task": "arith-vs-trunc",
        "dump_patch_name": PATCH_NAME,
        "dump_patch_sha256": patch_sha256(PATCH_NAME),
        "analyze_patch_name": ANALYZE_PATCH_NAME,
        "analyze_patch_sha256": patch_sha256(ANALYZE_PATCH_NAME),
        "started_utc": utc_timestamp(),
        "args": vars(args),
        "models": model_args,
    }
    write_json(RESULTS_ROOT / run_id / "run_manifest.json", run_manifest)

    configs: list[dict[str, Any]] = []
    for model_arg in model_args:
        config = build_model_config(args, run_id, model_arg)
        config["analyze_patch_name"] = ANALYZE_PATCH_NAME
        config["analyze_patch_sha256"] = patch_sha256(ANALYZE_PATCH_NAME)
        run_root = Path(config["output_dir"])
        results_root = Path(config["results_dir"])
        if args.force:
            if args.stage in {"all", "generate"}:
                shutil.rmtree(run_root, ignore_errors=True)
                shutil.rmtree(results_root, ignore_errors=True)
            elif args.stage == "dump":
                for path_name in ("trace_path", "trace_rows_path", "trace_segments_path"):
                    Path(config[path_name]).unlink(missing_ok=True)
                shutil.rmtree(Path(config["trace_segments_dir"]), ignore_errors=True)
                Path(run_root / "dump_result.json").unlink(missing_ok=True)
                shutil.rmtree(results_root, ignore_errors=True)
            elif args.stage == "analyze":
                shutil.rmtree(results_root, ignore_errors=True)
        run_root.mkdir(parents=True, exist_ok=True)
        results_root.mkdir(parents=True, exist_ok=True)
        write_json(results_root / "model_manifest.json", config)
        configs.append(config)

    dump_prepared = False
    analyze_prepared = False
    try:
        if args.stage in {"all", "generate", "dump"}:
            dump_prepared = prepare_version(PATCH_NAME)
            for config in configs:
                run_root = Path(config["output_dir"])
                config_dir = Path(config["config_dir"])
                if args.stage in {"all", "generate"}:
                    gen_config = dict(config, result_path=str(run_root / "generate_result.json"))
                    run_worker("generate", gen_config, config_dir)
                if args.stage in {"all", "dump"}:
                    dump_config = dict(config, result_path=str(run_root / "dump_result.json"))
                    run_worker("dump", dump_config, config_dir)
            if args.stage == "all":
                revert_version(PATCH_NAME)
                dump_prepared = False

        if args.stage in {"all", "analyze"}:
            analyze_prepared = prepare_version(ANALYZE_PATCH_NAME)
            for config in configs:
                has_flat_trace = Path(config["trace_path"]).exists()
                has_segment_trace = Path(config["trace_segments_path"]).exists()
                if not has_flat_trace and not has_segment_trace:
                    raise RuntimeError(
                        "missing trace for analysis: "
                        f"{config['trace_path']} or {config['trace_segments_path']}"
                    )
                analyze_trace(config)
    finally:
        if analyze_prepared and not args.keep_patch:
            revert_version(ANALYZE_PATCH_NAME)
        if dump_prepared and not args.keep_patch:
            revert_version(PATCH_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["generate", "dump"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", nargs="+", default=["qwen3-8b"])
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--subset-file", type=Path, default=DEFAULT_SUBSET_FILE)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--analyze-last-generated-tokens", type=int, default=1024)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", default="4096")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--analysis-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--analysis-chunk-rows", type=int, default=128)
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--write-points", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stage", choices=["all", "generate", "dump", "analyze"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-patch", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        if args.config is None:
            raise SystemExit("--worker requires --config")
        config = read_json(args.config)
        if args.worker == "generate":
            worker_generate(config)
        elif args.worker == "dump":
            worker_dump(config)
        raise SystemExit(0)
    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
