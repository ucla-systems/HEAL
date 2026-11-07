#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the ShareGPT official serving benchmark matrix.

The benchmark itself is delegated to tests/sharegpt-benchmark/run_sharegpt_serving.py so reported
TPOT/TTFT values use vLLM's official `bench serve` path. This wrapper handles
the current precision matrix, patch switching, and result aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import heal_policy  # noqa: E402

OUTPUT_ROOT = ROOT / "output" / "sharegpt-benchmark"
RESULTS_ROOT = TEST_DIR / "results"
PATCH_SCRIPT = ROOT / "scripts" / "vllm_incremental_build.sh"
SERVING_RUNNER = TEST_DIR / "run_sharegpt_serving.py"
DEFAULT_DATASET = ROOT / "output" / "sharegpt_bench" / "data" / "sharegpt_vicuna_subset4096.json"

MODEL_ALIASES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama31_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek_r1_distill_qwen_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}
MODEL_ORDER = [
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "llama-3.1-8b-instruct",
    "deepseek-r1-distill-qwen-14b",
]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    version: str
    dtype: str
    label: str
    tf32: bool = False
    heal_mode: str | None = None
    attention_backend: str | None = None


CASES: list[CaseSpec] = [
    CaseSpec(
        name="original_fp16",
        version="VLLM_ORIGINAL",
        dtype="float16",
        label="Original FP16",
        attention_backend="TRITON_ATTN",
    ),
    CaseSpec(
        name="original_bf16",
        version="VLLM_ORIGINAL",
        dtype="bfloat16",
        label="Original BF16",
        attention_backend="TRITON_ATTN",
    ),
    CaseSpec(
        name="layercast_fp32",
        version="VLLM_LAYERCAST",
        dtype="float32",
        label="LayerCast FP32",
        attention_backend="TRITON_ATTN",
    ),
    CaseSpec(
        name="layercast_tf32",
        version="VLLM_LAYERCAST",
        dtype="float32",
        label="LayerCast TF32",
        tf32=True,
        attention_backend="TRITON_ATTN",
    ),
    CaseSpec(
        name="heal_base",
        version="VLLM_HEAL",
        dtype="float16",
        label="HEAL-Base",
        heal_mode="base",
        attention_backend="TRITON_ATTN",
    ),
    CaseSpec(
        name="heal_plus",
        version="VLLM_HEAL",
        dtype="float16",
        label="HEAL+",
        heal_mode="plus",
        attention_backend="TRITON_ATTN",
    ),
    CaseSpec(
        name="heal_full",
        version="VLLM_HEAL",
        dtype="float16",
        label="HEAL-Full",
        heal_mode="full",
        attention_backend="TRITON_ATTN",
    ),
    CaseSpec(
        name="heal_plus_plus",
        version="VLLM_HEAL",
        dtype="float16",
        label="HEAL++",
        heal_mode="plus",
        attention_backend="TRITON_ATTN",
    ),
    CaseSpec(
        name="heal_full_plus",
        version="VLLM_HEAL",
        dtype="float16",
        label="HEAL-Full+",
        heal_mode="full",
        attention_backend="TRITON_ATTN",
    ),
]
CASES_BY_NAME = {case.name: case for case in CASES}
PATCH_ORDER = ("VLLM_ORIGINAL", "VLLM_LAYERCAST", "VLLM_HEAL")

HEAL_DYNAMIC_INT8_GEMM_WEIGHT_BY_MODEL_CASE: dict[tuple[str, str], bool] = {}
HEAL_FP16_GEMM_COMPENSATION_BY_MODEL_CASE: dict[tuple[str, str], bool] = {}
PAUSED_MODEL_CASES: dict[tuple[str, str], str] = {}


def heal_dynamic_int8_gemm_weight_for_case(model: str, case: CaseSpec) -> bool:
    return bool(
        HEAL_DYNAMIC_INT8_GEMM_WEIGHT_BY_MODEL_CASE.get(
            (resolve_model(model), case.name), False
        )
    )


def heal_fp16_gemm_compensation_for_case(model: str, case: CaseSpec) -> bool:
    return case.heal_mode == "full" or bool(
        HEAL_FP16_GEMM_COMPENSATION_BY_MODEL_CASE.get(
            (resolve_model(model), case.name), False
        )
    )


def _policy_model_label(model: str) -> str:
    return resolve_model(model).replace("/", "-")


def heal_plus_skip_qkv_layers_for_case(
    args: argparse.Namespace, case: CaseSpec, model: str | None = None
) -> str | None:
    if case.name not in {"heal_plus", "heal_plus_plus"}:
        return None
    value = getattr(args, "heal_skip_qkv_quant_layers", None)
    if value is None and model is not None and case.name == "heal_plus_plus":
        value = heal_policy.policy_skip_layers(
            args.heal_policy,
            model_arg=model,
            model_id=resolve_model(model),
            model_label=_policy_model_label(model),
            variant="HEAL++",
            field="HEAL_SKIP_QKV_QUANT_LAYERS",
        )
    return str(value) if value is not None else None


def heal_full_skip_fp16_layers_for_case(
    args: argparse.Namespace, case: CaseSpec, model: str | None = None
) -> str | None:
    if case.name not in {"heal_full", "heal_full_plus"}:
        return None
    value = getattr(args, "heal_skip_fp16_comp_layers", None)
    if value is None:
        value = getattr(args, "heal_skip_int8_comp_layers", None)
    if value is None and model is not None and case.name == "heal_full_plus":
        value = heal_policy.policy_skip_layers(
            args.heal_policy,
            model_arg=model,
            model_id=resolve_model(model),
            model_label=_policy_model_label(model),
            variant="HEAL-FULL+",
            field="HEAL_SKIP_FP16_COMP_LAYERS",
        )
    return str(value) if value is not None else None


def skip_reason_for_model_case(model: str, case: CaseSpec) -> str | None:
    return PAUSED_MODEL_CASES.get((resolve_model(model), case.name))


SUMMARY_FIELDS = [
    "model",
    "model_arg",
    "case",
    "label",
    "version",
    "dtype",
    "tf32",
    "heal_mode",
    "attention_backend",
    "status",
    "elapsed_s",
    "completed",
    "failed",
    "official_mean_tpot_ms",
    "official_median_tpot_ms",
    "official_p99_tpot_ms",
    "official_mean_ttft_ms",
    "official_mean_itl_ms",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "total_input_tokens",
    "total_output_tokens",
    "model_loading_memory_gib",
    "graph_capture_memory_gib",
    "model_load_plus_graph_memory_gib",
    "memory_usage_gib",
    "available_kv_cache_memory_gib",
    "gpu_kv_cache_size_tokens",
    "max_concurrency",
    "metrics_path",
    "output_dir",
    "error",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def model_slug(model: str) -> str:
    resolved = resolve_model(model)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in resolved).strip("_")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CaseSpec):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def vllm_git_sha() -> str:
    return git_output(["git", "-C", str(ROOT / "vllm"), "rev-parse", "HEAD"])


def ensure_vllm_clean() -> None:
    status = git_output(["git", "-C", str(ROOT / "vllm"), "status", "--porcelain"])
    if status:
        raise RuntimeError(f"vLLM worktree is not clean:\n{status}")


def active_patch_status() -> str:
    return subprocess.check_output([str(PATCH_SCRIPT), "status"], cwd=ROOT, text=True)


def active_patch_name() -> str | None:
    status_lines = active_patch_status().splitlines()
    first = status_lines[0] if status_lines else ""
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


def run_patch_command(action: str, version: str | None = None) -> None:
    cmd = [str(PATCH_SCRIPT), action]
    if version is not None:
        cmd.append(version)
    subprocess.run(cmd, cwd=ROOT, check=True)


def revert_active_patch_best_effort() -> None:
    try:
        if active_patch_name() is not None:
            log("reverting active vLLM patch")
            run_patch_command("revert-version")
    except Exception as exc:
        log(f"WARNING: failed to revert active patch automatically: {exc}")


def model_args_from_cli(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(MODEL_ORDER)
    if args.models:
        return list(args.models)
    return [args.model]


def case_args_from_cli(args: argparse.Namespace) -> list[CaseSpec]:
    names = args.cases or [case.name for case in CASES]
    return [CASES_BY_NAME[name] for name in names]


def cases_by_patch(cases: list[CaseSpec]) -> dict[str, list[CaseSpec]]:
    grouped: dict[str, list[CaseSpec]] = {version: [] for version in PATCH_ORDER}
    for case in cases:
        grouped.setdefault(case.version, []).append(case)
    return {version: grouped.get(version, []) for version in PATCH_ORDER if grouped.get(version)}


def precision_env(
    case: CaseSpec,
    model: str | None = None,
    args: argparse.Namespace | None = None,
) -> dict[str, str]:
    env = {
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1" if case.tf32 else "0",
        "VLLM_FLOAT32_MATMUL_PRECISION": "high" if case.tf32 else "highest",
    }
    if model is not None and heal_dynamic_int8_gemm_weight_for_case(model, case):
        env["HEAL_DYNAMIC_INT8_GEMM_WEIGHT"] = "1"
        env["HEAL_INT8_GEMM_COMPENSATION"] = "1"
    if model is not None and heal_fp16_gemm_compensation_for_case(model, case):
        env["HEAL_FP16_GEMM_COMPENSATION"] = "1"
    if args is not None:
        heal_plus_skip = heal_plus_skip_qkv_layers_for_case(args, case, model)
        heal_full_skip = heal_full_skip_fp16_layers_for_case(args, case, model)
        if heal_plus_skip:
            env["HEAL_SKIP_QKV_QUANT_LAYERS"] = heal_plus_skip
        if heal_full_skip:
            env["HEAL_SKIP_FP16_COMP_LAYERS"] = heal_full_skip
            env["HEAL_SKIP_INT8_COMP_LAYERS"] = heal_full_skip
    return env


def serving_output_name(args: argparse.Namespace, model: str, case: CaseSpec) -> str:
    # run_sharegpt_serving.py writes below output/sharegpt_serving. A
    # relative prefix keeps this test's artifacts under output/sharegpt-benchmark
    # while preserving the official serving runner.
    return "/".join(["..", "sharegpt-benchmark", args.run_id, model_slug(model), case.name])


def serving_output_dir(args: argparse.Namespace, model: str, case: CaseSpec) -> Path:
    return OUTPUT_ROOT / args.run_id / model_slug(model) / case.name


def build_serving_command(args: argparse.Namespace, model: str, case: CaseSpec) -> list[str]:
    cmd = [
        sys.executable,
        str(SERVING_RUNNER),
        "--version",
        case.version,
        "--model",
        resolve_model(model),
        "--dataset-json",
        str(args.dataset_json),
        "--max-prompts",
        str(args.max_prompts),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--dtype",
        case.dtype,
        "--temperature",
        str(args.temperature),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--seed",
        str(args.seed),
        "--request-rate",
        str(args.request_rate),
        "--server-timeout-sec",
        str(args.server_timeout_sec),
        "--output-name",
        serving_output_name(args, model, case),
    ]
    if case.attention_backend:
        cmd.extend(["--attention-backend", case.attention_backend])
    if case.version == "VLLM_LAYERCAST":
        cmd.append("--layercast-fp32")
    if case.heal_mode:
        cmd.extend(["--heal-mode", case.heal_mode])
    if heal_dynamic_int8_gemm_weight_for_case(model, case):
        cmd.append("--heal-dynamic-int8-gemm-weight")
    if heal_fp16_gemm_compensation_for_case(model, case):
        cmd.append("--heal-fp16-gemm-compensation")
    heal_plus_skip = heal_plus_skip_qkv_layers_for_case(args, case, model)
    if heal_plus_skip:
        cmd.extend(["--heal-plus-skip-qkv-layers", heal_plus_skip])
    heal_full_skip = heal_full_skip_fp16_layers_for_case(args, case, model)
    if heal_full_skip:
        cmd.extend(["--heal-full-skip-fp16-layers", heal_full_skip])
    if args.sharegpt_output_len is not None:
        cmd.extend(["--sharegpt-output-len", str(args.sharegpt_output_len)])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd



def elapsed_from_metrics(metrics: dict[str, Any], elapsed_s: float) -> float:
    if elapsed_s > 0:
        return elapsed_s
    for key in ("elapsed_s", "duration", "benchmark_duration_s"):
        value = metrics.get(key)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                return numeric
    official_path = metrics.get("official_metrics_path")
    if official_path:
        try:
            official = json.loads(Path(str(official_path)).read_text(encoding="utf-8"))
            duration = float(official.get("duration") or 0.0)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            duration = 0.0
        if duration > 0:
            return duration
    return elapsed_s

def row_from_metrics(
    *,
    args: argparse.Namespace,
    model_arg: str,
    case: CaseSpec,
    metrics_path: Path,
    elapsed_s: float,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    elapsed_s = elapsed_from_metrics(metrics, elapsed_s)
    memory = metrics.get("vllm_log_memory") or {}
    model_mem = memory.get("model_loading_memory_gib")
    graph_mem = memory.get("graph_capture_memory_gib")
    model_plus_graph = None
    if model_mem is not None:
        model_plus_graph = float(model_mem) + float(graph_mem or 0.0)
    memory_usage = model_plus_graph
    return {
        "model": resolve_model(model_arg),
        "model_arg": model_arg,
        "case": case.name,
        "label": case.label,
        "version": case.version,
        "dtype": case.dtype,
        "tf32": case.tf32,
        "heal_mode": case.heal_mode,
        "attention_backend": case.attention_backend,
        "status": "ok",
        "elapsed_s": round(elapsed_s, 3),
        "completed": metrics.get("completed"),
        "failed": metrics.get("failed"),
        "official_mean_tpot_ms": metrics.get("official_mean_tpot_ms"),
        "official_median_tpot_ms": metrics.get("official_median_tpot_ms"),
        "official_p99_tpot_ms": metrics.get("official_p99_tpot_ms"),
        "official_mean_ttft_ms": metrics.get("official_mean_ttft_ms"),
        "official_mean_itl_ms": metrics.get("official_mean_itl_ms"),
        "request_throughput": metrics.get("request_throughput"),
        "output_throughput": metrics.get("output_throughput"),
        "total_token_throughput": metrics.get("total_token_throughput"),
        "total_input_tokens": metrics.get("total_input_tokens"),
        "total_output_tokens": metrics.get("total_output_tokens"),
        "model_loading_memory_gib": model_mem,
        "graph_capture_memory_gib": graph_mem,
        "model_load_plus_graph_memory_gib": model_plus_graph,
        "memory_usage_gib": memory_usage,
        "available_kv_cache_memory_gib": memory.get("available_kv_cache_memory_gib"),
        "gpu_kv_cache_size_tokens": memory.get("gpu_kv_cache_size_tokens"),
        "max_concurrency": memory.get("max_concurrency"),
        "metrics_path": str(metrics_path),
        "output_dir": str(serving_output_dir(args, model_arg, case)),
        "error": None,
    }


def error_row(
    *,
    args: argparse.Namespace,
    model_arg: str,
    case: CaseSpec,
    error: BaseException,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "model": resolve_model(model_arg),
        "model_arg": model_arg,
        "case": case.name,
        "label": case.label,
        "version": case.version,
        "dtype": case.dtype,
        "tf32": case.tf32,
        "heal_mode": case.heal_mode,
        "attention_backend": case.attention_backend,
        "status": "error",
        "elapsed_s": round(elapsed_s, 3),
        "metrics_path": str(serving_output_dir(args, model_arg, case) / "metrics.json"),
        "output_dir": str(serving_output_dir(args, model_arg, case)),
        "error": f"{type(error).__name__}: {error}",
    }


def skipped_row(
    *,
    args: argparse.Namespace,
    model_arg: str,
    case: CaseSpec,
    reason: str,
) -> dict[str, Any]:
    return {
        "model": resolve_model(model_arg),
        "model_arg": model_arg,
        "case": case.name,
        "label": case.label,
        "version": case.version,
        "dtype": case.dtype,
        "tf32": case.tf32,
        "heal_mode": case.heal_mode,
        "attention_backend": case.attention_backend,
        "status": "skipped",
        "elapsed_s": 0.0,
        "metrics_path": str(serving_output_dir(args, model_arg, case) / "metrics.json"),
        "output_dir": str(serving_output_dir(args, model_arg, case)),
        "error": reason,
    }


def append_result_files(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    result_dir = RESULTS_ROOT / args.run_id
    write_json(result_dir / "summary.json", rows)
    write_csv(result_dir / "summary.csv", rows, SUMMARY_FIELDS)


def run_case(args: argparse.Namespace, model_arg: str, case: CaseSpec) -> dict[str, Any]:
    output_dir = serving_output_dir(args, model_arg, case)
    metrics_path = output_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        log(f"resume: {model_arg} {case.name} already has metrics")
        return row_from_metrics(
            args=args,
            model_arg=model_arg,
            case=case,
            metrics_path=metrics_path,
            elapsed_s=0.0,
        )

    cmd = build_serving_command(args, model_arg, case)
    env = dict(os.environ)
    env.update(precision_env(case, model_arg, args))
    start = time.monotonic()
    log(f"running {model_arg} {case.name}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    elapsed_s = time.monotonic() - start
    if not metrics_path.exists():
        raise FileNotFoundError(f"expected metrics file was not written: {metrics_path}")
    return row_from_metrics(
        args=args,
        model_arg=model_arg,
        case=case,
        metrics_path=metrics_path,
        elapsed_s=elapsed_s,
    )


def write_manifest(args: argparse.Namespace, models: list[str], cases: list[CaseSpec]) -> None:
    result_dir = RESULTS_ROOT / args.run_id
    payload = {
        "run_id": args.run_id,
        "created_at_utc": utc_timestamp(),
        "base_sha": vllm_git_sha() if (ROOT / "vllm" / ".git").exists() else None,
        "dataset_json": str(args.dataset_json),
        "models": [
            {"arg": model, "resolved": resolve_model(model), "slug": model_slug(model)}
            for model in models
        ],
        "cases": [asdict(case) for case in cases],
        "heal_dynamic_int8_gemm_weight_overrides": [
            {
                "model": resolve_model(model),
                "model_arg": model,
                "case": case.name,
            }
            for model in models
            for case in cases
            if heal_dynamic_int8_gemm_weight_for_case(model, case)
        ],
        "heal_fp16_gemm_compensation_overrides": [
            {
                "model": resolve_model(model),
                "model_arg": model,
                "case": case.name,
            }
            for model in models
            for case in cases
            if heal_fp16_gemm_compensation_for_case(model, case)
        ],
        "skipped_model_cases": [
            {
                "model": resolve_model(model),
                "model_arg": model,
                "case": case.name,
                "reason": skip_reason_for_model_case(model, case),
            }
            for model in models
            for case in cases
            if skip_reason_for_model_case(model, case)
        ],
        "defaults": {
            "max_prompts": args.max_prompts,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "sharegpt_output_len": args.sharegpt_output_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "request_rate": args.request_rate,
            "seed": args.seed,
            "temperature": args.temperature,
            "heal_skip_qkv_quant_layers": args.heal_skip_qkv_quant_layers,
            "heal_skip_fp16_comp_layers": args.heal_skip_fp16_comp_layers,
            "heal_skip_int8_comp_layers": args.heal_skip_int8_comp_layers,
        },
        "output_root": str(OUTPUT_ROOT / args.run_id),
        "result_dir": str(result_dir),
        "dry_run": args.dry_run,
    }
    write_json(result_dir / "manifest.json", jsonable(payload))


def dry_run(args: argparse.Namespace, models: list[str], cases: list[CaseSpec]) -> None:
    rows: list[dict[str, Any]] = []
    for model in models:
        for case in cases:
            skip_reason = skip_reason_for_model_case(model, case)
            row = {
                "model": resolve_model(model),
                "model_arg": model,
                "case": case.name,
                "label": case.label,
                "version": case.version,
                "dtype": case.dtype,
                "tf32": case.tf32,
                "heal_mode": case.heal_mode,
                "heal_dynamic_int8_gemm_weight": (
                    heal_dynamic_int8_gemm_weight_for_case(model, case)
                ),
                "heal_fp16_gemm_compensation": (
                    heal_fp16_gemm_compensation_for_case(model, case)
                ),
                "status": "skipped" if skip_reason else "dry_run",
                "metrics_path": str(serving_output_dir(args, model, case) / "metrics.json"),
                "output_dir": str(serving_output_dir(args, model, case)),
                "error": skip_reason,
                "command": None if skip_reason else build_serving_command(args, model, case),
                "env": None if skip_reason else precision_env(case, model, args),
            }
            rows.append(row)
    result_dir = RESULTS_ROOT / args.run_id
    write_json(result_dir / "dry_run_plan.json", jsonable(rows))
    log(f"wrote dry-run plan to {result_dir / 'dry_run_plan.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", default="qwen3-8b")
    model_group.add_argument("--models", nargs="+")
    model_group.add_argument("--all-models", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES_BY_NAME))
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-prompts", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--sharegpt-output-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--server-timeout-sec", type=int, default=1200)
    parser.add_argument("--run-id", default=f"sharegpt_{utc_timestamp()}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-vllm-clean-check", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--heal-skip-qkv-quant-layers",
        help="Comma-separated layers where the heal_plus case falls back to HEAL q/k/v handling.",
    )
    parser.add_argument(
        "--heal-skip-fp16-comp-layers",
        help="Comma-separated layers where the heal_full case falls back to HEAL+ MLP handling.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.heal_policy = heal_policy.load_heal_policy(
        args.heal_policy_json,
        root=ROOT,
    )
    args.dataset_json = args.dataset_json.expanduser()
    if not args.dataset_json.is_absolute():
        args.dataset_json = ROOT / args.dataset_json
    models = model_args_from_cli(args)
    cases = case_args_from_cli(args)
    write_manifest(args, models, cases)

    if args.dry_run:
        dry_run(args, models, cases)
        return
    if not args.dataset_json.exists():
        raise FileNotFoundError(f"ShareGPT dataset does not exist: {args.dataset_json}")
    if not args.skip_vllm_clean_check:
        ensure_vllm_clean()
    ensure_no_active_patch()

    rows: list[dict[str, Any]] = []
    try:
        grouped = cases_by_patch(cases)
        for version, version_cases in grouped.items():
            if not any(
                skip_reason_for_model_case(model, case) is None
                for model in models
                for case in version_cases
            ):
                log(f"skipping {version}: all selected model/case pairs are paused")
                continue
            log(f"preparing {version}")
            run_patch_command("prepare-version", version)
            try:
                for model in models:
                    for case in version_cases:
                        skip_reason = skip_reason_for_model_case(model, case)
                        if skip_reason:
                            log(f"skipping {model} {case.name}: {skip_reason}")
                            rows.append(
                                skipped_row(
                                    args=args,
                                    model_arg=model,
                                    case=case,
                                    reason=skip_reason,
                                )
                            )
                            append_result_files(args, rows)
                            continue
                        start = time.monotonic()
                        try:
                            row = run_case(args, model, case)
                        except Exception as exc:
                            row = error_row(
                                args=args,
                                model_arg=model,
                                case=case,
                                error=exc,
                                elapsed_s=time.monotonic() - start,
                            )
                            rows.append(row)
                            append_result_files(args, rows)
                            if not args.keep_going:
                                raise
                        else:
                            rows.append(row)
                            append_result_files(args, rows)
            finally:
                run_patch_command("revert-version")
    except Exception:
        revert_active_patch_best_effort()
        raise

    append_result_files(args, rows)
    log(f"wrote summary to {RESULTS_ROOT / args.run_id / 'summary.csv'}")


if __name__ == "__main__":
    main()
