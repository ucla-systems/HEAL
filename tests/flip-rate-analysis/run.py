# SPDX-License-Identifier: Apache-2.0
"""MedQA answer flip-rate runner using direct non-eager generation."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import heal_policy  # noqa: E402

VLLM_DIR = ROOT / "vllm"
PATCH_SCRIPT = ROOT / "scripts" / "vllm_incremental_build.sh"
OUTPUT_ROOT = ROOT / "output" / "flip-rate-analysis"
RESULTS_ROOT = TEST_DIR / "results"
DEFAULT_REFERENCE_GENERATION_DIR = (
    ROOT
    / "output"
    / "final-hidden-mse"
    / "medqa_ref"
    / "generation_artifacts"
)

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
class ModeSpec:
    name: str
    version: str
    dtype: str
    tf32: bool = False
    significand_bits: int | None = None
    heal_mode: str | None = None
    setting: str | None = None
    attention_backend: str | None = None


MODE_SPECS: list[ModeSpec] = [
    ModeSpec(
        name="layercast_fp32_generate",
        version="VLLM_LAYERCAST",
        dtype="float32",
        tf32=False,
        setting="layercast_fp32",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="layercast_tf32_generate",
        version="VLLM_LAYERCAST",
        dtype="float32",
        tf32=True,
        setting="layercast_tf32",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="original_fp16_generate",
        version="VLLM_ORIGINAL",
        dtype="float16",
        setting="native_fp16",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="original_bf16_generate",
        version="VLLM_ORIGINAL",
        dtype="bfloat16",
        setting="native_bf16",
        attention_backend="TRITON_ATTN",
    ),
] + [
    ModeSpec(
        name=f"significand_k{k:02d}_generate",
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
        name=f"heal_{mode}_generate",
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
        name="heal_plus_plus_generate",
        version="VLLM_HEAL",
        dtype="float16",
        tf32=False,
        heal_mode="plus",
        setting="heal_plus_plus",
        attention_backend="TRITON_ATTN",
    ),
    ModeSpec(
        name="heal_full_plus_generate",
        version="VLLM_HEAL",
        dtype="float16",
        tf32=False,
        heal_mode="full",
        setting="heal_full_plus",
        attention_backend="TRITON_ATTN",
    ),
]
MODE_SPECS_BY_NAME = {spec.name: spec for spec in MODE_SPECS}
PATCH_ORDER = (
    "VLLM_LAYERCAST",
    "VLLM_ORIGINAL",
    "VLLM_SIGNIFICAND_BITS",
    "VLLM_HEAL",
)

GENERATION_BATCH_SIZE_CAPS_BY_MODEL = {
    "Qwen/Qwen3-32B": 16,
}
GENERATION_BATCH_CAP_DISABLE_MIN_GPU_BYTES = 80 * 1024**3
HEAL_DYNAMIC_INT8_GEMM_WEIGHT_BY_MODEL_MODE: dict[tuple[str, str], bool] = {}
HEAL_FP16_GEMM_COMPENSATION_BY_MODEL_MODE: dict[tuple[str, str], bool] = {}
PAUSED_MODEL_MODES: dict[tuple[str, str], str] = {}


def skip_reason_for_model_mode(model_id: str, spec: ModeSpec) -> str | None:
    return PAUSED_MODEL_MODES.get((model_id, spec.name))


METRIC_FIELDS = [
    "model",
    "model_arg",
    "mode",
    "version",
    "dtype",
    "attention_backend",
    "tf32",
    "significand_bits",
    "heal_mode",
    "setting",
    "status",
    "wall_time_s",
    "worker_elapsed_s",
    "docs",
    "batches",
    "generated_tokens",
    "generation_batch_mode",
    "generation_batch_size",
    "requested_generation_batch_size",
    "max_length_count",
    "parse_fail_count",
    "answer_f_count",
    "answer_flip_count",
    "answer_flip_rate",
    "valid_reference_rows",
    "valid_reference_flip_count",
    "valid_reference_flip_rate",
    "reference_f_count",
    "answer_counts_json",
    "reference_answer_counts_json",
    "generated_path",
    "answers_path",
    "summary_path",
    "output_dir",
    "patch_sha256_before",
    "patch_sha256_after",
    "env_settings_json",
    "skip_reason",
]

FINAL_ANSWER_RE = re.compile(
    r"final\s+answer\s*[:\uFF1A\-]?\s*(?:is\s*)?[\(\[]?\s*([ABCD])\b",
    re.IGNORECASE,
)
ANSWER_LINE_RE = re.compile(
    r"^(?:answer\s*[:\uFF1A\-]?\s*)?[\(\[]?\s*([ABCD])\s*[\)\].,;!\s]*$",
    re.IGNORECASE,
)
VALID_ANSWERS = {"A", "B", "C", "D"}

MCR_FINAL_LINE_RE = re.compile(r"^\s*Final\s+(?:Answer|Label)\s*[:\uFF1A]\s*(.+?)\s*$", re.I | re.M)
MCR_FINAL_LABEL_RE = re.compile(r"^\s*Final\s+Label\s*[:\uFF1A]\s*(\d{1,2})\b", re.I | re.M)
MCR_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
MCR_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.I)
MCR_LEGAL_LIST_RE = re.compile(r"\[(.*?)\]")
MCR_CHOICE_0_4_RE = re.compile(r"\b([0-4])\b")
MCR_CHOICE_ABC_RE = re.compile(r"\b([ABC])\b", re.I)
MCR_FINCLAIM_RE = re.compile(r"\b([1-6])\b")
MCR_SENTIMENT_RE = re.compile(r"\b([1-3])\b")
MCR_FINCLAIM_NAME_TO_LABEL = {
    "reported factual data": "1",
    "rfd": "1",
    "attributed expert opinion": "2",
    "attributed expert opinion/analysis": "2",
    "aeoa": "2",
    "company forward-looking statement": "3",
    "fls": "3",
    "speculative statement": "4",
    "speculative statement/rumor": "4",
    "rumor": "4",
    "ssr": "4",
    "unsupported assertion": "5",
    "unsupported assertion/hype": "5",
    "hype": "5",
    "uah": "5",
    "general market observation": "6",
    "general market observation/trend description": "6",
    "trend description": "6",
    "gmo": "6",
}
MCR_SENTIMENT_NAME_TO_LABEL = {
    "positive": "1",
    "negative": "2",
    "neutral": "3",
}


def normalize_mcr_label_list(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        try:
            labels = sorted({int(label) for label in value}, reverse=True)
        except (TypeError, ValueError):
            return None
        return json.dumps(labels, separators=(",", ":"))
    text = str(value).strip()
    if not text:
        return None
    match = MCR_LEGAL_LIST_RE.search(text)
    if match:
        text = match.group(1)
    if not text.strip():
        return "[]"
    labels = sorted({int(label) for label in re.findall(r"\d+", text)}, reverse=True)
    labels = [label for label in labels if 0 <= label <= 7]
    return json.dumps(labels, separators=(",", ":"))


def normalize_mcr_answer(value: Any, *, target_kind: str | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if target_kind == "yes_no":
        match = MCR_YES_NO_RE.search(text)
        return match.group(1).lower() if match else re.sub(r"\s+", " ", text.lower()).strip(" .")
    if target_kind == "multi_label_list":
        return normalize_mcr_label_list(value)
    if target_kind == "multiple_choice_0_4":
        match = MCR_CHOICE_0_4_RE.search(text)
        return match.group(1) if match else None
    if target_kind == "multiple_choice_abc":
        match = MCR_CHOICE_ABC_RE.search(text)
        return match.group(1).upper() if match else None
    if target_kind == "finance_claim_category":
        match = MCR_FINCLAIM_RE.search(text)
        if match:
            return match.group(1)
        lowered = text.lower()
        for name, label in MCR_FINCLAIM_NAME_TO_LABEL.items():
            if name in lowered:
                return label
        return None
    if target_kind == "finance_sentiment":
        match = MCR_SENTIMENT_RE.search(text)
        if match:
            return match.group(1)
        lowered = text.lower()
        for name, label in MCR_SENTIMENT_NAME_TO_LABEL.items():
            if name in lowered:
                return label
        return None
    if target_kind in {"multiple_choice_abcd", "multiple_choice"}:
        return text[:1].upper() if text[:1].upper() in VALID_ANSWERS else text.upper()
    if target_kind == "label_number":
        match = MCR_NUMBER_RE.search(text)
        return str(int(match.group(0))) if match else text.lower()
    if target_kind == "numeric_or_text":
        match = MCR_NUMBER_RE.search(text.replace(",", ""))
        if match:
            number = float(match.group(0))
            return str(int(number)) if number.is_integer() else f"{number:.6g}"
    return re.sub(r"\s+", " ", text.lower()).strip(" .")


def parse_mcr_answer(text: str, *, target_kind: str | None = None) -> str | None:
    if target_kind in {"multiple_choice_abcd", "multiple_choice", None}:
        matches = FINAL_ANSWER_RE.findall(text)
        if matches:
            return matches[-1].upper()
        if target_kind in {"multiple_choice_abcd", "multiple_choice"}:
            return None
    if target_kind == "label_number":
        matches = MCR_FINAL_LABEL_RE.findall(text)
        if matches:
            return str(int(matches[-1]))
    matches = MCR_FINAL_LINE_RE.findall(text)
    if matches:
        return normalize_mcr_answer(matches[-1], target_kind=target_kind)
    if target_kind in {
        "label_number",
        "numeric_or_text",
        "yes_no",
        "multi_label_list",
        "multiple_choice_0_4",
        "multiple_choice_abc",
        "finance_claim_category",
        "finance_sentiment",
    }:
        return normalize_mcr_answer(text, target_kind=target_kind)
    return None


def target_kind_for_reference_row(row: dict[str, Any]) -> str | None:
    metadata = row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
    source = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    target_kind = row.get("target_kind") or metadata.get("target_kind") or source.get("target_kind")
    if target_kind:
        return str(target_kind)
    answer_type = str(metadata.get("answer_type") or source.get("answer_type") or "").strip()
    if answer_type in {
        "yes_no",
        "multi_label_list",
        "multiple_choice_0_4",
        "multiple_choice_abc",
        "multiple_choice_abcd",
        "finance_claim_category",
        "finance_sentiment",
    }:
        return answer_type
    if answer_type == "multiple_choice":
        return "multiple_choice_abcd"
    source_dataset = str(metadata.get("source_dataset") or source.get("source_dataset") or "")
    if source_dataset == "finqa":
        return "numeric_or_text"
    return None


def is_valid_reference_answer(answer: Any, *, answer_parser: str) -> bool:
    if answer_parser == "mcr-bench":
        return str(answer) != "F"
    return str(answer) in VALID_ANSWERS


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def model_label(model: str) -> str:
    return resolve_model(model).replace("/", "-")


def path_from_cli(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


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
            writer.writerow({field: row.get(field) for field in fieldnames})


def patch_hashes() -> dict[str, str]:
    import hashlib

    names = ["VLLM_ORIGINAL", "VLLM_LAYERCAST", "VLLM_SIGNIFICAND_BITS", "VLLM_HEAL"]
    hashes: dict[str, str] = {}
    for name in names:
        path = ROOT / "patches" / f"{name}.patch"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


def active_patch_status() -> str:
    return subprocess.check_output([str(PATCH_SCRIPT), "status"], cwd=ROOT, text=True)


def active_patch_name() -> str | None:
    status = active_patch_status()
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


def ensure_vllm_clean() -> None:
    status = subprocess.check_output(
        ["git", "-C", str(VLLM_DIR), "status", "--porcelain"],
        text=True,
    ).strip()
    if status:
        raise RuntimeError(f"vLLM worktree is not clean:\n{status}")


def run_patch_command(action: str, version: str | None = None) -> None:
    cmd = [str(PATCH_SCRIPT), action]
    if version is not None:
        cmd.append(version)
    subprocess.run(cmd, cwd=ROOT, check=True)


def prepare_version(version: str) -> None:
    ensure_vllm_clean()
    ensure_no_active_patch()
    log(f"prepare-version {version}")
    run_patch_command("prepare-version", version)


def revert_version(version: str) -> None:
    log(f"revert-version {version}")
    run_patch_command("revert-version", version)
    ensure_vllm_clean()


def revert_active_patch_best_effort() -> None:
    try:
        if active_patch_name() is not None:
            log("reverting active vLLM patch")
            run_patch_command("revert-version")
    except Exception as exc:  # pragma: no cover - recovery path
        log(f"WARNING: failed to revert active patch automatically: {exc}")


def configure_precision_env(env: dict[str, str], mode: dict[str, Any]) -> dict[str, str]:
    out = dict(env)
    for key in (
        "VLLM_VERSION_NAME",
        "VLLM_EXPERIMENT_OUTPUT_DIR",
        "VLLM_LAYERCAST_FORCE_FP32",
        "VLLM_DISABLE_COMPILE_CACHE",
        "VLLM_DISABLE_CUDA_GRAPH",
        "VLLM_SIGNIFICAND_BITS",
        "VLLM_SIGNIFICAND_BOUNDARY_SET",
        "VLLM_SIGNIFICAND_SKIP_CHANNEL_FILE",
        "VLLM_SIGNIFICAND_SKIP_CHANNEL_PERCENT",
        "VLLM_SIGNIFICAND_SKIP_CHANNEL_METRIC",
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
        "HEAL_FP32_ATTENTION_OUTPUT",
        "HEAP_V1",
        "HEAP_V1_FP32_OUTPUT_GEMM",
        "HEAP_V1_FP16_WEIGHT_SCALE",
        "HEAP_V1_INT16_Q_INPUT",
        "HEAP_V1_INT16_K_CACHE",
        "HEAP_V1_INT16_V_CACHE",
        "HEAP_V1_Q_SPLIT_FP16_DOT",
        "HEAP_V1_INT16_V_CACHE_SPLIT_FP16_DOT",
        "HEAP_V1_INT8_GEMM_SPLIT",
        "HEAL_PLUS_SKIP_QKV_QUANT_LAYERS",
        "HEAL_FULL_SKIP_INT8_COMP_LAYERS",
        "HEAL_FULL_SKIP_FP16_COMP_LAYERS",
        "TENSOR_TRACE_TENSOR_FILE",
        "VLLM_TENSOR_TRACE_TENSOR_FILE",
        "TENSOR_TRACE_STAGE_REGEX",
        "VLLM_TENSOR_TRACE_STAGE_REGEX",
    ):
        out.pop(key, None)
    out["PYTHONPATH"] = (
        str(VLLM_DIR)
        if not out.get("PYTHONPATH")
        else str(VLLM_DIR) + os.pathsep + out["PYTHONPATH"]
    )
    out["VLLM_VERSION_NAME"] = str(mode["version"])
    out["VLLM_EXPERIMENT_OUTPUT_DIR"] = str(mode["mode_dir"])
    out["VLLM_USE_V1"] = "1"
    out["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if mode.get("disable_cuda_graph"):
        out["VLLM_DISABLE_CUDA_GRAPH"] = "1"
    out["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1" if mode.get("tf32") else "0"
    out["VLLM_FLOAT32_MATMUL_PRECISION"] = "high" if mode.get("tf32") else "highest"
    if mode["version"] in {"VLLM_LAYERCAST", "VLLM_SIGNIFICAND_BITS"}:
        out["VLLM_LAYERCAST_FORCE_FP32"] = "1"
        out["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    if mode["version"] == "VLLM_SIGNIFICAND_BITS":
        out["VLLM_SIGNIFICAND_BITS"] = str(mode["significand_bits"])
        out["VLLM_SIGNIFICAND_BOUNDARY_SET"] = "all"
    if mode["version"] == "VLLM_HEAL":
        heal_mode = str(mode.get("heal_mode") or "full").strip().lower()
        if heal_mode not in {"base", "plus", "full"}:
            raise ValueError("HEAL mode must be one of: base, plus, full.")
        out["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        out["HEAL_MODE"] = heal_mode
        if mode.get("heal_dynamic_int8_gemm_weight") is not None:
            dynamic_int8 = bool(mode["heal_dynamic_int8_gemm_weight"])
            out["HEAL_DYNAMIC_INT8_GEMM_WEIGHT"] = "1" if dynamic_int8 else "0"
            if dynamic_int8:
                out["HEAL_INT8_GEMM_COMPENSATION"] = "1"
        if mode.get("heal_fp16_gemm_compensation") is not None:
            out["HEAL_FP16_GEMM_COMPENSATION"] = (
                "1" if bool(mode["heal_fp16_gemm_compensation"]) else "0"
            )
        if mode.get("heal_skip_qkv_quant_layers"):
            out["HEAL_SKIP_QKV_QUANT_LAYERS"] = str(mode["heal_skip_qkv_quant_layers"])
        fp16_skip_layers = mode.get("heal_skip_fp16_comp_layers")
        if fp16_skip_layers is None:
            fp16_skip_layers = mode.get("heal_skip_int8_comp_layers")
        if fp16_skip_layers:
            out["HEAL_SKIP_FP16_COMP_LAYERS"] = str(fp16_skip_layers)
            out["HEAL_SKIP_INT8_COMP_LAYERS"] = str(fp16_skip_layers)
    return out


def mode_env_settings(mode: dict[str, Any]) -> dict[str, str]:
    env = configure_precision_env({}, mode)
    keep = [
        "VLLM_VERSION_NAME",
        "VLLM_EXPERIMENT_OUTPUT_DIR",
        "VLLM_USE_V1",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_DISABLE_CUDA_GRAPH",
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
        "HEAL_FP16_GEMM_COMPENSATION_CHUNK_ROWS",
        "HEAL_INT8_GEMM_COMPENSATION",
        "HEAL_DYNAMIC_INT8_GEMM_WEIGHT",
        "HEAL_DYNAMIC_INT8_GEMM_WEIGHT_CHUNK_ROWS",
        "HEAL_FP16_GEMM_COMPENSATION",
        "HEAL_INT8_GEMM_STATS_FILE",
    ]
    return {key: env[key] for key in keep if key in env}


def apply_torch_precision(tf32: bool) -> None:
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    except Exception:
        return


def add_local_vllm_to_path() -> None:
    value = str(VLLM_DIR)
    if value not in sys.path:
        sys.path.insert(0, value)


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
    enable_chunked_prefill: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": mode["dtype"],
        "seed": seed,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": False,
        "enforce_eager": enforce_eager,
        "trust_remote_code": trust_remote_code,
        "disable_log_stats": True,
    }
    if mode.get("attention_backend"):
        kwargs["attention_backend"] = mode["attention_backend"]
    return kwargs


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


def generation_batch_size_for_model(args: argparse.Namespace, model_id: str) -> int:
    requested = int(args.generation_batch_size)
    cap = GENERATION_BATCH_SIZE_CAPS_BY_MODEL.get(model_id)
    if cap is None:
        return requested
    try:
        import torch

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory
            > GENERATION_BATCH_CAP_DISABLE_MIN_GPU_BYTES
        ):
            return requested
    except Exception:
        pass
    return min(requested, cap)


def spec_mode_dict(
    spec: ModeSpec,
    mode_dir: Path,
    *,
    model_id: str,
    disable_cuda_graph: bool,
    heal_skip_qkv_quant_layers: str | None = None,
    heal_skip_int8_comp_layers: str | None = None,
    heal_skip_fp16_comp_layers: str | None = None,
) -> dict[str, Any]:
    payload = asdict(spec)
    payload["mode_dir"] = str(mode_dir)
    payload["disable_cuda_graph"] = bool(disable_cuda_graph)
    if HEAL_DYNAMIC_INT8_GEMM_WEIGHT_BY_MODEL_MODE.get((model_id, spec.name), False):
        payload["heal_dynamic_int8_gemm_weight"] = True
    if spec.heal_mode == "full" or HEAL_FP16_GEMM_COMPENSATION_BY_MODEL_MODE.get(
        (model_id, spec.name), False
    ):
        payload["heal_fp16_gemm_compensation"] = True
    if spec.name in {"heal_plus_generate", "heal_plus_plus_generate"} and heal_skip_qkv_quant_layers:
        payload["heal_skip_qkv_quant_layers"] = heal_skip_qkv_quant_layers
    if spec.name in {"heal_full_generate", "heal_full_plus_generate"}:
        fp16_skip_layers = heal_skip_fp16_comp_layers or heal_skip_int8_comp_layers
        if fp16_skip_layers:
            payload["heal_skip_fp16_comp_layers"] = fp16_skip_layers
    return payload


def heal_skip_layers_for_spec(
    args: argparse.Namespace,
    *,
    model_arg: str,
    model_id: str,
    model_label_value: str,
    spec: ModeSpec,
) -> tuple[str | None, str | None]:
    qkv_skip_layers = args.heal_skip_qkv_quant_layers
    fp16_skip_layers = args.heal_skip_fp16_comp_layers or args.heal_skip_int8_comp_layers
    if spec.name == "heal_plus_plus_generate" and qkv_skip_layers is None:
        qkv_skip_layers = heal_policy.policy_skip_layers(
            args.heal_policy,
            model_arg=model_arg,
            model_id=model_id,
            model_label=model_label_value,
            variant="HEAL++",
            field="HEAL_SKIP_QKV_QUANT_LAYERS",
        )
    if spec.name == "heal_full_plus_generate" and fp16_skip_layers is None:
        fp16_skip_layers = heal_policy.policy_skip_layers(
            args.heal_policy,
            model_arg=model_arg,
            model_id=model_id,
            model_label=model_label_value,
            variant="HEAL-FULL+",
            field="HEAL_SKIP_FP16_COMP_LAYERS",
        )
    return qkv_skip_layers, fp16_skip_layers


def reference_artifact_for_model(args: argparse.Namespace, model_label_value: str) -> Path:
    base = path_from_cli(args.reference_generation_dir)
    if (base / "generated_tokens.jsonl").exists():
        return base
    return base / model_label_value


def load_reference_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    generated_path = path / "generated_tokens.jsonl"
    if not generated_path.exists():
        raise FileNotFoundError(f"missing FP32 reference generated_tokens.jsonl: {generated_path}")
    rows = read_jsonl(generated_path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"reference artifact has no rows: {generated_path}")
    for index, row in enumerate(rows):
        if "prompt_token_ids" not in row or "generated_token_ids" not in row:
            raise RuntimeError(f"reference row {index} is missing prompt/generated token ids")
    return rows


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def parse_answer_text(
    text: str,
    *,
    finish_reason: Any,
    token_count: int,
    max_new_tokens: int,
    answer_parser: str = "medqa",
    target_kind: str | None = None,
) -> dict[str, Any]:
    finish = str(finish_reason or "").strip().lower()
    hit_max_length = finish in {"length", "max_length", "max_tokens"} or token_count >= max_new_tokens
    if hit_max_length:
        return {
            "answer": "F",
            "parse_status": "max_length",
            "parse_source": "max_length",
            "hit_max_length": True,
        }

    if answer_parser == "mcr-bench":
        answer = parse_mcr_answer(text, target_kind=target_kind)
        if answer is not None:
            return {
                "answer": answer,
                "parse_status": "ok",
                "parse_source": "mcr_bench",
                "hit_max_length": False,
            }
        return {
            "answer": "F",
            "parse_status": "parse_failed",
            "parse_source": "none",
            "hit_max_length": False,
        }

    final_matches = list(FINAL_ANSWER_RE.finditer(text))
    if final_matches:
        answer = final_matches[-1].group(1).upper()
        return {
            "answer": answer,
            "parse_status": "ok",
            "parse_source": "final_answer",
            "hit_max_length": False,
        }

    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        match = ANSWER_LINE_RE.match(line)
        if match:
            answer = match.group(1).upper()
            return {
                "answer": answer,
                "parse_status": "ok",
                "parse_source": "last_answer_line",
                "hit_max_length": False,
            }

    return {
        "answer": "F",
        "parse_status": "parse_failed",
        "parse_source": "none",
        "hit_max_length": False,
    }


def reference_answer_for_row(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_new_tokens: int,
    answer_parser: str = "medqa",
    target_kind: str | None = None,
) -> dict[str, Any]:
    token_ids = [int(token_id) for token_id in row["generated_token_ids"]]
    text = decode_token_ids(tokenizer, token_ids)
    parsed = parse_answer_text(
        text,
        finish_reason=row.get("finish_reason"),
        token_count=len(token_ids),
        max_new_tokens=max_new_tokens,
        answer_parser=answer_parser,
        target_kind=target_kind,
    )
    parsed["text_tail"] = text[-1200:]
    return parsed


def summarize_answer_rows(
    rows: list[dict[str, Any]],
    *,
    answer_parser: str = "medqa",
) -> dict[str, Any]:
    docs = len(rows)
    generated_tokens = sum(int(row["generated_token_count"]) for row in rows)
    answer_counts = Counter(str(row["answer"]) for row in rows)
    reference_answer_counts = Counter(str(row["reference_answer"]) for row in rows)
    answer_flip_count = sum(1 for row in rows if row["answer"] != row["reference_answer"])
    valid_reference_rows = sum(
        1
        for row in rows
        if is_valid_reference_answer(row["reference_answer"], answer_parser=answer_parser)
    )
    valid_reference_flip_count = sum(
        1
        for row in rows
        if is_valid_reference_answer(row["reference_answer"], answer_parser=answer_parser)
        and row["answer"] != row["reference_answer"]
    )
    max_length_count = sum(1 for row in rows if row.get("hit_max_length"))
    parse_fail_count = sum(1 for row in rows if row.get("parse_status") == "parse_failed")
    answer_f_count = answer_counts.get("F", 0)
    reference_f_count = reference_answer_counts.get("F", 0)
    return {
        "docs": docs,
        "generated_tokens": generated_tokens,
        "max_length_count": max_length_count,
        "parse_fail_count": parse_fail_count,
        "answer_f_count": answer_f_count,
        "answer_flip_count": answer_flip_count,
        "answer_flip_rate": answer_flip_count / docs if docs else None,
        "valid_reference_rows": valid_reference_rows,
        "valid_reference_flip_count": valid_reference_flip_count,
        "valid_reference_flip_rate": (
            valid_reference_flip_count / valid_reference_rows if valid_reference_rows else None
        ),
        "reference_f_count": reference_f_count,
        "answer_counts": dict(sorted(answer_counts.items())),
        "reference_answer_counts": dict(sorted(reference_answer_counts.items())),
    }


def mode_outputs_complete(mode_dir: Path, expected_docs: int) -> bool:
    result_path = mode_dir / "worker_result.json"
    answers_path = mode_dir / "answers.jsonl"
    generated_path = mode_dir / "generated_tokens.jsonl"
    if not (result_path.exists() and answers_path.exists() and generated_path.exists()):
        return False
    try:
        return len(read_jsonl(answers_path)) == expected_docs and len(read_jsonl(generated_path)) == expected_docs
    except Exception:
        return False


def worker_generate(config: dict[str, Any]) -> None:
    started = time.perf_counter()
    mode = config["mode"]
    os.environ.update(configure_precision_env(dict(os.environ), mode))
    apply_torch_precision(bool(mode.get("tf32")))
    add_local_vllm_to_path()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    source_rows = read_jsonl(Path(config["reference_generated_path"]))
    if config.get("limit") is not None:
        source_rows = source_rows[: int(config["limit"])]
    if not source_rows:
        raise RuntimeError("generation worker received no source rows")

    max_new_tokens = int(config["max_new_tokens"])
    answer_parser = str(config.get("answer_parser", "medqa"))
    generation_batch_mode = str(config.get("generation_batch_mode", "chunked"))
    if generation_batch_mode not in {"chunked", "full", "full-reverse"}:
        raise ValueError(f"unknown generation_batch_mode: {generation_batch_mode}")
    generation_batch_size = int(config["generation_batch_size"])
    prompt_rows: list[dict[str, Any]] = []
    max_prompt_tokens = 0
    for fallback_doc_id, row in enumerate(source_rows):
        prompt_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
        max_prompt_tokens = max(max_prompt_tokens, len(prompt_ids))
        target_kind = target_kind_for_reference_row(row)
        reference = reference_answer_for_row(
            tokenizer,
            row,
            max_new_tokens=max_new_tokens,
            answer_parser=answer_parser,
            target_kind=target_kind,
        )
        source_metadata = row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
        prompt_rows.append(
            {
                "doc_id": int(row.get("doc_id", fallback_doc_id)),
                "subset_pos": int(row.get("subset_pos", fallback_doc_id)),
                "original_index": int(row.get("original_index", fallback_doc_id)),
                "id": row.get("id"),
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "reference_generated_token_ids": [
                    int(token_id) for token_id in row["generated_token_ids"]
                ],
                "reference_generated_token_count": int(
                    row.get("generated_token_count", len(row["generated_token_ids"]))
                ),
                "reference_finish_reason": row.get("finish_reason"),
                "reference_stop_reason": row.get("stop_reason"),
                "reference_answer": reference["answer"],
                "reference_parse_status": reference["parse_status"],
                "reference_parse_source": reference["parse_source"],
                "reference_hit_max_length": bool(reference["hit_max_length"]),
                "reference_text_tail": reference["text_tail"],
                "dataset_kind": row.get("dataset_kind") or ("mcr-bench" if answer_parser == "mcr-bench" else "medqa"),
                "source_metadata": source_metadata,
                "task": row.get("task") or source_metadata.get("task"),
                "target": row.get("target") or source_metadata.get("target"),
                "target_kind": target_kind,
            }
        )

    ordered_prompt_rows = list(reversed(prompt_rows)) if generation_batch_mode == "full-reverse" else list(prompt_rows)
    if generation_batch_mode in {"full", "full-reverse"}:
        generation_batches = [ordered_prompt_rows]
        generation_batch_size = len(ordered_prompt_rows)
    else:
        generation_batches = [
            ordered_prompt_rows[start : start + generation_batch_size]
            for start in range(0, len(ordered_prompt_rows), generation_batch_size)
        ]
    max_batch_prompt_tokens = max(
        sum(int(item["prompt_token_count"]) for item in batch)
        for batch in generation_batches
    )
    max_model_len = resolve_max_model_len(
        str(config["max_model_len"]),
        max_prompt_tokens + max_new_tokens,
    )
    enable_chunked_prefill = generation_batch_mode in {"full", "full-reverse"}
    max_num_batched_tokens_inputs = [
        int(config["generation_token_budget"]),
        max_prompt_tokens + 1,
        max_model_len,
    ]
    if not enable_chunked_prefill:
        max_num_batched_tokens_inputs.append(max_batch_prompt_tokens)
    max_num_batched_tokens = max(max_num_batched_tokens_inputs)
    log(
        "worker generate loading "
        f"mode={mode['name']} model={model_id} dtype={mode['dtype']} "
        f"batch_mode={generation_batch_mode} batch_size={generation_batch_size} "
        f"enforce_eager={bool(config['generation_enforce_eager'])} "
        f"chunked_prefill={enable_chunked_prefill} "
        f"max_model_len={max_model_len} max_num_batched_tokens={max_num_batched_tokens}"
    )
    load_start = time.perf_counter()
    llm = LLM(
        **llm_kwargs(
            model_id=model_id,
            mode=mode,
            seed=int(config["seed"]),
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=generation_batch_size,
            gpu_memory_utilization=float(config["gpu_memory_utilization"]),
            enforce_eager=bool(config["generation_enforce_eager"]),
            trust_remote_code=bool(config["trust_remote_code"]),
            enable_chunked_prefill=enable_chunked_prefill,
        )
    )
    load_elapsed = time.perf_counter() - load_start
    log(f"worker generate mode={mode['name']} model loaded in {load_elapsed:.1f}s")

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    use_token_prompts = bool(config["generation_use_token_prompts"])
    mode_dir = Path(config["mode_dir"])
    generated_path = mode_dir / "generated_tokens.jsonl"
    answers_path = mode_dir / "answers.jsonl"
    summary_path = mode_dir / "answer_summary.json"
    result_path = Path(config["result_path"])
    batch_stats: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []

    generation_order_index = 0
    try:
        for batch_id, batch in enumerate(generation_batches):
            if use_token_prompts:
                prompts = [TokensPrompt(prompt_token_ids=item["prompt_token_ids"]) for item in batch]
            else:
                prompts = [decode_token_ids(tokenizer, item["prompt_token_ids"]) for item in batch]
            batch_started = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            elapsed = time.perf_counter() - batch_started
            batch_generated_tokens = 0
            for item, output in zip(batch, outputs, strict=True):
                completion = output.outputs[0]
                token_ids = [int(token_id) for token_id in completion.token_ids]
                text = decode_token_ids(tokenizer, token_ids)
                parsed = parse_answer_text(
                    text,
                    finish_reason=completion.finish_reason,
                    token_count=len(token_ids),
                    max_new_tokens=max_new_tokens,
                    answer_parser=answer_parser,
                    target_kind=item.get("target_kind"),
                )
                batch_generated_tokens += len(token_ids)
                generated_row = {
                    "generation_batch_mode": generation_batch_mode,
                    "generation_batch_id": batch_id,
                    "generation_order_index": generation_order_index,
                    "doc_id": int(item["doc_id"]),
                    "subset_pos": int(item["subset_pos"]),
                    "original_index": int(item["original_index"]),
                    "id": item["id"],
                    "request_id": str(output.request_id),
                    "prompt_token_ids": item["prompt_token_ids"],
                    "prompt_token_count": int(item["prompt_token_count"]),
                    "generated_token_ids": token_ids,
                    "generated_token_count": len(token_ids),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "answer": parsed["answer"],
                    "parse_status": parsed["parse_status"],
                    "parse_source": parsed["parse_source"],
                    "hit_max_length": bool(parsed["hit_max_length"]),
                    "dataset_kind": item.get("dataset_kind"),
                    "source_metadata": item.get("source_metadata"),
                    "task": item.get("task"),
                    "target": item.get("target"),
                    "target_kind": item.get("target_kind"),
                    "generated_text_tail": text[-1200:],
                }
                answer_row = {
                    "mode": mode["name"],
                    "generation_batch_mode": generation_batch_mode,
                    "generation_batch_id": batch_id,
                    "generation_order_index": generation_order_index,
                    "doc_id": int(item["doc_id"]),
                    "subset_pos": int(item["subset_pos"]),
                    "original_index": int(item["original_index"]),
                    "id": item["id"],
                    "answer": parsed["answer"],
                    "parse_status": parsed["parse_status"],
                    "parse_source": parsed["parse_source"],
                    "hit_max_length": bool(parsed["hit_max_length"]),
                    "generated_token_count": len(token_ids),
                    "finish_reason": completion.finish_reason,
                    "reference_answer": item["reference_answer"],
                    "reference_parse_status": item["reference_parse_status"],
                    "reference_parse_source": item["reference_parse_source"],
                    "reference_hit_max_length": item["reference_hit_max_length"],
                    "reference_generated_token_count": item["reference_generated_token_count"],
                    "reference_finish_reason": item["reference_finish_reason"],
                    "dataset_kind": item.get("dataset_kind"),
                    "task": item.get("task"),
                    "target": item.get("target"),
                    "target_kind": item.get("target_kind"),
                    "answer_flip": parsed["answer"] != item["reference_answer"],
                    "generated_text_tail": text[-1200:],
                    "reference_text_tail": item["reference_text_tail"],
                }
                generated_rows.append(generated_row)
                answer_rows.append(answer_row)
                generation_order_index += 1
            batch_stats.append(
                {
                    "batch_id": batch_id,
                    "docs": len(batch),
                    "elapsed_s": elapsed,
                    "generated_tokens": batch_generated_tokens,
                }
            )
            log(
                f"worker generate mode={mode['name']} batch={batch_id} "
                f"docs={len(batch)} generated={batch_generated_tokens} elapsed={elapsed:.1f}s"
            )
    finally:
        shutdown_llm_and_distributed(llm)
        del llm
        gc.collect()

    summary = summarize_answer_rows(answer_rows, answer_parser=answer_parser)
    row_sort_key = lambda row: (int(row["subset_pos"]), int(row["doc_id"]), int(row["original_index"]))
    generated_rows = sorted(generated_rows, key=row_sort_key)
    answer_rows = sorted(answer_rows, key=row_sort_key)
    write_jsonl(generated_path, generated_rows)
    write_jsonl(answers_path, answer_rows)
    write_json(summary_path, summary)
    result = {
        "kind": "generate",
        "mode": mode["name"],
        "model": config["model_label"],
        "model_id": model_id,
        "docs": len(answer_rows),
        "batches": len(batch_stats),
        **summary,
        "generation_batch_mode": generation_batch_mode,
        "generation_batch_size": generation_batch_size,
        "requested_generation_batch_size": config.get("requested_generation_batch_size"),
        "elapsed_s": time.perf_counter() - started,
        "model_load_elapsed_s": load_elapsed,
        "generated_path": str(generated_path),
        "answers_path": str(answers_path),
        "summary_path": str(summary_path),
        "output_dir": str(mode_dir),
        "max_prompt_tokens": max_prompt_tokens,
        "max_batch_prompt_tokens": max_batch_prompt_tokens,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "batch_stats": batch_stats,
    }
    write_json(result_path, result)


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
    config_path = config_dir / f"{kind}_{mode['name']}.json"
    write_json(config_path, config)
    env = configure_precision_env(dict(os.environ), mode)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        kind,
        "--config",
        str(config_path),
    ]
    log(f"starting worker kind={kind} mode={mode['name']}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return read_json(result_path)


def selected_specs(args: argparse.Namespace) -> list[ModeSpec]:
    if not args.modes:
        return list(MODE_SPECS)
    unknown = [name for name in args.modes if name not in MODE_SPECS_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown mode(s): {', '.join(unknown)}")
    return [MODE_SPECS_BY_NAME[name] for name in args.modes]


def model_args_from_cli(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(MODEL_ORDER)
    if args.models:
        return args.models
    return [args.model]


def append_metric_defaults(
    *,
    row: dict[str, Any],
    model_arg: str,
    model_label_value: str,
    spec: ModeSpec,
    mode: dict[str, Any],
    patch_hashes_before: dict[str, str],
) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "model": model_label_value,
            "model_arg": model_arg,
            "mode": spec.name,
            "version": spec.version,
            "dtype": spec.dtype,
            "attention_backend": spec.attention_backend,
            "tf32": spec.tf32,
            "significand_bits": spec.significand_bits,
            "heal_mode": spec.heal_mode,
            "setting": spec.setting,
            "patch_sha256_before": patch_hashes_before.get(spec.version, ""),
            "patch_sha256_after": patch_hashes().get(spec.version, ""),
            "env_settings_json": json.dumps(mode_env_settings(mode), sort_keys=True),
            "answer_counts_json": json.dumps(out.get("answer_counts", {}), sort_keys=True),
            "reference_answer_counts_json": json.dumps(
                out.get("reference_answer_counts", {}), sort_keys=True
            ),
        }
    )
    for field in METRIC_FIELDS:
        out.setdefault(field, None)
    return out


def write_metrics(results_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        for key, value in list(csv_row.items()):
            if isinstance(value, (dict, list)):
                csv_row[key] = json.dumps(value, sort_keys=True)
        csv_rows.append(csv_row)
    write_csv(results_root / "metrics.csv", csv_rows, METRIC_FIELDS)
    write_json(results_root / "metrics.json", {"metrics": rows})


def dry_run_model(args: argparse.Namespace, run_id: str, model_arg: str) -> None:
    model_id = resolve_model(model_arg)
    label = model_label(model_arg)
    reference_dir = reference_artifact_for_model(args, label)
    source_rows = load_reference_rows(reference_dir, args.limit)
    run_root = OUTPUT_ROOT / run_id / label
    results_root = RESULTS_ROOT / run_id / label
    requested_specs = selected_specs(args)
    specs = [
        spec for spec in requested_specs if skip_reason_for_model_mode(model_id, spec) is None
    ]
    skipped_modes = [
        {"mode": spec.name, "reason": skip_reason_for_model_mode(model_id, spec)}
        for spec in requested_specs
        if skip_reason_for_model_mode(model_id, spec)
    ]
    if args.generation_batch_mode in {"full", "full-reverse"}:
        effective_generation_batch_size = len(source_rows)
    else:
        effective_generation_batch_size = generation_batch_size_for_model(args, model_id)
    plan = {
        "dry_run": True,
        "run_id": run_id,
        "model_arg": model_arg,
        "model_id": model_id,
        "model_label": label,
        "reference_generation_dir": str(reference_dir),
        "reference_rows": len(source_rows),
        "answer_parser": args.answer_parser,
        "output_root": str(run_root),
        "results_root": str(results_root),
        "requested_modes": [asdict(spec) for spec in requested_specs],
        "modes": [asdict(spec) for spec in specs],
        "skipped_modes": skipped_modes,
        "patch_order": list(PATCH_ORDER),
        "generation_batch_size": args.generation_batch_size,
        "generation_batch_mode": args.generation_batch_mode,
        "effective_generation_batch_size": effective_generation_batch_size,
        "generation_token_budget": args.generation_token_budget,
        "max_new_tokens": args.max_new_tokens,
        "generation_enforce_eager": args.generation_enforce_eager,
        "disable_cuda_graph": args.disable_cuda_graph,
        "generation_gpu_memory_utilization": args.generation_gpu_memory_utilization,
        "generation_use_token_prompts": args.generation_use_token_prompts,
        "patch_sha256": patch_hashes(),
        "active_patch_status": active_patch_status(),
    }
    write_json(results_root / "run_manifest.json", plan)
    log(f"dry-run model={label} docs={len(source_rows)} output={run_root} modes={len(specs)}")


def run_generation_mode(
    *,
    args: argparse.Namespace,
    model_arg: str,
    model_id: str,
    model_label_value: str,
    run_root: Path,
    results_root: Path,
    config_dir: Path,
    reference_dir: Path,
    source_rows: list[dict[str, Any]],
    spec: ModeSpec,
    patch_hashes_before: dict[str, str],
    metric_rows: list[dict[str, Any]],
) -> None:
    mode_dir = run_root / spec.name
    result_path = mode_dir / "worker_result.json"
    qkv_skip_layers, fp16_skip_layers = heal_skip_layers_for_spec(
        args,
        model_arg=model_arg,
        model_id=model_id,
        model_label_value=model_label_value,
        spec=spec,
    )
    mode = spec_mode_dict(
        spec,
        mode_dir,
        model_id=model_id,
        disable_cuda_graph=bool(args.disable_cuda_graph),
        heal_skip_qkv_quant_layers=qkv_skip_layers,
        heal_skip_fp16_comp_layers=fp16_skip_layers,
    )
    started = time.perf_counter()
    resumed_outputs = bool(args.resume and mode_outputs_complete(mode_dir, len(source_rows)))
    if resumed_outputs:
        log(f"resume: generation outputs already complete for mode={spec.name}")
        worker_result = read_json(result_path)
    else:
        if args.generation_batch_mode in {"full", "full-reverse"}:
            generation_batch_size = len(source_rows)
            log(
                f"generation batch mode for {model_label_value}: "
                f"{args.generation_batch_mode} -> {generation_batch_size} docs"
            )
        else:
            generation_batch_size = generation_batch_size_for_model(args, model_id)
            if generation_batch_size != int(args.generation_batch_size):
                log(
                    f"model-specific generation batch cap for {model_label_value}: "
                    f"{args.generation_batch_size} -> {generation_batch_size}"
                )
        config = {
            "model_arg": model_arg,
            "model_id": model_id,
            "model_label": model_label_value,
            "mode": mode,
            "mode_dir": str(mode_dir),
            "reference_generation_dir": str(reference_dir),
            "reference_generated_path": str(reference_dir / "generated_tokens.jsonl"),
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "answer_parser": args.answer_parser,
            "generation_batch_size": generation_batch_size,
            "requested_generation_batch_size": args.generation_batch_size,
            "generation_batch_mode": args.generation_batch_mode,
            "generation_token_budget": args.generation_token_budget,
            "generation_enforce_eager": args.generation_enforce_eager,
            "generation_use_token_prompts": args.generation_use_token_prompts,
            "gpu_memory_utilization": args.generation_gpu_memory_utilization,
            "generation_gpu_memory_utilization": args.generation_gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "trust_remote_code": args.trust_remote_code,
            "seed": args.seed,
            "result_path": str(result_path),
        }
        worker_result = run_worker_process(
            kind="generate",
            config=config,
            mode=mode,
            config_dir=config_dir,
        )
    metric = append_metric_defaults(
        row={
            "status": "ok",
            "wall_time_s": (
                worker_result.get("elapsed_s")
                if resumed_outputs and worker_result.get("elapsed_s") is not None
                else time.perf_counter() - started
            ),
            "worker_elapsed_s": worker_result.get("elapsed_s"),
            "docs": worker_result.get("docs"),
            "batches": worker_result.get("batches"),
            "generated_tokens": worker_result.get("generated_tokens"),
            "generation_batch_mode": worker_result.get("generation_batch_mode"),
            "generation_batch_size": worker_result.get("generation_batch_size"),
            "requested_generation_batch_size": worker_result.get("requested_generation_batch_size"),
            "max_length_count": worker_result.get("max_length_count"),
            "parse_fail_count": worker_result.get("parse_fail_count"),
            "answer_f_count": worker_result.get("answer_f_count"),
            "answer_flip_count": worker_result.get("answer_flip_count"),
            "answer_flip_rate": worker_result.get("answer_flip_rate"),
            "valid_reference_rows": worker_result.get("valid_reference_rows"),
            "valid_reference_flip_count": worker_result.get("valid_reference_flip_count"),
            "valid_reference_flip_rate": worker_result.get("valid_reference_flip_rate"),
            "reference_f_count": worker_result.get("reference_f_count"),
            "answer_counts": worker_result.get("answer_counts", {}),
            "reference_answer_counts": worker_result.get("reference_answer_counts", {}),
            "generated_path": worker_result.get("generated_path"),
            "answers_path": worker_result.get("answers_path"),
            "summary_path": worker_result.get("summary_path"),
            "output_dir": worker_result.get("output_dir"),
        },
        model_arg=model_arg,
        model_label_value=model_label_value,
        spec=spec,
        mode=mode,
        patch_hashes_before=patch_hashes_before,
    )
    metric_rows.append(metric)
    write_metrics(results_root, metric_rows)


def run_model(args: argparse.Namespace, run_id: str, model_arg: str) -> None:
    model_id = resolve_model(model_arg)
    label = model_label(model_arg)
    run_root = OUTPUT_ROOT / run_id / label
    results_root = RESULTS_ROOT / run_id / label
    config_dir = results_root / "worker_configs"
    reference_dir = reference_artifact_for_model(args, label)
    source_rows = load_reference_rows(reference_dir, args.limit)
    requested_specs = selected_specs(args)
    specs = [
        spec for spec in requested_specs if skip_reason_for_model_mode(model_id, spec) is None
    ]
    skipped_specs = [
        spec for spec in requested_specs if skip_reason_for_model_mode(model_id, spec)
    ]

    if run_root.exists() and any(run_root.iterdir()) and not args.resume:
        raise RuntimeError(f"output directory already exists; pass --resume: {run_root}")

    ensure_vllm_clean()
    ensure_no_active_patch()
    patch_hashes_before = patch_hashes()
    manifest = {
        "run_id": run_id,
        "model_arg": model_arg,
        "model_id": model_id,
        "model_label": label,
        "reference_generation_dir": str(reference_dir),
        "reference_generated_path": str(reference_dir / "generated_tokens.jsonl"),
        "reference_rows": len(source_rows),
        "answer_parser": args.answer_parser,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "output_root": str(run_root),
        "results_root": str(results_root),
        "requested_modes": [asdict(spec) for spec in requested_specs],
        "modes": [asdict(spec) for spec in specs],
        "skipped_modes": [
            {"mode": spec.name, "reason": skip_reason_for_model_mode(model_id, spec)}
            for spec in skipped_specs
        ],
        "patch_order": list(PATCH_ORDER),
        "patch_sha256_before": patch_hashes_before,
        "started_at": utc_timestamp(),
        "args": jsonable(vars(args)),
    }
    write_json(results_root / "run_manifest.json", manifest)

    metric_rows: list[dict[str, Any]] = []
    for spec in skipped_specs:
        qkv_skip_layers, fp16_skip_layers = heal_skip_layers_for_spec(
            args,
            model_arg=model_arg,
            model_id=model_id,
            model_label_value=label,
            spec=spec,
        )
        mode = spec_mode_dict(
            spec,
            run_root / spec.name,
            model_id=model_id,
            disable_cuda_graph=bool(args.disable_cuda_graph),
            heal_skip_qkv_quant_layers=qkv_skip_layers,
            heal_skip_fp16_comp_layers=fp16_skip_layers,
        )
        metric_rows.append(
            append_metric_defaults(
                row={
                    "status": "skipped",
                    "wall_time_s": 0.0,
                    "worker_elapsed_s": 0.0,
                    "output_dir": str(run_root / spec.name),
                    "skip_reason": skip_reason_for_model_mode(model_id, spec),
                },
                model_arg=model_arg,
                model_label_value=label,
                spec=spec,
                mode=mode,
                patch_hashes_before=patch_hashes_before,
            )
        )
    if metric_rows:
        write_metrics(results_root, metric_rows)
    try:
        for version in PATCH_ORDER:
            version_specs = [spec for spec in specs if spec.version == version]
            if not version_specs:
                continue
            prepare_version(version)
            try:
                for spec in version_specs:
                    run_generation_mode(
                        args=args,
                        model_arg=model_arg,
                        model_id=model_id,
                        model_label_value=label,
                        run_root=run_root,
                        results_root=results_root,
                        config_dir=config_dir,
                        reference_dir=reference_dir,
                        source_rows=source_rows,
                        spec=spec,
                        patch_hashes_before=patch_hashes_before,
                        metric_rows=metric_rows,
                    )
            finally:
                revert_version(version)
    except Exception:
        revert_active_patch_best_effort()
        raise

    patch_hashes_after = patch_hashes()
    if patch_hashes_after != patch_hashes_before:
        raise RuntimeError(
            "patch files changed during run; before="
            f"{patch_hashes_before} after={patch_hashes_after}"
        )
    manifest.update(
        {
            "ended_at": utc_timestamp(),
            "patch_sha256_after": patch_hashes_after,
            "metrics_csv": str(results_root / "metrics.csv"),
            "metrics_json": str(results_root / "metrics.json"),
        }
    )
    write_json(results_root / "run_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedQA direct generation answer flip-rate runner."
    )
    parser.add_argument("--worker", choices=["generate"], help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--reference-generation-dir",
        type=Path,
        default=DEFAULT_REFERENCE_GENERATION_DIR,
        help=(
            "FP32 LayerCast generation artifact parent or one model directory. "
            "Defaults to the current final-hidden-mse all-model artifact."
        ),
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--answer-parser",
        choices=["medqa", "mcr-bench"],
        default="medqa",
        help="Answer parser for generated completions. Defaults to MedQA A/B/C/D.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--generation-batch-size", type=int, default=200)
    parser.add_argument(
        "--generation-batch-mode",
        choices=["chunked", "full", "full-reverse"],
        default="chunked",
        help="Prompt submission mode for generation rounds.",
    )
    parser.add_argument("--generation-token-budget", type=int, default=4096)
    parser.add_argument(
        "--generation-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force eager generation. Default is non-eager direct generation.",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set VLLM_DISABLE_CUDA_GRAPH=1. Default leaves CUDA graph enabled.",
    )
    parser.add_argument(
        "--generation-use-token-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TokensPrompt(prompt_token_ids=...) from the FP32 artifact.",
    )
    parser.add_argument(
        "--heal-skip-qkv-quant-layers",
        help="Comma-separated layers where HEAL+ falls back to HEAL q/k/v handling.",
    )
    parser.add_argument(
        "--heal-skip-fp16-comp-layers",
        help="Comma-separated layers where HEAL-FULL falls back to HEAL+ MLP handling.",
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
    parser.add_argument(
        "--gpu-memory-utilization",
        "--generation-gpu-memory-utilization",
        dest="generation_gpu_memory_utilization",
        type=float,
        default=0.93,
        help="GPU memory utilization for every generation worker.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.heal_policy = heal_policy.load_heal_policy(
        args.heal_policy_json,
        root=ROOT,
    )
    if args.worker:
        if args.config is None:
            raise SystemExit("--config is required for worker mode")
        config = read_json(args.config)
        worker_generate(config)
        return 0

    run_id = args.run_id or utc_timestamp()
    selected_models = model_args_from_cli(args)
    if args.dry_run:
        for model_arg in selected_models:
            dry_run_model(args, run_id, model_arg)
        return 0

    for index, model_arg in enumerate(selected_models):
        label = model_label(model_arg)
        log(f"starting model {index + 1}/{len(selected_models)}: {label}")
        run_model(args, run_id, model_arg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
