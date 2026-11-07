# SPDX-License-Identifier: Apache-2.0
"""BF16 MedQA subset Q/K/V dynamic-range distribution runner."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
VLLM_DIR = ROOT / "vllm"
PATCH_SCRIPT = ROOT / "scripts" / "vllm_incremental_build.sh"
OUTPUT_ROOT = ROOT / "output" / "qkv-distribution"
RESULTS_ROOT = TEST_DIR / "results"
DEFAULT_SUBSET_FILE = (
    ROOT
    / "tests"
    / "error-contribution"
    / "subsets"
    / "medqa_calibration32_seed20260528.jsonl"
)
PATCH_NAME = "VLLM_QKV_DISTRIBUTION"
QKV_TENSOR_NAME = "qkv_absmax_over_absstd.pt"

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
MODEL_LABEL_ORDER = [
    "Qwen-Qwen3-8B",
    "Qwen-Qwen3-14B",
    "Qwen-Qwen3-32B",
    "meta-llama-Llama-3.1-8B-Instruct",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B",
]
MODEL_SHORT = {
    "Qwen-Qwen3-8B": "Qwen3 8B",
    "Qwen-Qwen3-14B": "Qwen3 14B",
    "Qwen-Qwen3-32B": "Qwen3 32B",
    "meta-llama-Llama-3.1-8B-Instruct": "Llama 8B",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B": "DS 14B",
}
GENERATION_BATCH_SIZE_CAPS_BY_MODEL = {
    "Qwen/Qwen3-32B": 16,
}
QKV_LABELS = ("q", "k", "v")
COLORS = {"q": "#f94144", "k": "#43aa8b", "v": "#277da1"}
DISPLAY_LABELS = {"q": "Q", "k": "K", "v": "V"}
PROBABILITY_THRESHOLDS = (16.0, 11.31)
SUMMARY_FIELDNAMES = [
    "model",
    "model_short",
    "qkv",
    "count",
    "mean",
    "std",
    "min",
    "p01",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "max",
    "p_le_16_0",
    "p_le_11_31",
]
HISTOGRAM_FIELDNAMES = [
    "model",
    "model_short",
    "qkv",
    "bin_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "count",
    "density",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def model_label(model: str) -> str:
    return resolve_model(model).replace("/", "-")


def model_args_from_cli(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(MODEL_ORDER)
    if args.models:
        return args.models
    return [args.model]


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_patch_name() -> str | None:
    status = subprocess.check_output([str(PATCH_SCRIPT), "status"], cwd=ROOT, text=True)
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


def add_local_vllm_to_path() -> None:
    value = str(VLLM_DIR)
    if value not in sys.path:
        sys.path.insert(0, value)


def path_from_cli(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_medqa_subset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _choice_value(row: dict[str, Any], idx: int, letter: str) -> Any:
    for key in (letter, letter.lower(), f"opa{letter.lower()}", f"ending{idx}"):
        value = row.get(key)
        if value is not None:
            return value
    return None


def medqa_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    choices = []
    for idx, letter in enumerate(("A", "B", "C", "D")):
        value = _choice_value(row, idx, letter)
        if value is not None:
            choices.append(f"{letter}. {value}")
    question = row.get("question") or row.get("sent1") or row.get("prompt") or ""
    user = "Question: " + str(question).strip()
    if choices:
        user += "\n\nOptions:\n" + "\n".join(choices)
    return [
        {
            "role": "system",
            "content": (
                "You are solving a US medical licensing multiple-choice question. "
                "Think step by step, then end with exactly one final line formatted "
                "as `Final Answer: X` where X is A, B, C, or D."
            ),
        },
        {"role": "user", "content": user},
    ]


def chat_prompt_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        return [int(token_id) for token_id in ids]
    prompt = "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
    prompt += "\n\nASSISTANT:"
    return [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]


def configure_worker_env(qkv_file: Path) -> None:
    os.environ["PYTHONPATH"] = (
        str(VLLM_DIR)
        if not os.environ.get("PYTHONPATH")
        else str(VLLM_DIR) + os.pathsep + os.environ["PYTHONPATH"]
    )
    os.environ["VLLM_QKV_DISTRIBUTION_FILE"] = str(qkv_file)
    os.environ["VLLM_DISABLE_CUDA_GRAPH"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["VLLM_FLOAT32_MATMUL_PRECISION"] = "highest"
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def shutdown_llm(llm: Any) -> None:
    for path in (
        ("llm_engine", "engine_core", "shutdown"),
        ("llm_engine", "shutdown"),
    ):
        obj = llm
        try:
            for attr in path:
                obj = getattr(obj, attr)
            obj()
            return
        except Exception:
            pass


def run_worker(args: argparse.Namespace) -> int:
    add_local_vllm_to_path()
    model_id = resolve_model(args.model)
    label = model_label(args.model)
    run_dir = OUTPUT_ROOT / args.run_id / label
    qkv_file = run_dir / QKV_TENSOR_NAME
    generated_path = run_dir / "generated_tokens.jsonl"
    summary_path = run_dir / "worker_summary.json"
    configure_worker_env(qkv_file)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.model_executor.models.qkv_distribution_trace import (
        clear_qkv_distribution_stats,
        flush_qkv_distribution_stats,
    )

    clear_qkv_distribution_stats()
    rows = load_medqa_subset(path_from_cli(args.subset_file), args.doc_limit)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_rows = []
    for index, row in enumerate(rows):
        doc_id = int(row.get("doc_id", index))
        prompt_ids = chat_prompt_ids(tokenizer, medqa_messages(row))
        prompt_rows.append({"doc_id": doc_id, "prompt_token_ids": prompt_ids})

    batch_size = args.generation_batch_size
    cap = GENERATION_BATCH_SIZE_CAPS_BY_MODEL.get(model_id)
    if cap is not None:
        batch_size = min(batch_size, cap)

    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": "bfloat16",
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.cpu_offload_gb > 0:
        llm_kwargs["cpu_offload_gb"] = args.cpu_offload_gb

    log(f"worker loading {label} docs={len(prompt_rows)} batch_size={batch_size}")
    started = time.perf_counter()
    llm = LLM(**llm_kwargs)
    generated_rows: list[dict[str, Any]] = []
    generated_tokens = 0
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    try:
        for batch_index, batch in enumerate(batched(prompt_rows, batch_size), start=1):
            prompts = [TokensPrompt(prompt_token_ids=row["prompt_token_ids"]) for row in batch]
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            for source, output in zip(batch, outputs, strict=True):
                token_ids: list[int] = []
                if output.outputs:
                    token_ids = [int(token_id) for token_id in output.outputs[0].token_ids]
                generated_tokens += len(token_ids)
                generated_rows.append(
                    {
                        "doc_id": source["doc_id"],
                        "prompt_tokens": len(source["prompt_token_ids"]),
                        "generated_token_ids": token_ids,
                        "generated_tokens": len(token_ids),
                    }
                )
            log(
                f"worker {label} batch={batch_index} docs_done={len(generated_rows)}/"
                f"{len(prompt_rows)} generated_tokens={generated_tokens}"
            )
        flush_qkv_distribution_stats(qkv_file)
    finally:
        shutdown_llm(llm)
        del llm
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    elapsed = time.perf_counter() - started
    write_jsonl(generated_path, generated_rows)
    summary = {
        "model": model_id,
        "model_arg": args.model,
        "model_label": label,
        "run_id": args.run_id,
        "status": "ok",
        "docs": len(prompt_rows),
        "generated_tokens": generated_tokens,
        "elapsed_s": elapsed,
        "qkv_tensor": qkv_file,
        "generated_path": generated_path,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_tokens": args.max_tokens,
        "batch_size": batch_size,
        "patch": PATCH_NAME,
    }
    write_json(summary_path, summary)
    log(f"worker complete {label} elapsed={elapsed:.1f}s qkv={qkv_file}")
    return 0


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 120,
            "font.size": 10.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")


def tensor_scalar(value: torch.Tensor) -> float:
    return float(value.item())


def load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a torch-save dict payload")
    return payload


def finite_flatten(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1).to(torch.float32)
    return flat[torch.isfinite(flat)]


def values_by_qkv(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    labels = [str(label).lower() for label in payload.get("qkv_labels", QKV_LABELS)]
    if "stats" in payload:
        stats = payload["stats"]
        if not torch.is_tensor(stats) or stats.ndim != 3:
            raise ValueError("payload['stats'] must have shape [layers, 3, tokens]")
        return {
            label: finite_flatten(stats[:, idx, :])
            for idx, label in enumerate(labels)
            if label in QKV_LABELS
        }
    stats_by_layer = payload.get("stats_by_layer")
    if not isinstance(stats_by_layer, dict):
        raise KeyError("payload must contain either 'stats' or 'stats_by_layer'")
    out: dict[str, list[torch.Tensor]] = {label: [] for label in QKV_LABELS}
    for layer_payload in stats_by_layer.values():
        if not isinstance(layer_payload, dict):
            continue
        for label in QKV_LABELS:
            tensor = layer_payload.get(label)
            if torch.is_tensor(tensor):
                out[label].append(finite_flatten(tensor))
    return {
        label: torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
        for label, chunks in out.items()
    }


def quantiles_1d(values: torch.Tensor, quantile_levels: list[float]) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("quantiles_1d expects a flat tensor")
    if values.numel() == 0:
        raise ValueError("cannot compute quantiles for an empty tensor")
    sorted_values = torch.sort(values.detach().to(device="cpu", dtype=torch.float32)).values
    if sorted_values.numel() == 1:
        return sorted_values.repeat(len(quantile_levels))
    positions = torch.tensor(quantile_levels, dtype=torch.float64) * (
        sorted_values.numel() - 1
    )
    lower = torch.floor(positions).to(torch.long)
    upper = torch.ceil(positions).to(torch.long)
    weight = (positions - lower.to(torch.float64)).to(torch.float32)
    lower_values = sorted_values[lower]
    upper_values = sorted_values[upper]
    return lower_values + (upper_values - lower_values) * weight


def summarize(model: str, label: str, values: torch.Tensor) -> dict[str, Any]:
    if values.numel() == 0:
        raise ValueError(f"no values for {model} {label}")
    quantiles = quantiles_1d(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    probabilities = {
        f"p_le_{str(threshold).replace('.', '_')}": tensor_scalar(
            (values <= threshold).to(torch.float32).mean()
        )
        for threshold in PROBABILITY_THRESHOLDS
    }
    return {
        "model": model,
        "model_short": MODEL_SHORT.get(model, model),
        "qkv": label,
        "count": int(values.numel()),
        "mean": tensor_scalar(values.mean()),
        "std": tensor_scalar(values.std(unbiased=False)),
        "min": tensor_scalar(values.min()),
        "p01": tensor_scalar(quantiles[0]),
        "p05": tensor_scalar(quantiles[1]),
        "p25": tensor_scalar(quantiles[2]),
        "median": tensor_scalar(quantiles[3]),
        "p75": tensor_scalar(quantiles[4]),
        "p95": tensor_scalar(quantiles[5]),
        "p99": tensor_scalar(quantiles[6]),
        "max": tensor_scalar(values.max()),
        **probabilities,
    }


def model_sort_key(path: Path) -> tuple[int, str]:
    model = path.parent.name
    return (MODEL_LABEL_ORDER.index(model) if model in MODEL_LABEL_ORDER else 999, model)


def discover_inputs(data_root: Path, models: list[str] | None) -> list[Path]:
    if models:
        labels = [model_label(model) for model in models]
        paths = [data_root / label / QKV_TENSOR_NAME for label in labels]
    else:
        paths = sorted(data_root.glob(f"*/{QKV_TENSOR_NAME}"), key=model_sort_key)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing qkv tensor(s):\n" + "\n".join(str(path) for path in missing)
        )
    if not paths:
        raise FileNotFoundError(f"no {QKV_TENSOR_NAME} tensors found under {data_root}")
    return paths


def global_edges(paths: list[Path], bins: int) -> torch.Tensor:
    mins: list[float] = []
    maxes: list[float] = []
    for path in paths:
        values = values_by_qkv(load_payload(path))
        finite = torch.cat([tensor for tensor in values.values() if tensor.numel()])
        mins.append(tensor_scalar(finite.min()))
        maxes.append(tensor_scalar(finite.max()))
    return torch.linspace(min(mins), max(maxes), bins + 1)


def build_histogram_rows(
    *,
    model: str,
    values_by_label: dict[str, torch.Tensor],
    edges: torch.Tensor,
) -> list[dict[str, Any]]:
    widths = edges[1:] - edges[:-1]
    centers = (edges[:-1] + edges[1:]) / 2
    rows: list[dict[str, Any]] = []
    for label in QKV_LABELS:
        values = values_by_label[label]
        counts = torch.histogram(values, bins=edges).hist.to(torch.float64)
        density = counts / (counts.sum() * widths.to(torch.float64))
        for idx in range(edges.numel() - 1):
            rows.append(
                {
                    "model": model,
                    "model_short": MODEL_SHORT.get(model, model),
                    "qkv": label,
                    "bin_index": idx,
                    "bin_left": tensor_scalar(edges[idx]),
                    "bin_right": tensor_scalar(edges[idx + 1]),
                    "bin_center": tensor_scalar(centers[idx]),
                    "count": int(counts[idx].item()),
                    "density": float(density[idx].item()),
                }
            )
    return rows


def probability_text(values_by_label: dict[str, torch.Tensor]) -> str:
    lines = []
    for threshold in PROBABILITY_THRESHOLDS:
        threshold_text = "16" if threshold == 16.0 else f"{threshold:g}"
        probs = {
            label: tensor_scalar((values_by_label[label] <= threshold).to(torch.float32).mean())
            for label in QKV_LABELS
        }
        lines.append(rf"$P(\mathrm{{AbsMax}}/\sigma \leq {threshold_text})$")
        lines.append(f"Q {probs['q']:.3f}   K {probs['k']:.3f}   V {probs['v']:.3f}")
    return "\n".join(lines)


def plot_distribution(
    *,
    model: str,
    values_by_label: dict[str, torch.Tensor],
    edges: torch.Tensor,
    output_pdf: Path,
    output_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.1, 2.55))
    widths = edges[1:] - edges[:-1]
    centers = (edges[:-1] + edges[1:]) / 2
    for label in QKV_LABELS:
        values = values_by_label[label]
        counts = torch.histogram(values, bins=edges).hist.to(torch.float64)
        density = counts / (counts.sum() * widths.to(torch.float64))
        ax.plot(
            centers.numpy(),
            density.numpy(),
            color=COLORS[label],
            linewidth=1.55,
            label=DISPLAY_LABELS[label],
        )
        ax.fill_between(centers.numpy(), density.numpy(), color=COLORS[label], alpha=0.08)
    ax.set_xlabel("AbsMax / AbsStd")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=True, loc="upper right", ncol=3, handlelength=1.4)
    ax.text(
        0.525,
        0.735,
        probability_text(values_by_label),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "linewidth": 0.6,
            "alpha": 0.92,
        },
    )
    fig.tight_layout()
    if model != "Qwen-Qwen3-14B":
        fig.text(
            0.070,
            0.065,
            MODEL_SHORT.get(model, model),
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
            color="#222222",
        )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def process_model_plot(path: Path, edges: torch.Tensor, output_dir: Path) -> dict[str, Any]:
    model = path.parent.name
    slug = slugify(model)
    pdf_path = output_dir / f"qkv_absmax_over_absstd_distribution__{slug}.pdf"
    png_path = output_dir / f"qkv_absmax_over_absstd_distribution__{slug}.png"
    summary_csv = output_dir / f"qkv_absmax_over_absstd_summary__{slug}.csv"
    histogram_csv = output_dir / f"qkv_absmax_over_absstd_histogram__{slug}.csv"
    payload = load_payload(path)
    values = values_by_qkv(payload)
    summary_rows = [summarize(model, label, values[label]) for label in QKV_LABELS]
    histogram_rows = build_histogram_rows(model=model, values_by_label=values, edges=edges)
    plot_distribution(
        model=model,
        values_by_label=values,
        edges=edges,
        output_pdf=pdf_path,
        output_png=png_path,
    )
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDNAMES)
    write_csv(histogram_csv, histogram_rows, HISTOGRAM_FIELDNAMES)
    return {
        "model": model,
        "model_short": MODEL_SHORT.get(model, model),
        "input": path,
        "pdf": pdf_path,
        "png": png_path,
        "summary_csv": summary_csv,
        "histogram_csv": histogram_csv,
        "source_metadata": payload.get("metadata", {}),
    }


def write_zip(zip_path: Path, files: list[Path], root: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))


def plot_run(
    *,
    data_root: Path,
    output_dir: Path,
    models: list[str] | None,
    bins: int,
) -> dict[str, Any]:
    paths = discover_inputs(data_root, models)
    configure_matplotlib()
    edges = global_edges(paths, bins)
    by_model_dir = output_dir / "plots" / "by_model"
    summary_all_csv = output_dir / "qkv_absmax_over_absstd_summary_all_models.csv"
    histogram_all_csv = output_dir / "qkv_absmax_over_absstd_histogram_all_models.csv"
    manifest_json = output_dir / "manifest.json"
    zip_path = output_dir / "qkv_absmax_over_absstd_all_model_plots.zip"
    summary_rows_all: list[dict[str, Any]] = []
    histogram_rows_all: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    for path in paths:
        result = process_model_plot(path, edges, by_model_dir)
        model_results.append(result)
        with Path(result["summary_csv"]).open("r", encoding="utf-8", newline="") as handle:
            summary_rows_all.extend(csv.DictReader(handle))
        with Path(result["histogram_csv"]).open("r", encoding="utf-8", newline="") as handle:
            histogram_rows_all.extend(csv.DictReader(handle))
        log(f"wrote {result['pdf']}")
    write_csv(summary_all_csv, summary_rows_all, SUMMARY_FIELDNAMES)
    write_csv(histogram_all_csv, histogram_rows_all, HISTOGRAM_FIELDNAMES)
    files_for_zip = [
        *(Path(result["pdf"]) for result in model_results),
        *(Path(result["png"]) for result in model_results),
        *(Path(result["summary_csv"]) for result in model_results),
        *(Path(result["histogram_csv"]) for result in model_results),
        summary_all_csv,
        histogram_all_csv,
    ]
    manifest = {
        "data_root": data_root,
        "inputs": paths,
        "bins": bins,
        "global_bin_min": tensor_scalar(edges[0]),
        "global_bin_max": tensor_scalar(edges[-1]),
        "summary_all_csv": summary_all_csv,
        "histogram_all_csv": histogram_all_csv,
        "zip": zip_path,
        "models": model_results,
    }
    write_json(manifest_json, manifest)
    files_for_zip.append(manifest_json)
    write_zip(zip_path, files_for_zip, TEST_DIR)
    log(f"wrote {summary_all_csv}")
    return manifest


def build_worker_command(args: argparse.Namespace, model: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        model,
        "--run-id",
        args.run_id,
        "--subset-file",
        str(path_from_cli(args.subset_file)),
        "--doc-limit",
        str(args.doc_limit),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--generation-batch-size",
        str(args.generation_batch_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cpu-offload-gb",
        str(args.cpu_offload_gb),
        "--seed",
        str(args.seed),
    ]
    if args.attention_backend:
        cmd.extend(["--attention-backend", args.attention_backend])
    cmd.append("--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code")
    return cmd


def run_full(args: argparse.Namespace) -> int:
    models = model_args_from_cli(args)
    run_dir = OUTPUT_ROOT / args.run_id
    results_dir = RESULTS_ROOT / args.run_id
    plan = {
        "run_id": args.run_id,
        "models": models,
        "resolved_models": [resolve_model(model) for model in models],
        "patch": PATCH_NAME,
        "patch_file": ROOT / "patches" / f"{PATCH_NAME}.patch",
        "patch_sha256": sha256_file(ROOT / "patches" / f"{PATCH_NAME}.patch"),
        "output_dir": run_dir,
        "results_dir": results_dir,
        "subset_file": path_from_cli(args.subset_file),
        "doc_limit": args.doc_limit,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_tokens": args.max_tokens,
    }
    write_json(results_dir / "plan.json", plan)
    if args.dry_run:
        print(json.dumps(jsonable(plan), indent=2, sort_keys=True))
        return 0

    ensure_no_active_patch()
    try:
        log(f"applying {PATCH_NAME}")
        run_patch_command("prepare-version", PATCH_NAME)
        for model in models:
            cmd = build_worker_command(args, model)
            log("running " + " ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        revert_active_patch_best_effort()
    if not args.skip_plot:
        plot_run(data_root=run_dir, output_dir=results_dir, models=models, bins=args.bins)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--run-id", default=f"qkv_distribution_{utc_timestamp()}")
    parser.add_argument("--subset-file", type=Path, default=DEFAULT_SUBSET_FILE)
    parser.add_argument("--doc-limit", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bins", type=int, default=220)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--results-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    if args.plot_only:
        models = model_args_from_cli(args) if (args.models or args.all_models) else None
        data_root = path_from_cli(args.data_root) if args.data_root else OUTPUT_ROOT / args.run_id
        results_dir = path_from_cli(args.results_dir) if args.results_dir else RESULTS_ROOT / args.run_id
        plot_run(data_root=data_root, output_dir=results_dir, models=models, bins=args.bins)
        return 0
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
