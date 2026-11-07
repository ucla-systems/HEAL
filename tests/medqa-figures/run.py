#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build MedQA error-vs-flip and mantissa-vs-flip figures from runner outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
FINAL_HIDDEN_RESULTS = ROOT / "tests" / "final-hidden-mse" / "results"
FLIP_RESULTS = ROOT / "tests" / "flip-rate-analysis" / "results"
DEFAULT_GPU_SLUG = "NVIDIA_H100_80GB_HBM3"

MODEL_ORDER = [
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

SETTING_LABELS = {
    "native_bf16": "BF16",
    "native_fp16": "FP16",
    "layercast_tf32": "TF32 (+LayerCast)",
    "layercast_fp32": "FP32 (+LayerCast)",
    "heal_base": "HEAL-Base",
    "heal_plus": "HEAL+",
    "heal_plus_plus": "HEAL++",
    "heal_full": "HEAL-Full",
    "heal_full_plus": "HEAL-Full+",
}
BASELINE_POINT_BITS = {
    "native_bf16": 7,
    "native_fp16": 10,
    "layercast_tf32": 10,
}
BASELINE_MARKERS = {
    "native_bf16": "s",
    "native_fp16": "^",
    "layercast_tf32": "D",
}
BASELINE_COLORS = {
    "native_bf16": "#009e73",
    "native_fp16": "#d55e00",
    "layercast_tf32": "#cc79a7",
}
GROUP_COLORS = {
    "fp": "#555555",
    "mantissa": "#0072b2",
    "heal": "#009e73",
    "native": "#d55e00",
}


@dataclass(frozen=True)
class MetricRow:
    run_id: str
    model: str
    setting: str
    mode: str
    status: str
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-hidden-run-id", action="append", default=[])
    parser.add_argument("--flip-run-id", action="append", default=[])
    parser.add_argument("--final-hidden-results-root", type=Path, default=FINAL_HIDDEN_RESULTS)
    parser.add_argument("--flip-results-root", type=Path, default=FLIP_RESULTS)
    parser.add_argument("--gpu-slug", default=DEFAULT_GPU_SLUG)
    parser.add_argument("--gpu-label", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--png-only", action="store_true")
    return parser.parse_args()


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def model_sort_key(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else 999, model)


def setting_sort_key(setting: str) -> tuple[int, int, str]:
    if setting == "layercast_fp32":
        return (0, 0, setting)
    if setting == "layercast_tf32":
        return (0, 1, setting)
    if setting.startswith("mantissa_k"):
        return (1, int(setting.removeprefix("mantissa_k") or 999), setting)
    if setting == "native_bf16":
        return (2, 0, setting)
    if setting == "native_fp16":
        return (2, 1, setting)
    if setting.startswith("heal_"):
        order = ["heal_base", "heal_plus", "heal_plus_plus", "heal_full", "heal_full_plus"]
        return (3, order.index(setting) if setting in order else 999, setting)
    return (9, 999, setting)


def gpu_label_from_slug(slug: str) -> str:
    upper = slug.upper()
    if "H100" in upper:
        return "H100"
    if "A100" in upper:
        return "A100"
    if "MI300" in upper:
        return "MI300X"
    return slug


def normalize_model(value: str) -> str:
    return value.replace("/", "-")


def normalize_setting(row: dict[str, Any]) -> str | None:
    setting = str(row.get("setting") or row.get("mode") or "")
    mode = str(row.get("mode") or "")
    if setting in {"layercast_fp32_ref", "layercast_fp32_generate_ref"}:
        return None
    if setting in {"original_bf16", "original_bf16_prefill", "original_bf16_generate"}:
        return "native_bf16"
    if setting in {"original_fp16", "original_fp16_prefill", "original_fp16_generate"}:
        return "native_fp16"
    if setting.startswith("significand_k"):
        bits = re.search(r"significand_k0*(\d+)", setting)
        return f"mantissa_k{int(bits.group(1))}" if bits else None
    if mode.startswith("significand_k"):
        bits = re.search(r"significand_k0*(\d+)", mode)
        return f"mantissa_k{int(bits.group(1))}" if bits else None
    if setting.startswith("mantissa_k"):
        bits = re.search(r"mantissa_k0*(\d+)", setting)
        return f"mantissa_k{int(bits.group(1))}" if bits else setting
    return setting or None


def load_metrics_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("metrics"), list):
        return list(data["metrics"])
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        return [data]
    return []


def load_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_run_metrics(root: Path, run_ids: Iterable[str]) -> Iterable[MetricRow]:
    for run_id in run_ids:
        run_dir = root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"missing run directory: {run_dir}")
        metric_paths = sorted(run_dir.glob("*/metrics.json"))
        if not metric_paths:
            metric_paths = sorted(run_dir.glob("*/metrics.csv"))
        for metric_path in metric_paths:
            rows = load_metrics_json(metric_path) if metric_path.suffix == ".json" else load_metrics_csv(metric_path)
            for row in rows:
                status = str(row.get("status") or "")
                setting = normalize_setting(row)
                if not setting:
                    continue
                model = normalize_model(str(row.get("model") or metric_path.parent.name))
                yield MetricRow(
                    run_id=run_id,
                    model=model,
                    setting=setting,
                    mode=str(row.get("mode") or ""),
                    status=status,
                    payload=row,
                )


def select_latest_ok(rows: Iterable[MetricRow]) -> dict[tuple[str, str], MetricRow]:
    selected: dict[tuple[str, str], MetricRow] = {}
    for row in rows:
        if row.status and row.status != "ok":
            continue
        selected[(row.model, row.setting)] = row
    return selected


def flip_rate(row: MetricRow) -> float:
    payload = row.payload
    value = to_float(payload.get("valid_reference_flip_rate"))
    if math.isfinite(value):
        return value
    value = to_float(payload.get("answer_flip_rate"))
    if math.isfinite(value):
        return value
    pct = to_float(payload.get("valid_reference_flip_rate_pct"))
    if math.isfinite(pct):
        return pct / 100.0
    pct = to_float(payload.get("answer_flip_rate_pct"))
    if math.isfinite(pct):
        return pct / 100.0
    return math.nan


def mantissa_bits(setting: str) -> int | None:
    match = re.fullmatch(r"mantissa_k(\d+)", setting)
    if match:
        return int(match.group(1))
    return BASELINE_POINT_BITS.get(setting)


def setting_group(setting: str) -> str:
    if setting.startswith("mantissa_k"):
        return "mantissa"
    if setting.startswith("heal_"):
        return "heal"
    if setting.startswith("native_"):
        return "native"
    return "fp"


def is_fresh_native_triton(setting: str, *metric_rows: MetricRow) -> bool:
    if setting not in {"native_bf16", "native_fp16"}:
        return True
    return all(str(row.payload.get("attention_backend") or "") == "TRITON_ATTN" for row in metric_rows)


def joined_rows(final_rows: dict[tuple[str, str], MetricRow], flip_rows: dict[tuple[str, str], MetricRow], gpu_label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    keys = sorted(set(final_rows) & set(flip_rows), key=lambda item: (model_sort_key(item[0]), setting_sort_key(item[1])))
    for model, setting in keys:
        mse_row = final_rows[(model, setting)]
        flip_row = flip_rows[(model, setting)]
        if not is_fresh_native_triton(setting, mse_row, flip_row):
            continue
        mse = to_float(mse_row.payload.get("mse"))
        std = to_float(mse_row.payload.get("std_hidden_mse"), 0.0)
        count = to_float(mse_row.payload.get("compared_hidden_rows"), 0.0)
        rate = flip_rate(flip_row)
        if not (math.isfinite(mse) and mse > 0 and math.isfinite(rate)):
            continue
        stderr = std / math.sqrt(count) if math.isfinite(std) and count > 0 else 0.0
        bits = mantissa_bits(setting)
        rows.append(
            {
                "gpu": gpu_label,
                "model": model,
                "model_short": MODEL_SHORT.get(model, model),
                "setting": setting,
                "setting_label": SETTING_LABELS.get(setting, setting),
                "setting_group": setting_group(setting),
                "mantissa_bits": "" if bits is None else str(bits),
                "mean_hidden_mse": f"{mse:.12g}",
                "std_hidden_mse": f"{std:.12g}" if math.isfinite(std) else "",
                "mse_row_count": f"{int(count)}" if count > 0 else "",
                "stderr_hidden_mse": f"{stderr:.12g}",
                "answer_flip_rate": f"{rate:.12g}",
                "answer_flip_rate_pct": f"{rate * 100.0:.12g}",
                "final_hidden_run_id": mse_row.run_id,
                "flip_run_id": flip_row.run_id,
                "final_hidden_mode": mse_row.mode,
                "flip_mode": flip_row.mode,
                "final_hidden_attention_backend": str(mse_row.payload.get("attention_backend") or ""),
                "flip_attention_backend": str(flip_row.payload.get("attention_backend") or ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "font.size": 10.5,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "legend.title_fontsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_mantissa(rows: list[dict[str, str]], output_root: Path, png_only: bool) -> list[Path]:
    import matplotlib.pyplot as plt

    plot_dir = output_root / "plots" / "mantissa_bits_vs_flip_rate"
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_csv(plot_dir / "mantissa_bits_vs_flip_rate_plot_data.csv", rows)
    paths: list[Path] = []
    for model in sorted({row["model"] for row in rows}, key=model_sort_key):
        model_rows = [row for row in rows if row["model"] == model]
        sweep = [row for row in model_rows if row["setting"].startswith("mantissa_k") and row["mantissa_bits"]]
        baseline = [row for row in model_rows if row["setting"] in BASELINE_POINT_BITS]
        fp32 = [row for row in model_rows if row["setting"] == "layercast_fp32"]
        if not sweep and not baseline:
            continue
        fig, ax = plt.subplots(figsize=(4.05, 2.55))
        if sweep:
            sweep = sorted(sweep, key=lambda row: int(row["mantissa_bits"]))
            ax.plot(
                [int(row["mantissa_bits"]) for row in sweep],
                [float(row["answer_flip_rate"]) for row in sweep],
                color="#0072b2",
                marker="o",
                markersize=4.8,
                linewidth=1.6,
                alpha=0.92,
            )
        for row in baseline:
            ax.scatter(
                [int(row["mantissa_bits"])],
                [float(row["answer_flip_rate"])],
                color=BASELINE_COLORS[row["setting"]],
                marker=BASELINE_MARKERS[row["setting"]],
                s=82,
                linewidths=0.7,
                edgecolors="white",
                zorder=3,
                label=row["setting_label"],
            )
        for row in fp32:
            ax.axhline(float(row["answer_flip_rate"]), color="#555555", linestyle="--", linewidth=1.2, alpha=0.7, label="FP32")
        ax.set_xlabel("# Significand Bits")
        ax.set_ylabel("Answer Flip Rate")
        ax.set_xticks(list(range(3, 24)))
        ax.set_xlim(2.6, 23.4)
        y_values = [float(row["answer_flip_rate"]) for row in model_rows]
        y_max = max(y_values) if y_values else 0.0
        ax.set_ylim(bottom=0.0, top=min(0.6, max(0.05, y_max * 1.15)))
        ax.grid(True, which="major", axis="y", color="#dddddd", linewidth=0.7)
        ax.grid(False, which="both", axis="x")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            dedup = dict(zip(labels, handles))
            ax.legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True)
        ax.text(0.03, 0.05, MODEL_SHORT.get(model, model), transform=ax.transAxes, fontsize=10.5, fontweight="semibold")
        fig.tight_layout()
        stem = f"mantissa_bits_vs_flip_rate__{slugify(model)}"
        png = plot_dir / f"{stem}.png"
        fig.savefig(png, bbox_inches="tight")
        paths.append(png)
        if not png_only:
            pdf = plot_dir / f"{stem}.pdf"
            fig.savefig(pdf, bbox_inches="tight")
            paths.append(pdf)
        plt.close(fig)
    return paths


def plot_error_vs_flip(rows: list[dict[str, str]], output_root: Path, png_only: bool) -> list[Path]:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plot_dir = output_root / "plots" / "error_vs_flip_rate_by_model_errorbars"
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_csv(plot_dir / "error_vs_flip_rate_by_model_errorbars_plot_data.csv", rows)
    paths: list[Path] = []
    for model in sorted({row["model"] for row in rows}, key=model_sort_key):
        model_rows = [row for row in rows if row["model"] == model]
        fig, ax = plt.subplots(figsize=(4.05, 2.7))
        for row in sorted(model_rows, key=lambda row: setting_sort_key(row["setting"])):
            mean = float(row["mean_hidden_mse"])
            stderr = float(row["stderr_hidden_mse"] or 0.0)
            lower = min(stderr, mean * 0.95)
            upper = stderr
            color = GROUP_COLORS.get(row["setting_group"], "#555555")
            ax.errorbar(
                mean,
                float(row["answer_flip_rate"]),
                xerr=[[lower], [upper]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.35,
                capsize=3.0,
                capthick=1.0,
                markersize=4.8,
                alpha=0.86,
                zorder=2,
            )
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel("Final Hidden MSE")
        ax.set_ylabel("Answer Flip Rate")
        y_values = [float(row["answer_flip_rate"]) for row in model_rows]
        y_max = max(y_values) if y_values else 0.0
        ax.set_ylim(bottom=0.0, top=min(0.6, max(0.05, y_max * 1.12)))
        ax.grid(True, which="major", axis="y", color="#dddddd", linewidth=0.7)
        ax.grid(False, which="both", axis="x")
        handles = []
        for group, label in [("fp", "FP32/TF32"), ("mantissa", "Significand"), ("native", "Native"), ("heal", "HEAL")]:
            if any(row["setting_group"] == group for row in model_rows):
                handles.append(plt.Line2D([0], [0], color=GROUP_COLORS[group], marker="o", linestyle="none", markersize=5.5, label=label))
        if handles:
            ax.legend(handles=handles, loc="upper left", frameon=True)
        ax.text(0.03, 0.05, MODEL_SHORT.get(model, model), transform=ax.transAxes, fontsize=10.5, fontweight="semibold")
        fig.tight_layout()
        stem = f"error_vs_flip_rate_errorbars__{slugify(model)}"
        png = plot_dir / f"{stem}.png"
        fig.savefig(png, bbox_inches="tight")
        paths.append(png)
        if not png_only:
            pdf = plot_dir / f"{stem}.pdf"
            fig.savefig(pdf, bbox_inches="tight")
            paths.append(pdf)
        plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    if not args.final_hidden_run_id:
        raise SystemExit("at least one --final-hidden-run-id is required")
    if not args.flip_run_id:
        raise SystemExit("at least one --flip-run-id is required")
    output_root = args.output_root or (ROOT / "figures" / args.gpu_slug / "fig-error-vs-flip-and-mantissa-vs-flip")
    gpu_label = args.gpu_label or gpu_label_from_slug(args.gpu_slug)

    final_rows = select_latest_ok(iter_run_metrics(args.final_hidden_results_root, args.final_hidden_run_id))
    flip_rows = select_latest_ok(iter_run_metrics(args.flip_results_root, args.flip_run_id))
    rows = joined_rows(final_rows, flip_rows, gpu_label)
    if not rows and not args.allow_missing:
        raise SystemExit("no joined final-hidden/flip rows found")

    data_path = output_root / "plots" / "medqa_error_flip_joined_data.csv"
    write_csv(data_path, rows)
    configure_matplotlib()
    written = [data_path]
    written.extend(plot_mantissa(rows, output_root, args.png_only))
    written.extend(plot_error_vs_flip(rows, output_root, args.png_only))
    for written_path in written:
        print(f"wrote {written_path}")
    print(f"joined_rows={len(rows)} final_rows={len(final_rows)} flip_rows={len(flip_rows)}")


if __name__ == "__main__":
    main()
