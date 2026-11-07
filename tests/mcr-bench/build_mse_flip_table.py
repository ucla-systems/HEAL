#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export MCR-Bench MSE/flip-rate results into paper table data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FINAL_RESULTS = ROOT / "tests" / "final-hidden-mse" / "results"
FLIP_RESULTS = ROOT / "tests" / "flip-rate-analysis" / "results"
FINAL_OUTPUTS = ROOT / "output" / "final-hidden-mse"
FLIP_OUTPUTS = ROOT / "output" / "flip-rate-analysis"
TABLE_DATA = ROOT / "tables" / "data"
MSE_FLIP_DATA = TABLE_DATA / "mse-flip-rate"
EXEC_TIME_DATA = TABLE_DATA / "execution-time"
MSE_FLIP_TEX = ROOT / "tables" / "mse-flip-rate.tex"
EXEC_TIME_TEX = ROOT / "tables" / "execution-time.tex"

MODE_ROWS = [
    ("original_bf16_prefill", "original_bf16_generate", "BF16"),
    ("original_fp16_prefill", "original_fp16_generate", "FP16"),
    ("layercast_tf32_prefill", "layercast_tf32_generate", "TF32 (+LayerCast)"),
    ("layercast_fp32_prefill", "layercast_fp32_generate", "FP32 (+LayerCast)"),
    ("__midrule__", "__midrule__", ""),
    ("heal_base_prefill", "heal_base_generate", "HEAL-Base"),
    ("heal_plus_prefill", "heal_plus_generate", "HEAL+"),
    ("heal_plus_plus_prefill", "heal_plus_plus_generate", "HEAL++"),
    ("heal_full_plus_prefill", "heal_full_plus_generate", "HEAL-Full"),
]
TABLE_PREFILL_MODES = {mode for mode, _, _ in MODE_ROWS if mode != "__midrule__"}
TABLE_FLIP_MODES = {mode for _, mode, _ in MODE_ROWS if mode != "__midrule__"}

MODEL_INFO = [
    ("qwen3-8b", "Qwen3-8B", "Qwen3-8B"),
    ("qwen3-14b", "Qwen3-14B", "Qwen3-14B"),
    ("qwen3-32b", "Qwen3-32B", "Qwen3-32B"),
    ("llama-3.1-8b-instruct", "Llama-8B", "Llama-8B"),
    ("deepseek-r1-distill-qwen-14b", "DeepSeek-14B", "DS-14B"),
]
MODEL_BY_ARG = {arg: (table, exec_label) for arg, table, exec_label in MODEL_INFO}
MODEL_ORDER = [table for _, table, _ in MODEL_INFO]
EXEC_LABEL_BY_TABLE = {table: exec_label for _, table, exec_label in MODEL_INFO}
MODEL_DIR_BY_TABLE = {
    "Qwen3-8B": "Qwen-Qwen3-8B",
    "Qwen3-14B": "Qwen-Qwen3-14B",
    "Qwen3-32B": "Qwen-Qwen3-32B",
    "Llama-8B": "meta-llama-Llama-3.1-8B-Instruct",
    "DeepSeek-14B": "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Base MCR run id used for output filenames.")
    parser.add_argument(
        "--final-source-run-id",
        help="Base run id that produced <id>_<domain>_final_hidden_mse. Defaults to --run-id.",
    )
    parser.add_argument(
        "--flip-source-run-id",
        action="append",
        default=[],
        help=(
            "Base run id that produced <id>_<domain>_flip_rate. Pass multiple "
            "times to aggregate generation flip-rate rounds. Defaults to --run-id."
        ),
    )
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--skip-tex", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite_values(values: list[Any]) -> list[float]:
    return [value for value in (to_float(value) for value in values) if math.isfinite(value)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def standard_error(values: list[float]) -> float:
    return sample_std(values) / math.sqrt(len(values)) if values else math.nan


def fmt_num(value: Any, digits: int = 6) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def fmt_mse(value: Any) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return "--"
    if number == 0:
        return "0"
    text = f"{number:.1e}"
    mantissa, exponent = text.split("e")
    return f"{mantissa}e{int(exponent)}"


def fmt_flip(mean_value: Any, std_value: Any = 0.0) -> str:
    avg = to_float(mean_value)
    std = to_float(std_value)
    if not math.isfinite(avg):
        return "$--$"
    if not math.isfinite(std):
        std = 0.0
    return f"${avg:.1f}\\pm{std:.1f}$"


def add_elapsed(elapsed_by_model: dict[str, float], model: str, value: Any) -> None:
    number = to_float(value)
    if math.isfinite(number):
        elapsed_by_model[model] += number


def model_name(row: dict[str, str]) -> str | None:
    model_arg = row.get("model_arg", "")
    if model_arg in MODEL_BY_ARG:
        return MODEL_BY_ARG[model_arg][0]
    raw = row.get("model", "")
    for table_name, model_dir in MODEL_DIR_BY_TABLE.items():
        if raw == model_dir or raw == table_name:
            return table_name
    return None



def maybe_read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def read_worker_elapsed(output_root: Path, run_id: str, model: str, mode: str) -> str:
    path = output_root / run_id / MODEL_DIR_BY_TABLE[model] / mode / "worker_result.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    elapsed = data.get("elapsed_s")
    return "" if elapsed is None else str(elapsed)


def previous_mse_rows(args: argparse.Namespace, domain: str, final_run: str) -> list[tuple[str, str, dict[str, str]]]:
    tag = f"mcr-bench-{domain}-{args.gpu.lower()}"
    rows: list[tuple[str, str, dict[str, str]]] = []
    for row in maybe_read_csv(MSE_FLIP_DATA / f"final-hidden-mse-{tag}.csv"):
        model = row.get("model", "")
        mode = row.get("mode", "")
        if row.get("source_run_id") != final_run or model not in MODEL_ORDER or mode not in TABLE_PREFILL_MODES:
            continue
        enriched = dict(row)
        enriched.setdefault("model", row.get("model_results_dir", MODEL_DIR_BY_TABLE[model]))
        enriched["worker_elapsed_s"] = read_worker_elapsed(FINAL_OUTPUTS, final_run, model, mode)
        rows.append((model, mode, enriched))
    return rows


def previous_flip_rows(args: argparse.Namespace, domain: str, flip_runs: list[str]) -> list[tuple[str, str, dict[str, str]]]:
    tag = f"mcr-bench-{domain}-{args.gpu.lower()}"
    wanted = set(flip_runs)
    rows: list[tuple[str, str, dict[str, str]]] = []
    for row in maybe_read_csv(MSE_FLIP_DATA / f"answer-flip-rate-{tag}.csv"):
        model = row.get("model", "")
        mode = row.get("mode", "")
        source_ids = [source for source in row.get("source_run_ids", "").split(",") if source]
        matching = [source for source in source_ids if source in wanted]
        if not matching or model not in MODEL_ORDER or mode not in TABLE_FLIP_MODES:
            continue
        def pct_to_rate(field: str) -> str:
            value = to_float(row.get(field))
            return "" if not math.isfinite(value) else str(value / 100.0)

        enriched = {
            "mode": mode,
            "status": row.get("status", ""),
            "docs": row.get("docs", ""),
            "answer_flip_count": row.get("answer_flip_count", ""),
            "answer_flip_rate": pct_to_rate("answer_flip_rate_pct"),
            "valid_reference_rows": row.get("valid_reference_rows", ""),
            "valid_reference_flip_count": row.get("valid_reference_flip_count", ""),
            "valid_reference_flip_rate": pct_to_rate("valid_reference_flip_rate_pct"),
            "answer_f_count": row.get("answer_f_count", ""),
            "max_length_count": row.get("max_length_count", ""),
            "parse_fail_count": row.get("parse_fail_count", ""),
            "worker_elapsed_s": row.get("elapsed_s", ""),
            "_source_run_id": matching[0],
        }
        rows.append((model, mode, enriched))
    return rows


def load_model_metrics(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        raise FileNotFoundError(f"missing result root: {root}")
    for metrics_path in sorted(root.glob("*/metrics.csv")):
        rows.extend(read_csv(metrics_path))
    if not rows:
        raise FileNotFoundError(f"no metrics.csv found under {root}")
    return rows


def actual_final_run_id(args: argparse.Namespace, domain: str) -> str:
    base = args.final_source_run_id or args.run_id
    return f"{base}_{domain}_final_hidden_mse"


def actual_flip_run_ids(args: argparse.Namespace, domain: str) -> list[str]:
    bases = args.flip_source_run_id or [args.run_id]
    return [f"{base}_{domain}_flip_rate" for base in bases]


def collect_domain(args: argparse.Namespace, domain: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final_run = actual_final_run_id(args, domain)
    flip_runs = actual_flip_run_ids(args, domain)
    final_rows = load_model_metrics(FINAL_RESULTS / final_run)

    final_by_key: dict[tuple[str, str], dict[str, str]] = {}
    flip_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    elapsed_by_model = {model: 0.0 for model in MODEL_ORDER}

    for model, mode, row in previous_mse_rows(args, domain, final_run):
        final_by_key[(model, mode)] = row

    for row in final_rows:
        model = model_name(row)
        mode = row.get("mode", "")
        if model is None or row.get("status") != "ok":
            continue
        if mode in TABLE_PREFILL_MODES:
            final_by_key[(model, mode)] = row

    def add_flip_row(model: str, mode: str, row: dict[str, str]) -> None:
        source = row.get("_source_run_id", "")
        rows = flip_by_key.setdefault((model, mode), [])
        if source:
            rows[:] = [existing for existing in rows if existing.get("_source_run_id") != source]
        rows.append(row)

    for model, mode, row in previous_flip_rows(args, domain, flip_runs):
        add_flip_row(model, mode, row)

    for flip_run in flip_runs:
        for row in load_model_metrics(FLIP_RESULTS / flip_run):
            model = model_name(row)
            mode = row.get("mode", "")
            if model is None or row.get("status") != "ok":
                continue
            if mode in TABLE_FLIP_MODES:
                enriched = dict(row)
                enriched["_source_run_id"] = flip_run
                add_flip_row(model, mode, enriched)

    mse_records: list[dict[str, Any]] = []
    flip_records: list[dict[str, Any]] = []
    for prefill_mode, flip_mode, approach in MODE_ROWS:
        if prefill_mode == "__midrule__":
            continue
        for model in MODEL_ORDER:
            final = final_by_key.get((model, prefill_mode))
            flips = flip_by_key.get((model, flip_mode), [])
            if final is None:
                raise RuntimeError(f"missing final-hidden row: {domain} {model} {prefill_mode}")
            if len(flips) != len(flip_runs):
                raise RuntimeError(
                    f"missing flip-rate row(s): {domain} {model} {flip_mode} "
                    f"has {len(flips)}, expected {len(flip_runs)}"
                )
            add_elapsed(elapsed_by_model, model, final.get("worker_elapsed_s"))
            for row in flips:
                add_elapsed(elapsed_by_model, model, row.get("worker_elapsed_s"))

            mse_records.append(
                {
                    "source_run_id": final_run,
                    "domain": domain,
                    "gpu": args.gpu,
                    "model": model,
                    "model_results_dir": final.get("model_results_dir", final.get("model", "")),
                    "approach": approach,
                    "mode": prefill_mode,
                    "status": final.get("status", ""),
                    "mse": final.get("mse", ""),
                    "mse_table_value": fmt_mse(final.get("mse")),
                    "compared_hidden_rows": final.get("compared_hidden_rows", ""),
                    "hidden_size": final.get("hidden_size", ""),
                    "metrics_csv": str(FINAL_RESULTS / final_run / MODEL_DIR_BY_TABLE[model] / "metrics.csv"),
                }
            )

            answer_flip_rates = finite_values([100.0 * to_float(row.get("answer_flip_rate")) for row in flips])
            valid_flip_rates = finite_values([100.0 * to_float(row.get("valid_reference_flip_rate")) for row in flips])
            answer_flip_counts = finite_values([row.get("answer_flip_count") for row in flips])
            valid_flip_counts = finite_values([row.get("valid_reference_flip_count") for row in flips])
            first = flips[0]
            flip_records.append(
                {
                    "run_id": "+".join(flip_runs),
                    "source_run_ids": ",".join(row["_source_run_id"] for row in flips),
                    "rounds": str(len(flips)),
                    "domain": domain,
                    "gpu": args.gpu,
                    "model": model,
                    "configuration": approach,
                    "mode": flip_mode,
                    "status": first.get("status", ""),
                    "docs": first.get("docs", ""),
                    "answer_flip_count": fmt_num(mean(answer_flip_counts), 3) if answer_flip_counts else "",
                    "answer_flip_rate_pct": f"{mean(answer_flip_rates):.6f}" if answer_flip_rates else "",
                    "answer_flip_rate_pct_std": f"{population_std(answer_flip_rates):.6f}" if answer_flip_rates else "",
                    "answer_flip_rate_pct_stderr": f"{standard_error(answer_flip_rates):.6f}" if answer_flip_rates else "",
                    "answer_f_count": first.get("answer_f_count", ""),
                    "max_length_count": first.get("max_length_count", ""),
                    "parse_fail_count": first.get("parse_fail_count", ""),
                    "valid_reference_rows": first.get("valid_reference_rows", ""),
                    "valid_reference_flip_count": fmt_num(mean(valid_flip_counts), 3) if valid_flip_counts else "",
                    "valid_reference_flip_rate_pct": f"{mean(valid_flip_rates):.6f}" if valid_flip_rates else "",
                    "valid_reference_flip_rate_pct_std": f"{population_std(valid_flip_rates):.6f}" if valid_flip_rates else "",
                    "valid_reference_flip_rate_pct_stderr": f"{standard_error(valid_flip_rates):.6f}" if valid_flip_rates else "",
                    "elapsed_s": fmt_num(sum(finite_values([row.get("worker_elapsed_s") for row in flips])), 6),
                    "metrics_csv": str(FLIP_RESULTS / first["_source_run_id"] / MODEL_DIR_BY_TABLE[model] / "metrics.csv"),
                }
            )

    exec_records = [
        {
            "run_id": args.run_id,
            "final_source_run_id": args.final_source_run_id or args.run_id,
            "flip_source_run_ids": ",".join(args.flip_source_run_id or [args.run_id]),
            "domain": domain,
            "task": "MCR-Bench",
            "gpu": args.gpu,
            "model": EXEC_LABEL_BY_TABLE[model],
            "gpu_hours": f"{elapsed_by_model[model] / 3600.0:.6f}",
        }
        for model in MODEL_ORDER
    ]
    return mse_records, flip_records, exec_records


def write_domain_outputs(args: argparse.Namespace, domain: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mse_records, flip_records, exec_records = collect_domain(args, domain)
    tag = f"mcr-bench-{domain}-{args.gpu.lower()}"
    write_csv(
        MSE_FLIP_DATA / f"final-hidden-mse-{tag}.csv",
        [
            "source_run_id",
            "domain",
            "gpu",
            "model",
            "model_results_dir",
            "approach",
            "mode",
            "status",
            "mse",
            "mse_table_value",
            "compared_hidden_rows",
            "hidden_size",
            "metrics_csv",
        ],
        mse_records,
    )
    write_csv(
        MSE_FLIP_DATA / f"answer-flip-rate-{tag}.csv",
        [
            "run_id",
            "source_run_ids",
            "rounds",
            "domain",
            "gpu",
            "model",
            "configuration",
            "mode",
            "status",
            "docs",
            "answer_flip_count",
            "answer_flip_rate_pct",
            "answer_flip_rate_pct_std",
            "answer_flip_rate_pct_stderr",
            "answer_f_count",
            "max_length_count",
            "parse_fail_count",
            "valid_reference_rows",
            "valid_reference_flip_count",
            "valid_reference_flip_rate_pct",
            "valid_reference_flip_rate_pct_std",
            "valid_reference_flip_rate_pct_stderr",
            "elapsed_s",
            "metrics_csv",
        ],
        flip_records,
    )
    write_csv(
        EXEC_TIME_DATA / f"mcr-bench-{domain}-{args.gpu.lower()}.csv",
        ["run_id", "final_source_run_id", "flip_source_run_ids", "domain", "task", "gpu", "model", "gpu_hours"],
        exec_records,
    )
    (MSE_FLIP_DATA / f"final-hidden-mse-{tag}.json").write_text(
        json.dumps({"run_id": args.run_id, "domain": domain, "gpu": args.gpu, "records": mse_records}, indent=2) + "\n",
        encoding="utf-8",
    )
    (MSE_FLIP_DATA / f"answer-flip-rate-{tag}.json").write_text(
        json.dumps({"run_id": args.run_id, "domain": domain, "gpu": args.gpu, "records": flip_records}, indent=2) + "\n",
        encoding="utf-8",
    )
    (EXEC_TIME_DATA / f"mcr-bench-{domain}-{args.gpu.lower()}.json").write_text(
        json.dumps({"run_id": args.run_id, "domain": domain, "task": "MCR-Bench", "gpu": args.gpu, "records": exec_records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return mse_records, flip_records, exec_records


def render_mse_flip_table(mse_records: list[dict[str, Any]], flip_records: list[dict[str, Any]]) -> str:
    mse = {(row["approach"], row["model"]): row for row in mse_records}
    flip = {(row["configuration"], row["model"]): row for row in flip_records}
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Final MSE and answer flip rate across five models on MCR-Bench.}",
        r"\label{tab:mse_flip_rate}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|cc|cc|cc|cc|cc}",
        r"\toprule",
        r"\multirow{2}{*}{Approach} & \multicolumn{2}{c|}{Qwen3-8B} & \multicolumn{2}{c|}{Qwen3-14B} & \multicolumn{2}{c|}{Qwen3-32B} & \multicolumn{2}{c|}{Llama-8B} & \multicolumn{2}{c}{DeepSeek-14B} \\",
        r" & MSE & Flip Rate \% & MSE & Flip Rate \% & MSE & Flip Rate \% & MSE & Flip Rate \% & MSE & Flip Rate \% \\",
        r"\midrule",
    ]
    for prefill_mode, _, approach in MODE_ROWS:
        if prefill_mode == "__midrule__":
            lines.append(r"\midrule")
            continue
        cells = []
        for model in MODEL_ORDER:
            mse_row = mse.get((approach, model))
            flip_row = flip.get((approach, model))
            cells.append(mse_row.get("mse_table_value", "--") if mse_row else "--")
            cells.append(
                fmt_flip(
                    flip_row.get("answer_flip_rate_pct"),
                    flip_row.get("answer_flip_rate_pct_stderr") or flip_row.get("answer_flip_rate_pct_std"),
                )
                if flip_row
                else "$--$"
            )
        lines.append(f"{approach:<19} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{flushleft}",
            r"\footnotesize $^*$ \textit{Llama-8B} refers to Llama-3.1-8B-Instruct and \textit{DeepSeek-14B} refers to DeepSeek-Distill-Qwen-14B.",
            r"\end{flushleft}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def update_execution_time_tex(exec_records: list[dict[str, Any]], gpu: str) -> None:
    if not EXEC_TIME_TEX.exists():
        return
    values = {row["model"]: to_float(row["gpu_hours"]) for row in exec_records}
    ordered = ["Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "Llama-8B", "DS-14B"]
    if not all(model in values and math.isfinite(values[model]) for model in ordered):
        return
    replacement = "& {}   & {} \\\\".format(
        gpu,
        " & ".join(f"{values[model]:.1f}" for model in ordered),
    )
    lines = EXEC_TIME_TEX.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_mcr = False
    replaced = False
    for line in lines:
        if line.startswith("MCR-Bench "):
            in_mcr = True
            output.append(line)
            continue
        if in_mcr and line.strip().startswith(f"& {gpu}"):
            output.append(replacement)
            replaced = True
            in_mcr = False
            continue
        if in_mcr and line.strip().startswith("& ") and not replaced:
            output.append(replacement)
            replaced = True
            in_mcr = False
        output.append(line)
    EXEC_TIME_TEX.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    last_mse: list[dict[str, Any]] = []
    last_flip: list[dict[str, Any]] = []
    last_exec: list[dict[str, Any]] = []
    for domain in args.domains:
        last_mse, last_flip, last_exec = write_domain_outputs(args, domain)
        print(f"wrote MCR-Bench table data for domain={domain} gpu={args.gpu}")
    if not args.skip_tex and last_mse and last_flip:
        MSE_FLIP_TEX.write_text(render_mse_flip_table(last_mse, last_flip), encoding="utf-8")
        update_execution_time_tex(last_exec, args.gpu)
        print(f"wrote {MSE_FLIP_TEX}")
        print(f"updated {EXEC_TIME_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
