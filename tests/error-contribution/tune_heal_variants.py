# SPDX-License-Identifier: Apache-2.0
"""Tune HEAL++ and HEAL-FULL+ layer downgrade sets from attribution CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
FINAL_HIDDEN_DIR = ROOT / "tests" / "final-hidden-mse"
if str(FINAL_HIDDEN_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_HIDDEN_DIR))
import common as fhm_common  # noqa: E402

RESULTS_ROOT = TEST_DIR / "results"
DEFAULT_CONTRIBUTION_RUN_ID = "error_contribution_calibration32"
DEFAULT_OUTPUT_RUN_ID = "heal_variant_tuning"
DEFAULT_VALIDATION_RUN_PREFIX = "heal_variant_tuning_validation"
DEFAULT_VALIDATION_SUBSET = TEST_DIR / "subsets" / "medqa_validation8_seed20260528.jsonl"
DEFAULT_REFERENCE_GENERATION_DIR = (
    ROOT
    / "output"
    / "final-hidden-mse"
    / "medqa_ref"
    / "generation_artifacts"
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def path_from_cli(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
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


def model_args_from_cli(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(fhm_common.MODEL_ORDER)
    if args.models:
        return args.models
    return [args.model]


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def contribution_csv_path(run_id: str, model_label: str) -> Path:
    return RESULTS_ROOT / run_id / model_label / "heal_plus_recursive_truncation_contribution_ratios.csv"


def load_layer_rankings(*, contribution_run_id: str, model_label: str) -> dict[str, Any]:
    csv_path = contribution_csv_path(contribution_run_id, model_label)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing HEAL+ contribution CSV: {csv_path}")
    rows = read_csv(csv_path)
    if not rows:
        raise RuntimeError(f"empty contribution CSV: {csv_path}")
    layer_rows: list[dict[str, Any]] = []
    for row in rows:
        layer = int(row["layer"])
        attn = _float(row, "attention_compute_final_share")
        kv = _float(row, "kv_final_share")
        mlp = _float(row, "mlp_compute_final_share")
        layer_rows.append(
            {
                "layer": layer,
                "attention_compute_final_share": attn,
                "kv_final_share": kv,
                "attention_plus_kv_final_share": attn + kv,
                "mlp_compute_final_share": mlp,
            }
        )
    attention_kv_order = [
        row["layer"]
        for row in sorted(
            layer_rows,
            key=lambda row: (row["attention_plus_kv_final_share"], row["layer"]),
        )
    ]
    mlp_order = [
        row["layer"]
        for row in sorted(
            layer_rows,
            key=lambda row: (row["mlp_compute_final_share"], row["layer"]),
        )
    ]
    layer_index_order = [
        row["layer"] for row in sorted(layer_rows, key=lambda row: row["layer"])
    ]
    return {
        "csv_path": csv_path,
        "rows": layer_rows,
        "layer_index_order": layer_index_order,
        "attention_kv_order": attention_kv_order,
        "mlp_order": mlp_order,
    }


def layer_list(layers: list[int]) -> str:
    return ",".join(str(layer) for layer in layers)


def final_hidden_metrics_path(run_id: str, model_label: str) -> Path:
    return ROOT / "tests" / "final-hidden-mse" / "results" / run_id / model_label / "metrics.json"


def load_mode_metric(*, run_id: str, model_label: str, mode: str) -> dict[str, Any]:
    path = final_hidden_metrics_path(run_id, model_label)
    if not path.exists():
        raise FileNotFoundError(f"missing validation metrics: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("metrics", []):
        if row.get("mode") == mode and row.get("status") == "ok":
            return row
    raise RuntimeError(f"metrics for mode={mode} not found in {path}")


def validation_command(
    *,
    args: argparse.Namespace,
    model_arg: str,
    run_id: str,
    mode: str,
    skip_qkv_layers: str | None = None,
    skip_int8_layers: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tests" / "final-hidden-mse" / "run.py"),
        "--model",
        model_arg,
        "--stage",
        "replay",
        "--generation-dir",
        str(path_from_cli(args.reference_generation_dir)),
        "--subset-file",
        str(path_from_cli(args.validation_subset_file)),
        "--limit",
        str(args.validation_limit),
        "--replay-generated-token-limit",
        str(args.max_replay_tokens_per_doc),
        "--prefill-modes",
        mode,
        "--run-id",
        run_id,
        "--prefill-batch-size",
        str(args.prefill_batch_size),
        "--prefill-token-budget",
        str(args.prefill_token_budget),
        "--replay-gpu-memory-utilization",
        str(args.replay_gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--skip-gate",
        "--resume",
    ]
    if skip_qkv_layers:
        cmd.extend(["--heal-skip-qkv-quant-layers", skip_qkv_layers])
    if skip_int8_layers:
        cmd.extend(["--heal-skip-fp16-comp-layers", skip_int8_layers])
    if args.reuse_prepared_vllm_heal:
        cmd.extend(["--prepared-vllm-version", "VLLM_HEAL"])
    return cmd


def run_validation(
    *,
    args: argparse.Namespace,
    model_arg: str,
    model_label: str,
    run_id: str,
    mode: str,
    skip_qkv_layers: str | None = None,
    skip_int8_layers: str | None = None,
) -> dict[str, Any]:
    cmd = validation_command(
        args=args,
        model_arg=model_arg,
        run_id=run_id,
        mode=mode,
        skip_qkv_layers=skip_qkv_layers,
        skip_int8_layers=skip_int8_layers,
    )
    if args.dry_run:
        log("dry-run: " + " ".join(cmd))
        return {
            "run_id": run_id,
            "mode": mode,
            "median_hidden_mse": 0.0,
            "mse": 0.0,
            "compared_hidden_rows": 0,
            "dry_run": True,
        }
    log(f"validation start model={model_label} run_id={run_id} mode={mode}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    metric = dict(load_mode_metric(run_id=run_id, model_label=model_label, mode=mode))
    metric["run_id"] = run_id
    metric["skip_qkv_layers"] = skip_qkv_layers or ""
    metric["skip_int8_layers"] = skip_int8_layers or ""
    log(
        f"validation done model={model_label} run_id={run_id} "
        f"mode={mode} median={float(metric['median_hidden_mse']):.6g}"
    )
    return metric


def evaluate_k(
    *,
    args: argparse.Namespace,
    model_arg: str,
    model_label: str,
    variant: str,
    orders: dict[str, list[int]],
    baseline_median: float,
    threshold: float,
    allowed_mse_ratio: float,
    mode: str,
    k: int,
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    mid_evaluations: list[dict[str, Any]] = []
    metric_cache: dict[tuple[int, ...], dict[str, Any]] = {}
    for order_name, order in orders.items():
        selected = order[:k]
        selected_text = layer_list(selected)
        selected_key = tuple(sorted(selected))
        if selected_key in metric_cache:
            metric = dict(metric_cache[selected_key])
        else:
            run_id = (
                f"{args.validation_run_prefix}_{model_label}_"
                f"{variant}_{order_name}_k{k:02d}"
            )
            metric = run_validation(
                args=args,
                model_arg=model_arg,
                model_label=model_label,
                run_id=run_id,
                mode=mode,
                skip_qkv_layers=selected_text if variant == "heal_plus_plus" else None,
                skip_int8_layers=selected_text if variant == "heal_full_plus" else None,
            )
            metric_cache[selected_key] = dict(metric)
        median = float(metric.get("median_hidden_mse") or 0.0)
        ok = median <= threshold if baseline_median > 0.0 else True
        evaluation = {
            "variant": variant,
            "order": order_name,
            "k": k,
            "layers": selected_text,
            "median_hidden_mse": median,
            "baseline_median_hidden_mse": baseline_median,
            "allowed_mse_ratio": float(allowed_mse_ratio),
            "mse_ratio": median / baseline_median if baseline_median > 0.0 else 0.0,
            "ok": ok,
            "run_id": metric.get("run_id", ""),
        }
        evaluations.append(evaluation)
        mid_evaluations.append(evaluation)
    passing = [row for row in mid_evaluations if row["ok"]]
    return {
        "k": k,
        "ok": bool(passing),
        "best_metric": min(
            passing,
            key=lambda row: (float(row["median_hidden_mse"]), str(row["order"])),
        ) if passing else None,
        "evaluations": mid_evaluations,
    }


def run_binary_search(
    *,
    args: argparse.Namespace,
    model_arg: str,
    model_label: str,
    variant: str,
    orders: dict[str, list[int]],
    baseline_median: float,
    allowed_mse_ratio: float,
    mode: str,
) -> dict[str, Any]:
    if not orders:
        raise ValueError("at least one downgrade order is required")
    max_k = min(len(order) for order in orders.values())
    threshold = baseline_median * float(allowed_mse_ratio)
    if max_k <= 0:
        return {
            "variant": variant,
            "mode": mode,
            "orders": orders,
            "search_strategy": "exponential_from_4_then_binary_interval",
            "baseline_median_hidden_mse": baseline_median,
            "threshold_median_hidden_mse": threshold,
            "allowed_mse_ratio": float(allowed_mse_ratio),
            "best_k": 0,
            "best_order": "",
            "best_layers": [],
            "best_layers_csv": "",
            "best_evaluation": None,
            "evaluations": [],
        }
    evaluations: list[dict[str, Any]] = []
    checked: dict[int, dict[str, Any]] = {}

    def check(k: int) -> dict[str, Any]:
        k = max(0, min(int(k), max_k))
        if k not in checked:
            checked[k] = evaluate_k(
                args=args,
                model_arg=model_arg,
                model_label=model_label,
                variant=variant,
                orders=orders,
                baseline_median=baseline_median,
                threshold=threshold,
                allowed_mse_ratio=allowed_mse_ratio,
                mode=mode,
                k=k,
                evaluations=evaluations,
            )
        return checked[k]

    best_k = 0
    best_metric: dict[str, Any] | None = None
    first_k = min(4, max_k)
    probe = check(first_k)
    if probe["ok"]:
        best_k = first_k
        best_metric = probe["best_metric"]
        lo = first_k
        hi = max_k + 1
        next_k = first_k * 2
        while next_k <= max_k:
            probe = check(next_k)
            if probe["ok"]:
                best_k = next_k
                best_metric = probe["best_metric"]
                lo = next_k
                next_k *= 2
            else:
                hi = next_k
                break
        else:
            hi = max_k + 1
        # Search the open interval (lo, hi). If every power-of-two probe passed,
        # this finishes the tail between the largest probe and max_k.
        left = lo + 1
        right = hi - 1
    else:
        left = 0
        right = first_k - 1

    while left <= right:
        mid = (left + right) // 2
        probe = check(mid)
        if probe["ok"]:
            best_k = mid
            best_metric = probe["best_metric"]
            left = mid + 1
        else:
            right = mid - 1

    best_layers_csv = str(best_metric.get("layers", "")) if best_metric else ""
    best_layers = [int(layer) for layer in best_layers_csv.split(",") if layer]
    return {
        "variant": variant,
        "mode": mode,
        "orders": orders,
        "search_strategy": "exponential_from_4_then_binary_interval",
        "baseline_median_hidden_mse": baseline_median,
        "threshold_median_hidden_mse": threshold,
        "allowed_mse_ratio": float(allowed_mse_ratio),
        "best_k": best_k,
        "best_order": best_metric.get("order", "") if best_metric else "",
        "best_layers": best_layers,
        "best_layers_csv": best_layers_csv,
        "best_evaluation": best_metric,
        "evaluations": evaluations,
    }


def tune_model(args: argparse.Namespace, model_arg: str) -> dict[str, Any]:
    model_label = fhm_common.model_label(model_arg)
    variants = set(args.variants)
    rankings = load_layer_rankings(
        contribution_run_id=args.contribution_run_id,
        model_label=model_label,
    )
    out_dir = RESULTS_ROOT / args.output_run_id / model_label
    out_dir.mkdir(parents=True, exist_ok=True)

    heal_plus_baseline: dict[str, Any] | None = None
    heal_full_baseline: dict[str, Any] | None = None
    heal_plus_plus: dict[str, Any] | None = None
    heal_full_plus: dict[str, Any] | None = None

    if "heal_plus_plus" in variants:
        heal_plus_baseline_run_id = f"{args.validation_run_prefix}_{model_label}_heal_plus_baseline"
        heal_plus_baseline = run_validation(
            args=args,
            model_arg=model_arg,
            model_label=model_label,
            run_id=heal_plus_baseline_run_id,
            mode="heal_plus_prefill",
        )
        heal_plus_median = float(heal_plus_baseline.get("median_hidden_mse") or 0.0)
        heal_plus_plus = run_binary_search(
            args=args,
            model_arg=model_arg,
            model_label=model_label,
            variant="heal_plus_plus",
            orders={
                "prefix": rankings["layer_index_order"],
                "calibration_low": rankings["attention_kv_order"],
            },
            baseline_median=heal_plus_median,
            allowed_mse_ratio=float(args.heal_plus_plus_allowed_mse_ratio),
            mode="heal_plus_prefill",
        )

    if "heal_full_plus" in variants:
        heal_full_baseline_run_id = f"{args.validation_run_prefix}_{model_label}_heal_full_baseline"
        heal_full_baseline = run_validation(
            args=args,
            model_arg=model_arg,
            model_label=model_label,
            run_id=heal_full_baseline_run_id,
            mode="heal_full_prefill",
        )
        heal_full_median = float(heal_full_baseline.get("median_hidden_mse") or 0.0)
        heal_full_plus = run_binary_search(
            args=args,
            model_arg=model_arg,
            model_label=model_label,
            variant="heal_full_plus",
            orders={
                "prefix": rankings["layer_index_order"],
                "calibration_low": rankings["mlp_order"],
            },
            baseline_median=heal_full_median,
            allowed_mse_ratio=float(args.heal_full_plus_allowed_mse_ratio),
            mode="heal_full_prefill",
        )

    evaluation_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    for result in (heal_plus_plus, heal_full_plus):
        if result is None:
            continue
        for row in result["evaluations"]:
            evaluation_rows.append({"model": model_label, **row})
        for order_name, order in result["orders"].items():
            for rank, layer in enumerate(order):
                order_rows.append(
                    {
                        "model": model_label,
                        "variant": result["variant"],
                        "order": order_name,
                        "rank": rank,
                        "layer": layer,
                    }
                )
    if evaluation_rows:
        write_csv(out_dir / "validation_evaluations.csv", evaluation_rows)
    if order_rows:
        write_csv(out_dir / "selection_orders.csv", order_rows)
    write_csv(out_dir / "contribution_layer_ranking.csv", rankings["rows"])
    result = {
        "model_arg": model_arg,
        "model_label": model_label,
        "contribution_run_id": args.contribution_run_id,
        "contribution_csv": rankings["csv_path"],
        "validation_subset_file": path_from_cli(args.validation_subset_file),
        "max_replay_tokens_per_doc": int(args.max_replay_tokens_per_doc),
        "variant_allowed_mse_ratios": {
            "heal_plus_plus": float(args.heal_plus_plus_allowed_mse_ratio),
            "heal_full_plus": float(args.heal_full_plus_allowed_mse_ratio),
        },
        "heal_plus_baseline": heal_plus_baseline,
        "heal_full_baseline": heal_full_baseline,
        "heal_plus_plus": heal_plus_plus,
        "heal_full_plus": heal_full_plus,
    }
    write_json(out_dir / "heal_variant_tuning.json", result)
    env_payload: dict[str, Any] = {}
    if heal_plus_plus is not None:
        env_payload["HEAL++"] = {
            "selection_order": heal_plus_plus["best_order"],
            "HEAL_MODE": "plus",
            "HEAL_SKIP_QKV_QUANT_LAYERS": heal_plus_plus["best_layers_csv"],
        }
    if heal_full_plus is not None:
        env_payload["HEAL-FULL+"] = {
            "selection_order": heal_full_plus["best_order"],
            "table_label": "HEAL-FULL",
            "HEAL_MODE": "full",
            "HEAL_FP16_GEMM_COMPENSATION": "1",
            "HEAL_SKIP_FP16_COMP_LAYERS": heal_full_plus["best_layers_csv"],
            "HEAL_SKIP_INT8_COMP_LAYERS": heal_full_plus["best_layers_csv"],
        }
    write_json(out_dir / "heal_variant_env.json", env_payload)
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--contribution-run-id", default=DEFAULT_CONTRIBUTION_RUN_ID)
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--validation-run-prefix", default=DEFAULT_VALIDATION_RUN_PREFIX)
    parser.add_argument("--reference-generation-dir", type=Path, default=DEFAULT_REFERENCE_GENERATION_DIR)
    parser.add_argument("--validation-subset-file", type=Path, default=DEFAULT_VALIDATION_SUBSET)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--max-replay-tokens-per-doc", type=int, default=1024)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("heal_plus_plus", "heal_full_plus"),
        default=("heal_plus_plus", "heal_full_plus"),
        help="Tuned variants to evaluate. Use heal_plus_plus alone to refresh only HEAL++.",
    )
    parser.add_argument("--heal-plus-plus-allowed-mse-ratio", type=float, default=None)
    parser.add_argument("--heal-full-plus-allowed-mse-ratio", type=float, default=None)
    parser.add_argument("--prefill-batch-size", type=int, default=2)
    parser.add_argument("--prefill-token-budget", type=int, default=6000)
    parser.add_argument("--replay-gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--max-model-len", default="auto")
    parser.add_argument(
        "--reuse-prepared-vllm-heal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepare VLLM_HEAL once and reuse it for all validation subprocesses.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.heal_plus_plus_allowed_mse_ratio is None:
        args.heal_plus_plus_allowed_mse_ratio = 1.20
    if args.heal_full_plus_allowed_mse_ratio is None:
        args.heal_full_plus_allowed_mse_ratio = 1.50
    return args


def main() -> int:
    args = parse_args()
    models = model_args_from_cli(args)
    results = []
    prepared_heal = False
    try:
        if args.reuse_prepared_vllm_heal and not args.dry_run:
            log("prepare shared VLLM_HEAL")
            fhm_common.ensure_vllm_clean()
            fhm_common.run_patch_command("prepare-version", "VLLM_HEAL")
            prepared_heal = True
        for index, model_arg in enumerate(models):
            log(f"tuning model {index + 1}/{len(models)}: {fhm_common.model_label(model_arg)}")
            results.append(tune_model(args, model_arg))
    finally:
        if prepared_heal:
            log("revert shared VLLM_HEAL")
            fhm_common.run_patch_command("revert-version")
            fhm_common.ensure_vllm_clean()
    summary_rows = []
    for result in results:
        row: dict[str, Any] = {"model": result["model_label"]}
        heal_plus_plus = result.get("heal_plus_plus")
        if heal_plus_plus is not None:
            row.update(
                {
                    "heal_plus_plus_k": heal_plus_plus["best_k"],
                    "heal_plus_plus_order": heal_plus_plus["best_order"],
                    "heal_plus_plus_layers": heal_plus_plus["best_layers_csv"],
                    "heal_plus_plus_allowed_mse_ratio": heal_plus_plus["allowed_mse_ratio"],
                    "heal_plus_plus_baseline_median_hidden_mse": heal_plus_plus["baseline_median_hidden_mse"],
                }
            )
        heal_full_plus = result.get("heal_full_plus")
        if heal_full_plus is not None:
            row.update(
                {
                    "heal_full_plus_k": heal_full_plus["best_k"],
                    "heal_full_plus_order": heal_full_plus["best_order"],
                    "heal_full_plus_layers": heal_full_plus["best_layers_csv"],
                    "heal_full_plus_allowed_mse_ratio": heal_full_plus["allowed_mse_ratio"],
                    "heal_full_plus_baseline_median_hidden_mse": heal_full_plus["baseline_median_hidden_mse"],
                }
            )
        summary_rows.append(row)
    out_dir = RESULTS_ROOT / args.output_run_id
    write_csv(out_dir / "heal_variant_tuning_summary.csv", summary_rows)
    write_json(out_dir / "heal_variant_tuning_summary.json", {"models": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
