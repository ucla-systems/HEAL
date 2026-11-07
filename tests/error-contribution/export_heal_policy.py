# SPDX-License-Identifier: Apache-2.0
"""Export tuned HEAL++/HEAL-FULL+ layer settings as a portable policy JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
FINAL_HIDDEN_DIR = ROOT / "tests" / "final-hidden-mse"
if str(FINAL_HIDDEN_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_HIDDEN_DIR))
import common as fhm_common  # noqa: E402

RESULTS_ROOT = TEST_DIR / "results"
POLICY_ROOT = TEST_DIR / "policies"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def model_results_from_run(run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "heal_variant_tuning_summary.json"
    if summary_path.exists():
        payload = read_json(summary_path)
        return [row for row in payload.get("models", []) if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*/heal_variant_tuning.json")):
        rows.append(read_json(path))
    if not rows:
        raise FileNotFoundError(f"no HEAL tuning JSON found under {run_dir}")
    return rows


def env_entry(result: dict[str, Any]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    heal_plus_plus = result.get("heal_plus_plus")
    if isinstance(heal_plus_plus, dict):
        variants["HEAL++"] = {
            "selection_order": heal_plus_plus.get("best_order", ""),
            "HEAL_MODE": "plus",
            "HEAL_SKIP_QKV_QUANT_LAYERS": heal_plus_plus.get("best_layers_csv", ""),
            "best_k": heal_plus_plus.get("best_k", 0),
            "allowed_mse_ratio": heal_plus_plus.get("allowed_mse_ratio"),
        }
    heal_full_plus = result.get("heal_full_plus")
    if isinstance(heal_full_plus, dict):
        variants["HEAL-FULL+"] = {
            "selection_order": heal_full_plus.get("best_order", ""),
            "table_label": "HEAL-FULL",
            "HEAL_MODE": "full",
            "HEAL_FP16_GEMM_COMPENSATION": "1",
            "HEAL_SKIP_FP16_COMP_LAYERS": heal_full_plus.get("best_layers_csv", ""),
            "HEAL_SKIP_INT8_COMP_LAYERS": heal_full_plus.get("best_layers_csv", ""),
            "best_k": heal_full_plus.get("best_k", 0),
            "allowed_mse_ratio": heal_full_plus.get("allowed_mse_ratio"),
        }
    return variants

def build_policy(*, run_id: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for result in results:
        model_arg = str(result.get("model_arg") or "")
        model_label = str(result.get("model_label") or result.get("model") or "")
        if not model_arg and model_label:
            model_arg = model_label
        if not model_label:
            model_label = fhm_common.model_label(model_arg)
        model_id = fhm_common.resolve_model(model_arg)
        models[model_label] = {
            "model_arg": model_arg,
            "model_id": model_id,
            "model_label": model_label,
            "aliases": sorted({model_arg, model_id, model_label, model_id.replace("/", "-")}),
            "contribution_run_id": result.get("contribution_run_id"),
            "allowed_mse_ratio": result.get("allowed_mse_ratio"),
            "variant_allowed_mse_ratios": result.get("variant_allowed_mse_ratios", {}),
            "validation_subset_file": str(result.get("validation_subset_file", "")),
            "variants": env_entry(result),
        }
    return {
        "policy_version": 1,
        "source_tuning_run_id": run_id,
        "models": models,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuning-run-id",
        default="heal_variant_tuning",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = RESULTS_ROOT / args.tuning_run_id
    policy = build_policy(
        run_id=args.tuning_run_id,
        results=model_results_from_run(run_dir),
    )
    output = args.output or POLICY_ROOT / f"{args.tuning_run_id}.json"
    write_json(output, policy)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
