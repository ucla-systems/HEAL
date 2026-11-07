# SPDX-License-Identifier: Apache-2.0
"""MCR-Bench adapter for final-hidden MSE and flip-rate-analysis runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MCR_ROOT = ROOT / "MCR-Bench"
FINAL_HIDDEN_RUNNER = ROOT / "tests" / "final-hidden-mse" / "run.py"
FLIP_RATE_RUNNER = ROOT / "tests" / "flip-rate-analysis" / "run.py"

DOMAIN_DATASETS = {
    "all": MCR_ROOT / "combined_mcr_bench.jsonl",
    "medical": MCR_ROOT / "med_data" / "combined_med_data.jsonl",
    "legal": MCR_ROOT / "legal_data" / "combined_legal_data.jsonl",
    "finance": MCR_ROOT / "financial_data" / "combined_financial_data.jsonl",
}

# MCR-Bench table scope mirrors the reported precision and HEAL configurations.
# The untuned HEAL-Full baseline remains available through explicit
# --prefill-modes/--modes but is not part of the default table workflow.
TABLE_PREFILL_MODES = [
    "original_bf16_prefill",
    "original_fp16_prefill",
    "layercast_tf32_prefill",
    "layercast_fp32_prefill",
    "heal_base_prefill",
    "heal_plus_prefill",
    "heal_plus_plus_prefill",
    "heal_full_plus_prefill",
]
TABLE_FLIP_MODES = [
    "original_bf16_generate",
    "original_fp16_generate",
    "layercast_tf32_generate",
    "layercast_fp32_generate",
    "heal_base_generate",
    "heal_plus_generate",
    "heal_plus_plus_generate",
    "heal_full_plus_generate",
]


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def model_args(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return ["--all-models"]
    if args.models:
        return ["--models", *args.models]
    if args.model:
        return ["--model", args.model]
    return []


def common_precision_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.heal_skip_qkv_quant_layers:
        out += ["--heal-skip-qkv-quant-layers", args.heal_skip_qkv_quant_layers]
    if args.heal_skip_fp16_comp_layers:
        out += ["--heal-skip-fp16-comp-layers", args.heal_skip_fp16_comp_layers]
    if args.heal_skip_int8_comp_layers:
        out += ["--heal-skip-int8-comp-layers", args.heal_skip_int8_comp_layers]
    if args.heal_policy_json:
        out += ["--heal-policy-json", str(args.heal_policy_json)]
    return out


def domain_scoped_dir(base: Path | None, args: argparse.Namespace, domain: str) -> Path | None:
    if base is None:
        return None
    if len(args.domains) > 1:
        return base / domain
    return base


def generation_dir_for_domain(args: argparse.Namespace, domain: str) -> Path | None:
    return domain_scoped_dir(args.generation_dir, args, domain)


def reference_generation_dir_for_domain(args: argparse.Namespace, domain: str) -> Path | None:
    if args.reference_generation_dir is not None:
        return domain_scoped_dir(args.reference_generation_dir, args, domain)
    return generation_dir_for_domain(args, domain)


def final_hidden_cmd(args: argparse.Namespace, *, domain: str, run_id: str) -> list[str]:
    cmd = [
        sys.executable,
        str(FINAL_HIDDEN_RUNNER),
        "--run-id",
        f"{run_id}_{domain}_final_hidden_mse",
        "--subset-file",
        str(DOMAIN_DATASETS[domain]),
        "--dataset-kind",
        "mcr-bench",
        "--mcr-max-source-chars",
        str(args.mcr_max_source_chars),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    generation_dir = generation_dir_for_domain(args, domain)
    if generation_dir is not None:
        cmd += ["--generation-dir", str(generation_dir)]
    cmd += [
        "--generation-token-budget",
        str(args.max_new_tokens),
        "--replay-generated-token-limit",
        str(args.max_new_tokens),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    cmd += ["--prefill-modes", *(args.prefill_modes or TABLE_PREFILL_MODES)]
    if args.stage:
        cmd += ["--stage", args.stage]
    if args.resume:
        cmd += ["--resume"]
    return cmd + model_args(args) + common_precision_args(args)


def flip_rate_cmd(args: argparse.Namespace, *, domain: str, run_id: str) -> list[str]:
    reference_dir = reference_generation_dir_for_domain(args, domain)
    if reference_dir is None:
        reference_dir = (
            ROOT
            / "output"
            / "final-hidden-mse"
            / f"{run_id}_{domain}_final_hidden_mse"
            / "generation_artifacts"
        )
    cmd = [
        sys.executable,
        str(FLIP_RATE_RUNNER),
        "--run-id",
        f"{run_id}_{domain}_flip_rate",
        "--reference-generation-dir",
        str(reference_dir),
        "--answer-parser",
        "mcr-bench",
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    cmd += ["--generation-token-budget", str(args.max_new_tokens)]
    cmd += ["--generation-batch-size", str(args.generation_batch_size)]
    cmd += ["--generation-batch-mode", args.generation_batch_mode]
    cmd += ["--generation-gpu-memory-utilization", str(args.generation_gpu_memory_utilization)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    cmd += ["--modes", *(args.modes or TABLE_FLIP_MODES)]
    if args.resume:
        cmd += ["--resume"]
    return cmd + model_args(args) + common_precision_args(args)


def run_or_print(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final-hidden MSE and flip-rate-analysis on MCR-Bench."
    )
    parser.add_argument(
        "--task",
        choices=["all", "final-hidden-mse", "flip-rate-analysis"],
        default="all",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(DOMAIN_DATASETS),
        default=["all"],
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stage", choices=["all", "generate", "replay"])
    parser.add_argument("--generation-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--prefill-modes",
        "--final-hidden-modes",
        nargs="+",
        dest="prefill_modes",
    )
    parser.add_argument("--modes", "--flip-modes", nargs="+", dest="modes")
    parser.add_argument("--reference-generation-dir", type=Path)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument(
        "--generation-batch-mode",
        choices=["chunked", "full", "full-reverse"],
        default="chunked",
    )
    parser.add_argument("--generation-gpu-memory-utilization", type=float, default=0.93)
    parser.add_argument("--mcr-max-source-chars", type=int, default=12000)
    parser.add_argument("--heal-skip-qkv-quant-layers")
    parser.add_argument("--heal-skip-fp16-comp-layers")
    parser.add_argument("--heal-skip-int8-comp-layers")
    parser.add_argument("--heal-policy-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"mcr_bench_{utc_timestamp()}"
    missing = [domain for domain in args.domains if not DOMAIN_DATASETS[domain].exists()]
    if missing:
        raise FileNotFoundError(
            "missing MCR-Bench dataset(s): "
            + ", ".join(str(DOMAIN_DATASETS[domain]) for domain in missing)
        )
    for domain in args.domains:
        if args.task in {"all", "final-hidden-mse"}:
            run_or_print(final_hidden_cmd(args, domain=domain, run_id=run_id), dry_run=args.dry_run)
        if args.task in {"all", "flip-rate-analysis"}:
            run_or_print(flip_rate_cmd(args, domain=domain, run_id=run_id), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
