# SPDX-License-Identifier: Apache-2.0
"""Materialize validation reference hidden states from generated-token artifacts.

This avoids re-decoding validation generations: it filters an existing
generated_tokens.jsonl, rebuilds replay prompts, then runs LayerCast FP32
prefill to save the reference hidden_states expected by final-hidden-mse
replay. vLLM must already be prepared with VLLM_LAYERCAST.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FHM_DIR = ROOT / "tests" / "final-hidden-mse"
sys.path.insert(0, str(FHM_DIR))

spec = importlib.util.spec_from_file_location("final_hidden_mse_run", FHM_DIR / "run.py")
fhm = importlib.util.module_from_spec(spec)
sys.modules["final_hidden_mse_run"] = fhm
assert spec.loader is not None
spec.loader.exec_module(fhm)
common = fhm.common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--source-generation-dir", type=Path, required=True)
    parser.add_argument(
        "--output-run-id",
        default="medqa_validation_reference",
    )
    parser.add_argument("--subset-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-replay-tokens-per-doc", type=int, default=1024)
    parser.add_argument("--prefill-batch-size", type=int, default=1)
    parser.add_argument("--prefill-token-budget", type=int, default=2048)
    parser.add_argument("--max-model-len", default="2048")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--replay-top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-trace-batches", action="store_true")
    return parser.parse_args()


def select_models(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(common.MODEL_ORDER)
    if args.models:
        return list(args.models)
    return ["qwen3-8b"]


def summarize_rows(
    replay_rows: list[dict[str, Any]],
    model_label: str,
    source_path: Path,
    gen_dir: Path,
    started: float,
) -> dict[str, Any]:
    summary = fhm.replay_row_summary(replay_rows)
    return {
        "kind": "generate",
        "mode": fhm.GENERATION_SPEC.name,
        "model": model_label,
        "docs": len(replay_rows),
        "batches": 0,
        **summary,
        "elapsed_s": time.perf_counter() - started,
        "generated_source_jsonl": str(source_path),
        "generated_path": str(gen_dir / "generated_tokens.jsonl"),
        "replay_manifest_path": str(gen_dir / "replay_manifest.jsonl"),
        "replay_positions_path": str(gen_dir / "replay_positions.csv"),
        "output_dir": str(gen_dir),
        "materialized_reference_hidden_by": "layercast_fp32_prefill",
    }


def materialize_model(args: argparse.Namespace, model_arg: str) -> None:
    model_id = common.resolve_model(model_arg)
    model_label = common.model_label(model_arg)
    output_root = common.OUTPUT_ROOT / args.output_run_id
    results_root = common.RESULTS_ROOT / args.output_run_id / model_label
    gen_dir = output_root / "generation_artifacts" / model_label
    source_dir = args.source_generation_dir / model_label
    source_generated = source_dir / "generated_tokens.jsonl"
    if not source_generated.exists():
        raise FileNotFoundError(source_generated)

    started = time.perf_counter()
    generated_rows = common.read_jsonl(source_generated)
    generated_rows = fhm.filter_rows_by_subset_file(generated_rows, args.subset_file)
    if args.limit is not None and args.limit < len(generated_rows):
        generated_rows = generated_rows[: args.limit]
    if not generated_rows:
        raise RuntimeError(f"no generated rows selected for {model_label}")

    replay_rows = fhm.build_replay_rows_from_generated(
        generated_rows,
        replay_generated_token_limit=args.max_replay_tokens_per_doc,
        trim_self_circulation=True,
    )
    if not replay_rows:
        raise RuntimeError(f"no replay rows selected for {model_label}")

    if gen_dir.exists() and any(gen_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"output exists; pass --resume: {gen_dir}")
    gen_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    common.write_jsonl(gen_dir / "generated_tokens.jsonl", generated_rows)
    common.write_jsonl(gen_dir / "replay_manifest.jsonl", replay_rows)
    common.write_jsonl(gen_dir / "generation_batches.jsonl", [])
    common.write_replay_positions(gen_dir / "replay_positions.csv", replay_rows)
    common.write_json(
        gen_dir / "worker_result.json",
        summarize_rows(replay_rows, model_label, source_generated, gen_dir, started),
    )

    if common.hidden_outputs_complete(gen_dir / "hidden_states", replay_rows):
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{model_label}: hidden_states already complete",
            flush=True,
        )
        return

    batch_manifest = common.build_batch_manifest(
        replay_rows,
        batch_size=int(args.prefill_batch_size),
        token_budget=int(args.prefill_token_budget),
    )
    replay_manifest_path = gen_dir / "replay_manifest.jsonl"
    batch_manifest_path = gen_dir / "batch_manifest.jsonl"
    common.write_jsonl(batch_manifest_path, batch_manifest)

    mode = fhm.spec_mode_dict(fhm.GENERATION_SPEC, gen_dir)
    config = {
        "model_arg": model_arg,
        "model_id": model_id,
        "model_label": model_label,
        "mode": mode,
        "mode_dir": str(gen_dir),
        "replay_manifest_path": str(replay_manifest_path),
        "batch_manifest_path": str(batch_manifest_path),
        "result_path": str(results_root / "materialize_reference_hidden_result.json"),
        "seed": int(args.seed),
        "max_model_len": str(args.max_model_len),
        "prefill_token_budget": int(args.prefill_token_budget),
        "prefill_batch_size": int(args.prefill_batch_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "trust_remote_code": bool(args.trust_remote_code),
        "replay_top_k": int(args.replay_top_k),
        "keep_trace_batches": bool(args.keep_trace_batches),
    }
    common.write_json(results_root / "materialize_reference_hidden_config.json", config)
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] materialize {model_label}: "
        f"docs={len(replay_rows)} batches={len(batch_manifest)} "
        f"max_replay={max(int(r['replay_token_count']) for r in replay_rows)}",
        flush=True,
    )
    fhm.run_worker_process(
        kind="prefill",
        config=config,
        mode=mode,
        config_dir=results_root / "worker_configs",
    )
    if not common.hidden_outputs_complete(gen_dir / "hidden_states", replay_rows):
        raise RuntimeError(
            f"hidden_states incomplete after materialization: {gen_dir / 'hidden_states'}"
        )


def main() -> int:
    args = parse_args()
    active = common.active_patch_status()
    if "Active patch: VLLM_LAYERCAST" not in active:
        raise RuntimeError("VLLM_LAYERCAST must be active before running this script")
    for model in select_models(args):
        materialize_model(args, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
