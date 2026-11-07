# SPDX-License-Identifier: Apache-2.0
"""Run vLLM's official serving benchmark for ShareGPT TPOT metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
VLLM_CWD = ROOT / "vllm"

PERFORMANCE_VERSIONS = (
    "VLLM_ORIGINAL",
    "VLLM_LAYERCAST",
    "VLLM_HEAL",
    "VLLM_SUMMARIZED",
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


def output_path(*parts: str) -> Path:
    path = OUTPUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class VllmLogMetrics:
    MODEL_RE = re.compile(
        r"Model loading took (?P<memory>[0-9.]+) GiB(?: memory)? and "
        r"(?P<seconds>[0-9.]+) seconds"
    )
    KV_MEMORY_RE = re.compile(r"Available KV cache memory: (?P<memory>[0-9.]+) GiB")
    KV_SIZE_RE = re.compile(r"GPU KV cache size: (?P<tokens>[0-9,]+) tokens")
    CONCURRENCY_RE = re.compile(
        r"Maximum concurrency for (?P<tokens>[0-9,]+) tokens per request: "
        r"(?P<concurrency>[0-9.]+)x"
    )
    GRAPH_CAPTURE_RE = re.compile(
        r"Graph capturing finished .* took (?P<memory>-?[0-9.]+) GiB"
    )

    def __init__(self) -> None:
        self.model_loading_memory_gib: float | None = None
        self.model_loading_seconds: float | None = None
        self.available_kv_cache_memory_gib: float | None = None
        self.gpu_kv_cache_size_tokens: int | None = None
        self.max_concurrency_tokens_per_request: int | None = None
        self.max_concurrency: float | None = None
        self.graph_capture_memory_gib: float | None = None

    @staticmethod
    def _int_with_commas(value: str) -> int:
        return int(value.replace(",", ""))

    def record_line(self, message: str) -> None:
        if match := self.MODEL_RE.search(message):
            self.model_loading_memory_gib = float(match.group("memory"))
            self.model_loading_seconds = float(match.group("seconds"))
        if match := self.KV_MEMORY_RE.search(message):
            self.available_kv_cache_memory_gib = float(match.group("memory"))
        if match := self.KV_SIZE_RE.search(message):
            self.gpu_kv_cache_size_tokens = self._int_with_commas(match.group("tokens"))
        if match := self.CONCURRENCY_RE.search(message):
            self.max_concurrency_tokens_per_request = self._int_with_commas(
                match.group("tokens")
            )
            self.max_concurrency = float(match.group("concurrency"))
        if match := self.GRAPH_CAPTURE_RE.search(message):
            self.graph_capture_memory_gib = float(match.group("memory"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_loading_memory_gib": self.model_loading_memory_gib,
            "model_loading_seconds": self.model_loading_seconds,
            "available_kv_cache_memory_gib": self.available_kv_cache_memory_gib,
            "gpu_kv_cache_size_tokens": self.gpu_kv_cache_size_tokens,
            "max_concurrency_tokens_per_request": self.max_concurrency_tokens_per_request,
            "max_concurrency": self.max_concurrency,
            "graph_capture_memory_gib": self.graph_capture_memory_gib,
        }


def parse_vllm_log_memory(lines: list[str]) -> dict[str, Any]:
    metrics = VllmLogMetrics()
    for line in lines:
        metrics.record_line(line)
    return metrics.as_dict()


def git_rev_parse(ref: str = "HEAD") -> str:
    return subprocess.check_output(
        ["git", "-C", str(VLLM_CWD), "rev-parse", ref],
        text=True,
    ).strip()


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _normalize_heal_mode(heal_mode: str | None) -> str:
    mode = (heal_mode or "full").strip().lower()
    if mode not in {"base", "plus", "full"}:
        raise ValueError("HEAL mode must be one of: base, plus, full.")
    return mode


def attention_config_for_run(
    version: str,
    dtype: str,
    *,
    heal_mode: str | None = None,
) -> dict[str, str] | None:
    normalized_heal_mode = (
        _normalize_heal_mode(heal_mode) if version == "VLLM_HEAL" else None
    )
    if version == "VLLM_LAYERCAST" or (
        version == "VLLM_HEAL" and normalized_heal_mode in {"plus", "full"}
    ):
        return {"backend": "TRITON_ATTN"}
    if dtype in {"float", "float32"}:
        return {"backend": "TRITON_ATTN"}
    return None


def normalize_dtype_for_version(version: str, dtype: str) -> str:
    normalized = dtype.lower()
    if version == "VLLM_LAYERCAST":
        if normalized == "auto":
            return "float32"
        if normalized not in {"float", "float32"}:
            raise ValueError(
                f"{version} is a LayerCast FP32 execution-path implementation; "
                f"requested dtype={dtype!r} is not supported."
            )
    if version == "VLLM_HEAL":
        if normalized == "auto":
            return "float16"
        if normalized not in {"float16", "half"}:
            raise ValueError(
                f"{version} is a 16-bit FP16-path implementation; "
                f"requested dtype={dtype!r} is not supported."
            )
    return dtype


def normalize_dtype_for_precision_options(
    version: str,
    dtype: str,
    *,
    significand_bits: int | None = None,
    layercast_fp32: bool = False,
) -> str:
    if version == "VLLM_SUMMARIZED" and (
        significand_bits is not None or layercast_fp32
    ):
        normalized = dtype.lower()
        if normalized == "auto":
            return "float32"
        if normalized not in {"float", "float32"}:
            raise ValueError(
                "VLLM_SUMMARIZED LayerCast/Significand modes support only "
                f"dtype=auto or dtype=float32; requested dtype={dtype!r}."
            )
    return normalize_dtype_for_version(version, dtype)


def set_common_env(
    *,
    version: str,
    output_dir: Path,
    layercast_fp32: bool = False,
    significand_bits: int | None = None,
    heal_plus_skip_qkv_layers: str | None = None,
    heal_full_skip_int8_layers: str | None = None,
    heal_full_skip_fp16_layers: str | None = None,
    heal_mode: str | None = None,
    heal_dynamic_int8_gemm_weight: bool = False,
    heal_fp16_gemm_compensation: bool = False,
) -> dict[str, str]:
    env = dict(os.environ)
    env["VLLM_VERSION_NAME"] = version
    env["VLLM_EXPERIMENT_OUTPUT_DIR"] = str(output_dir)
    env.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    env.setdefault("VLLM_FLOAT32_MATMUL_PRECISION", "highest")

    heal_full_skip_layers = heal_full_skip_fp16_layers or heal_full_skip_int8_layers

    if heal_dynamic_int8_gemm_weight and heal_fp16_gemm_compensation:
        raise ValueError(
            "HEAL dynamic INT8 GEMM weight and FP16 GEMM compensation "
            "are mutually exclusive."
        )
    if significand_bits is not None:
        if version != "VLLM_SUMMARIZED":
            raise ValueError("--significand-bits is only supported by VLLM_SUMMARIZED")
        bits_value = str(significand_bits)
        env["VLLM_SIGNIFICAND_BITS"] = bits_value
        env.setdefault("VLLM_SIGNIFICAND_BOUNDARY_SET", "all")

    if version == "VLLM_LAYERCAST":
        env["VLLM_LAYERCAST_FORCE_FP32"] = "1"
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    elif version == "VLLM_SUMMARIZED" and (significand_bits is not None or layercast_fp32):
        env["VLLM_LAYERCAST_FORCE_FP32"] = "1"
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    elif version == "VLLM_HEAL":
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        heal_mode_value = _normalize_heal_mode(heal_mode)
        env["HEAL_MODE"] = heal_mode_value
        if heal_plus_skip_qkv_layers:
            env["HEAL_SKIP_QKV_QUANT_LAYERS"] = heal_plus_skip_qkv_layers
        if heal_full_skip_layers:
            env["HEAL_SKIP_FP16_COMP_LAYERS"] = heal_full_skip_layers
            env["HEAL_SKIP_INT8_COMP_LAYERS"] = heal_full_skip_layers
        if heal_dynamic_int8_gemm_weight:
            env["HEAL_DYNAMIC_INT8_GEMM_WEIGHT"] = "1"
            env["HEAL_INT8_GEMM_COMPENSATION"] = "1"
        if heal_fp16_gemm_compensation or (
            heal_mode_value == "full" and not heal_dynamic_int8_gemm_weight
        ):
            env["HEAL_FP16_GEMM_COMPENSATION"] = "1"

    return env


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(base_url: str, proc: subprocess.Popen[Any], timeout_sec: int) -> None:
    deadline = time.monotonic() + timeout_sec
    urls = (f"{base_url}/v1/models", f"{base_url}/health")
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if 200 <= response.status < 500:
                        return
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {base_url}: {last_error}")


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-json", type=Path)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--sharegpt-output-len", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--attention-backend")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--server-timeout-sec", type=int, default=900)
    parser.add_argument("--port", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--layercast-fp32", action="store_true")
    parser.add_argument("--significand-bits", type=int)
    parser.add_argument("--heal-mode", choices=["base", "plus", "full"])
    parser.add_argument("--heal-dynamic-int8-gemm-weight", action="store_true")
    parser.add_argument("--heal-fp16-gemm-compensation", action="store_true")
    parser.add_argument("--heal-plus-skip-qkv-layers")
    parser.add_argument("--heal-full-skip-fp16-layers")
    parser.add_argument("--heal-full-skip-int8-layers")
    parser.add_argument("--output-name")
    return parser.parse_args()


def _server_cmd(args: argparse.Namespace, port: int, dtype: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        resolve_model(args.model),
        "--dtype",
        dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    attention_backend = args.attention_backend
    if attention_backend is None:
        attention_config = attention_config_for_run(
            args.version, dtype, heal_mode=args.heal_mode
        )
        if attention_config and attention_config.get("backend"):
            attention_backend = attention_config["backend"]
    if attention_backend:
        cmd.extend(["--attention-backend", attention_backend])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    return cmd


def _benchmark_cmd(args: argparse.Namespace, run_dir: Path, port: int, served_name: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--model",
        served_name,
    ]
    if args.dataset_json is not None:
        cmd.extend(
            [
                "--dataset-name",
                "sharegpt",
                "--dataset-path",
                str(args.dataset_json.expanduser().resolve()),
            ]
        )
        if args.sharegpt_output_len is not None:
            cmd.extend(["--sharegpt-output-len", str(args.sharegpt_output_len)])
    else:
        cmd.extend(
            [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(min(args.max_model_len // 2, 1024)),
                "--random-output-len",
                str(args.max_tokens),
                "--random-range-ratio",
                "0",
            ]
        )
    cmd.extend(
        [
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--num-prompts",
            str(args.max_prompts),
            "--request-rate",
            str(args.request_rate),
            "--seed",
            str(args.seed),
            "--percentile-metrics",
            "ttft,tpot,itl",
            "--metric-percentiles",
            "50,90,99",
            "--save-result",
            "--result-dir",
            str(run_dir),
            "--result-filename",
            "official_serving_metrics.json",
            "--disable-tqdm",
        ]
    )
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    return cmd


def main() -> None:
    args = parse_args()
    if args.version not in PERFORMANCE_VERSIONS:
        raise ValueError(
            f"{args.version} is not a performance version. "
            f"Use one of: {', '.join(PERFORMANCE_VERSIONS)}"
        )
    if args.heal_dynamic_int8_gemm_weight and args.heal_fp16_gemm_compensation:
        raise ValueError(
            "--heal-dynamic-int8-gemm-weight and "
            "--heal-fp16-gemm-compensation are mutually exclusive"
        )

    dtype = normalize_dtype_for_precision_options(
        args.version,
        args.dtype,
        significand_bits=args.significand_bits,
        layercast_fp32=args.layercast_fp32,
    )

    run_dir = output_path(
        "sharegpt_serving", args.output_name or f"{args.version}_{int(time.time())}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    env = set_common_env(
        version=args.version,
        output_dir=run_dir,
        layercast_fp32=args.layercast_fp32,
        significand_bits=args.significand_bits,
        heal_mode=args.heal_mode,
        heal_dynamic_int8_gemm_weight=args.heal_dynamic_int8_gemm_weight,
        heal_fp16_gemm_compensation=args.heal_fp16_gemm_compensation,
        heal_plus_skip_qkv_layers=args.heal_plus_skip_qkv_layers,
        heal_full_skip_fp16_layers=args.heal_full_skip_fp16_layers,
        heal_full_skip_int8_layers=args.heal_full_skip_int8_layers,
    )
    env["VLLM_USE_V1"] = "1"

    port = args.port or _free_port()
    served_name = resolve_model(args.model)
    base_url = f"http://127.0.0.1:{port}"
    server_log = run_dir / "server.log"
    benchmark_log = run_dir / "benchmark.log"
    official_metrics = run_dir / "official_serving_metrics.json"
    server_cmd = _server_cmd(args, port, dtype)
    bench_cmd = _benchmark_cmd(args, run_dir, port, served_name)

    proc: subprocess.Popen[Any] | None = None
    failure: Exception | None = None
    try:
        with server_log.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                server_cmd,
                cwd=VLLM_CWD,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_server(base_url, proc, args.server_timeout_sec)
            with benchmark_log.open("w", encoding="utf-8") as bench_log:
                subprocess.run(
                    bench_cmd,
                    cwd=VLLM_CWD,
                    env=env,
                    stdout=bench_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    except Exception as exc:
        failure = exc
        raise
    finally:
        if proc is not None and proc.poll() is None:
            _terminate_process_group(proc)
        if failure is not None:
            server_lines = (
                server_log.read_text(encoding="utf-8", errors="replace").splitlines()
                if server_log.exists()
                else []
            )
            write_json(
                run_dir / "failure.json",
                {
                    "base_sha": git_rev_parse(),
                    "version": args.version,
                    "model": args.model,
                    "resolved_model": resolve_model(args.model),
                    "dataset": "sharegpt" if args.dataset_json else "random_serving_smoke",
                    "benchmark_mode": "official_vllm_bench_serve",
                    "dtype": dtype,
                    "requested_dtype": args.dtype,
                    "heal_mode": args.heal_mode if args.version == "VLLM_HEAL" else None,
                    "error": f"{type(failure).__name__}: {failure}",
                    "server_log": str(server_log),
                    "benchmark_log": str(benchmark_log),
                    "server_command": server_cmd,
                    "benchmark_command": bench_cmd,
                    "command_cwd": str(VLLM_CWD),
                    "vllm_log_memory": parse_vllm_log_memory(server_lines),
                    "env": {
                        key: value
                        for key, value in env.items()
                        if key.startswith(("VLLM_", "HEAP_", "HEAL_"))
                    },
                },
            )

    serving = json.loads(official_metrics.read_text(encoding="utf-8"))
    if serving.get("failed", 0) not in {0, 0.0}:
        raise RuntimeError(
            f"official serving benchmark reported failed requests: {serving.get('failed')}"
        )
    server_lines = server_log.read_text(encoding="utf-8", errors="replace").splitlines()
    result = {
        "base_sha": git_rev_parse(),
        "version": args.version,
        "model": args.model,
        "resolved_model": resolve_model(args.model),
        "dataset": "sharegpt" if args.dataset_json else "random_serving_smoke",
        "benchmark_mode": "official_vllm_bench_serve",
        "dtype": dtype,
        "requested_dtype": args.dtype,
        "heal_mode": args.heal_mode if args.version == "VLLM_HEAL" else None,
        "prompt_count": args.max_prompts,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "sharegpt_output_len": args.sharegpt_output_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "official_metrics_path": str(official_metrics),
        "completed": serving.get("completed"),
        "failed": serving.get("failed"),
        "official_mean_tpot_ms": serving.get("mean_tpot_ms"),
        "official_median_tpot_ms": serving.get("median_tpot_ms"),
        "official_p99_tpot_ms": serving.get("p99_tpot_ms"),
        "official_mean_ttft_ms": serving.get("mean_ttft_ms"),
        "official_mean_itl_ms": serving.get("mean_itl_ms"),
        "request_throughput": serving.get("request_throughput"),
        "output_throughput": serving.get("output_throughput"),
        "total_token_throughput": serving.get("total_token_throughput"),
        "total_input_tokens": serving.get("total_input_tokens"),
        "total_output_tokens": serving.get("total_output_tokens"),
        "vllm_log_memory": parse_vllm_log_memory(server_lines),
        "server_log": str(server_log),
        "benchmark_log": str(benchmark_log),
        "server_command": server_cmd,
        "benchmark_command": bench_cmd,
        "command_cwd": str(VLLM_CWD),
        "env": {
            key: value
            for key, value in env.items()
            if key.startswith(("VLLM_", "HEAP_", "HEAL_"))
        },
    }
    write_json(run_dir / "metrics.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
