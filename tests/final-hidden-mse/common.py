# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the final-hidden MSE replay runners."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
VLLM_DIR = ROOT / "vllm"
PATCH_SCRIPT = ROOT / "scripts" / "vllm_incremental_build.sh"
OUTPUT_ROOT = ROOT / "output" / "final-hidden-mse"
DATA_ROOT = TEST_DIR / "data"
RESULTS_ROOT = TEST_DIR / "results"
DEFAULT_SUBSET_FILE = (
    ROOT
    / "tests"
    / "error-contribution"
    / "subsets"
    / "medqa_calibration32_seed20260528.jsonl"
)
DEFAULT_MCR_BENCH_ROOT = ROOT / "MCR-Bench"
DEFAULT_H100_QWEN3_8B_BASELINE: Path | None = None

FINAL_HIDDEN_STAGE = "forward.norm.hidden"
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


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def model_label(model: str) -> str:
    return resolve_model(model).replace("/", "-")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_hashes() -> dict[str, str]:
    names = [
        "VLLM_ORIGINAL",
        "VLLM_LAYERCAST",
        "VLLM_SIGNIFICAND_BITS",
        "VLLM_HEAL",
    ]
    return {name: sha256_file(ROOT / "patches" / f"{name}.patch") for name in names}


def ensure_vllm_clean() -> None:
    status = subprocess.check_output(
        ["git", "-C", str(VLLM_DIR), "status", "--porcelain"],
        text=True,
    ).strip()
    if status:
        raise RuntimeError(f"vLLM worktree is not clean:\n{status}")


def active_patch_status() -> str:
    status = subprocess.check_output(
        [str(PATCH_SCRIPT), "status"],
        cwd=ROOT,
        text=True,
    )
    return status


def run_patch_command(action: str, version: str | None = None) -> None:
    cmd = [str(PATCH_SCRIPT), action]
    if version is not None:
        cmd.append(version)
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_local_vllm_to_path() -> None:
    value = str(VLLM_DIR)
    if value not in sys.path:
        sys.path.insert(0, value)


def load_medqa_subset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_subset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return load_medqa_subset(path, limit=limit)


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


def final_answer_contract(answer_format: str) -> str:
    return (
        "You may reason, but before you stop you must write a separate "
        f"last line exactly in this format: `{answer_format}`. "
        "Do not put any text after that final line."
    )


def _option_lines(row: dict[str, Any]) -> list[str]:
    raw_options = row.get("answer_options") or []
    choices: list[str] = []
    if isinstance(raw_options, list):
        for option in raw_options:
            if not isinstance(option, dict):
                continue
            label = str(option.get("label", "")).strip()
            text = str(option.get("text", "")).strip()
            if label and text:
                choices.append(f"{label}. {text}")
    return choices


def _target_from_abcd_row(row: dict[str, Any]) -> str | None:
    correct_option = row.get("correct_option")
    if isinstance(correct_option, str) and correct_option.strip():
        return correct_option.strip()[:1].upper()
    for key in ("answer", "target", "label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip()[:1].upper() in {"A", "B", "C", "D"}:
            return value.strip()[:1].upper()
    answer_idx = row.get("answer_idx")
    if isinstance(answer_idx, str) and answer_idx[:1].upper() in {"A", "B", "C", "D"}:
        return answer_idx[:1].upper()
    label = row.get("label")
    if isinstance(label, int) and 0 <= label <= 3:
        return "ABCD"[label]
    if isinstance(label, str) and label.isdigit() and 0 <= int(label) <= 3:
        return "ABCD"[int(label)]
    return None


def mcr_bench_task(row: dict[str, Any]) -> str:
    task = str(
        row.get("task") or row.get("mcr_task") or row.get("mcr_domain") or ""
    ).strip().lower()
    if task in {"medical", "legal", "finance"}:
        return task
    source = str(row.get("source_dataset") or "").strip().lower()
    legal_task = str(row.get("legal_task") or "").strip().lower()
    finance_task = str(row.get("finance_task") or "").strip().lower()
    if legal_task or source in {
        "lexglue_unfair_tos",
        "casehold",
        "consumer_contracts",
        "insurance_policy",
        "barexam",
    }:
        return "legal"
    if finance_task or source in {"finclaim", "financial_sentiment", "finqa"}:
        return "finance"
    return "medical"


def mcr_bench_target(row: dict[str, Any]) -> tuple[str | None, str]:
    task = mcr_bench_task(row)
    answer_type = str(row.get("answer_type") or "").strip()
    correct_option = row.get("correct_option")
    if task == "medical":
        if answer_type == "yes_no":
            target = correct_option if correct_option is not None else row.get("answer")
            return (None if target is None else str(target).strip()), "yes_no"
        return _target_from_abcd_row(row), "multiple_choice_abcd"
    if task == "legal":
        if answer_type == "multi_label_list":
            return (None if correct_option is None else str(correct_option)), "multi_label_list"
        if answer_type == "multiple_choice_0_4":
            return (None if correct_option is None else str(correct_option)), "multiple_choice_0_4"
        if answer_type == "yes_no":
            return (None if correct_option is None else str(correct_option)), "yes_no"
        if answer_type == "multiple_choice_abc":
            return (None if correct_option is None else str(correct_option)), "multiple_choice_abc"
        if answer_type in {"multiple_choice_abcd", "multiple_choice"}:
            return (None if correct_option is None else str(correct_option)), "multiple_choice_abcd"
        label = row.get("label")
        if isinstance(label, int):
            return str(label + 1), "label_number"
        if isinstance(label, str) and label.isdigit():
            return str(int(label) + 1), "label_number"
        return None, "label_number"
    if task == "finance":
        if answer_type == "finance_claim_category":
            return (None if correct_option is None else str(correct_option)), "finance_claim_category"
        if answer_type == "finance_sentiment":
            return (None if correct_option is None else str(correct_option)), "finance_sentiment"
        answer = correct_option if correct_option is not None else row.get("answer")
        return (None if answer is None else str(answer).strip()), "numeric_or_text"
    return None, "text"


def mcr_bench_source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    target, target_kind = mcr_bench_target(row)
    return {
        "task": mcr_bench_task(row),
        "target": target,
        "target_kind": target_kind,
        "mcr_domain": row.get("mcr_domain"),
        "source_dataset": row.get("source_dataset"),
        "answer_type": row.get("answer_type"),
        "answer_options": row.get("answer_options"),
        "legal_task": row.get("legal_task"),
        "finance_task": row.get("finance_task"),
    }


def _mcr_medical_messages(
    row: dict[str, Any],
    *,
    max_source_chars: int,
) -> list[dict[str, str]]:
    context_question = str(
        row.get("context_question")
        or row.get("question")
        or row.get("sent1")
        or row.get("prompt")
        or ""
    ).strip()
    if max_source_chars > 0:
        context_question = context_question[:max_source_chars]

    answer_type = str(row.get("answer_type") or "multiple_choice").strip()
    raw_options = row.get("answer_options") or []
    choices: list[str] = []
    if answer_type == "yes_no":
        choices = ["yes", "no"]
    elif isinstance(raw_options, list):
        for option in raw_options:
            if not isinstance(option, dict):
                continue
            label = str(option.get("label", "")).strip()
            text = str(option.get("text", "")).strip()
            if label and text:
                choices.append(f"{label}. {text}")
    if not choices:
        for idx, letter in enumerate(("A", "B", "C", "D")):
            value = _choice_value(row, idx, letter)
            if value is not None:
                choices.append(f"{letter}. {value}")

    labels = [choice.split(".", 1)[0] for choice in choices]
    label_text = ", ".join(labels) if labels else "the best answer"
    answer_format = "Final Answer: X"
    user = "Question:\n" + context_question
    user += f"\n\nAnswer type: {answer_type}"
    if choices:
        user += "\n\nOptions:\n" + "\n".join(choices)
    user += (
        "\n\nFinal response requirement: your last line must be exactly "
        f"`{answer_format}`, where X is one of: {label_text}."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are solving medical exam and biomedical reasoning questions. "
                "Use only the provided context, question, and answer options. "
                + final_answer_contract(answer_format)
                + f" X must be one of: {label_text}."
            ),
        },
        {"role": "user", "content": user},
    ]


def _mcr_legal_messages(
    row: dict[str, Any],
    *,
    max_source_chars: int,
) -> list[dict[str, str]]:
    legal_task = str(row.get("legal_task") or row.get("source_dataset") or "").strip()
    choices = _option_lines(row)

    if legal_task == "lexglue_unfair_tos":
        text = str(row.get("clause") or row.get("context_question") or row.get("text") or "").strip()
        if max_source_chars > 0:
            text = text[:max_source_chars]
        answer_format = "Final Answer: [n,...]"
        system = (
            "You are reviewing a terms-of-service clause for unfair contract terms. "
            "Identify every unfair term type that applies. "
            + final_answer_contract(answer_format)
            + " Use [] if no unfair type applies. If multiple labels apply, list "
            "the numbers in descending order."
        )
        user = "Clause:\n" + text
        if choices:
            user += "\n\nLabel meanings:\n" + "\n".join(choices)
        user += (
            "\n\nFinal response requirement: your last line must be exactly "
            "`Final Answer: [n,...]`, for example `Final Answer: []` or "
            "`Final Answer: [7,2]`."
        )
    elif legal_task == "casehold":
        text = str(row.get("citing_prompt") or row.get("context_question") or "").strip()
        if max_source_chars > 0:
            text = text[:max_source_chars]
        answer_format = "Final Answer: N"
        system = (
            "You are selecting the legal holding that best completes the citation "
            "context. "
            + final_answer_contract(answer_format)
            + " N must be 0, 1, 2, 3, or 4."
        )
        user = "Citation context:\n" + text
        if choices:
            user += "\n\nCandidate holdings:\n" + "\n".join(choices)
        user += (
            "\n\nFinal response requirement: your last line must be exactly "
            "`Final Answer: N`, where N is 0, 1, 2, 3, or 4."
        )
    elif legal_task == "consumer_contracts":
        contract = str(row.get("contract") or "").strip()
        question = str(row.get("question") or "").strip()
        if contract and question:
            if max_source_chars > 0:
                contract = contract[:max_source_chars]
            text = f"Contract:\n{contract}\n\nQuestion:\n{question}"
        else:
            text = str(row.get("context_question") or "").strip()
        answer_format = "Final Answer: X"
        system = (
            "You are answering a yes/no question about a consumer contract. "
            "Use only the contract text. "
            + final_answer_contract(answer_format)
            + " X must be either yes or no."
        )
        user = text + "\n\nOptions:\nyes\nno"
        user += (
            "\n\nFinal response requirement: your last line must be exactly "
            "`Final Answer: yes` or `Final Answer: no`."
        )
    elif legal_task == "insurance_policy":
        policy = str(row.get("policy") or "").strip()
        claim = str(row.get("claim") or "").strip()
        if policy or claim:
            if max_source_chars > 0:
                policy = policy[:max_source_chars]
                claim = claim[:max_source_chars]
            text = f"Policy:\n{policy}\n\nClaim:\n{claim}\n\nIs the claim covered by the policy?"
        else:
            text = str(row.get("context_question") or "").strip()
        answer_format = "Final Answer: X"
        system = (
            "You are deciding whether an insurance claim is covered by a policy. "
            "Compare the claim to the policy language. "
            + final_answer_contract(answer_format)
            + " X must be A, B, or C."
        )
        user = text
        if choices:
            user += "\n\nOptions:\n" + "\n".join(choices)
        user += (
            "\n\nFinal response requirement: your last line must be exactly "
            "`Final Answer: X`, where X is A, B, or C."
        )
    elif legal_task == "barexam":
        text = str(row.get("question") or row.get("context_question") or "").strip()
        if max_source_chars > 0:
            text = text[:max_source_chars]
        answer_format = "Final Answer: X"
        system = (
            "You are answering a bar exam multiple-choice legal question. "
            "Choose the single best answer from the provided choices. "
            + final_answer_contract(answer_format)
            + " X must be A, B, C, or D."
        )
        user = "Question:\n" + text
        if choices:
            user += "\n\nChoices:\n" + "\n".join(choices)
        user += (
            "\n\nFinal response requirement: your last line must be exactly "
            "`Final Answer: X`, where X is A, B, C, or D."
        )
    else:
        text = str(
            row.get("context_question")
            or row.get("text")
            or row.get("citing_prompt")
            or row.get("contract")
            or ""
        ).strip()
        if max_source_chars > 0:
            text = text[:max_source_chars]
        answer_format = "Final Answer: ..."
        system = (
            "You are answering a legal reasoning question. Use only the provided "
            "text and options. "
            + final_answer_contract(answer_format)
        )
        user = text
        if choices:
            user += "\n\nOptions:\n" + "\n".join(choices)
        user += "\n\nFinal response requirement: your last line must start with `Final Answer:`."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _mcr_finance_messages(
    row: dict[str, Any],
    *,
    max_source_chars: int,
) -> list[dict[str, str]]:
    finance_task = str(row.get("finance_task") or row.get("source_dataset") or "").strip()
    choices = _option_lines(row)

    if finance_task == "finclaim":
        text = str(row.get("context_question") or row.get("claim_text") or "").strip()
        if max_source_chars > 0:
            text = text[:max_source_chars]
        answer_format = "Final Answer: N"
        system = (
            "You are classifying the evidentiary basis for a financial claim. "
            "Choose exactly one category from the provided list. "
            + final_answer_contract(answer_format)
            + " N must be 1, 2, 3, 4, 5, or 6."
        )
        user = "Financial claim:\n" + text
        if choices:
            user += "\n\nCategories:\n" + "\n".join(choices)
        user += (
            "\n\nFinal response requirement: your last line must be exactly "
            "`Final Answer: N`, where N is 1, 2, 3, 4, 5, or 6."
        )
    elif finance_task == "financial_sentiment":
        text = str(row.get("context_question") or row.get("Sentence") or "").strip()
        if max_source_chars > 0:
            text = text[:max_source_chars]
        answer_format = "Final Answer: N"
        system = (
            "You are classifying the sentiment of a financial text. Choose exactly "
            "one category from the provided list. "
            + final_answer_contract(answer_format)
            + " N must be 1, 2, or 3."
        )
        user = "Financial text:\n" + text
        if choices:
            user += "\n\nCategories:\n" + "\n".join(choices)
        user += (
            "\n\nFinal response requirement: your last line must be exactly "
            "`Final Answer: N`, where N is 1, 2, or 3."
        )
    elif finance_task == "finqa":
        pre_text = str(row.get("pre_text") or "").strip()
        table = str(row.get("table") or "").strip()
        post_text = str(row.get("post_text") or "").strip()
        question = str(row.get("question") or "").strip()
        if not any((pre_text, table, post_text, question)):
            user = str(
                row.get("context_question")
                or row.get("query")
                or row.get("text")
                or row.get("input")
                or ""
            ).strip()
            if max_source_chars > 0:
                user = user[:max_source_chars]
        else:
            if max_source_chars > 0:
                pre_text = pre_text[:max_source_chars]
                post_text = post_text[:max_source_chars]
            user = (
                "Pre-Table Financial Context:\n"
                f"{pre_text}\n\nFinancial Table:\n{table}\n\n"
                f"Post-Table Financial Context:\n{post_text}\n\nQuestion:\n{question}"
            )
        system = (
            "You are a financial reasoning assistant. Use both the text and the "
            "table to answer the question. Perform any needed calculations carefully. "
            + final_answer_contract("Final Answer: X")
            + " X should be the final numeric answer only."
        )
        user += (
            "\n\nFinal response requirement: your last line must be exactly "
            "`Final Answer: X`, where X is the final numeric answer only."
        )
    else:
        user = str(
            row.get("query")
            or row.get("text")
            or row.get("input")
            or row.get("context_question")
            or ""
        ).strip()
        if max_source_chars > 0:
            user = user[:max_source_chars]
        system = (
            "You are answering financial QA questions. Use the provided context. "
            + final_answer_contract("Final Answer: X")
        )
        user += (
            "\n\nFinal response requirement: your last line must start with "
            "`Final Answer:` and contain only the answer after the colon."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def mcr_bench_messages(
    row: dict[str, Any],
    *,
    max_source_chars: int = 12000,
) -> list[dict[str, str]]:
    task = mcr_bench_task(row)
    if task == "medical":
        return _mcr_medical_messages(row, max_source_chars=max_source_chars)
    if task == "legal":
        return _mcr_legal_messages(row, max_source_chars=max_source_chars)
    if task == "finance":
        return _mcr_finance_messages(row, max_source_chars=max_source_chars)
    raise ValueError(f"unknown MCR-Bench task: {task}")


def messages_for_row(
    row: dict[str, Any],
    *,
    dataset_kind: str = "medqa",
    max_source_chars: int = 12000,
) -> list[dict[str, str]]:
    if dataset_kind == "medqa":
        return medqa_messages(row)
    if dataset_kind == "mcr-bench":
        return mcr_bench_messages(row, max_source_chars=max_source_chars)
    raise ValueError(f"unknown dataset kind: {dataset_kind}")


def chat_prompt_and_ids(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    if getattr(tokenizer, "chat_template", None):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt = "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
        prompt += "\n\nASSISTANT:"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, [int(token_id) for token_id in token_ids]


def configure_precision_env(env: dict[str, str], mode: dict[str, Any]) -> dict[str, str]:
    out = dict(env)
    for key in (
        "VLLM_VERSION_NAME",
        "VLLM_EXPERIMENT_OUTPUT_DIR",
        "VLLM_LAYERCAST_FORCE_FP32",
        "VLLM_DISABLE_COMPILE_CACHE",
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
    out["VLLM_DISABLE_CUDA_GRAPH"] = "1"
    out["VLLM_USE_V1"] = "1"
    out["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
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
        if mode.get("heal_skip_qkv_quant_layers") is not None:
            out["HEAL_SKIP_QKV_QUANT_LAYERS"] = str(
                mode["heal_skip_qkv_quant_layers"]
            )
        fp16_skip_layers = mode.get("heal_skip_fp16_comp_layers")
        if fp16_skip_layers is None:
            fp16_skip_layers = mode.get("heal_skip_int8_comp_layers")
        if fp16_skip_layers is not None:
            out["HEAL_SKIP_FP16_COMP_LAYERS"] = str(fp16_skip_layers)
            # The current HEAL patch still reads this compensation-layer name.
            out["HEAL_SKIP_INT8_COMP_LAYERS"] = str(fp16_skip_layers)
        if mode.get("heal_dynamic_int8_gemm_weight") is not None:
            dynamic_int8 = bool(mode["heal_dynamic_int8_gemm_weight"])
            out["HEAL_DYNAMIC_INT8_GEMM_WEIGHT"] = "1" if dynamic_int8 else "0"
            if dynamic_int8:
                out["HEAL_INT8_GEMM_COMPENSATION"] = "1"
        if mode.get("heal_fp16_gemm_compensation") is not None:
            out["HEAL_FP16_GEMM_COMPENSATION"] = (
                "1" if bool(mode["heal_fp16_gemm_compensation"]) else "0"
            )
    return out


def apply_torch_precision(tf32: bool) -> None:
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    except Exception:
        return


def enable_tensor_trace(trace_file: Path) -> None:
    os.environ["TENSOR_TRACE_TENSOR_FILE"] = str(trace_file)
    os.environ["VLLM_TENSOR_TRACE_TENSOR_FILE"] = str(trace_file)
    os.environ["TENSOR_TRACE_STAGE_REGEX"] = rf"^{re.escape(FINAL_HIDDEN_STAGE)}$"
    os.environ["VLLM_TENSOR_TRACE_STAGE_REGEX"] = rf"^{re.escape(FINAL_HIDDEN_STAGE)}$"
    clear_tensor_trace()


def clear_tensor_trace() -> None:
    try:
        from vllm.model_executor.models.tensor_trace import clear_tensor_trace_tensors

        clear_tensor_trace_tensors()
    except Exception:
        return


def flush_tensor_trace() -> None:
    try:
        from vllm.model_executor.models.tensor_trace import flush_tensor_trace_tensors

        flush_tensor_trace_tensors()
    except Exception as exc:
        raise RuntimeError("failed to flush tensor trace") from exc


def _trace_keys(path: Path, stage: str = FINAL_HIDDEN_STAGE) -> list[str]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        metadata = read_json(meta_path)
        rows = metadata.get(stage, [])
        keys = [row["key"] for row in rows if isinstance(row, dict) and "key" in row]
        if keys:
            return keys
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    exact = [key for key in keys if key == stage]
    chunks = sorted(key for key in keys if key.startswith(stage + "__chunk_"))
    return exact or chunks


def load_trace_stage_chunk_metadata(
    path: Path,
    stage: str = FINAL_HIDDEN_STAGE,
) -> list[dict[str, Any]]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return []
    metadata = read_json(meta_path)
    rows = metadata.get(stage, [])
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def load_trace_stage(path: Path, stage: str = FINAL_HIDDEN_STAGE) -> torch.Tensor:
    keys = _trace_keys(path, stage)
    if not keys:
        raise RuntimeError(f"trace stage {stage!r} not found in {path}")
    tensors = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in keys:
            tensors.append(handle.get_tensor(key).to(torch.float32))
    if len(tensors) == 1:
        tensor = tensors[0]
    else:
        tensor = torch.cat(tensors, dim=0)
    if tensor.ndim != 2:
        raise RuntimeError(f"expected 2D hidden tensor in {path}, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def load_trace_stage_chunks(
    path: Path,
    stage: str = FINAL_HIDDEN_STAGE,
) -> list[torch.Tensor]:
    keys = _trace_keys(path, stage)
    if not keys:
        raise RuntimeError(f"trace stage {stage!r} not found in {path}")
    chunks = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in keys:
            tensor = handle.get_tensor(key).to(torch.float32)
            if tensor.ndim != 2:
                raise RuntimeError(
                    f"expected 2D hidden tensor in {path}, got {tuple(tensor.shape)}"
                )
            chunks.append(tensor.contiguous())
    return chunks


def build_batch_manifest(
    replay_rows: list[dict[str, Any]],
    *,
    batch_size: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        batch_id = len(batches)
        batches.append(
            {
                "batch_id": batch_id,
                "doc_indices": [int(row["doc_id"]) for row in current],
                "total_replay_tokens": current_tokens,
                "max_replay_tokens": max(int(row["replay_token_count"]) for row in current),
                "batch_size": len(current),
            }
        )
        current = []
        current_tokens = 0

    for row in replay_rows:
        row_tokens = int(row["replay_token_count"])
        if current and (
            len(current) >= batch_size or current_tokens + row_tokens > token_budget
        ):
            flush()
        current.append(row)
        current_tokens += row_tokens
        if row_tokens > token_budget:
            flush()
    flush()
    return batches


def write_replay_positions(path: Path, replay_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "subset_pos",
        "original_index",
        "row_index",
        "token_pos",
        "absolute_token_pos",
        "hidden_row_index",
        "prefix_token_count",
        "expected_token_id",
        "generated_token_count",
        "replay_token_count",
    ]
    rows: list[dict[str, Any]] = []
    for item in replay_rows:
        prefix = int(item["prompt_token_count"])
        generated = item["generated_token_ids"]
        for token_pos, token_id in enumerate(generated):
            hidden_row_index = prefix - 1 + token_pos
            rows.append(
                {
                    "doc_id": item["doc_id"],
                    "subset_pos": item["subset_pos"],
                    "original_index": item["original_index"],
                    "row_index": token_pos,
                    "token_pos": token_pos,
                    "absolute_token_pos": hidden_row_index,
                    "hidden_row_index": hidden_row_index,
                    "prefix_token_count": prefix + token_pos,
                    "expected_token_id": int(token_id),
                    "generated_token_count": len(generated),
                    "replay_token_count": item["replay_token_count"],
                }
            )
    write_csv(path, rows, fieldnames)


def hidden_outputs_complete(hidden_dir: Path, replay_rows: list[dict[str, Any]]) -> bool:
    if not hidden_dir.exists():
        return False
    for row in replay_rows:
        path = hidden_dir / f"doc_{int(row['doc_id'])}.safetensors"
        if not path.exists():
            return False
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                shape = handle.get_tensor("hidden_states").shape
        except Exception:
            return False
        if int(shape[0]) != int(row["generated_token_count"]):
            return False
    return True


def split_replay_trace_to_doc_hidden(
    *,
    trace_path: Path,
    batch_rows: list[dict[str, Any]],
    hidden_dir: Path,
) -> dict[str, Any]:
    hidden = load_trace_stage(trace_path)
    expected_rows = sum(int(row["replay_token_count"]) for row in batch_rows)
    if hidden.shape[0] < expected_rows:
        raise RuntimeError(
            f"{trace_path} has {hidden.shape[0]} rows, expected at least {expected_rows}"
        )
    hidden_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    written = []
    for row in batch_rows:
        doc_id = int(row["doc_id"])
        prompt_tokens = int(row["prompt_token_count"])
        generated_tokens = int(row["generated_token_count"])
        replay_tokens = int(row["replay_token_count"])
        doc_hidden = hidden[start : start + replay_tokens]
        if prompt_tokens <= 0:
            raise RuntimeError(f"doc {doc_id} has no prompt tokens")
        generated_start = prompt_tokens - 1
        generated_hidden = doc_hidden[generated_start : generated_start + generated_tokens]
        if generated_hidden.shape[0] != generated_tokens:
            raise RuntimeError(
                f"doc {doc_id} generated hidden rows mismatch: "
                f"{generated_hidden.shape[0]} vs {generated_tokens}"
            )
        out_path = hidden_dir / f"doc_{doc_id}.safetensors"
        save_file({"hidden_states": generated_hidden.contiguous()}, str(out_path))
        written.append(
            {
                "doc_id": doc_id,
                "path": str(out_path),
                "shape": list(generated_hidden.shape),
            }
        )
        start += replay_tokens
    return {
        "trace_path": str(trace_path),
        "trace_rows": int(hidden.shape[0]),
        "expected_replay_rows": expected_rows,
        "ignored_extra_rows": int(hidden.shape[0] - expected_rows),
        "hidden_size": int(hidden.shape[1]),
        "written": written,
    }


def _request_id_aliases(request_id: str) -> list[str]:
    aliases = [request_id]
    prefix, sep, suffix = request_id.rpartition("-")
    if sep and prefix and len(suffix) == 8 and all(
        char in "0123456789abcdefABCDEF" for char in suffix
    ):
        aliases.append(prefix)
    return aliases


def _split_generation_trace_to_doc_hidden_with_metadata(
    *,
    trace_path: Path,
    chunks: list[torch.Tensor],
    chunk_metadata: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    hidden_dir: Path,
) -> dict[str, Any]:
    if len(chunks) != len(chunk_metadata):
        raise RuntimeError(
            f"{trace_path} metadata chunk count {len(chunk_metadata)} does not "
            f"match tensor chunk count {len(chunks)}"
        )
    hidden_size = int(chunks[0].shape[1])
    hidden_dir.mkdir(parents=True, exist_ok=True)

    request_rows: dict[str, dict[str, Any]] = {}
    target_positions: dict[str, dict[int, int]] = {}
    doc_rows: dict[int, dict[int, torch.Tensor]] = {}
    for fallback_index, row in enumerate(batch_rows):
        request_id = row.get("request_id")
        if request_id is None:
            raise RuntimeError(
                f"{trace_path} has row metadata, but doc "
                f"{int(row['doc_id'])} has no request_id in replay manifest"
            )
        request_key = str(request_id)
        if request_key in request_rows:
            raise RuntimeError(f"duplicate request_id in batch rows: {request_key}")
        prompt_tokens = int(row["prompt_token_count"])
        generated_tokens = int(row["generated_token_count"])
        request_rows[request_key] = row
        target_positions[request_key] = {
            prompt_tokens - 1 + token_pos: token_pos
            for token_pos in range(generated_tokens)
        }
        doc_rows[int(row["doc_id"])] = {}

    trace_rows = 0
    matched_rows = 0
    ignored_rows = 0
    duplicate_generation_rows = 0
    padding_rows = 0
    for chunk_index, (chunk, meta_entry) in enumerate(
        zip(chunks, chunk_metadata, strict=True)
    ):
        chunk_rows = int(chunk.shape[0])
        trace_rows += chunk_rows
        row_metadata = meta_entry.get("row_metadata")
        if not isinstance(row_metadata, dict):
            raise RuntimeError(
                f"{trace_path} chunk {chunk_index} is missing tensor row metadata"
            )
        request_ids = row_metadata.get("request_ids")
        row_request_indices = row_metadata.get("row_request_indices")
        row_token_positions = row_metadata.get("row_token_positions")
        if not isinstance(request_ids, list):
            raise RuntimeError(f"{trace_path} chunk {chunk_index} has no request_ids")
        if not isinstance(row_request_indices, list) or not isinstance(
            row_token_positions, list
        ):
            raise RuntimeError(
                f"{trace_path} chunk {chunk_index} has incomplete row metadata"
            )
        if len(row_request_indices) < chunk_rows or len(row_token_positions) < chunk_rows:
            raise RuntimeError(
                f"{trace_path} chunk {chunk_index} metadata rows are shorter than "
                f"tensor rows: {len(row_request_indices)}/{len(row_token_positions)} "
                f"vs {chunk_rows}"
            )
        request_keys = []
        for request_id in request_ids:
            raw_request_key = str(request_id)
            request_key = next(
                (
                    alias
                    for alias in _request_id_aliases(raw_request_key)
                    if alias in target_positions
                ),
                raw_request_key,
            )
            request_keys.append(request_key)
        for row_index in range(chunk_rows):
            request_index = int(row_request_indices[row_index])
            if request_index < 0:
                padding_rows += 1
                continue
            if request_index >= len(request_keys):
                raise RuntimeError(
                    f"{trace_path} chunk {chunk_index} row {row_index} has "
                    f"request index {request_index}, but only {len(request_keys)} "
                    "request ids are recorded"
                )
            request_key = request_keys[request_index]
            positions_for_request = target_positions.get(request_key)
            if not positions_for_request:
                ignored_rows += 1
                continue
            token_position = int(row_token_positions[row_index])
            generated_token_pos = positions_for_request.get(token_position)
            if generated_token_pos is None:
                ignored_rows += 1
                continue
            doc_id = int(request_rows[request_key]["doc_id"])
            rows_for_doc = doc_rows[doc_id]
            if generated_token_pos in rows_for_doc:
                duplicate_generation_rows += 1
                continue
            rows_for_doc[generated_token_pos] = chunk[row_index].to(torch.float32)
            matched_rows += 1

    expected_rows = sum(int(row["generated_token_count"]) for row in batch_rows)
    written = []
    missing: list[dict[str, Any]] = []
    for row in batch_rows:
        doc_id = int(row["doc_id"])
        generated_tokens = int(row["generated_token_count"])
        rows_for_doc = doc_rows[doc_id]
        missing_positions = [
            token_pos for token_pos in range(generated_tokens) if token_pos not in rows_for_doc
        ]
        if missing_positions:
            missing.append(
                {
                    "doc_id": doc_id,
                    "request_id": row.get("request_id"),
                    "missing_token_positions": missing_positions[:20],
                    "missing_count": len(missing_positions),
                }
            )
            continue
        generated_hidden = (
            torch.stack(
                [rows_for_doc[token_pos] for token_pos in range(generated_tokens)],
                dim=0,
            ).contiguous()
            if generated_tokens
            else torch.empty((0, hidden_size), dtype=torch.float32)
        )
        out_path = hidden_dir / f"doc_{doc_id}.safetensors"
        save_file({"hidden_states": generated_hidden}, str(out_path))
        written.append(
            {
                "doc_id": doc_id,
                "path": str(out_path),
                "shape": list(generated_hidden.shape),
            }
        )
    if missing:
        raise RuntimeError(
            f"{trace_path} metadata split matched {matched_rows}/{expected_rows} "
            f"generated hidden rows; missing={missing[:5]}"
        )

    return {
        "trace_path": str(trace_path),
        "split_source": "row_metadata",
        "trace_rows": int(trace_rows),
        "matched_generation_rows": int(matched_rows),
        "ignored_non_generation_rows": int(ignored_rows),
        "duplicate_generation_rows": int(duplicate_generation_rows),
        "padding_rows": int(padding_rows),
        "generated_hidden_rows": int(expected_rows),
        "hidden_size": hidden_size,
        "written": written,
    }


def split_generation_trace_to_doc_hidden(
    *,
    trace_path: Path,
    batch_rows: list[dict[str, Any]],
    hidden_dir: Path,
) -> dict[str, Any]:
    """Split generate-time hidden rows into per-doc generated-token hidden files.

    vLLM generation chunks can interleave decode rows for already-prefilled
    requests with prefill rows for later requests in the same Python
    ``LLM.generate`` batch. Within each model forward chunk, vLLM emits decode
    rows for active requests first, then spends the remaining token budget on
    waiting prompts in request order.

    The first generated token for a request is predicted from the final prompt
    row. Later generated-token hidden rows come from decode rows. Requests that
    finish by a stop token can consume one extra scheduler row after the emitted
    generated tokens; that row is used only for alignment and is not written.
    """

    chunks = load_trace_stage_chunks(trace_path)
    if not chunks:
        raise RuntimeError(f"no trace chunks found in {trace_path}")
    chunk_metadata = load_trace_stage_chunk_metadata(trace_path)
    metadata_split_error: str | None = None
    if any(isinstance(row.get("row_metadata"), dict) for row in chunk_metadata):
        try:
            return _split_generation_trace_to_doc_hidden_with_metadata(
                trace_path=trace_path,
                chunks=chunks,
                chunk_metadata=chunk_metadata,
                batch_rows=batch_rows,
                hidden_dir=hidden_dir,
            )
        except RuntimeError as exc:
            metadata_error = str(exc)
            recoverable_metadata_error = (
                "duplicate hidden row" in metadata_error
                or "metadata split matched" in metadata_error
            )
            if not recoverable_metadata_error:
                raise
            # Some generation traces contain scheduler rows whose metadata
            # reuses request slots after swap-removal, making row positions
            # ambiguous. The scheduler-shape splitter below was written for
            # exactly that generation path and is safer for those traces.
            metadata_split_error = metadata_error
    hidden_size = int(chunks[0].shape[1])
    prompt_total = sum(int(row["prompt_token_count"]) for row in batch_rows)
    if prompt_total <= 0:
        raise RuntimeError("generation batch has no prompt tokens")

    hidden_dir.mkdir(parents=True, exist_ok=True)
    doc_rows: dict[int, list[torch.Tensor]] = {
        int(row["doc_id"]): [] for row in batch_rows
    }
    prompt_seen = [0 for _ in batch_rows]
    scheduler_seen = [0 for _ in batch_rows]
    written_seen = [0 for _ in batch_rows]
    write_limits = [int(row["generated_token_count"]) for row in batch_rows]
    scheduler_limits = []
    for row in batch_rows:
        full_generated = int(row.get("full_generated_token_count",
                                     row["generated_token_count"]))
        finish_reason = row.get("original_finish_reason", row.get("finish_reason"))
        stop_extra = 1 if finish_reason == "stop" else 0
        scheduler_limits.append(full_generated + stop_extra)

    active: list[int] = []
    waiting_index = 0
    consumed_prompt_rows = 0
    consumed_decode_rows = 0

    def consume_generation_row(doc_index: int, hidden_row: torch.Tensor) -> None:
        doc_id = int(batch_rows[doc_index]["doc_id"])
        scheduler_seen[doc_index] += 1
        if written_seen[doc_index] < write_limits[doc_index]:
            doc_rows[doc_id].append(hidden_row)
            written_seen[doc_index] += 1

    for chunk_index, chunk in enumerate(chunks):
        cursor = 0
        chunk_rows = int(chunk.shape[0])
        while len(active) > chunk_rows:
            removed = False
            active_pos = 0
            while active_pos < len(active):
                doc_index = active[active_pos]
                if written_seen[doc_index] < write_limits[doc_index]:
                    active_pos += 1
                    continue
                active[active_pos] = active[-1]
                active.pop()
                removed = True
                break
            if not removed:
                break
        if chunk_rows < len(active):
            raise RuntimeError(
                f"{trace_path} chunk {chunk_index} has {chunk_rows} rows, "
                f"but {len(active)} active generation rows are expected"
            )

        for doc_index in active:
            consume_generation_row(doc_index, chunk[cursor])
            consumed_decode_rows += 1
            cursor += 1

        # vLLM's running-request container removes completed entries by filling
        # the slot with the current tail entry. Preserving that swap-remove
        # behavior is necessary to keep later decode rows aligned to requests.
        active_pos = 0
        while active_pos < len(active):
            doc_index = active[active_pos]
            if scheduler_seen[doc_index] < scheduler_limits[doc_index]:
                active_pos += 1
                continue
            active[active_pos] = active[-1]
            active.pop()

        while cursor < chunk_rows and waiting_index < len(batch_rows):
            row = batch_rows[waiting_index]
            prompt_tokens = int(row["prompt_token_count"])
            if prompt_tokens <= 0:
                raise RuntimeError(f"doc {int(row['doc_id'])} has no prompt tokens")
            need = prompt_tokens - prompt_seen[waiting_index]
            take = min(need, chunk_rows - cursor)
            if take <= 0:
                raise RuntimeError(
                    f"internal scheduling error at chunk {chunk_index}, "
                    f"doc {int(row['doc_id'])}"
                )
            prompt_seen[waiting_index] += take
            consumed_prompt_rows += take
            cursor += take
            if prompt_seen[waiting_index] == prompt_tokens:
                if scheduler_limits[waiting_index] > 0:
                    consume_generation_row(waiting_index, chunk[cursor - 1])
                    if scheduler_seen[waiting_index] < scheduler_limits[waiting_index]:
                        active.append(waiting_index)
                waiting_index += 1

        if cursor != chunk_rows:
            raise RuntimeError(
                f"{trace_path} chunk {chunk_index} has {chunk_rows - cursor} "
                "unassigned rows after decode and prompt scheduling"
            )

    if consumed_prompt_rows != prompt_total:
        raise RuntimeError(
            f"{trace_path} consumed {consumed_prompt_rows} prompt rows, "
            f"expected {prompt_total}"
        )
    if waiting_index != len(batch_rows):
        raise RuntimeError(
            f"{trace_path} finished with {len(batch_rows) - waiting_index} "
            "prompts not fully scheduled"
        )
    scheduler_expected = sum(scheduler_limits)
    scheduler_actual = sum(scheduler_seen)
    if scheduler_actual != scheduler_expected:
        raise RuntimeError(
            f"{trace_path} consumed {scheduler_actual} generation scheduler rows, "
            f"expected {scheduler_expected}"
        )
    if active:
        raise RuntimeError(f"{trace_path} finished with {len(active)} active requests")

    written = []
    for row in batch_rows:
        doc_id = int(row["doc_id"])
        generated_tokens = int(row["generated_token_count"])
        rows = doc_rows[doc_id]
        if len(rows) != generated_tokens:
            raise RuntimeError(
                f"doc {doc_id} generation hidden rows mismatch: "
                f"{len(rows)} vs {generated_tokens}"
            )
        generated_hidden = (
            torch.stack(rows, dim=0).contiguous()
            if rows
            else torch.empty((0, hidden_size), dtype=torch.float32)
        )
        out_path = hidden_dir / f"doc_{doc_id}.safetensors"
        save_file({"hidden_states": generated_hidden}, str(out_path))
        written.append(
            {
                "doc_id": doc_id,
                "path": str(out_path),
                "shape": list(generated_hidden.shape),
            }
        )

    result = {
        "trace_path": str(trace_path),
        "split_source": (
            "scheduler_fallback_after_metadata"
            if metadata_split_error is not None
            else "scheduler"
        ),
        "trace_rows": int(sum(chunk.shape[0] for chunk in chunks)),
        "prefill_rows": int(consumed_prompt_rows),
        "decode_rows": int(consumed_decode_rows),
        "scheduler_generation_rows": int(scheduler_actual),
        "ignored_generation_rows": int(scheduler_actual - sum(write_limits)),
        "generated_hidden_rows": int(
            sum(int(row["generated_token_count"]) for row in batch_rows)
        ),
        "hidden_size": hidden_size,
        "written": written,
    }
    if metadata_split_error is not None:
        result["metadata_split_error"] = metadata_split_error
    return result


def load_doc_hidden(hidden_dir: Path, doc_id: int) -> torch.Tensor:
    path = hidden_dir / f"doc_{doc_id}.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor("hidden_states").to(torch.float32)


def _quantile(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, q).item())


def compute_hidden_mse(
    *,
    reference_hidden_dir: Path,
    candidate_hidden_dir: Path,
    replay_rows: list[dict[str, Any]],
    data_dir: Path,
    mode_name: str,
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    mse_tensors: list[torch.Tensor] = []
    doc_ids: list[torch.Tensor] = []
    token_positions: list[torch.Tensor] = []
    sample_rows: list[dict[str, Any]] = []
    sq_sum = 0.0
    abs_sum = 0.0
    elem_count = 0
    max_abs = 0.0
    hidden_size: int | None = None

    for row in replay_rows:
        doc_id = int(row["doc_id"])
        ref = load_doc_hidden(reference_hidden_dir, doc_id)
        cand = load_doc_hidden(candidate_hidden_dir, doc_id)
        if ref.shape != cand.shape:
            raise RuntimeError(
                f"hidden shape mismatch for doc {doc_id}: {tuple(ref.shape)} vs {tuple(cand.shape)}"
            )
        if hidden_size is None:
            hidden_size = int(ref.shape[1]) if ref.ndim == 2 else None
        diff = cand - ref
        row_mse = diff.square().mean(dim=1) if diff.numel() else torch.empty(0)
        row_abs = diff.abs().mean(dim=1) if diff.numel() else torch.empty(0)
        mse_tensors.append(row_mse.to(torch.float32).contiguous())
        doc_ids.append(torch.full((row_mse.numel(),), doc_id, dtype=torch.int32))
        token_positions.append(torch.arange(row_mse.numel(), dtype=torch.int32))
        sq_sum += float(diff.square().sum().item())
        abs_sum += float(diff.abs().sum().item())
        elem_count += diff.numel()
        if diff.numel():
            max_abs = max(max_abs, float(diff.abs().max().item()))
        sample_rows.append(
            {
                "mode": mode_name,
                "doc_id": doc_id,
                "subset_pos": row["subset_pos"],
                "original_index": row["original_index"],
                "generated_tokens": int(row["generated_token_count"]),
                "full_generated_tokens": int(
                    row.get("full_generated_token_count", row["generated_token_count"])
                ),
                "limit_truncated_tokens": int(
                    row.get("limit_truncated_generated_token_count", 0)
                ),
                "self_circulation_trimmed": bool(
                    row.get("self_circulation_trimmed", False)
                ),
                "self_circulation_period": int(
                    row.get("self_circulation_period", 0)
                ),
                "self_circulation_cycle_start": int(
                    row.get("self_circulation_cycle_start", 0)
                ),
                "self_circulation_dropped_tokens": int(
                    row.get("self_circulation_dropped_token_count", 0)
                ),
                "hidden_rows": int(row_mse.numel()),
                "mean_hidden_mse": float(row_mse.mean().item()) if row_mse.numel() else None,
                "median_hidden_mse": _quantile(row_mse, 0.5),
                "p90_hidden_mse": _quantile(row_mse, 0.9),
                "max_hidden_mse": float(row_mse.max().item()) if row_mse.numel() else None,
                "mean_hidden_mae": float(row_abs.mean().item()) if row_abs.numel() else None,
            }
        )

    mse_all = torch.cat(mse_tensors) if mse_tensors else torch.empty(0, dtype=torch.float32)
    doc_all = torch.cat(doc_ids) if doc_ids else torch.empty(0, dtype=torch.int32)
    pos_all = (
        torch.cat(token_positions)
        if token_positions
        else torch.empty(0, dtype=torch.int32)
    )
    torch.save(
        {
            "mse": mse_all,
            "doc_id": doc_all,
            "token_pos": pos_all,
        },
        data_dir / "token_mse.pt",
    )
    write_json(
        data_dir / "token_mse.meta.json",
        {
            "mode": mode_name,
            "rows": int(mse_all.numel()),
            "hidden_size": hidden_size,
            "reference_hidden_dir": str(reference_hidden_dir),
            "candidate_hidden_dir": str(candidate_hidden_dir),
            "self_circulation_docs": sum(
                1 for row in replay_rows if row.get("self_circulation_trimmed")
            ),
            "self_circulation_dropped_tokens": sum(
                int(row.get("self_circulation_dropped_token_count", 0))
                for row in replay_rows
            ),
            "limit_truncated_generated_tokens": sum(
                int(row.get("limit_truncated_generated_token_count", 0))
                for row in replay_rows
            ),
        },
    )
    write_csv(
        data_dir / "sample_mse.csv",
        sample_rows,
        [
            "mode",
            "doc_id",
            "subset_pos",
            "original_index",
            "generated_tokens",
            "full_generated_tokens",
            "limit_truncated_tokens",
            "self_circulation_trimmed",
            "self_circulation_period",
            "self_circulation_cycle_start",
            "self_circulation_dropped_tokens",
            "hidden_rows",
            "mean_hidden_mse",
            "median_hidden_mse",
            "p90_hidden_mse",
            "max_hidden_mse",
            "mean_hidden_mae",
        ],
    )

    metrics = {
        "mode": mode_name,
        "compared_hidden_rows": int(mse_all.numel()),
        "hidden_size": hidden_size,
        "mse": sq_sum / elem_count if elem_count else None,
        "mae": abs_sum / elem_count if elem_count else None,
        "max_abs": max_abs,
        "std_hidden_mse": float(mse_all.std(unbiased=False).item()) if mse_all.numel() else None,
        "median_hidden_mse": _quantile(mse_all, 0.5),
        "p90_hidden_mse": _quantile(mse_all, 0.9),
        "p95_hidden_mse": _quantile(mse_all, 0.95),
        "p99_hidden_mse": _quantile(mse_all, 0.99),
        "max_hidden_mse": float(mse_all.max().item()) if mse_all.numel() else None,
        "token_mse_file": str(data_dir / "token_mse.pt"),
        "sample_mse_file": str(data_dir / "sample_mse.csv"),
    }
    return metrics


def load_h100_qwen3_gate_baseline(path: Path | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    if path is None or not path.exists():
        return values
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") != "Qwen-Qwen3-8B":
                continue
            setting = row.get("setting", "")
            value = row.get("mean_hidden_mse")
            if setting and value:
                values[setting] = float(value)
    return values
