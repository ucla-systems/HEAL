# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for applying tuned HEAL++/HEAL-FULL+ layer policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _resolve_path(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def load_heal_policy(path: str | Path | None, *, root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    policy_path = _resolve_path(path, root=root)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["_policy_path"] = str(policy_path)
    return payload


def _env_from_tuning_result(result: dict[str, Any]) -> dict[str, Any]:
    heal_plus_plus = result.get("heal_plus_plus", {})
    heal_full_plus = result.get("heal_full_plus", {})
    return {
        "HEAL++": {
            "selection_order": heal_plus_plus.get("best_order", ""),
            "HEAL_MODE": "plus",
            "HEAL_SKIP_QKV_QUANT_LAYERS": heal_plus_plus.get("best_layers_csv", ""),
        },
        "HEAL-FULL+": {
            "selection_order": heal_full_plus.get("best_order", ""),
            "table_label": "HEAL-FULL",
            "HEAL_MODE": "full",
            "HEAL_FP16_GEMM_COMPENSATION": "1",
            "HEAL_SKIP_FP16_COMP_LAYERS": heal_full_plus.get("best_layers_csv", ""),
            "HEAL_SKIP_INT8_COMP_LAYERS": heal_full_plus.get("best_layers_csv", ""),
        },
    }


def _model_entry_candidates(policy: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    models = policy.get("models", policy)
    if isinstance(models, dict):
        return [
            (str(key), value)
            for key, value in models.items()
            if isinstance(value, dict)
        ]
    if isinstance(models, list):
        entries: list[tuple[str, dict[str, Any]]] = []
        for result in models:
            if not isinstance(result, dict):
                continue
            label = str(result.get("model_label") or result.get("model") or "")
            if not label:
                continue
            entries.append(
                (
                    label,
                    {
                        "model_arg": result.get("model_arg"),
                        "model_label": label,
                        "model_id": result.get("model_id"),
                        "variants": _env_from_tuning_result(result),
                    },
                )
            )
        return entries
    return []


def _variants_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    variants = entry.get("variants")
    if isinstance(variants, dict):
        return variants
    return entry


def _variant_aliases(name: str) -> tuple[str, ...]:
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "heal++":
        return ("HEAL++", "heal_plus_plus", "heal++")
    if normalized in {"heal-full+", "heal-full-plus", "heal-fullplus"}:
        return ("HEAL-FULL+", "heal_full_plus", "HEAL-Full+", "heal-full-plus")
    return (name,)


def policy_for_model(
    policy: dict[str, Any] | None,
    *,
    model_arg: str | None = None,
    model_id: str | None = None,
    model_label: str | None = None,
) -> dict[str, Any] | None:
    if policy is None:
        return None
    keys = {
        str(value)
        for value in (model_arg, model_id, model_label)
        if value is not None and str(value)
    }
    keys.update(key.replace("/", "-") for key in list(keys))
    keys.update(key.replace("/", "_") for key in list(keys))
    for key, entry in _model_entry_candidates(policy):
        aliases = {key}
        raw_aliases = entry.get("aliases", [])
        if isinstance(raw_aliases, list):
            aliases.update(str(alias) for alias in raw_aliases)
        for field in ("model_arg", "model_id", "model_label", "model"):
            value = entry.get(field)
            if value:
                aliases.add(str(value))
        aliases.update(alias.replace("/", "-") for alias in list(aliases))
        aliases.update(alias.replace("/", "_") for alias in list(aliases))
        if keys & aliases:
            return _variants_from_entry(entry)
    path = policy.get("_policy_path", "<inline>")
    wanted = ", ".join(sorted(keys)) or "<unknown>"
    raise KeyError(f"HEAL policy {path} has no entry for model {wanted}")


def policy_skip_layers(
    policy: dict[str, Any] | None,
    *,
    model_arg: str | None = None,
    model_id: str | None = None,
    model_label: str | None = None,
    variant: str,
    field: str,
) -> str | None:
    variants = policy_for_model(
        policy,
        model_arg=model_arg,
        model_id=model_id,
        model_label=model_label,
    )
    if variants is None:
        return None
    variant_payload: dict[str, Any] | None = None
    for name in _variant_aliases(variant):
        value = variants.get(name)
        if isinstance(value, dict):
            variant_payload = value
            break
    if variant_payload is None:
        path = policy.get("_policy_path", "<inline>") if policy else "<none>"
        raise KeyError(f"HEAL policy {path} has no {variant} entry for {model_label or model_id or model_arg}")
    if field not in variant_payload:
        path = policy.get("_policy_path", "<inline>") if policy else "<none>"
        raise KeyError(f"HEAL policy {path} {variant} entry has no {field}")
    value = variant_payload[field]
    return "" if value is None else str(value)
