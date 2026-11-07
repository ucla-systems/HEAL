#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge portable HEAL policy JSON files. Later inputs override duplicate models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")




VARIANT_RATIO_KEYS = {"HEAL++": "heal_plus_plus", "HEAL-FULL+": "heal_full_plus"}


def fill_variant_thresholds(model_payload: dict[str, Any]) -> None:
    ratios = model_payload.get("variant_allowed_mse_ratios", {})
    if not isinstance(ratios, dict):
        ratios = {}
    variants = model_payload.get("variants", {})
    if not isinstance(variants, dict):
        return
    default_ratio = model_payload.get("allowed_mse_ratio")
    for variant_name, ratio_key in VARIANT_RATIO_KEYS.items():
        variant = variants.get(variant_name)
        if not isinstance(variant, dict) or variant.get("allowed_mse_ratio") is not None:
            continue
        ratio = ratios.get(ratio_key, default_ratio)
        if ratio is not None:
            variant["allowed_mse_ratio"] = ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged: dict[str, Any] = {"policy_version": 1, "source_tuning_run_id": args.source_id or "merged", "models": {}}
    source_ids: list[str] = []
    for path in args.input:
        payload = read_json(path)
        if payload.get("source_tuning_run_id"):
            source_ids.append(str(payload["source_tuning_run_id"]))
        models = payload.get("models", {})
        if not isinstance(models, dict):
            raise TypeError(f"policy {path} has non-dict models")
        for model_label, model_payload in models.items():
            if model_label not in merged["models"]:
                fill_variant_thresholds(model_payload)
                merged["models"][model_label] = model_payload
                continue
            existing = merged["models"][model_label]
            if not isinstance(existing, dict) or not isinstance(model_payload, dict):
                merged["models"][model_label] = model_payload
                continue
            existing_variants = existing.get("variants", {})
            new_variants = model_payload.get("variants", {})
            combined = dict(existing)
            for key, value in model_payload.items():
                if key != "variants":
                    combined[key] = value
            aliases = set(existing.get("aliases", [])) | set(model_payload.get("aliases", []))
            if aliases:
                combined["aliases"] = sorted(str(alias) for alias in aliases)
            if isinstance(existing_variants, dict) and isinstance(new_variants, dict):
                combined["variants"] = {**existing_variants, **new_variants}
            else:
                combined["variants"] = new_variants or existing_variants
            fill_variant_thresholds(combined)
            merged["models"][model_label] = combined
    if args.source_id is None:
        merged["source_tuning_run_id"] = "+".join(dict.fromkeys(source_ids))
    write_json(args.output, merged)
    print(args.output)
    print("models=" + str(len(merged["models"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
