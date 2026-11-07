# error-contribution

MedQA residual/KV error-contribution replay runner. It uses the FP32 LayerCast generation artifacts from `tests/final-hidden-mse`. The sampled diagnostic path can sample generated-token positions with a fixed seed; HEAL-family attribution should use `--full-replay-attribution` with `--max-replay-tokens-per-doc 1024` so prefix-KV means are computed from complete per-document token prefixes.

The FP32 LayerCast reference stores selected rows for:

- `layer_<i>.input_layernorm.output.residual`
- `layer_<i>.post_attention_layernorm.output.residual`
- `attention.k_full.<i>` and `attention.v_full.<i>`
- `forward.norm.input`
- `forward.norm.hidden`

BF16/FP16/HEAL-family candidate replays use Triton attention, read the matching FP32 reference batch, and write compact per-stage per-row MSE payloads. They also keep only the selected raw candidate tensors needed for the current attribution split (`attention.q`, full-prefix K/V, `attention.core_output`, MLP input/output, and the FP32 MLP oracle). For current HEAL-family attribution, those payloads are aggregated into:

- `*_residual_kv_mse_points.csv`: one row per selected token/layer, with residual MSE, KV MSE, local update MSE, cross terms, and prefix-KV MSE.
- `*_residual_kv_wls_by_layer.csv`: per-layer WLS coefficients and diagnostic means.
- `*_recursive_truncation_contribution_ratios.{csv,png}`: stacked Attention/MLP/KV final-share bars with a right-axis layer-ending MSE line.
- `streaming-panels/*.png`: 16-layer scatter panels for MLP, attention input, attention prefix-KV, and KV input relations.

The current contribution model is fit per layer from per-token scalar rows:

- `kv_mse ~= input_residual_mse`
- `attention_delta_mse ~= input_residual_mse + prefix_kv_mse`
- `mlp_delta_mse ~= post_attention_residual_mse`

Here `attention_delta_mse` and `mlp_delta_mse` are `MSE(after - before)`. The reported cross term is `E[before * (after - before)]`, computed from `mean_product - before_mse`. `prefix_kv_mse` is a within-document prefix mean and includes the token's own KV row. The local fits use normalized Theil-Sen slopes with a non-negative weighted-median intercept. MSE jump outliers are excluded before fitting and plotting layer MSE.

The sampled path and PDF diagnostics are useful for quick checks. With `--write-pdfs`, the runner can also write `*_residual_mse_diagnostics.pdf` and `*_local_propagation_regression_diagnostics.pdf` under `figures/<gpu>/error-contribution/<model>/`.

Dry-run plan check:

```bash
python tests/error-contribution/run.py \
  --model qwen3-8b \
  --limit 2 \
  --dry-run
```

Full run example:

```bash
python tests/error-contribution/run.py \
  --all-models \
  --reference-generation-dir output/final-hidden-mse/<reference_run>/generation_artifacts \
  --run-id error_contribution_$(date -u +%Y%m%dT%H%M%SZ)
```

Before running, `vllm/` must be clean and no incremental-build patch should be active. The script applies one selected patch, runs the requested modes, then reverts the patch. Default reference/BF16/FP16 modes use `VLLM_ERROR_CONTRIBUTION`; `heal_base`, `heal_plus`, and `heal_full` use the additional `VLLM_ERROR_CONTRIBUTION_HEAL` patch and should be run separately from the reference/BF16/FP16 modes.

## HEAL Variant Tuning

`tune_heal_variants.py` is the canonical selector for HEAL++ and HEAL-FULL+ layer downgrades. It reads the calibration matrix selected by `--contribution-run-id` and uses `medqa_validation8_seed20260528.jsonl` as the held-out MSE validation set.

The current selection rule uses variant-specific median final-hidden-MSE budgets relative to the non-downgraded baseline for the same mode: 1.2x for HEAL++ and 1.5x for HEAL-FULL+. For each candidate `k`, the tuner evaluates two downgrade orders:

- `prefix`: degrade layer indices `0..k-1`.
- `calibration_low`: degrade the `k` lowest-contribution layers from the calibration matrix (`attention + kv` share for HEAL++, MLP share for HEAL-FULL+).

The search treats a `k` as feasible when either order stays within that variant median-MSE budget, and records the lower-MSE passing order as `best_order`. It probes `k=4` first, doubles while feasible, then binary-searches the bracket between the largest feasible probe and the first failed probe; if `k=4` fails, it binary-searches below 4. Outputs include `validation_evaluations.csv`, `selection_orders.csv`, `heal_variant_tuning.json`, `heal_variant_env.json`, and a run-level `heal_variant_tuning_summary.csv`.

Example:

```bash
python3.12 tests/error-contribution/tune_heal_variants.py   --all-models   --reference-generation-dir output/final-hidden-mse/<reference_run>/generation_artifacts
```

## Streaming HEAL Attribution

Recommended HEAL-family run shape for one model:

```bash
python3.12 tests/error-contribution/run.py \
  --model qwen3-8b \
  --run-id error_contribution_streaming_YYYYMMDD \
  --full-replay-attribution \
  --limit 10 \
  --max-replay-tokens-per-doc 1024 \
  --prefill-batch-size 10 \
  --prefill-token-budget 10240 \
  --candidate-modes heal_base heal_plus heal_full \
  --cleanup-trace-batches
```

For large models, run the reference first with `--skip-candidates`, then run HEAL-family candidates with `--skip-reference --cleanup-trace-batches`. This keeps only manifests and compact CSV/PNG outputs under `output/error-contribution`.

For memory-heavy runs, candidates can also be run one mode at a time with `--skip-reference`; keep `reference_batches` until the last candidate finishes, then remove the remaining temporary `reference_batches` and `mse_batches` directories.


Policy export for other machines:

```bash
python3.12 tests/error-contribution/export_heal_policy.py   --tuning-run-id heal_variant_tuning_pluspp1p2_fullplus1p5_YYYYMMDD   --output tests/error-contribution/policies/heal_variant_tuning_pluspp1p2_fullplus1p5_YYYYMMDD.json
```

Policy JSON files under `tests/error-contribution/policies/` are generated local artifacts and are not committed. Copy only the exported JSON to another machine and pass it as `--heal-policy-json`; the downstream runners will apply the per-model settings to `heal_plus_plus_*` and `heal_full_plus_*` modes.
