# MCR-Bench

This directory contains the MCR-Bench data used by the workspace tests:
4,500 questions drawn from existing medical, legal, and financial datasets,
1,500 per domain. The data is kept in the same JSONL format consumed by
`tests/mcr-bench/run.py`.

Datasets:

- `combined_mcr_bench.jsonl`: all 4,500 questions, used by default.
- `med_data/combined_med_data.jsonl`: 1,500 medical questions.
- `legal_data/combined_legal_data.jsonl`: 1,500 legal questions.
- `financial_data/combined_financial_data.jsonl`: 1,500 financial questions.

## Sampling

Questions are sampled with the fixed seed `20260726` from the named test
splits of each source. PubMedQA, CareQA, and FinClaimRadar do not publish a
named held-out test split for the relevant data, so their complete evaluation
pools are used instead. Exact duplicate prompts are removed within each source
before sampling.

| Domain | Source | Questions |
|---|---|---:|
| Medical | MedQA | 500 |
| Medical | CareQA | 500 |
| Medical | PubMedQA | 500 |
| Legal | CaseHOLD | 625 |
| Legal | LexGLUE Unfair-ToS | 625 |
| Legal | LegalBench Insurance Policy | 133 |
| Legal | BarExam QA | 117 |
| Finance | FinQA | 500 |
| Finance | FinClaimRadar | 500 |
| Finance | Financial PhraseBank | 270 |
| Finance | FiQA-SA sentiment | 230 |

The combined file interleaves domains and sources so that every prefix stays
close to the final distribution; the domain files keep the same relative
order. Each row records its upstream dataset, revision, split, and index, and
carries a deterministic gold answer (`correct_option` and `scoring`).

Generated model outputs, plots, and table exports belong outside this data
directory, usually under `output/`, `figures/`, or `tables/`.
