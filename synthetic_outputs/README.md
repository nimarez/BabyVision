# Synthetic Outputs

This folder contains the packaged synthetic BabyVision hard-case datasets.

## Kept Datasets

- `gemini_wrong_hard_cases_mixed_20260524T032756Z/`
  - Primary mixed set.
  - 50 examples total: 46 rendered cases and 4 AI-generated cases.
  - Includes `meta_data.jsonl`, `hard_cases_manifest.jsonl`, `hard_cases_manifest.json`, and per-example image / prompt / eval files.
  - Includes perceptual hash reports against `data/babyvision_data/images`.

- `gemini_wrong_hard_cases_20260524T022754Z/`
  - Rendered-only hard-case set.
  - 50 examples selected from deterministic rendered candidates where Gemini 3.1 Pro gave a clean wrong answer.

- `ai_generated_hard_cases_20260524T032756Z/`
  - AI-generated hard-case supplement.
  - 4 clean hard cases retained from AI-generated candidate batches.

## Cleanup Notes

Removed intermediate candidate and staging folders:

- `targeted_weaknesses_20260524T011657Z/`
- `gemini_wrong_hard_cases_20260524T013053Z/`
- `rendered_weakness_candidates_20260524T015111Z/`
- `rendered_weakness_candidates_20260524T021559Z/`
- `ai_weakness_candidates_20260524T023150Z/`
- `ai_weakness_candidates_20260524T030421Z/`
