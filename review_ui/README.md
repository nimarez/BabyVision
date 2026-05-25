# BabyVision Review UI

This is a local, dependency-free UI for reviewing BabyVision synthetic outputs.
It is meant for checking two things per generated case:

- whether the image follows the generation prompt
- whether the question, answer, and rubric are correct

## Run

From the repo root:

```bash
python3 review_ui/server.py --port 8787
```

Open `http://127.0.0.1:8787`.

The server uses only the Python standard library. If port `8787` is already in
use, choose another port:

```bash
python3 review_ui/server.py --port 8788
```

## Review Workflow

1. Select a synthetic dataset from the left sidebar.
2. Pick a case from the case list, or filter by search, task type, or review status.
3. Inspect the image against the displayed generation prompt and generation trace.
4. Mark `Prompt Adherence` as `Pass`, `Fail`, or `Unsure`.
5. Inspect the question, ground-truth answer, model answer, and model rationale.
6. Mark `Answer And Rubric` as `Pass`, `Fail`, or `Unsure`.
7. Add notes or corrected answer/question text when needed.
8. Click `Save Review`.

The sidebar counters update as reviews are saved. The status filter can be used
to return to unreviewed cases or failures.

## Inputs

The UI discovers datasets from:

```text
synthetic_outputs/*/hard_cases_manifest.json
```

Each manifest entry should point to the image, `.prompt.json`, and `.eval.json`
files for the case.

## Outputs

Review state is written under `synthetic_outputs/reviews/`:

- `synthetic_outputs/reviews/<dataset>.reviews.jsonl`
- `synthetic_outputs/reviews/<dataset>.latest.json`

The JSONL file is append-only audit history. The `latest.json` file stores the
current review state by case id and is what the UI loads on refresh.

## Notes

The server only serves files inside this repo. It does not call external model
APIs, and it does not modify the generated synthetic examples.
