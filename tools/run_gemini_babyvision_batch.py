#!/usr/bin/env python3
"""Prepare, submit, and collect a Vertex Gemini batch run for BabyVision."""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import CreateBatchJobConfig, JobState

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "babyvision_eval"))
from utils import extract_boxed_answer, format_choices  # noqa: E402


DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_LOCATION = "global"
DEFAULT_METADATA = REPO_ROOT / "data/babyvision_data/meta_data.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "babyvision_eval/results/gemini-3.1-pro-preview/batch_high_thinking"
COMPLETED_STATES = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def gcloud_config_value(name: str) -> str | None:
    try:
        value = capture(["gcloud", "config", "get-value", name])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return value if value and value != "(unset)" else None


def image_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type in {"image/png", "image/jpeg", "image/webp"}:
        return mime_type
    raise ValueError(f"Unsupported image type for {path}")


def load_tasks(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_question(item: dict[str, Any]) -> tuple[str, str]:
    if item["ansType"] == "blank":
        return item["question"], item["blankAns"]
    question = item["question"] + "\nChoices:\n" + format_choices(item["options"])
    answer = chr(65 + int(item["choiceAns"]))
    return question, answer


def normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(text.strip().lower().split())


def local_exact_judge(extracted_answer: str | None, answer: str) -> bool:
    return normalize_answer(extracted_answer) == normalize_answer(answer)


def request_for_task(
    item: dict[str, Any],
    *,
    metadata_path: Path,
    gcs_image_prefix: str,
    max_output_tokens: int,
    thinking_level: str,
    thinking_budget: int | None,
    media_resolution: str,
) -> dict[str, Any]:
    question, _ = make_question(item)
    benchmark_question = (
        question
        + "\nThink about the question and give your final answer in "
        "\\boxed{Answer} format. Keep the visible response concise."
    )
    thinking_config: dict[str, Any] = {}
    if thinking_budget is not None:
        thinking_config["thinkingBudget"] = thinking_budget
    else:
        thinking_config["thinkingLevel"] = thinking_level
    local_image = metadata_path.parent / item["image"]
    gcs_image_uri = f"{gcs_image_prefix.rstrip('/')}/{item['image']}"
    return {
        "key": str(item["taskId"]),
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "fileUri": gcs_image_uri,
                                "mimeType": image_mime_type(local_image),
                            }
                        },
                        {"text": benchmark_question},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT"],
                "mediaResolution": f"MEDIA_RESOLUTION_{media_resolution}",
                "thinkingConfig": thinking_config,
            },
        },
    }


def write_batch_input(
    path: Path,
    *,
    tasks: list[dict[str, Any]],
    metadata_path: Path,
    gcs_image_prefix: str,
    max_output_tokens: int,
    thinking_level: str,
    thinking_budget: int | None,
    media_resolution: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in tasks:
            request = request_for_task(
                item,
                metadata_path=metadata_path,
                gcs_image_prefix=gcs_image_prefix,
                max_output_tokens=max_output_tokens,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                media_resolution=media_resolution,
            )
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")


def text_from_response(response: dict[str, Any]) -> str:
    parts = (
        response.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    return "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()


def response_from_output_row(row: dict[str, Any]) -> tuple[str | None, dict[str, Any], str | None]:
    key = row.get("key")
    if "response" in row:
        return key, row["response"], row.get("status")
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        return key or prediction.get("key"), prediction.get("response", prediction), row.get("status")
    return key, row, row.get("status")


def collect_outputs(
    *,
    output_dir: Path,
    local_output_dir: Path,
    metadata_path: Path,
) -> Path:
    gcs_outputs = output_dir / "gcs_outputs"
    if gcs_outputs.exists():
        shutil.rmtree(gcs_outputs)
    gcs_outputs.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((output_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    run(["gcloud", "storage", "cp", "-r", manifest["gcs_output_uri"].rstrip("/") + "/*", str(gcs_outputs)])

    metadata = {int(item["taskId"]): item for item in load_tasks(metadata_path)}
    output_rows = []
    for path in sorted(gcs_outputs.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    output_rows.append(json.loads(line))

    results_by_id = {}
    for row in output_rows:
        key, response, status = response_from_output_row(row)
        if key is None:
            continue
        task_id = int(key)
        item = metadata[task_id]
        question, answer = make_question(item)
        benchmark_question = (
            question
            + "\nThink about the question and give your final answer in "
            "\\boxed{Answer} format."
        )
        model_output = text_from_response(response)
        extracted_answer = extract_boxed_answer(model_output)
        results_by_id[task_id] = {
            "Id": task_id,
            "Question": benchmark_question,
            "Model": manifest["model"],
            "JudgeModel": "local_exact",
            "ModelResult": model_output,
            "GroundTruth": answer,
            "ExtractedAnswer": extracted_answer,
            "LLMJudgeResult": local_exact_judge(extracted_answer, answer),
            "JudgeResult": "local_exact",
            "Type": item["type"],
            "Subtype": item["subtype"],
            "AnswerType": item["ansType"],
            "Image": item["image"],
            "BatchJob": manifest["job_name"],
            "BatchStatus": status,
            "UsageMetadata": response.get("usageMetadata", {}),
            "Error": None if model_output else status or "empty_model_output",
        }

    ordered = [results_by_id[int(item["taskId"])] for item in load_tasks(metadata_path)]
    result_path = local_output_dir / "model_results_run_1.json"
    result_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BabyVision through Vertex Gemini batch inference.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bucket", default="construction-takeoff-496319-document-ai")
    parser.add_argument("--gcs-prefix", default=None)
    parser.add_argument("--project", default=gcloud_config_value("project"))
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=2048,
        help=(
            "Explicit Gemini thinking token budget. Use -1 to omit it and use "
            "--thinking-level instead."
        ),
    )
    parser.add_argument(
        "--thinking-level",
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        default="HIGH",
    )
    parser.add_argument(
        "--media-resolution",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="HIGH",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Existing batch_manifest.json for status or collection.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.project:
        print("No Google Cloud project found. Pass --project.", file=sys.stderr)
        return 2

    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        client = genai.Client(
            vertexai=True,
            project=manifest.get("project", args.project),
            location=manifest.get("location", args.location),
        )
        job = client.batches.get(name=manifest["job_name"])
        manifest["job_state"] = str(job.state)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Job: {job.name}")
        print(f"State: {job.state}")

        if args.wait:
            while job.state not in COMPLETED_STATES:
                time.sleep(args.poll_seconds)
                job = client.batches.get(name=job.name)
                manifest["job_state"] = str(job.state)
                args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                print(f"Job state: {job.state}")

        if args.collect:
            if job.state != JobState.JOB_STATE_SUCCEEDED:
                print(f"Job is not ready to collect: {job.state}", file=sys.stderr)
                return 1
            result_path = collect_outputs(
                output_dir=args.manifest.parent,
                local_output_dir=args.manifest.parent,
                metadata_path=args.metadata,
            )
            print(f"Collected results: {result_path}")
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gcs_prefix = args.gcs_prefix or f"babyvision-batch/gemini-3.1-pro-preview-high/{run_id}"
    gcs_root = f"gs://{args.bucket}/{gcs_prefix.strip('/')}"
    gcs_image_prefix = f"{gcs_root}/data"
    gcs_input_uri = f"{gcs_root}/input/babyvision_requests.jsonl"
    gcs_output_uri = f"{gcs_root}/output"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_input = args.output_dir / "babyvision_requests.jsonl"
    manifest_path = args.output_dir / "batch_manifest.json"
    tasks = load_tasks(args.metadata)
    thinking_budget = None if args.thinking_budget < 0 else args.thinking_budget

    write_batch_input(
        local_input,
        tasks=tasks,
        metadata_path=args.metadata,
        gcs_image_prefix=gcs_image_prefix,
        max_output_tokens=args.max_output_tokens,
        thinking_level=args.thinking_level,
        thinking_budget=thinking_budget,
        media_resolution=args.media_resolution,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": args.project,
        "location": args.location,
        "model": args.model,
        "tasks": len(tasks),
        "thinking_level": args.thinking_level,
        "thinking_budget": thinking_budget,
        "media_resolution": args.media_resolution,
        "max_output_tokens": args.max_output_tokens,
        "gcs_input_uri": gcs_input_uri,
        "gcs_output_uri": gcs_output_uri,
        "job_name": None,
        "job_state": None,
    }

    if args.submit:
        run([
            "gcloud",
            "storage",
            "rsync",
            "-r",
            str(args.metadata.parent),
            gcs_image_prefix,
        ])
        run(["gcloud", "storage", "cp", str(local_input), gcs_input_uri])

        client = genai.Client(vertexai=True, project=args.project, location=args.location)
        job = client.batches.create(
            model=args.model,
            src=gcs_input_uri,
            config=CreateBatchJobConfig(
                display_name=f"babyvision-{args.model}-{run_id}",
                dest=gcs_output_uri,
            ),
        )
        manifest["job_name"] = job.name
        manifest["job_state"] = str(job.state)
        print(f"Submitted batch job: {job.name}")
        print(f"Initial state: {job.state}")

        if args.wait:
            while job.state not in COMPLETED_STATES:
                time.sleep(args.poll_seconds)
                job = client.batches.get(name=job.name)
                manifest["job_state"] = str(job.state)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                print(f"Job state: {job.state}")
            if args.collect and job.state == JobState.JOB_STATE_SUCCEEDED:
                result_path = collect_outputs(
                    output_dir=args.output_dir,
                    local_output_dir=args.output_dir,
                    metadata_path=args.metadata,
                )
                print(f"Collected results: {result_path}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Input JSONL: {local_input}")
    print(f"Manifest: {manifest_path}")
    print(f"GCS input: {gcs_input_uri}")
    print(f"GCS output: {gcs_output_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
