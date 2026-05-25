#!/usr/bin/env python3
"""Evaluate Vertex AI Gemini models on the BabyVision MLLM benchmark."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types
from google.genai.types import GenerateContentConfig, Modality
from tqdm import tqdm

from utils import extract_boxed_answer, format_choices


DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_LOCATION = "global"


def gcloud_config_value(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    value = result.stdout.strip()
    return value if value and value != "(unset)" else None


def image_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type in {"image/png", "image/jpeg", "image/webp"}:
        return mime_type
    raise ValueError(f"Unsupported image type for {path}; use PNG, JPEG, or WebP.")


def enum_value(enum_cls, name: str | None):
    if not name or name == "NONE":
        return None
    return getattr(enum_cls, name)


def response_text(response) -> str:
    if not response.candidates:
        raise RuntimeError("Model response did not include candidates.")
    candidate = response.candidates[0]
    parts = candidate.content.parts or [] if candidate.content else []
    text = "\n".join(part.text for part in parts if part.text).strip()
    if not text:
        finish_reason = getattr(candidate, "finish_reason", None)
        raise RuntimeError(f"Model response did not include text. finish_reason={finish_reason}")
    return text


def normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"^answer\s*[:：]\s*", "", text)
    text = text.strip("`'\" ")
    text = re.sub(r"\s+", "", text)
    text = text.replace("（", "(").replace("）", ")").replace("，", ",")
    text = re.sub(r"^\(([a-z])\)$", r"\1", text)
    text = re.sub(r"^\((-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?)+)\)$", r"\1", text)
    return text.rstrip(".")


def is_correct(extracted_answer: str | None, ground_truth: str) -> bool:
    normalized_model = normalize_answer(extracted_answer)
    normalized_truth = normalize_answer(ground_truth)
    if normalized_model == normalized_truth:
        return True

    model_number = re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_model)
    truth_number = re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_truth)
    if model_number and truth_number:
        return float(normalized_model) == float(normalized_truth)

    return False


def build_task(raw: dict[str, Any], data_dir: Path, *, final_only: bool) -> dict[str, Any]:
    image_path = data_dir / raw["image"]
    if raw["ansType"] == "blank":
        question = raw["question"]
        answer = raw["blankAns"]
    else:
        question = raw["question"] + "\nChoices:\n" + format_choices(raw["options"])
        answer = chr(65 + int(raw["choiceAns"]))

    if final_only:
        question = (
            question
            + "\nReturn only the final answer in exactly \\boxed{Answer} format. "
            + "Do not include explanation."
        )
    else:
        question = (
            question
            + "\nThink about the question, then give your final answer in exactly "
            + "\\boxed{Answer} format. Keep the final answer concise."
        )

    return {
        "Id": raw["taskId"],
        "Question": question,
        "GroundTruth": answer,
        "Type": raw["type"],
        "Subtype": raw["subtype"],
        "ImagePath": str(image_path),
    }


def generate_answer(
    client,
    task: dict[str, Any],
    *,
    model: str,
    max_output_tokens: int,
    thinking_level: str | None,
    media_resolution: str | None,
) -> tuple[str, Any]:
    image_path = Path(task["ImagePath"])
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=image_path.read_bytes(),
                    mime_type=image_mime_type(image_path),
                ),
                types.Part.from_text(text=task["Question"]),
            ],
        )
    ]

    thinking_config = None
    if thinking_level:
        thinking_config = types.ThinkingConfig(
            thinking_level=enum_value(types.ThinkingLevel, thinking_level)
        )

    config = GenerateContentConfig(
        response_modalities=[Modality.TEXT],
        max_output_tokens=max_output_tokens,
        temperature=0,
        thinking_config=thinking_config,
        media_resolution=enum_value(
            types.MediaResolution,
            f"MEDIA_RESOLUTION_{media_resolution}" if media_resolution else None,
        ),
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except errors.ClientError as exc:
        if thinking_config and "thinking_level is not supported" in str(exc):
            config.thinking_config = None
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        else:
            raise

    return response_text(response), getattr(response, "usage_metadata", None)


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if not usage:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {}


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        existing = json.load(f)
    return {int(item["Id"]): item for item in existing}


def write_results(path: Path, results_by_id: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [results_by_id[key] for key in sorted(results_by_id)]
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    correct = sum(1 for item in results if item.get("LLMJudgeResult"))
    accuracy = correct / total if total else 0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    by_type: dict[str, list[bool]] = {}
    for item in results:
        by_type.setdefault(item["Type"], []).append(bool(item.get("LLMJudgeResult")))
    for task_type in sorted(by_type):
        values = by_type[task_type]
        type_accuracy = sum(values) / len(values)
        print(f"  {task_type}: {type_accuracy * 100:.2f}% ({sum(values)}/{len(values)})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Vertex AI Gemini model on BabyVision."
    )
    parser.add_argument(
        "--test-json-path",
        default="../data/babyvision_data/meta_data.jsonl",
        help="Path to BabyVision meta_data.jsonl.",
    )
    parser.add_argument(
        "--output",
        default="results/gemini_3_1_pro_preview_vertex_exact.json",
        help="Output JSON path.",
    )
    parser.add_argument("--project", default=gcloud_config_value("project"))
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--limit-per-type",
        type=int,
        help="Select up to this many tasks from each BabyVision type.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--type", help="Optional exact task type filter.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--verbose-reasoning",
        action="store_true",
        help="Use the original reasoning-style prompt instead of final-answer-only.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--thinking-level",
        choices=["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"],
        default="HIGH",
    )
    parser.add_argument(
        "--media-resolution",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="HIGH",
    )
    args = parser.parse_args()

    if not args.project:
        print(
            "No Google Cloud project found. Pass --project or run "
            "`gcloud config set project PROJECT_ID`.",
            file=sys.stderr,
        )
        return 2

    test_json_path = Path(args.test_json_path)
    data_dir = test_json_path.parent
    output_path = Path(args.output)

    raw_tasks = [
        json.loads(line)
        for line in test_json_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [
        build_task(raw, data_dir, final_only=not args.verbose_reasoning)
        for raw in raw_tasks
    ]
    if args.type:
        tasks = [task for task in tasks if task["Type"] == args.type]
    if args.limit_per_type is not None:
        selected = []
        counts: dict[str, int] = {}
        for task in tasks:
            task_type = task["Type"]
            if counts.get(task_type, 0) >= args.limit_per_type:
                continue
            selected.append(task)
            counts[task_type] = counts.get(task_type, 0) + 1
        tasks = selected
    if args.offset:
        tasks = tasks[args.offset :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    results_by_id = load_existing(output_path) if args.resume else {}
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    thinking_level = None if args.thinking_level == "NONE" else args.thinking_level

    print(f"Model: {args.model}")
    print(f"Project/location: {args.project}/{args.location}")
    print(f"Tasks selected: {len(tasks)}")
    print(f"Output: {output_path}")

    for task in tqdm(tasks, desc="Evaluating"):
        task_id = int(task["Id"])
        if task_id in results_by_id:
            continue

        error = ""
        model_output = ""
        usage = None
        for attempt in range(1, args.retries + 1):
            try:
                model_output, usage = generate_answer(
                    client,
                    task,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                    thinking_level=thinking_level,
                    media_resolution=args.media_resolution,
                )
                break
            except Exception as exc:  # noqa: BLE001 - preserve per-task failures.
                error = str(exc)
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 30))

        extracted_answer = extract_boxed_answer(model_output)
        correct = is_correct(extracted_answer, task["GroundTruth"])
        result = {
            "Id": task_id,
            "Question": task["Question"],
            "ModelReasoning": "",
            "ModelResult": model_output,
            "GroundTruth": task["GroundTruth"],
            "ExtractedAnswer": extracted_answer,
            "LLMJudgeResult": correct,
            "ScoringMethod": "normalized_exact_match",
            "Type": task["Type"],
            "Subtype": task["Subtype"],
            "ModelName": args.model,
            "VertexProject": args.project,
            "VertexLocation": args.location,
            "UsageMetadata": usage_to_dict(usage),
        }
        if error and not model_output:
            result["Error"] = error
        results_by_id[task_id] = result
        write_results(output_path, results_by_id)
        if args.sleep:
            time.sleep(args.sleep)

    final_results = [results_by_id[int(task["Id"])] for task in tasks if int(task["Id"]) in results_by_id]
    print_summary(final_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
