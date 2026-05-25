#!/usr/bin/env python3
"""Run BabyVision evaluation with Vertex AI Gemini models."""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types
from google.genai.types import GenerateContentConfig, Modality
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "babyvision_eval"))
from utils import LLM_JUDGE_PROMPT, extract_boxed_answer, format_choices  # noqa: E402


DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_LOCATION = "global"
DEFAULT_METADATA = REPO_ROOT / "data/babyvision_data/meta_data.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "babyvision_eval/results/gemini-3.1-pro-preview"

thread_local = threading.local()


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


def get_client(project: str, location: str) -> genai.Client:
    client = getattr(thread_local, "client", None)
    if client is None:
        client = genai.Client(vertexai=True, project=project, location=location)
        thread_local.client = client
    return client


def image_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type in {"image/png", "image/jpeg", "image/webp"}:
        return mime_type
    raise ValueError(f"Unsupported image type for {path}")


def thinking_config(level: str | None) -> types.ThinkingConfig | None:
    if not level or level == "NONE":
        return None
    return types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, level))


def response_text(response) -> str:
    if not response.candidates:
        return ""
    content = response.candidates[0].content
    if not content or not content.parts:
        return ""
    return "\n".join(part.text for part in content.parts if part.text).strip()


def usage_to_dict(response) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {}


def generate_text_with_usage(
    client: genai.Client,
    *,
    model: str,
    contents: list[types.Content] | str,
    max_output_tokens: int,
    thinking_level: str,
    media_resolution: str | None,
    temperature: float,
) -> str:
    media_resolution_value = None
    if media_resolution and media_resolution != "NONE":
        media_resolution_value = getattr(
            types.MediaResolution, f"MEDIA_RESOLUTION_{media_resolution}"
        )
    config = GenerateContentConfig(
        response_modalities=[Modality.TEXT],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        thinking_config=thinking_config(thinking_level),
        media_resolution=media_resolution_value,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except errors.ClientError as exc:
        if thinking_level != "NONE" and "thinking_level is not supported" in str(exc):
            config.thinking_config = None
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        else:
            raise
    return response_text(response), usage_to_dict(response)


def generate_text(
    client: genai.Client,
    *,
    model: str,
    contents: list[types.Content] | str,
    max_output_tokens: int,
    thinking_level: str,
    media_resolution: str | None,
    temperature: float,
) -> str:
    text, _ = generate_text_with_usage(
        client,
        model=model,
        contents=contents,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
        media_resolution=media_resolution,
        temperature=temperature,
    )
    return text


def call_with_retries(fn, *, retries: int, task_id: int) -> Any:
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - keep benchmark resilient.
            last_error = exc
            if attempt == retries - 1:
                break
            print(
                f"task {task_id}: {exc}; retrying in {delay:.0f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"task {task_id} failed after {retries} attempts") from last_error


def load_tasks(metadata_path: Path, limit: int | None) -> list[dict[str, Any]]:
    tasks = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
            if limit and len(tasks) >= limit:
                break
    return tasks


def make_question(item: dict[str, Any]) -> tuple[str, str]:
    if item["ansType"] == "blank":
        return item["question"], item["blankAns"]
    question = item["question"] + "\nChoices:\n" + format_choices(item["options"])
    answer = chr(65 + int(item["choiceAns"]))
    return question, answer


def local_exact_judge(extracted_answer: str | None, answer: str) -> bool:
    if extracted_answer is None:
        return False
    normalize = lambda text: "".join(str(text).lower().split())
    return normalize(extracted_answer) == normalize(answer)


def judge_answer(
    client: genai.Client,
    *,
    judge_model: str,
    question: str,
    answer: str,
    extracted_answer: str | None,
    retries: int,
    thinking_level: str,
) -> tuple[bool, str]:
    if extracted_answer is None:
        return False, "no_extracted_answer"

    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        groundtruth=answer,
        modeloutput=extracted_answer,
    )

    def call() -> str:
        return generate_text(
            client,
            model=judge_model,
            contents=prompt,
            max_output_tokens=64,
            thinking_level=thinking_level,
            media_resolution=None,
            temperature=0.0,
        )

    judge_output = call_with_retries(call, retries=retries, task_id=-1)
    clean = judge_output.strip().lower()
    if "true" in clean:
        return True, judge_output
    if "false" in clean:
        return False, judge_output
    return local_exact_judge(extracted_answer, answer), f"unparsed_judge: {judge_output}"


def process_task(
    item: dict[str, Any],
    *,
    metadata_path: Path,
    project: str,
    location: str,
    model: str,
    judge_model: str | None,
    max_output_tokens: int,
    thinking_level: str,
    judge_thinking_level: str,
    media_resolution: str,
    retries: int,
) -> dict[str, Any]:
    client = get_client(project, location)
    question, answer = make_question(item)
    benchmark_question = (
        question
        + "\nThink about the question and give your final answer in "
        "\\boxed{Answer} format."
    )
    image_path = metadata_path.parent / item["image"]
    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type=image_mime_type(image_path),
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                image_part,
                types.Part.from_text(text=benchmark_question),
            ],
        )
    ]

    def model_call() -> tuple[str, dict[str, Any]]:
        return generate_text_with_usage(
            client,
            model=model,
            contents=contents,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            media_resolution=media_resolution,
            temperature=0.0,
        )

    try:
        model_output, usage_metadata = call_with_retries(
            model_call, retries=retries, task_id=item["taskId"]
        )
        extracted_answer = extract_boxed_answer(model_output)
        if judge_model:
            judged, judge_output = judge_answer(
                client,
                judge_model=judge_model,
                question=benchmark_question,
                answer=answer,
                extracted_answer=extracted_answer,
                retries=retries,
                thinking_level=judge_thinking_level,
            )
        else:
            judged = local_exact_judge(extracted_answer, answer)
            judge_output = "local_exact"
        error = None
    except Exception as exc:  # noqa: BLE001 - preserve failed rows.
        model_output = ""
        usage_metadata = {}
        extracted_answer = None
        judged = False
        judge_output = ""
        error = str(exc)

    return {
        "Id": item["taskId"],
        "Question": benchmark_question,
        "Model": model,
        "JudgeModel": judge_model or "local_exact",
        "ModelResult": model_output,
        "GroundTruth": answer,
        "ExtractedAnswer": extracted_answer,
        "LLMJudgeResult": judged,
        "JudgeResult": judge_output,
        "Type": item["type"],
        "Subtype": item["subtype"],
        "AnswerType": item["ansType"],
        "Image": item["image"],
        "UsageMetadata": usage_metadata,
        "Error": error,
    }


def read_existing(jsonl_path: Path) -> dict[int, dict[str, Any]]:
    if not jsonl_path.exists():
        return {}
    results = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                results[int(row["Id"])] = row
    return results


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BabyVision with Vertex Gemini.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="model_results_run_1")
    parser.add_argument("--project", default=gcloud_config_value("project"))
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument(
        "--media-resolution",
        choices=["NONE", "LOW", "MEDIUM", "HIGH"],
        default="HIGH",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"],
        default="HIGH",
    )
    parser.add_argument(
        "--judge-thinking-level",
        choices=["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"],
        default="LOW",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.project:
        print("No Google Cloud project found. Pass --project.", file=sys.stderr)
        return 2

    tasks = load_tasks(args.metadata, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.run_name}.jsonl"
    json_path = args.output_dir / f"{args.run_name}.json"

    results_by_id = read_existing(jsonl_path) if args.resume else {}
    pending = [task for task in tasks if int(task["taskId"]) not in results_by_id]
    judge_model = None if args.no_judge else args.judge_model

    print(f"Model: {args.model}")
    print(f"Judge: {judge_model or 'local_exact'}")
    print(f"Tasks: {len(tasks)} ({len(pending)} pending)")
    print(f"Output: {json_path}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(
                process_task,
                task,
                metadata_path=args.metadata,
                project=args.project,
                location=args.location,
                model=args.model,
                judge_model=judge_model,
                max_output_tokens=args.max_output_tokens,
                thinking_level=args.thinking_level,
                judge_thinking_level=args.judge_thinking_level,
                media_resolution=args.media_resolution,
                retries=args.retries,
            ): task
            for task in pending
        }
        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Gemini BabyVision",
        ):
            row = future.result()
            results_by_id[int(row["Id"])] = row
            append_jsonl(jsonl_path, row)

    ordered = [results_by_id[int(task["taskId"])] for task in tasks]
    json_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    correct = sum(1 for row in ordered if row["LLMJudgeResult"])
    errors = sum(1 for row in ordered if row.get("Error"))
    print(f"Correct: {correct}/{len(ordered)} ({correct / len(ordered):.2%})")
    print(f"Errored rows: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
