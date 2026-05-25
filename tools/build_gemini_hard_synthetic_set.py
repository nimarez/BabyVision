#!/usr/bin/env python3
"""Generate visual logical puzzle synthetic cases that Gemini answers incorrectly."""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import errors, types
from google.genai.types import GenerateContentConfig, Modality
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "babyvision_eval"))
from utils import extract_boxed_answer  # noqa: E402


DEFAULT_PROJECT = "construction-takeoff-496319"
DEFAULT_LOCATION = "global"
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
DEFAULT_EVAL_MODEL = "gemini-3.1-pro-preview"


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def image_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type in {"image/png", "image/jpeg", "image/webp"}:
        return mime_type
    raise ValueError(f"Unsupported image type for {path}")


def normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def local_exact_judge(extracted_answer: str | None, answer: str) -> bool:
    return normalize_answer(extracted_answer) == normalize_answer(answer)


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_json_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def response_text(response: Any) -> str:
    if not response.candidates:
        return ""
    content = response.candidates[0].content
    if not content or not content.parts:
        return ""
    return "\n".join(part.text for part in content.parts if part.text).strip()


def usage_to_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {}


def thinking_config(level: str) -> types.ThinkingConfig | None:
    if level == "NONE":
        return None
    return types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, level))


def call_with_backoff(
    label: str,
    fn: Callable[[], Any],
    *,
    retries: int,
    base_delay: float,
) -> Any:
    delay = base_delay
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except (errors.APIError, errors.ClientError, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            print(
                f"{label}: {exc}; retrying in {delay:.0f}s "
                f"({attempt}/{retries})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"{label} failed after {retries} attempts") from last_error


def save_image_response(response: Any, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text_parts: list[str] = []
    image_count = 0
    if not response.candidates:
        raise RuntimeError("Image response had no candidates.")
    for part in response.candidates[0].content.parts:
        if part.text:
            text_parts.append(part.text)
        elif part.inline_data:
            image_count += 1
            image = Image.open(BytesIO(part.inline_data.data))
            target = path if image_count == 1 else path.with_name(
                f"{path.stem}_{image_count}{path.suffix}"
            )
            image.save(target)
    if image_count == 0:
        raise RuntimeError("Image response did not include an image.")
    return text_parts


def generate_image(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    output_path: Path,
    retries: int,
) -> list[str]:
    def call() -> Any:
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=GenerateContentConfig(
                response_modalities=[Modality.TEXT, Modality.IMAGE],
            ),
        )

    response = call_with_backoff(
        f"generate {output_path.name}",
        call,
        retries=retries,
        base_delay=90,
    )
    return save_image_response(response, output_path)


def generate_text(
    client: genai.Client,
    *,
    model: str,
    image_path: Path,
    prompt: str,
    max_output_tokens: int,
    thinking_level: str,
    media_resolution: str,
    retries: int,
    label: str,
) -> tuple[str, dict[str, Any]]:
    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type=image_mime_type(image_path),
    )
    prompt_part = types.Part.from_text(text=prompt)
    contents = [types.Content(role="user", parts=[prompt_part, image_part])]
    media_resolution_value = getattr(
        types.MediaResolution, f"MEDIA_RESOLUTION_{media_resolution}"
    )

    def call() -> Any:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=GenerateContentConfig(
                response_modalities=[Modality.TEXT],
                max_output_tokens=max_output_tokens,
                temperature=0.0,
                thinking_config=thinking_config(thinking_level),
                media_resolution=media_resolution_value,
            ),
        )

    response = call_with_backoff(label, call, retries=retries, base_delay=10)
    return response_text(response), usage_to_dict(response)


def extract_fallback_answer(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed.strip()
    clean = text.strip()
    letter_matches = re.findall(r"\b[A-D]\b", clean.upper())
    if letter_matches:
        return letter_matches[-1]
    number_matches = re.findall(r"\b\d+\b", clean)
    if number_matches:
        return number_matches[-1]
    return None


def base_style() -> str:
    return (
        "Create one BabyVision benchmark puzzle image. Use a clean white "
        "worksheet background, crisp vector-like shapes, high contrast, "
        "printed puzzle layout, and no photorealism. Do not include an answer "
        "key, explanation, or highlighted solution."
    )


def connect_lines_spec(rng: random.Random, case_id: str) -> dict[str, Any]:
    answer = str(rng.randint(1, 6))
    start = rng.choice(["A", "B", "C", "D", "E", "F"])
    prompt = (
        f"{base_style()} Draw a hard line-tracing puzzle with six start dots "
        "labeled A-F on the left and six endpoint dots labeled 1-6 on the "
        f"right. The continuous path starting at {start} must terminate at "
        f"endpoint {answer}. Include many close crossings, under-over visual "
        "ambiguity, and at least five bends, but every path must be continuous "
        "and unbroken. Use thin black paths and small colored labels. Do not "
        "draw arrows or show the solution. Variant id: "
        f"{case_id}."
    )
    return {
        "case_id": case_id,
        "type": "Visual Tracking",
        "subtype": "Connect the lines",
        "question": (
            f"Start at label {start} and follow its continuous path. "
            "Which numbered endpoint does it reach?"
        ),
        "answer": answer,
        "prompt": prompt,
    }


def maze_spec(rng: random.Random, case_id: str) -> dict[str, Any]:
    answer = str(rng.randint(1, 4))
    prompt = (
        f"{base_style()} Draw a square black-line maze with entrance S on the "
        "left edge and four exits labeled 1, 2, 3, 4 around the right and "
        f"bottom edges. Only exit {answer} must be reachable from S. Include "
        "dead ends, tight parallel corridors, and no drawn solution path. "
        "Make walls unambiguous and labels outside the maze. Variant id: "
        f"{case_id}."
    )
    return {
        "case_id": case_id,
        "type": "Visual Tracking",
        "subtype": "Maze",
        "question": "Starting from S, which numbered exit can be reached without crossing a wall?",
        "answer": answer,
        "prompt": prompt,
    }


def cube_unfold_spec(rng: random.Random, case_id: str) -> dict[str, Any]:
    answer = rng.choice(["A", "B", "C", "D"])
    symbols = ["yellow star", "green plus", "black crescent", "purple stripe"]
    rng.shuffle(symbols)
    correct_symbol = symbols[0]
    prompt = (
        f"{base_style()} Draw a cube-net spatial puzzle. Show a cross-shaped "
        "cube net with a red circle on the center face and a blue triangle on "
        "the face directly above it. The face that folds to the right side "
        f"must contain a {correct_symbol}. Below the net show four answer "
        "options A-D, each a small square with one symbol. Option "
        f"{answer} must show the {correct_symbol} and be the only correct "
        "option. The other options should be plausible symbols from the net. "
        "Keep labels clear. Variant id: "
        f"{case_id}."
    )
    return {
        "case_id": case_id,
        "type": "Spatial Perception",
        "subtype": "3D Cube Unfold",
        "question": (
            "Fold the cube net with the red circle as the front face and the "
            "blue triangle as the top face. Which option is on the right face?"
        ),
        "answer": answer,
        "prompt": prompt,
    }


def pattern_3d_spec(rng: random.Random, case_id: str) -> dict[str, Any]:
    answer = rng.choice(["A", "B", "C", "D"])
    prompt = (
        f"{base_style()} Draw a 3 by 3 grid of isometric cube-cluster tiles. "
        "The top-right tile is missing and shown as a dashed blank with a "
        "question mark. The visible tiles follow two simultaneous rules: top "
        "color rotates across columns and side shading alternates across rows. "
        f"Below the grid show options A-D. Option {answer} must be the only "
        "tile that satisfies both rules; the distractors should each violate "
        "one subtle rule. Avoid duplicate options. Variant id: "
        f"{case_id}."
    )
    return {
        "case_id": case_id,
        "type": "Spatial Perception",
        "subtype": "3D Pattern Completion",
        "question": "Which option completes the missing isometric tile?",
        "answer": answer,
        "prompt": prompt,
    }


def count_blocks_spec(rng: random.Random, case_id: str) -> dict[str, Any]:
    heights = [
        [rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)]
        for _ in range(3)
    ]
    answer = str(sum(sum(row) for row in heights))
    prompt = (
        f"{base_style()} Draw one isometric stack of unit cubes on a 3 by 3 "
        "footprint. The column heights by row from back to front are "
        f"{heights[0]}, {heights[1]}, {heights[2]}, for a total of {answer} "
        "cubes including hidden support cubes. Use subtle gray side shading "
        "and visible top faces. Do not write the total count in the image. "
        "Variant id: "
        f"{case_id}."
    )
    return {
        "case_id": case_id,
        "type": "Spatial Perception",
        "subtype": "Count 3D blocks",
        "question": "How many unit cubes are in the stacked block structure, including hidden support cubes?",
        "answer": answer,
        "prompt": prompt,
    }


def find_same_spec(rng: random.Random, case_id: str) -> dict[str, Any]:
    answer = rng.choice(["A", "B", "C", "D"])
    prompt = (
        f"{base_style()} Draw a fine-grained matching puzzle. At the top, show "
        "one reference icon: a small rounded square with a notch, one interior "
        "dot, and a tiny tail. Below it show four options labeled A-D. Option "
        f"{answer} must be exactly identical to the reference. The other "
        "options must each differ by exactly one subtle feature: notch side, "
        "dot position, tail direction, or tail length. Keep all options crisp "
        "and similarly sized. Variant id: "
        f"{case_id}."
    )
    return {
        "case_id": case_id,
        "type": "Fine-grained Discrimination",
        "subtype": "Find the same",
        "question": "Which option is exactly identical to the reference icon?",
        "answer": answer,
        "prompt": prompt,
    }


def find_shadow_spec(rng: random.Random, case_id: str) -> dict[str, Any]:
    answer = rng.choice(["A", "B", "C", "D"])
    prompt = (
        f"{base_style()} Draw a shadow-matching puzzle. At the top show one "
        "irregular colored object made of connected rectangles, with a hook, "
        "a lower notch, and a small side tab. Below it show four black "
        f"silhouette options labeled A-D. Option {answer} must match exactly "
        "when color is ignored. The other options should be near-matches: one "
        "mirrored, one rotated, and one missing a small feature. Variant id: "
        f"{case_id}."
    )
    return {
        "case_id": case_id,
        "type": "Fine-grained Discrimination",
        "subtype": "Find the shadow",
        "question": "Which shadow option matches the object exactly after ignoring color?",
        "answer": answer,
        "prompt": prompt,
    }


def overlay_spec(rng: random.Random, case_id: str) -> dict[str, Any]:
    answer = rng.choice(["A", "B", "C", "D"])
    prompt = (
        f"{base_style()} Draw a visual overlay puzzle. Show a target square at "
        "the top containing black and blue marks made by overlaying two "
        "transparent layers. Below it show four candidate pairs labeled A-D, "
        f"where each pair has two small layer squares. Pair {answer} must be "
        "the only pair that overlays exactly to form the target. Distractors "
        "should add, omit, or shift exactly one mark. Variant id: "
        f"{case_id}."
    )
    return {
        "case_id": case_id,
        "type": "Visual Pattern Recognition",
        "subtype": "Overlay Patterns",
        "question": "Which option pair overlays to make the target pattern?",
        "answer": answer,
        "prompt": prompt,
    }


SPEC_BUILDERS = [
    connect_lines_spec,
    cube_unfold_spec,
    pattern_3d_spec,
    find_same_spec,
    maze_spec,
    count_blocks_spec,
    find_shadow_spec,
    overlay_spec,
]


def make_spec(rng: random.Random, candidate_num: int) -> dict[str, Any]:
    builder = SPEC_BUILDERS[(candidate_num - 1) % len(SPEC_BUILDERS)]
    suffix = rng.randint(100000, 999999)
    return builder(rng, f"candidate_{candidate_num:04d}_{suffix}")


def validation_prompt(spec: dict[str, Any]) -> str:
    return f"""Inspect this generated visual logical puzzle image and return JSON only.
Use this schema:
{{
  "valid": true,
  "supports_intended_answer": true,
  "visual_answer": "{spec['answer']}",
  "reasoning_trace": "brief visual reasoning",
  "quality_notes": "ambiguity or artifact notes"
}}
Question: {spec['question']}
Intended answer: {spec['answer']}
The image is valid only if the question can be answered unambiguously from the visible puzzle and the intended answer is visibly supported. If labels, options, paths, walls, or shapes are malformed, set valid to false.
"""


def eval_prompt(spec: dict[str, Any]) -> str:
    return f"""Answer this visual logical puzzle.
Question: {spec['question']}
Think privately and keep the visible response concise. Put the final answer only in \\boxed{{...}}.
"""


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def existing_retained_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def copy_case_files(
    *,
    image_path: Path,
    prompt_path: Path,
    validation_path: Path,
    eval_path: Path,
    hard_dir: Path,
    hard_index: int,
) -> dict[str, str]:
    hard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hard_{hard_index:03d}_{image_path.stem}"
    targets = {
        "image": hard_dir / f"{stem}{image_path.suffix}",
        "prompt": hard_dir / f"{stem}.prompt.json",
        "validation": hard_dir / f"{stem}.validation.json",
        "eval": hard_dir / f"{stem}.eval.json",
    }
    shutil.copy2(image_path, targets["image"])
    shutil.copy2(prompt_path, targets["prompt"])
    shutil.copy2(validation_path, targets["validation"])
    shutil.copy2(eval_path, targets["eval"])
    return {key: str(value) for key, value in targets.items()}


def load_seed_specs(seed_manifest: Path | None) -> list[dict[str, Any]]:
    if not seed_manifest:
        return []
    data = json.loads(seed_manifest.read_text(encoding="utf-8"))
    root = seed_manifest.parent
    seed_specs = []
    for index, example in enumerate(data.get("examples", []), start=1):
        image_path = root / f"{index:02d}_{example['id']}.png"
        if not image_path.exists():
            continue
        seed_specs.append(
            {
                "case_id": example["id"],
                "type": example["type"],
                "subtype": example["subtype"],
                "question": example["question"],
                "answer": example["answer"],
                "prompt": example["prompt"],
                "seed_image_path": str(image_path),
            }
        )
    return seed_specs


def process_candidate(
    *,
    candidate_num: int,
    spec: dict[str, Any],
    output_dir: Path,
    image_client: genai.Client,
    eval_client: genai.Client,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_dir = output_dir / "candidates"
    base = candidate_dir / f"candidate_{candidate_num:04d}_{spec['case_id']}"
    image_path = base.with_suffix(".png")
    prompt_path = base.with_suffix(".prompt.json")
    validation_path = base.with_suffix(".validation.json")
    eval_path = base.with_suffix(".eval.json")

    response_text_parts: list[str] = []
    if spec.get("seed_image_path"):
        image_path = Path(spec["seed_image_path"])
        prompt_path = image_path.with_suffix(".prompt.json")
    elif not image_path.exists():
        response_text_parts = generate_image(
            image_client,
            model=args.image_model,
            prompt=spec["prompt"],
            output_path=image_path,
            retries=args.retries,
        )
        write_json(
            prompt_path,
            {
                "created_at": now_iso(),
                "generator": "tools/build_gemini_hard_synthetic_set.py",
                "model": args.image_model,
                "project": args.project,
                "location": args.location,
                "output_image": str(image_path),
                "case_spec": spec,
                "prompt": spec["prompt"],
                "response_text": response_text_parts,
            },
        )

    validation_text, validation_usage = generate_text(
        eval_client,
        model=args.eval_model,
        image_path=image_path,
        prompt=validation_prompt(spec),
        max_output_tokens=args.validation_max_output_tokens,
        thinking_level=args.thinking_level,
        media_resolution=args.media_resolution,
        retries=args.retries,
        label=f"validate {image_path.name}",
    )
    try:
        validation = parse_json_object(validation_text)
    except Exception as exc:  # noqa: BLE001
        validation = {
            "valid": False,
            "supports_intended_answer": False,
            "visual_answer": None,
            "reasoning_trace": "",
            "quality_notes": f"validation_json_parse_error: {exc}; raw={validation_text}",
        }
    validation_record = {
        "created_at": now_iso(),
        "model": args.eval_model,
        "prompt": validation_prompt(spec),
        "raw_output": validation_text,
        "parsed": validation,
        "usage_metadata": validation_usage,
    }
    write_json(validation_path, validation_record)

    is_valid = bool(validation.get("valid")) and bool(
        validation.get("supports_intended_answer")
    )
    if not is_valid:
        return {
            "status": "rejected_invalid",
            "candidate_num": candidate_num,
            "case_spec": spec,
            "image": str(image_path),
            "prompt_record": str(prompt_path),
            "validation": str(validation_path),
            "reason": validation.get("quality_notes") or validation,
        }

    model_text, usage = generate_text(
        eval_client,
        model=args.eval_model,
        image_path=image_path,
        prompt=eval_prompt(spec),
        max_output_tokens=args.eval_max_output_tokens,
        thinking_level=args.thinking_level,
        media_resolution=args.media_resolution,
        retries=args.retries,
        label=f"eval {image_path.name}",
    )
    extracted = extract_fallback_answer(model_text)
    correct = local_exact_judge(extracted, spec["answer"])
    eval_record = {
        "created_at": now_iso(),
        "model": args.eval_model,
        "project": args.project,
        "location": args.location,
        "thinking_level": args.thinking_level,
        "media_resolution": args.media_resolution,
        "max_output_tokens": args.eval_max_output_tokens,
        "question": spec["question"],
        "ground_truth": spec["answer"],
        "prompt": eval_prompt(spec),
        "model_output": model_text,
        "extracted_answer": extracted,
        "correct": correct,
        "usage_metadata": usage,
    }
    write_json(eval_path, eval_record)

    status = "rejected_correct" if correct else "retained_wrong"
    return {
        "status": status,
        "candidate_num": candidate_num,
        "case_spec": spec,
        "image": str(image_path),
        "prompt_record": str(prompt_path),
        "validation": str(validation_path),
        "eval": str(eval_path),
        "ground_truth": spec["answer"],
        "extracted_answer": extracted,
        "model_output": model_text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and retain valid synthetic cases that Gemini gets wrong."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--project", default=gcloud_config_value("project") or DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)
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
    parser.add_argument("--eval-max-output-tokens", type=int, default=8192)
    parser.add_argument("--validation-max-output-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = args.output_dir / "run_config.json"
    all_records_path = args.output_dir / "all_candidates.jsonl"
    retained_path = args.output_dir / "retained_hard_cases.jsonl"
    hard_dir = args.output_dir / "hard_cases"

    if not run_config_path.exists():
        write_json(
            run_config_path,
            {
                "created_at": now_iso(),
                "target_count": args.target_count,
                "max_candidates": args.max_candidates,
                "project": args.project,
                "location": args.location,
                "image_model": args.image_model,
                "eval_model": args.eval_model,
                "thinking_level": args.thinking_level,
                "media_resolution": args.media_resolution,
                "eval_max_output_tokens": args.eval_max_output_tokens,
                "validation_max_output_tokens": args.validation_max_output_tokens,
                "seed": args.seed,
                "seed_manifest": str(args.seed_manifest) if args.seed_manifest else None,
            },
        )

    rng = random.Random(args.seed)
    image_client = genai.Client(vertexai=True, project=args.project, location=args.location)
    eval_client = genai.Client(vertexai=True, project=args.project, location=args.location)
    seed_specs = load_seed_specs(args.seed_manifest)
    retained_count = existing_retained_count(retained_path)

    print(
        f"Starting hard-case build: retained={retained_count}, "
        f"target={args.target_count}",
        flush=True,
    )

    for candidate_num in range(1, args.max_candidates + 1):
        if retained_count >= args.target_count:
            break
        spec = seed_specs[candidate_num - 1] if candidate_num <= len(seed_specs) else make_spec(rng, candidate_num)
        print(
            f"[{candidate_num}/{args.max_candidates}] "
            f"{spec['type']} / {spec['subtype']} target={spec['answer']} "
            f"retained={retained_count}/{args.target_count}",
            flush=True,
        )
        record = process_candidate(
            candidate_num=candidate_num,
            spec=spec,
            output_dir=args.output_dir,
            image_client=image_client,
            eval_client=eval_client,
            args=args,
        )
        append_jsonl(all_records_path, record)
        if record["status"] == "retained_wrong":
            retained_count += 1
            copied = copy_case_files(
                image_path=Path(record["image"]),
                prompt_path=Path(record["prompt_record"]),
                validation_path=Path(record["validation"]),
                eval_path=Path(record["eval"]),
                hard_dir=hard_dir,
                hard_index=retained_count,
            )
            retained_record = {
                **record,
                "retained_index": retained_count,
                "hard_case_files": copied,
            }
            append_jsonl(retained_path, retained_record)
            print(
                f"  retained hard case {retained_count}: "
                f"truth={record['ground_truth']} extracted={record['extracted_answer']}",
                flush=True,
            )
        else:
            print(f"  {record['status']}", flush=True)

    print(
        f"Finished loop with retained={retained_count}/{args.target_count}. "
        f"Output: {args.output_dir}",
        flush=True,
    )
    return 0 if retained_count >= args.target_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
