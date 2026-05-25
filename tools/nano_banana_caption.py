#!/usr/bin/env python3
"""Caption/trace a visual logical puzzle image with Vertex AI Gemini Pro."""

from __future__ import annotations

import argparse
import mimetypes
import subprocess
import sys
from pathlib import Path

from google import genai
from google.genai import errors
from google.genai import types
from google.genai.types import GenerateContentConfig, Modality


DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_FALLBACK_MODELS = "gemini-2.5-pro"
DEFAULT_LOCATION = "global"
DEFAULT_PROMPT = """\
Inspect this visual logical puzzle image and return concise JSON only.
Use this schema:
{
  "type": "Visual Tracking | Fine-grained Discrimination | Visual Pattern Recognition | Spatial Perception",
  "subtype": "short subtype name",
  "question": "benchmark question a model should answer",
  "answer": "ground truth answer",
  "reasoning_trace": "brief visual reasoning that justifies the answer"
}
If the image is ambiguous or has multiple plausible answers, set "answer" to "needs_review" and explain why in reasoning_trace.
"""


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


def generate_caption(
    client,
    model: str,
    contents,
    *,
    max_output_tokens: int | None,
    thinking_level: str | None,
):
    thinking_config = None
    if thinking_level:
        thinking_config = types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, thinking_level)
        )

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=GenerateContentConfig(
            response_modalities=[Modality.TEXT],
            max_output_tokens=max_output_tokens,
            thinking_config=thinking_config,
        ),
    )


def response_text(response) -> str:
    if not response.candidates:
        raise RuntimeError("Model response did not include candidates.")
    parts = response.candidates[0].content.parts or []
    text = "\n".join(part.text for part in parts if part.text).strip()
    if not text:
        raise RuntimeError("Model response did not include text.")
    return strip_json_fence(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Caption and produce a reasoning trace for a puzzle image with Gemini Pro."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output")
    parser.add_argument("--project", default=gcloud_config_value("project"))
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Vertex AI captioning/reasoning model. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--fallback-model",
        default=DEFAULT_FALLBACK_MODELS,
        help=(
            "Comma-separated fallback model list if --model is unavailable or returns "
            f"no text. Defaults to {DEFAULT_FALLBACK_MODELS}."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens for captioning. Defaults to 4096.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"],
        default="HIGH",
        help="Gemini thinking level for models that support it. Defaults to HIGH.",
    )
    args = parser.parse_args()

    if not args.project:
        print(
            "No Google Cloud project found. Pass --project or run "
            "`gcloud config set project PROJECT_ID`.",
            file=sys.stderr,
        )
        return 2

    image_path = Path(args.image)
    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type=image_mime_type(image_path),
    )
    prompt_part = types.Part.from_text(text=args.prompt)
    contents = [types.Content(role="user", parts=[prompt_part, image_part])]

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    fallback_models = [
        model.strip()
        for model in args.fallback_model.split(",")
        if model.strip() and model.strip() != args.model
    ]
    models_to_try = [args.model, *fallback_models]
    errors_seen: list[str] = []
    text = ""
    model_used = ""
    thinking_level = None if args.thinking_level == "NONE" else args.thinking_level
    for index, model in enumerate(models_to_try):
        try:
            try:
                response = generate_caption(
                    client,
                    model,
                    contents,
                    max_output_tokens=args.max_output_tokens,
                    thinking_level=thinking_level,
                )
            except errors.ClientError as exc:
                if thinking_level and "thinking_level is not supported" in str(exc):
                    response = generate_caption(
                        client,
                        model,
                        contents,
                        max_output_tokens=args.max_output_tokens,
                        thinking_level=None,
                    )
                else:
                    raise
            text = response_text(response)
            model_used = model
            break
        except (errors.ClientError, RuntimeError) as exc:
            errors_seen.append(f"{model}: {exc}")
            if index < len(models_to_try) - 1:
                print(
                    f"{model} unavailable or empty ({exc}); falling back to "
                    f"{models_to_try[index + 1]}.",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    "All caption models failed: " + " | ".join(errors_seen)
                ) from exc

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(f"model_used: {model_used}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
