#!/usr/bin/env python3
"""Generate a single visual logical puzzle synthetic image with Vertex AI Gemini."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from google import genai
from google.genai.types import GenerateContentConfig, Modality
from PIL import Image


DEFAULT_MODEL = "gemini-3-pro-image-preview"
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


def write_response_parts(response, output_path: Path) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_parts: list[str] = []
    image_count = 0

    for part in response.candidates[0].content.parts:
        if part.text:
            text_parts.append(part.text)
        elif part.inline_data:
            image_count += 1
            image = Image.open(BytesIO(part.inline_data.data))
            if image_count == 1:
                image.save(output_path)
            else:
                numbered = output_path.with_name(
                    f"{output_path.stem}_{image_count}{output_path.suffix}"
                )
                image.save(numbered)

    if image_count == 0:
        raise RuntimeError("Model response did not include an image.")

    return text_parts


def default_prompt_record_path(output_path: Path) -> Path:
    return output_path.with_suffix(".prompt.json")


def write_prompt_record(
    path: Path,
    *,
    prompt: str,
    output_path: Path,
    model: str,
    project: str,
    location: str,
    response_text: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/nano_banana_generate.py",
        "model": model,
        "project": project,
        "location": location,
        "output_image": str(output_path),
        "prompt": prompt,
        "response_text": response_text,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a visual logical puzzle synthetic image with Vertex AI Gemini Pro Image."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="synthetic_outputs/nano_banana_pro.png")
    parser.add_argument(
        "--prompt-output",
        help="Path for the prompt record JSON. Defaults to a sibling .prompt.json file.",
    )
    parser.add_argument(
        "--no-prompt-record",
        action="store_true",
        help="Do not write the prompt record sidecar.",
    )
    parser.add_argument("--project", default=gcloud_config_value("project"))
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Vertex AI image model. Defaults to {DEFAULT_MODEL}.",
    )
    args = parser.parse_args()

    if not args.project:
        print(
            "No Google Cloud project found. Pass --project or run "
            "`gcloud config set project PROJECT_ID`.",
            file=sys.stderr,
        )
        return 2

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    response = client.models.generate_content(
        model=args.model,
        contents=args.prompt,
        config=GenerateContentConfig(
            response_modalities=[Modality.TEXT, Modality.IMAGE],
        ),
    )

    output_path = Path(args.output)
    text_parts = write_response_parts(response, output_path)
    if not args.no_prompt_record:
        prompt_record_path = (
            Path(args.prompt_output)
            if args.prompt_output
            else default_prompt_record_path(output_path)
        )
        write_prompt_record(
            prompt_record_path,
            prompt=args.prompt,
            output_path=output_path,
            model=args.model,
            project=args.project,
            location=args.location,
            response_text=text_parts,
        )
        print(f"saved prompt: {prompt_record_path}")
    for text in text_parts:
        print(text)

    print(f"saved image: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
