#!/usr/bin/env python3
"""Generate AI-created BabyVision weakness candidates with prompt records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors
from google.genai.types import GenerateContentConfig, Modality
from PIL import Image


DEFAULT_PROJECT = "construction-takeoff-496319"
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-3-pro-image-preview"


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


def base_prompt() -> str:
    return (
        "Create one BabyVision-style visual reasoning benchmark image. "
        "Use a clean printed worksheet style on a white background, crisp "
        "geometric shapes, small labels, no photorealism, no watermark, and "
        "no explanatory answer key. It must be unambiguous and solvable from "
        "the image alone. Do not include the written question in the image."
    )


CASES: list[dict[str, Any]] = [
    {
        "slug": "metro_switch_route",
        "type": "Visual Tracking",
        "subtype": "Metro map",
        "answer": "D",
        "question": "Follow the blue route from START through every switch. Which terminal option does it reach?",
        "prompt": (
            f"{base_prompt()} Draw a metro-map route tracing puzzle with a START "
            "station on the left and four terminal stations labeled A, B, C, D "
            "on the right. Draw several colored route lines with crossings and "
            "small switch diamonds. The blue route from START must terminate at "
            "D. Other route segments should look plausible and cross nearby, "
            "but the blue route must remain continuous. Do not highlight D as "
            "the answer."
        ),
    },
    {
        "slug": "tangled_threads",
        "type": "Visual Tracking",
        "subtype": "Connect the lines",
        "answer": "2",
        "question": "Start at the red dot and follow the same continuous thread. Which numbered endpoint does it reach?",
        "prompt": (
            f"{base_prompt()} Draw a tangled thread tracing puzzle. There is a "
            "red start dot on the left and endpoints numbered 1, 2, 3, 4, 5 "
            "on the right. The red thread must be one continuous thin curve "
            "that passes through close overlaps and ends at endpoint 2. Use "
            "other gray threads as distractors. Use small bridge gaps or "
            "over-under cues so the correct red thread is valid but hard."
        ),
    },
    {
        "slug": "maze_ports",
        "type": "Visual Tracking",
        "subtype": "Maze",
        "answer": "3",
        "question": "Starting at S, which numbered port can be reached without crossing a wall?",
        "prompt": (
            f"{base_prompt()} Draw a compact square maze with entrance S and "
            "five ports numbered 1 to 5 around the boundary. Exactly port 3 "
            "must be reachable from S. Include loops, blocked corridors, and "
            "narrow parallel passages. Keep walls black and paths white."
        ),
    },
    {
        "slug": "shadow_tabs",
        "type": "Fine-grained Discrimination",
        "subtype": "Find the shadow",
        "answer": "C",
        "question": "Which shadow option matches the object exactly after ignoring color?",
        "prompt": (
            f"{base_prompt()} Draw one irregular colored object made of connected "
            "rounded rectangles, with a notch, a side tab, and a tiny step on "
            "one edge. Below it draw four black silhouettes labeled A, B, C, D. "
            "Option C must match the object exactly. The others should be near "
            "misses: mirrored, missing the tiny step, and shifted side tab."
        ),
    },
    {
        "slug": "same_icon_microdetail",
        "type": "Fine-grained Discrimination",
        "subtype": "Find the same",
        "answer": "B",
        "question": "Which option is exactly identical to the reference icon?",
        "prompt": (
            f"{base_prompt()} Draw a reference icon at top: an abstract bug-like "
            "glyph with two antennae, three dots, one short tail, and one small "
            "missing notch. Below it draw options A, B, C, D. Option B must be "
            "exactly identical to the reference. Each other option differs by "
            "one subtle detail: dot count, notch side, antenna length, or tail "
            "angle."
        ),
    },
    {
        "slug": "same_texture_tile",
        "type": "Fine-grained Discrimination",
        "subtype": "Find the same",
        "answer": "D",
        "question": "Which option tile exactly matches the reference tile?",
        "prompt": (
            f"{base_prompt()} Draw a reference 4x4 mini tile containing small "
            "black, blue, and orange marks. Below it draw four candidate 4x4 "
            "mini tiles labeled A-D. Option D must be the exact match. The "
            "other options should each swap one tiny mark, rotate one mark, or "
            "change one color."
        ),
    },
    {
        "slug": "overlay_sparse_marks",
        "type": "Visual Pattern Recognition",
        "subtype": "Overlay Patterns",
        "answer": "A",
        "question": "Which option pair overlays to make the target pattern?",
        "prompt": (
            f"{base_prompt()} Draw a target square at the top containing sparse "
            "marks: one black crescent, a blue diagonal line, two black dots, "
            "and a tiny orange square. Below draw four candidate layer pairs "
            "labeled A, B, C, D. Each option is two small transparent layer "
            "squares side by side. Pair A must overlay exactly to form the "
            "target; B-D each add, omit, or shift one mark."
        ),
    },
    {
        "slug": "overlay_grid_symbols",
        "type": "Visual Pattern Recognition",
        "subtype": "Overlay Patterns",
        "answer": "C",
        "question": "Which two-layer option reconstructs the target grid?",
        "prompt": (
            f"{base_prompt()} Draw a 5x5 target grid with a few cells containing "
            "small shapes. Below draw options A-D, each showing two 5x5 layer "
            "grids. Option C must be the only pair whose union exactly matches "
            "the target. Distractors should be very close, with one extra or "
            "missing symbol."
        ),
    },
    {
        "slug": "cube_net_symbols",
        "type": "Spatial Perception",
        "subtype": "3D Cube Unfold",
        "answer": "B",
        "question": "Fold the cube net. If the red circle is front and the blue triangle is top, which option is on the right face?",
        "prompt": (
            f"{base_prompt()} Draw a cube net with six square faces. The center "
            "face has a red circle and the face above it has a blue triangle. "
            "Arrange the other faces with distinct symbols. The face that folds "
            "to the right side must correspond to option B. Below the net draw "
            "four option squares labeled A-D with the face symbols. Do not mark "
            "the correct option."
        ),
    },
    {
        "slug": "cube_net_color_edges",
        "type": "Spatial Perception",
        "subtype": "3D Cube Unfold",
        "answer": "D",
        "question": "After folding the net, which option shows the face opposite the green square?",
        "prompt": (
            f"{base_prompt()} Draw a cube-net puzzle with colored faces and small "
            "edge marks. Ask visually by layout only: the green square's opposite "
            "face must correspond to option D. Show options A-D as colored/symbol "
            "face squares below. Use a valid foldable cube net."
        ),
    },
    {
        "slug": "isometric_missing_tile",
        "type": "Spatial Perception",
        "subtype": "3D Pattern Completion",
        "answer": "A",
        "question": "Which option completes the missing isometric tile?",
        "prompt": (
            f"{base_prompt()} Draw a 3x3 grid of isometric cube-cluster tiles with "
            "the bottom-right tile missing. The sequence should combine color "
            "rotation and left/right face shading. Below draw options A-D. Option "
            "A must be the only tile satisfying both rules; B-D should violate "
            "one subtle rule."
        ),
    },
    {
        "slug": "count_hidden_cubes",
        "type": "Spatial Perception",
        "subtype": "Count 3D blocks",
        "answer": "19",
        "question": "How many unit cubes are in the stack, including hidden support cubes?",
        "prompt": (
            f"{base_prompt()} Draw an isometric stack of unit cubes on a small "
            "footprint. The total number of cubes including hidden supports must "
            "be exactly 19. Make the visible arrangement plausible but hard to "
            "count, with partial occlusion and different column heights. Do not "
            "write the number 19 in the image."
        ),
    },
    {
        "slug": "paper_fold_holes",
        "type": "Spatial Perception",
        "subtype": "Paper Folding",
        "answer": "C",
        "question": "After unfolding the paper, which option shows the hole pattern?",
        "prompt": (
            f"{base_prompt()} Draw a paper-folding hole-punch puzzle. Show three "
            "fold steps on the left and the final folded paper with two punched "
            "holes. On the right show unfolded options A-D. Option C must show "
            "the correct unfolded hole pattern. Distractors should be plausible "
            "mirror or rotation errors."
        ),
    },
    {
        "slug": "rotation_pattern",
        "type": "Visual Pattern Recognition",
        "subtype": "Rotation Patterns",
        "answer": "B",
        "question": "Which option continues the rotation pattern?",
        "prompt": (
            f"{base_prompt()} Draw a sequence of five abstract arrow-and-dot "
            "tiles, with the sixth tile missing. The rule is rotation plus one "
            "dot moving corner-to-corner. Below draw options A-D. Option B must "
            "continue the pattern; distractors should each satisfy only one part "
            "of the rule."
        ),
    },
    {
        "slug": "mirror_pattern",
        "type": "Visual Pattern Recognition",
        "subtype": "Mirroring Patterns",
        "answer": "D",
        "question": "Which option is the correct mirror completion?",
        "prompt": (
            f"{base_prompt()} Draw a left-half/right-half mirror completion puzzle "
            "with an irregular arrangement of small icons. One quadrant is blank. "
            "Below draw options A-D. Option D must complete the mirror symmetry; "
            "other options should be near misses with one misplaced icon."
        ),
    },
    {
        "slug": "shadow_rotation_combo",
        "type": "Fine-grained Discrimination",
        "subtype": "Find the shadow",
        "answer": "A",
        "question": "Which option is the exact shadow of the object without rotating it?",
        "prompt": (
            f"{base_prompt()} Draw a colored object composed of a crescent-like "
            "piece, a short bar, and two small square protrusions. Draw four "
            "black shadow options A-D below. Option A is the exact unrotated "
            "shadow. The others are rotated, mirrored, or missing a protrusion."
        ),
    },
    {
        "slug": "route_letters",
        "type": "Visual Tracking",
        "subtype": "Lines Observation",
        "answer": "G",
        "question": "Follow the dashed path from START. Which letter does it pass last before the finish?",
        "prompt": (
            f"{base_prompt()} Draw a path-observation puzzle. A dashed path from "
            "START to FINISH winds through scattered letters A-H. The last letter "
            "the path passes before FINISH must be G. Add nearby distractor "
            "letters and crossing paths, but the dashed path remains clear."
        ),
    },
]

MIRROR_TAIL_CASES: list[dict[str, Any]] = [
    {
        "slug": "mirror_quadrant_icons_dense",
        "type": "Visual Pattern Recognition",
        "subtype": "Mirroring Patterns",
        "answer": "C",
        "question": "Which option completes the missing quadrant by mirror symmetry?",
        "prompt": (
            f"{base_prompt()} Draw a 2x2 mirror-completion puzzle. Three "
            "quadrants are filled with 6-8 tiny geometric icons each, arranged "
            "around a vertical and horizontal mirror axis. The top-right quadrant "
            "is blank with a question mark. Below draw options A-D. Option C "
            "must be the only exact mirror completion. Distractors should be very "
            "close with one icon swapped or one position slightly wrong."
        ),
    },
    {
        "slug": "mirror_left_right_symbols",
        "type": "Visual Pattern Recognition",
        "subtype": "Mirroring Patterns",
        "answer": "A",
        "question": "Which option is the exact left-right mirror of the reference panel?",
        "prompt": (
            f"{base_prompt()} Draw a reference panel containing a small asymmetric "
            "arrangement of arrows, dots, triangles, and a crescent. Below draw "
            "four candidate panels A-D. Option A must be the exact horizontal "
            "mirror of the reference. Other options should rotate one symbol, "
            "miss one dot, or mirror only part of the panel."
        ),
    },
    {
        "slug": "mirror_diagonal_completion",
        "type": "Visual Pattern Recognition",
        "subtype": "Mirroring Patterns",
        "answer": "D",
        "question": "Which option completes the pattern across the diagonal mirror line?",
        "prompt": (
            f"{base_prompt()} Draw a square pattern with a dashed diagonal mirror "
            "line from top-left to bottom-right. One triangular region is blank. "
            "Place small colored marks and arrows on the filled side. Below draw "
            "options A-D. Option D must be the exact diagonal mirror completion; "
            "distractors should be plausible near misses."
        ),
    },
    {
        "slug": "paper_fold_four_holes",
        "type": "Spatial Perception",
        "subtype": "Paper Folding",
        "answer": "B",
        "question": "After unfolding the paper, which option shows the final hole pattern?",
        "prompt": (
            f"{base_prompt()} Draw a paper-folding puzzle with two fold steps and "
            "two hole punches on the folded paper. Show unfolded options A-D as "
            "grids of holes. Option B must be correct. Distractors should reflect "
            "common mirror and rotation mistakes."
        ),
    },
    {
        "slug": "paper_fold_corner_cut",
        "type": "Spatial Perception",
        "subtype": "Paper Folding",
        "answer": "D",
        "question": "Which option shows the paper after unfolding?",
        "prompt": (
            f"{base_prompt()} Draw a paper folding and corner-cut puzzle. Show a "
            "square paper folded twice, then one triangular corner cut and one "
            "small circular punch. Show unfolded options A-D. Option D must be "
            "the correct unfolded result."
        ),
    },
    {
        "slug": "overlay_color_layers_complex",
        "type": "Visual Pattern Recognition",
        "subtype": "Overlay Patterns",
        "answer": "C",
        "question": "Which option pair overlays to reproduce the target?",
        "prompt": (
            f"{base_prompt()} Draw a target square made of two transparent layers "
            "with black strokes, blue dots, and orange mini-squares. Below show "
            "four layer-pair options A-D. Option C must overlay exactly. Each "
            "distractor should be off by exactly one shifted mark."
        ),
    },
    {
        "slug": "overlay_rotated_grid",
        "type": "Visual Pattern Recognition",
        "subtype": "Overlay Patterns",
        "answer": "B",
        "question": "Which pair of transparent tiles overlays to match the target grid?",
        "prompt": (
            f"{base_prompt()} Draw a 4x4 target grid containing small symbols. "
            "Below draw four candidate pairs A-D, each with two transparent 4x4 "
            "tiles. Option B must exactly overlay to the target; distractors "
            "should be near-matches with one rotated or missing symbol."
        ),
    },
    {
        "slug": "rotation_double_rule",
        "type": "Visual Pattern Recognition",
        "subtype": "Rotation Patterns",
        "answer": "A",
        "question": "Which option continues the two-rule rotation sequence?",
        "prompt": (
            f"{base_prompt()} Draw a six-tile sequence where the final tile is "
            "missing. Each tile has an asymmetric arrow and two small dots. The "
            "arrow rotates 90 degrees each step while the dots swap diagonals. "
            "Below draw options A-D. Option A must satisfy both rules."
        ),
    },
    {
        "slug": "rotation_clock_hands",
        "type": "Visual Pattern Recognition",
        "subtype": "Rotation Patterns",
        "answer": "D",
        "question": "Which option is the next clock-hand pattern?",
        "prompt": (
            f"{base_prompt()} Draw a pattern sequence of clock-like tiles with "
            "two hands and one colored marker. The missing next tile requires "
            "tracking two rotations at different speeds. Below draw A-D. Option "
            "D must be correct."
        ),
    },
    {
        "slug": "shadow_notched_object",
        "type": "Fine-grained Discrimination",
        "subtype": "Find the shadow",
        "answer": "B",
        "question": "Which silhouette is the exact shadow of the object?",
        "prompt": (
            f"{base_prompt()} Draw a colored asymmetric object with several small "
            "notches and protrusions. Below draw black silhouettes A-D. Option B "
            "must be exact; the others should be near matches with tiny missing "
            "or mirrored details."
        ),
    },
    {
        "slug": "same_microglyph_options",
        "type": "Fine-grained Discrimination",
        "subtype": "Find the same",
        "answer": "C",
        "question": "Which option exactly matches the reference micro-glyph?",
        "prompt": (
            f"{base_prompt()} Draw a reference micro-glyph made from tiny line "
            "segments, dots, and one notch. Below draw options A-D. Option C "
            "must be identical; other options each differ by one tiny feature."
        ),
    },
    {
        "slug": "cube_unfold_opposite_face",
        "type": "Spatial Perception",
        "subtype": "3D Cube Unfold",
        "answer": "A",
        "question": "After folding the net, which option is opposite the striped face?",
        "prompt": (
            f"{base_prompt()} Draw a valid cube net with symbols on every face. "
            "One face has diagonal stripes. Below draw options A-D as face "
            "symbols. Option A must be the face opposite the striped face after "
            "folding. Distractors should be adjacent faces."
        ),
    },
]


def save_response_image(response: Any, output_path: Path) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_parts: list[str] = []
    image_count = 0
    if not response.candidates:
        raise RuntimeError("No candidates in image response.")
    for part in response.candidates[0].content.parts:
        if part.text:
            text_parts.append(part.text)
        elif part.inline_data:
            image_count += 1
            image = Image.open(BytesIO(part.inline_data.data))
            path = output_path if image_count == 1 else output_path.with_name(
                f"{output_path.stem}_{image_count}{output_path.suffix}"
            )
            image.save(path)
    if image_count == 0:
        raise RuntimeError("Image response did not include an image.")
    return text_parts


def call_with_retries(label: str, fn, retries: int) -> Any:
    delay = 90
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except (errors.APIError, errors.ClientError, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            print(
                f"{label}: {exc}; retrying in {delay}s ({attempt}/{retries})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"{label} failed after {retries} attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI-created BabyVision candidates.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", default=gcloud_config_value("project") or DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--profile",
        choices=["diverse", "mirror-tail"],
        default="diverse",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    metadata_path = args.output_dir / "meta_data.jsonl"
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    rows = []
    cases = MIRROR_TAIL_CASES if args.profile == "mirror-tail" else CASES
    limit = args.limit or len(cases)

    for index, case in enumerate(cases[:limit], start=1):
        output_path = image_dir / f"{index:06d}_{case['slug']}.png"
        prompt_path = output_path.with_suffix(".prompt.json")
        print(f"[{index}/{min(limit, len(cases))}] {case['subtype']} -> {output_path}", flush=True)
        if output_path.exists() and prompt_path.exists():
            response_text = json.loads(prompt_path.read_text()).get("response_text", [])
        else:
            def call():
                return client.models.generate_content(
                    model=args.model,
                    contents=case["prompt"],
                    config=GenerateContentConfig(
                        response_modalities=[Modality.TEXT, Modality.IMAGE],
                    ),
                )

            response = call_with_retries(output_path.name, call, args.retries)
            response_text = save_response_image(response, output_path)
            prompt_path.write_text(
                json.dumps(
                    {
                        "created_at": now_iso(),
                        "generator": "tools/generate_ai_weakness_candidates.py",
                        "model": args.model,
                        "project": args.project,
                        "location": args.location,
                        "output_image": str(output_path),
                        "case": case,
                        "prompt": case["prompt"],
                        "response_text": response_text,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        rows.append(
            {
                "taskId": index,
                "image": f"images/{output_path.name}",
                "question": case["question"],
                "type": case["type"],
                "subtype": case["subtype"],
                "ansType": "blank",
                "blankAns": case["answer"],
                "options": [],
            }
        )

    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "generator": "tools/generate_ai_weakness_candidates.py",
                "model": args.model,
                "project": args.project,
                "location": args.location,
                "profile": args.profile,
                "count": len(rows),
                "metadata": str(metadata_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} AI-generated candidates to {args.output_dir}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
