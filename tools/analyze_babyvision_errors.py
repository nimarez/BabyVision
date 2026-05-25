#!/usr/bin/env python3
"""Summarize BabyVision failures into task dimensions for synthetic data design."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


KEYWORD_DIMENSIONS = {
    "coordinate_search": ["row", "column", "(x,y)", "coordinate"],
    "counting": ["count", "how many", "number of"],
    "3d_spatial": ["3d", "cube", "block", "unfold", "fold", "view"],
    "path_tracking": ["maze", "line", "path", "metro", "connect"],
    "pattern_completion": ["complete", "completion", "missing", "pattern"],
    "rotation_or_shadow": ["rotate", "rotation", "shadow", "mirror"],
    "fine_detail": ["different", "same", "small", "letter", "number"],
    "overlay_reconstruction": ["overlay", "reconstruct", "reconstruction"],
}


def load_metadata(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[int(item["taskId"])] = item
    return rows


def load_results(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_bucket(metadata_dir: Path, image_name: str) -> dict[str, Any]:
    path = metadata_dir / image_name
    with Image.open(path) as image:
        width, height = image.size
    aspect = width / height
    if aspect < 0.8:
        aspect_bucket = "portrait"
    elif aspect > 1.25:
        aspect_bucket = "landscape"
    else:
        aspect_bucket = "square-ish"
    area = width * height
    if area < 512 * 512:
        area_bucket = "small"
    elif area < 1024 * 1024:
        area_bucket = "medium"
    else:
        area_bucket = "large"
    return {
        "width": width,
        "height": height,
        "aspect_bucket": aspect_bucket,
        "area_bucket": area_bucket,
    }


def dimensions_for(item: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            str(item.get("type", "")),
            str(item.get("subtype", "")),
            str(item.get("question", "")),
        ]
    ).lower()
    dims = [
        dimension
        for dimension, keywords in KEYWORD_DIMENSIONS.items()
        if any(keyword in text for keyword in keywords)
    ]
    return dims or ["other"]


def grouped_accuracy(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        value = str(row[key])
        totals[value]["total"] += 1
        if row["correct"]:
            totals[value]["correct"] += 1
    summary = []
    for value, counts in totals.items():
        total = counts["total"]
        correct = counts["correct"]
        summary.append(
            {
                key: value,
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "failures": total - correct,
            }
        )
    return sorted(summary, key=lambda item: (item["accuracy"], -item["total"], item[key]))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# BabyVision Gemini Error Analysis",
        "",
        f"- Results: `{report['results_path']}`",
        f"- Model: `{report['model']}`",
        f"- Total: {report['overall']['total']}",
        f"- Correct: {report['overall']['correct']}",
        f"- Accuracy: {report['overall']['accuracy']:.2%}",
        "",
        "## Weakest Subtypes",
        "",
    ]
    for row in report["by_subtype"][:12]:
        lines.append(
            f"- {row['Subtype']}: {row['accuracy']:.2%} "
            f"({row['correct']}/{row['total']}, failures={row['failures']})"
        )
    lines.extend(["", "## Weakest Dimensions", ""])
    for row in report["by_dimension"][:12]:
        lines.append(
            f"- {row['dimension']}: {row['accuracy']:.2%} "
            f"({row['correct']}/{row['total']}, failures={row['failures']})"
        )
    lines.extend(["", "## Synthetic Data Targets", ""])
    for target in report["synthetic_targets"]:
        lines.append(
            f"- {target['priority']}. {target['target']}: "
            f"{target['accuracy']:.2%} accuracy over {target['total']} tasks. "
            f"Prompt focus: {target['prompt_focus']}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BabyVision model failures.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/babyvision_data/meta_data.jsonl"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    results = load_results(args.results)
    metadata_dir = args.metadata.parent

    enriched = []
    for row in results:
        item = metadata[int(row["Id"])]
        image_info = image_bucket(metadata_dir, item["image"])
        dims = dimensions_for(item)
        enriched_row = {
            **row,
            "correct": bool(row["LLMJudgeResult"]),
            "Subtype": item["subtype"],
            "Type": item["type"],
            "AnswerType": item["ansType"],
            "Image": item["image"],
            "option_count": len(item.get("options") or []),
            "dimensions": dims,
            **image_info,
        }
        enriched.append(enriched_row)

    total = len(enriched)
    correct = sum(row["correct"] for row in enriched)

    dimension_rows = []
    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        for dimension in row["dimensions"]:
            by_dimension[dimension].append(row)
    for dimension, rows in by_dimension.items():
        dim_total = len(rows)
        dim_correct = sum(row["correct"] for row in rows)
        dimension_rows.append(
            {
                "dimension": dimension,
                "total": dim_total,
                "correct": dim_correct,
                "accuracy": dim_correct / dim_total if dim_total else 0.0,
                "failures": dim_total - dim_correct,
            }
        )
    dimension_rows = sorted(
        dimension_rows,
        key=lambda item: (item["accuracy"], -item["total"], item["dimension"]),
    )

    by_subtype = grouped_accuracy(enriched, "Subtype")
    synthetic_targets = []
    for priority, row in enumerate(by_subtype[:8], start=1):
        subtype_failures = [
            item for item in enriched if item["Subtype"] == row["Subtype"] and not item["correct"]
        ]
        dimension_counts = Counter(
            dimension
            for item in subtype_failures
            for dimension in item["dimensions"]
        )
        focus = ", ".join(dimension for dimension, _ in dimension_counts.most_common(3))
        synthetic_targets.append(
            {
                "priority": priority,
                "target": row["Subtype"],
                "accuracy": row["accuracy"],
                "total": row["total"],
                "failures": row["failures"],
                "prompt_focus": focus or row["Subtype"],
            }
        )

    report = {
        "results_path": str(args.results),
        "model": (
            results[0].get("Model")
            or results[0].get("ModelName")
            or "unknown"
            if results
            else "unknown"
        ),
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "failures": total - correct,
        },
        "by_type": grouped_accuracy(enriched, "Type"),
        "by_subtype": by_subtype,
        "by_answer_type": grouped_accuracy(enriched, "AnswerType"),
        "by_aspect_bucket": grouped_accuracy(enriched, "aspect_bucket"),
        "by_area_bucket": grouped_accuracy(enriched, "area_bucket"),
        "by_dimension": dimension_rows,
        "synthetic_targets": synthetic_targets,
        "failed_examples": [
            {
                "Id": row["Id"],
                "Type": row["Type"],
                "Subtype": row["Subtype"],
                "GroundTruth": row["GroundTruth"],
                "ExtractedAnswer": row["ExtractedAnswer"],
                "Image": row["Image"],
                "dimensions": row["dimensions"],
            }
            for row in enriched
            if not row["correct"]
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.output_md, report)
    print(json.dumps(report["overall"], indent=2))
    print("Weakest subtypes:")
    for row in by_subtype[:8]:
        print(
            f"  {row['Subtype']}: {row['accuracy']:.2%} "
            f"({row['correct']}/{row['total']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
