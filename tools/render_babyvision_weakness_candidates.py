#!/usr/bin/env python3
"""Render deterministic visual logical puzzle weakness candidates with known answers."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont, fill=(20, 20, 20)) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=font, fill=fill)


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_case(
    *,
    root: Path,
    task_id: int,
    image: Image.Image,
    spec: dict[str, Any],
    prompt: str,
    reasoning_trace: str,
) -> dict[str, Any]:
    image_name = f"{task_id:06d}_{spec['slug']}.png"
    image_path = root / "images" / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    prompt_path = image_path.with_suffix(".prompt.json")
    write_record(
        prompt_path,
        {
            "created_at": now_iso(),
            "generator": "tools/render_babyvision_weakness_candidates.py",
            "output_image": str(image_path),
            "prompt": prompt,
            "type": spec["type"],
            "subtype": spec["subtype"],
            "question": spec["question"],
            "answer": spec["answer"],
            "reasoning_trace": reasoning_trace,
            "parameters": spec.get("parameters", {}),
        },
    )
    return {
        "taskId": task_id,
        "image": f"images/{image_name}",
        "question": spec["question"],
        "type": spec["type"],
        "subtype": spec["subtype"],
        "ansType": "blank",
        "blankAns": spec["answer"],
        "options": [],
    }


def draw_grid_labels(draw: ImageDraw.ImageDraw, rows: int, cols: int, left: int, top: int, cell: int, font: ImageFont.ImageFont) -> None:
    for r in range(rows):
        text_center(draw, (left - 22, top + r * cell + cell / 2), str(r + 1), font, fill=(70, 70, 70))
    for c in range(cols):
        text_center(draw, (left + c * cell + cell / 2, top - 22), str(c + 1), font, fill=(70, 70, 70))


def render_dense_find_different(rng: random.Random, task_id: int) -> tuple[Image.Image, dict[str, Any], str, str]:
    rows = rng.randint(12, 17)
    cols = rng.randint(18, 27)
    cell = min(44, int((1320 - 90) / cols), int((900 - 110) / rows))
    left = 70
    top = 70
    same, odd = rng.choice([("C", "G"), ("O", "Q"), ("P", "R"), ("E", "F"), ("B", "8")])
    target_r = rng.randint(1, rows)
    target_c = rng.randint(1, cols)
    image = Image.new("RGB", (1400, 980), "white")
    draw = ImageDraw.Draw(image)
    small = load_font(18)
    glyph = load_font(max(18, int(cell * 0.58)), bold=True)
    title = load_font(26, bold=True)
    draw.text((70, 24), "Find the single different symbol", font=title, fill=(20, 20, 20))
    draw_grid_labels(draw, rows, cols, left, top, cell, small)
    for r in range(rows):
        for c in range(cols):
            x0 = left + c * cell
            y0 = top + r * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline=(210, 210, 210), width=1)
            char = odd if (r + 1 == target_r and c + 1 == target_c) else same
            text_center(draw, (x0 + cell / 2, y0 + cell / 2), char, glyph)
    answer = f"({target_r},{target_c})"
    spec = {
        "slug": "dense_find_different",
        "type": "Fine-grained Discrimination",
        "subtype": "Find the different",
        "question": f"The grid contains many '{same}' symbols and one different symbol. What is the coordinate of the different symbol? Use (row,column).",
        "answer": answer,
        "parameters": {"rows": rows, "cols": cols, "same": same, "odd": odd, "target": answer},
    }
    prompt = "Rendered dense coordinate-search puzzle with one subtly different symbol."
    trace = f"All cells show {same} except row {target_r}, column {target_c}, which shows {odd}."
    return image, spec, prompt, trace


def icon(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: int, *, notch: str, dot: str, tail: str, fill=(250, 250, 250), outline=(20, 20, 20)) -> None:
    s = scale
    x0, y0 = cx - s, cy - s
    x1, y1 = cx + s, cy + s
    draw.rounded_rectangle([x0, y0, x1, y1], radius=max(3, s // 4), fill=fill, outline=outline, width=2)
    notch_points = {
        "tr": [(x1 - s * 0.55, y0), (x1, y0), (x1, y0 + s * 0.55)],
        "tl": [(x0 + s * 0.55, y0), (x0, y0), (x0, y0 + s * 0.55)],
        "br": [(x1 - s * 0.55, y1), (x1, y1), (x1, y1 - s * 0.55)],
        "bl": [(x0 + s * 0.55, y1), (x0, y1), (x0, y1 - s * 0.55)],
    }
    draw.polygon(notch_points[notch], fill="white")
    dot_offsets = {
        "lower_left": (-0.45, 0.42),
        "lower_right": (0.45, 0.42),
        "upper_left": (-0.45, -0.42),
        "upper_right": (0.45, -0.42),
    }
    dx, dy = dot_offsets[dot]
    rr = max(3, s // 6)
    draw.ellipse([cx + dx * s - rr, cy + dy * s - rr, cx + dx * s + rr, cy + dy * s + rr], fill=(45, 105, 190))
    tail_vectors = {
        "down_left": (-0.8, 1.0),
        "down_right": (0.8, 1.0),
        "up_left": (-0.8, -1.0),
        "up_right": (0.8, -1.0),
    }
    tx, ty = tail_vectors[tail]
    draw.line([cx, cy + s, cx + tx * s, cy + s + ty * s * 0.55], fill=outline, width=3)


def render_find_same_grid(rng: random.Random, task_id: int) -> tuple[Image.Image, dict[str, Any], str, str]:
    rows, cols = 7, 9
    cell = 92
    left, top = 260, 190
    image = Image.new("RGB", (1250, 920), "white")
    draw = ImageDraw.Draw(image)
    title = load_font(28, bold=True)
    small = load_font(18)
    draw.text((55, 35), "Find the exact match", font=title, fill=(20, 20, 20))
    ref = {"notch": rng.choice(["tr", "tl", "br", "bl"]), "dot": rng.choice(["lower_left", "lower_right", "upper_left", "upper_right"]), "tail": rng.choice(["down_left", "down_right", "up_left", "up_right"])}
    draw.text((70, 125), "Reference", font=load_font(22), fill=(40, 40, 40))
    icon(draw, 145, 240, 35, **ref)
    target_r = rng.randint(1, rows)
    target_c = rng.randint(1, cols)
    draw_grid_labels(draw, rows, cols, left, top, cell, small)
    choices = {
        "notch": ["tr", "tl", "br", "bl"],
        "dot": ["lower_left", "lower_right", "upper_left", "upper_right"],
        "tail": ["down_left", "down_right", "up_left", "up_right"],
    }
    for r in range(rows):
        for c in range(cols):
            x0 = left + c * cell
            y0 = top + r * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline=(210, 210, 210), width=1)
            params = dict(ref)
            if not (r + 1 == target_r and c + 1 == target_c):
                feature = rng.choice(["notch", "dot", "tail"])
                params[feature] = rng.choice([v for v in choices[feature] if v != params[feature]])
            icon(draw, int(x0 + cell / 2), int(y0 + cell / 2), 26, **params)
    answer = f"({target_r},{target_c})"
    spec = {
        "slug": "dense_find_same",
        "type": "Fine-grained Discrimination",
        "subtype": "Find the same",
        "question": "Which grid coordinate contains the icon exactly identical to the reference icon? Use (row,column).",
        "answer": answer,
        "parameters": {"rows": rows, "cols": cols, "target": answer, "reference": ref},
    }
    prompt = "Rendered dense exact-match icon puzzle with one true match and many near-matches."
    trace = f"The only icon with the same notch, dot position, and tail direction is at row {target_r}, column {target_c}."
    return image, spec, prompt, trace


def render_line_tracking(rng: random.Random, task_id: int) -> tuple[Image.Image, dict[str, Any], str, str]:
    image = Image.new("RGB", (1250, 850), "white")
    draw = ImageDraw.Draw(image)
    title = load_font(28, bold=True)
    label_font = load_font(25, bold=True)
    draw.text((55, 30), "Follow the line", font=title, fill=(20, 20, 20))
    starts = list("ABCDEF")
    endpoints = list(range(1, 7))
    perm = endpoints[:]
    rng.shuffle(perm)
    target_i = rng.randrange(6)
    target_start = starts[target_i]
    answer = str(perm[target_i])
    start_y = [125 + i * 110 for i in range(6)]
    end_y = [125 + i * 110 for i in range(6)]
    x_left, x_right = 115, 1135
    for i, s in enumerate(starts):
        draw.ellipse([x_left - 9, start_y[i] - 9, x_left + 9, start_y[i] + 9], fill=(20, 20, 20))
        text_center(draw, (55, start_y[i]), s, label_font)
    for i, n in enumerate(endpoints):
        draw.ellipse([x_right - 9, end_y[i] - 9, x_right + 9, end_y[i] + 9], fill=(20, 20, 20))
        text_center(draw, (1192, end_y[i]), str(n), label_font)
    for i, endpoint in enumerate(perm):
        y0 = start_y[i]
        y1 = end_y[endpoint - 1]
        mids = []
        for x in [260, 430, 610, 790, 960]:
            offset = rng.choice([-170, -110, -60, 0, 60, 110, 170])
            mids.append((x, max(85, min(760, (y0 + y1) / 2 + offset))))
        points = [(x_left, y0), *mids, (x_right, y1)]
        draw.line(points, fill=(25, 25, 25), width=3, joint="curve")
    spec = {
        "slug": "connect_lines",
        "type": "Visual Tracking",
        "subtype": "Connect the lines",
        "question": f"Start at label {target_start} and follow its continuous line. Which numbered endpoint does it reach?",
        "answer": answer,
        "parameters": {"mapping": dict(zip(starts, map(str, perm), strict=True)), "target_start": target_start},
    }
    prompt = "Rendered line-tracking puzzle with six crossing continuous paths."
    trace = f"The generated mapping sends {target_start} to endpoint {answer}."
    return image, spec, prompt, trace


def iso_xy(origin: tuple[int, int], i: int, j: int, z: int, size: int) -> tuple[int, int]:
    ox, oy = origin
    return (int(ox + (i - j) * size), int(oy + (i + j) * size * 0.55 - z * size * 0.9))


def cube(draw: ImageDraw.ImageDraw, origin: tuple[int, int], i: int, j: int, z: int, size: int) -> None:
    x, y = iso_xy(origin, i, j, z, size)
    top = [(x, y - size // 2), (x + size, y), (x, y + size // 2), (x - size, y)]
    left = [(x - size, y), (x, y + size // 2), (x, y + size // 2 + int(size * 0.9)), (x - size, y + int(size * 0.9))]
    right = [(x + size, y), (x, y + size // 2), (x, y + size // 2 + int(size * 0.9)), (x + size, y + int(size * 0.9))]
    draw.polygon(left, fill=(180, 188, 198), outline=(65, 70, 75))
    draw.polygon(right, fill=(150, 160, 172), outline=(65, 70, 75))
    draw.polygon(top, fill=(230, 235, 240), outline=(65, 70, 75))


def render_count_blocks(rng: random.Random, task_id: int) -> tuple[Image.Image, dict[str, Any], str, str]:
    rows = cols = 4
    heights = [[rng.randint(1, 4) for _ in range(cols)] for _ in range(rows)]
    answer = str(sum(sum(row) for row in heights))
    image = Image.new("RGB", (1150, 920), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "Count the cubes", font=load_font(28, bold=True), fill=(20, 20, 20))
    origin = (575, 245)
    size = 42
    for z in range(4):
        for i in range(rows):
            for j in range(cols):
                if heights[i][j] > z:
                    cube(draw, origin, j, i, z, size)
    spec = {
        "slug": "count_3d_blocks",
        "type": "Spatial Perception",
        "subtype": "Count 3D blocks",
        "question": "How many unit cubes are in the stacked block structure, including hidden support cubes?",
        "answer": answer,
        "parameters": {"heights_back_to_front": heights, "total": answer},
    }
    prompt = "Rendered isometric unit-cube counting puzzle with hidden support cubes."
    trace = f"Summing the 4 by 4 column heights {heights} gives {answer} cubes."
    return image, spec, prompt, trace


def draw_symbol(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], symbol: str, *, fill: tuple[int, int, int] | str = (20, 20, 20)) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    w, h = x1 - x0, y1 - y0
    if symbol == "star":
        pts = []
        for k in range(10):
            ang = -math.pi / 2 + k * math.pi / 5
            r = min(w, h) * (0.35 if k % 2 == 0 else 0.16)
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        draw.polygon(pts, fill=fill)
    elif symbol == "triangle":
        draw.polygon([(cx, y0 + h * 0.2), (x0 + w * 0.2, y1 - h * 0.2), (x1 - w * 0.2, y1 - h * 0.2)], fill=fill)
    elif symbol == "circle":
        draw.ellipse([x0 + w * 0.22, y0 + h * 0.22, x1 - w * 0.22, y1 - h * 0.22], fill=fill)
    elif symbol == "plus":
        t = w * 0.13
        draw.rectangle([cx - t, y0 + h * 0.22, cx + t, y1 - h * 0.22], fill=fill)
        draw.rectangle([x0 + w * 0.22, cy - t, x1 - w * 0.22, cy + t], fill=fill)
    elif symbol == "stripe":
        draw.line([x0 + w * 0.25, y1 - h * 0.22, x1 - w * 0.2, y0 + h * 0.2], fill=fill, width=max(3, int(w * 0.1)))
    else:
        draw.arc([x0 + w * 0.22, y0 + h * 0.18, x1 - w * 0.12, y1 - h * 0.18], 80, 280, fill=fill, width=max(3, int(w * 0.1)))


def render_cube_unfold(rng: random.Random, task_id: int) -> tuple[Image.Image, dict[str, Any], str, str]:
    image = Image.new("RGB", (1150, 860), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "Fold the cube net", font=load_font(28, bold=True), fill=(20, 20, 20))
    symbols = ["star", "plus", "circle", "crescent", "stripe", "triangle"]
    right_symbol = rng.choice(symbols)
    answer = rng.choice(["A", "B", "C", "D"])
    net = {
        (0, 0): ("circle", (210, 55, 55)),
        (0, -1): ("triangle", (55, 105, 210)),
        (1, 0): (right_symbol, (210, 170, 25)),
        (-1, 0): (rng.choice(symbols), (30, 30, 30)),
        (0, 1): (rng.choice(symbols), (65, 150, 80)),
        (2, 0): (rng.choice(symbols), (120, 80, 180)),
    }
    size = 105
    cx, cy = 475, 250
    for (gx, gy), (sym, color) in net.items():
        x0 = cx + gx * size
        y0 = cy + gy * size
        draw.rectangle([x0, y0, x0 + size, y0 + size], outline=(30, 30, 30), width=3)
        draw_symbol(draw, (x0 + 8, y0 + 8, x0 + size - 8, y0 + size - 8), sym, fill=color)
    option_symbols = [s for s in symbols if s != right_symbol]
    rng.shuffle(option_symbols)
    labels = ["A", "B", "C", "D"]
    option_map = {label: option_symbols.pop() for label in labels}
    option_map[answer] = right_symbol
    for idx, label in enumerate(labels):
        x0 = 270 + idx * 160
        y0 = 620
        draw.text((x0, y0 - 36), label, font=load_font(24, bold=True), fill=(20, 20, 20))
        draw.rectangle([x0, y0, x0 + 95, y0 + 95], outline=(40, 40, 40), width=2)
        draw_symbol(draw, (x0 + 8, y0 + 8, x0 + 87, y0 + 87), option_map[label], fill=(30, 30, 30))
    spec = {
        "slug": "cube_unfold",
        "type": "Spatial Perception",
        "subtype": "3D Cube Unfold",
        "question": "Fold the cube net with the red circle as the front face and the blue triangle as the top face. Which option is on the right face?",
        "answer": answer,
        "parameters": {"right_symbol": right_symbol, "option_symbols": option_map},
    }
    prompt = "Rendered cube-net puzzle with labeled answer options."
    trace = f"The square to the right of the front red-circle face folds to the right face and matches option {answer}."
    return image, spec, prompt, trace


def render_overlay(rng: random.Random, task_id: int) -> tuple[Image.Image, dict[str, Any], str, str]:
    image = Image.new("RGB", (1250, 920), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "Overlay the two layers", font=load_font(28, bold=True), fill=(20, 20, 20))
    labels = ["A", "B", "C", "D"]
    answer = rng.choice(labels)
    marks = [
        ("vbar", rng.randint(25, 65)),
        ("hline", rng.randint(30, 70)),
        ("dot", (rng.randint(25, 75), rng.randint(25, 75))),
        ("diag", 0),
        ("circle", (rng.randint(30, 70), rng.randint(30, 70))),
    ]
    layer1 = marks[:2]
    layer2 = marks[2:]

    def draw_marks(box: tuple[int, int, int, int], selected: list[tuple[str, Any]]) -> None:
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline=(40, 40, 40), width=2)
        w, h = x1 - x0, y1 - y0
        for kind, value in selected:
            if kind == "vbar":
                x = x0 + int(w * value / 100)
                draw.rectangle([x - 7, y0 + 12, x + 7, y1 - 12], fill=(20, 20, 20))
            elif kind == "hline":
                y = y0 + int(h * value / 100)
                draw.line([x0 + 12, y, x1 - 12, y], fill=(45, 105, 190), width=5)
            elif kind == "dot":
                x = x0 + int(w * value[0] / 100)
                y = y0 + int(h * value[1] / 100)
                draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(45, 105, 190))
            elif kind == "diag":
                draw.line([x0 + 15, y1 - 18, x1 - 15, y0 + 18], fill=(45, 105, 190), width=4)
            elif kind == "circle":
                x = x0 + int(w * value[0] / 100)
                y = y0 + int(h * value[1] / 100)
                draw.ellipse([x - 13, y - 13, x + 13, y + 13], outline=(20, 20, 20), width=4)

    draw.text((560, 90), "Target", font=load_font(22, bold=True), fill=(20, 20, 20))
    draw_marks((500, 130, 730, 360), marks)
    for idx, label in enumerate(labels):
        x0 = 105 + idx * 285
        y0 = 560
        draw.text((x0, y0 - 42), label, font=load_font(24, bold=True), fill=(20, 20, 20))
        l1 = list(layer1)
        l2 = list(layer2)
        if label != answer:
            changed = rng.choice([1, 2])
            target_layer = l1 if changed == 1 else l2
            if target_layer:
                target_layer.pop(rng.randrange(len(target_layer)))
            target_layer.append(("dot", (rng.randint(15, 85), rng.randint(15, 85))))
        draw_marks((x0, y0, x0 + 95, y0 + 95), l1)
        draw_marks((x0 + 120, y0, x0 + 215, y0 + 95), l2)
    spec = {
        "slug": "overlay_patterns",
        "type": "Visual Pattern Recognition",
        "subtype": "Overlay Patterns",
        "question": "Which option pair overlays to make the target pattern?",
        "answer": answer,
        "parameters": {"answer": answer},
    }
    prompt = "Rendered overlay reconstruction puzzle with one correct layer pair."
    trace = f"Only option {answer} contains the two layers whose combined marks equal the target."
    return image, spec, prompt, trace


RENDERERS = [
    render_dense_find_different,
    render_find_same_grid,
    render_line_tracking,
    render_count_blocks,
    render_cube_unfold,
    render_overlay,
]

HARD_TAIL_RENDERERS = [
    render_line_tracking,
    render_find_same_grid,
    render_count_blocks,
    render_line_tracking,
    render_overlay,
    render_line_tracking,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synthetic BabyVision weakness candidates.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument(
        "--profile",
        choices=["balanced", "hard-tail"],
        default="balanced",
        help="Renderer mix. hard-tail emphasizes line tracking and exact matching.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    metadata_path = args.output_dir / "meta_data.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()
    renderers = HARD_TAIL_RENDERERS if args.profile == "hard-tail" else RENDERERS

    with metadata_path.open("w", encoding="utf-8") as handle:
        for task_id in range(1, args.count + 1):
            renderer = renderers[(task_id - 1) % len(renderers)]
            image, spec, prompt, trace = renderer(rng, task_id)
            row = save_case(
                root=args.output_dir,
                task_id=task_id,
                image=image,
                spec=spec,
                prompt=prompt,
                reasoning_trace=trace,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_record(
        args.output_dir / "render_manifest.json",
        {
            "created_at": now_iso(),
            "generator": "tools/render_babyvision_weakness_candidates.py",
            "count": args.count,
            "seed": args.seed,
            "profile": args.profile,
            "metadata": str(metadata_path),
            "renderers": [fn.__name__ for fn in renderers],
        },
    )
    print(f"wrote {args.count} candidates to {args.output_dir}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
