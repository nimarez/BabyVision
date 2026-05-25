#!/usr/bin/env python3
"""Local review server for BabyVision synthetic outputs."""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent
SYNTHETIC_DIR = ROOT / "synthetic_outputs"
REVIEWS_DIR = SYNTHETIC_DIR / "reviews"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def ensure_inside(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if resolved != parent_resolved and parent_resolved not in resolved.parents:
        raise ValueError("path outside allowed root")
    return resolved


def review_paths(dataset_id: str) -> tuple[Path, Path]:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in dataset_id)
    return REVIEWS_DIR / f"{safe_name}.reviews.jsonl", REVIEWS_DIR / f"{safe_name}.latest.json"


def load_latest_reviews(dataset_id: str) -> dict:
    _, latest_path = review_paths(dataset_id)
    if not latest_path.exists():
        return {}
    payload = read_json(latest_path)
    return payload.get("reviews", {})


def find_datasets() -> list[dict]:
    datasets = []
    if not SYNTHETIC_DIR.exists():
        return datasets

    for manifest_path in sorted(SYNTHETIC_DIR.glob("*/hard_cases_manifest.json")):
        dataset_dir = manifest_path.parent
        dataset_id = dataset_dir.name
        manifest = read_json(manifest_path)
        latest = load_latest_reviews(dataset_id)
        counts = {
            "prompt_pass": 0,
            "prompt_fail": 0,
            "rubric_pass": 0,
            "rubric_fail": 0,
            "reviewed": 0,
        }
        for review in latest.values():
            if review.get("prompt_adherence") == "pass":
                counts["prompt_pass"] += 1
            if review.get("prompt_adherence") == "fail":
                counts["prompt_fail"] += 1
            if review.get("answer_rubric") == "pass":
                counts["rubric_pass"] += 1
            if review.get("answer_rubric") == "fail":
                counts["rubric_fail"] += 1
            if review.get("prompt_adherence") or review.get("answer_rubric"):
                counts["reviewed"] += 1

        datasets.append(
            {
                "id": dataset_id,
                "path": str(dataset_dir.relative_to(ROOT)),
                "created_at": manifest.get("created_at"),
                "count": manifest.get("count", len(manifest.get("cases", []))),
                "selection_rule": manifest.get("selection_rule", ""),
                "source_counts": manifest.get("source_counts", {}),
                "subtype_counts": manifest.get("subtype_counts", {}),
                "review_counts": counts,
            }
        )
    return datasets


def case_identity(case: dict, fallback: int) -> str:
    return str(
        case.get("mixed_case_id")
        or case.get("hard_case_id")
        or case.get("source_task_id")
        or fallback
    )


def load_dataset_cases(dataset_id: str) -> dict:
    manifest_path = ensure_inside(SYNTHETIC_DIR / dataset_id / "hard_cases_manifest.json", SYNTHETIC_DIR)
    if not manifest_path.exists():
        raise FileNotFoundError(dataset_id)

    manifest = read_json(manifest_path)
    latest = load_latest_reviews(dataset_id)
    cases = []
    for index, case in enumerate(manifest.get("cases", []), start=1):
        case_id = case_identity(case, index)
        enriched = dict(case)
        enriched["case_id"] = case_id
        enriched["index"] = index
        enriched["image_url"] = "/file/" + case.get("image", "")

        for field, target in (("prompt_json", "prompt_data"), ("eval_json", "eval_data")):
            path_value = case.get(field)
            if path_value:
                path = ensure_inside(ROOT / path_value, ROOT)
                enriched[target] = read_json(path) if path.exists() else {}
            else:
                enriched[target] = {}

        prompt_data = enriched.get("prompt_data", {})
        eval_data = enriched.get("eval_data", {})
        prompt_case = prompt_data.get("case", {}) if isinstance(prompt_data.get("case"), dict) else {}
        enriched["generated_prompt"] = (
            prompt_data.get("prompt")
            or prompt_case.get("prompt")
            or ""
        )
        enriched["reasoning_trace"] = prompt_data.get("reasoning_trace", "")
        enriched["generation_response_text"] = prompt_data.get("response_text", [])
        enriched["model_result"] = eval_data.get("model_result", "")
        enriched["review"] = latest.get(case_id, {})
        cases.append(enriched)

    return {
        "dataset": {
            "id": dataset_id,
            "path": str(manifest_path.parent.relative_to(ROOT)),
            "created_at": manifest.get("created_at"),
            "selection_rule": manifest.get("selection_rule", ""),
            "count": manifest.get("count", len(cases)),
            "source_counts": manifest.get("source_counts", {}),
            "subtype_counts": manifest.get("subtype_counts", {}),
        },
        "cases": cases,
    }


def save_review(payload: dict) -> dict:
    dataset_id = str(payload.get("dataset_id", "")).strip()
    case_id = str(payload.get("case_id", "")).strip()
    if not dataset_id or not case_id:
        raise ValueError("dataset_id and case_id are required")

    ensure_inside(SYNTHETIC_DIR / dataset_id, SYNTHETIC_DIR)
    if not (SYNTHETIC_DIR / dataset_id / "hard_cases_manifest.json").exists():
        raise FileNotFoundError(dataset_id)

    allowed = {
        "prompt_adherence",
        "answer_rubric",
        "prompt_notes",
        "rubric_notes",
        "corrected_answer",
        "corrected_question",
        "severity",
        "reviewer",
    }
    review = {key: payload.get(key, "") for key in allowed}
    review.update(
        {
            "dataset_id": dataset_id,
            "case_id": case_id,
            "updated_at": utc_now(),
        }
    )

    jsonl_path, latest_path = review_paths(dataset_id)
    latest_payload = {"dataset_id": dataset_id, "updated_at": review["updated_at"], "reviews": {}}
    if latest_path.exists():
        latest_payload = read_json(latest_path)
        latest_payload.setdefault("reviews", {})

    latest_payload["updated_at"] = review["updated_at"]
    latest_payload["reviews"][case_id] = review

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(review, ensure_ascii=False) + "\n")
    write_json(latest_path, latest_payload)
    return review


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "BabyVisionReview/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: dict | list, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/datasets":
                self.send_json({"datasets": find_datasets()})
            elif path.startswith("/api/datasets/") and path.endswith("/cases"):
                dataset_id = path.removeprefix("/api/datasets/").removesuffix("/cases").strip("/")
                self.send_json(load_dataset_cases(dataset_id))
            elif path.startswith("/file/"):
                self.serve_repo_file(path.removeprefix("/file/"))
            else:
                self.serve_static(path)
        except FileNotFoundError:
            self.send_error_json(404, "not found")
        except ValueError as exc:
            self.send_error_json(400, str(exc))
        except Exception as exc:
            self.send_error_json(500, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/reviews":
            self.send_error_json(404, "not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json({"review": save_review(payload)})
        except json.JSONDecodeError:
            self.send_error_json(400, "invalid json")
        except FileNotFoundError:
            self.send_error_json(404, "dataset not found")
        except ValueError as exc:
            self.send_error_json(400, str(exc))
        except Exception as exc:
            self.send_error_json(500, str(exc))

    def serve_repo_file(self, rel_path: str) -> None:
        path = ensure_inside(ROOT / rel_path, ROOT)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(rel_path)
        self.send_file(path)

    def serve_static(self, request_path: str) -> None:
        if request_path in ("", "/"):
            request_path = "/index.html"
        path = ensure_inside(UI_DIR / request_path.lstrip("/"), UI_DIR)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(request_path)
        self.send_file(path)

    def send_file(self, path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(path))
        content_type = content_type or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the BabyVision synthetic review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Serving BabyVision review UI at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
