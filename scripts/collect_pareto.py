"""Collect result JSON files and draw a lightweight accuracy-cost Pareto SVG."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


FIELDS = [
    "file",
    "dataset",
    "method",
    "seed",
    "n_tasks",
    "accuracy",
    "ece",
    "mean_latency_s",
    "mean_real_prompt_tokens",
    "mean_token_surrogate",
    "mean_samples",
    "mean_verified_candidates",
    "oracle_candidate_accuracy",
    "always_first_accuracy",
    "highest_count_accuracy",
]


def _row(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)["summary"]
    out = {field: summary.get(field, "") for field in FIELDS}
    out["file"] = path
    return out


def _float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _write_svg(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    width, height = 760, 480
    left, right, top, bottom = 72, 24, 28, 64
    xs = [_float(r, "mean_token_surrogate") for r in rows]
    ys = [_float(r, "accuracy") for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 0.1

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y: float) -> float:
        return height - bottom - (y - ymin) / (ymax - ymin) * (height - top - bottom)

    colors = {
        "acdan": "#1f77b4",
        "bon": "#d62728",
        "sc": "#ff7f0e",
        "asc": "#2ca02c",
        "cot": "#7f7f7f",
    }
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Mean token surrogate / cost</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">Accuracy</text>',
        f'<text x="{left}" y="{top-8}" font-family="sans-serif" font-size="13">ACDAN Pareto candidates</text>',
    ]
    for r in rows:
        method = str(r.get("method", ""))
        x, y = sx(_float(r, "mean_token_surrogate")), sy(_float(r, "accuracy"))
        color = colors.get(method, "#9467bd")
        label = f'{r.get("dataset","")} {method}'
        elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" opacity="0.85"><title>{label}</title></circle>')
    legend_x = width - 145
    for i, (method, color) in enumerate(colors.items()):
        y = top + 18 + i * 20
        elems.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        elems.append(f'<text x="{legend_x+12}" y="{y+4}" font-family="sans-serif" font-size="12">{method}</text>')
    elems.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elems), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect ACDAN result summaries.")
    parser.add_argument("--glob", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--svg", default=None)
    args = parser.parse_args()

    rows = [_row(path) for path in sorted(glob.glob(args.glob))]
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {csv_path}")
    if args.svg:
        _write_svg(rows, Path(args.svg))
        print(f"wrote svg -> {args.svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
