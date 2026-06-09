from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def flatten(prefix: str, obj: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flatten(name, value, out)
        else:
            out[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CONCORD run metrics.")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="runs/metrics_summary.csv")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(args.runs).glob("*/metrics.json")):
        row: dict[str, Any] = {"run": metrics_path.parent.name}
        with metrics_path.open("r", encoding="utf-8") as f:
            flatten("", json.load(f), row)
        eval_path = metrics_path.parent / "eval_metrics.json"
        if eval_path.exists():
            with eval_path.open("r", encoding="utf-8") as f:
                flatten("eval", json.load(f), row)
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
