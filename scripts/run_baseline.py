from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_sweeps(arguments: dict[str, Any], sweeps: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not sweeps:
        return [arguments]
    keys = list(sweeps)
    runs = []
    for values in itertools.product(*(sweeps[key] for key in keys)):
        merged = dict(arguments)
        merged.update(dict(zip(keys, values)))
        runs.append(merged)
    return runs


def to_argv(arguments: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in arguments.items():
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an external baseline with a recorded CONCORD comparison config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    baseline = cfg["baseline"]
    repo_dir = Path(baseline["repo_dir"])
    entrypoint = baseline.get("entrypoint", "run.py")
    python = baseline.get("python", "python")
    output_dir = Path(cfg.get("output_dir", "runs/baselines")) / baseline["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = expand_sweeps(dict(cfg.get("arguments", {})), dict(cfg.get("sweeps", {})))
    commands = []
    for run_args in runs:
        cmd = [python, entrypoint, *to_argv(run_args)]
        commands.append({"cwd": str(repo_dir), "cmd": cmd})
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_dir, check=True)

    with (output_dir / "commands.json").open("w", encoding="utf-8") as f:
        json.dump(commands, f, indent=2)


if __name__ == "__main__":
    main()
