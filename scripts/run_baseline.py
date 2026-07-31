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


def resolve_templates(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(arguments)
    for _ in range(3):
        values = {**context, **resolved}
        updated = {
            key: value.format(**values) if isinstance(value, str) else value
            for key, value in resolved.items()
        }
        if updated == resolved:
            return updated
        resolved = updated
    return resolved


def verify_revision(repo_dir: Path, expected: str) -> str:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True,
    ).strip()
    if actual != expected:
        raise RuntimeError(
            f"{repo_dir} is at {actual}, but the baseline manifest requires {expected}"
        )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an external baseline with a versioned CONCORD comparison config."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--allow-revision-mismatch",
        action="store_true",
        help="Run even if the external repository is not at the pinned commit.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    baseline = cfg["baseline"]
    repo_dir = Path(baseline["repo_dir"])
    entrypoint = baseline.get("entrypoint", "run.py")
    python = baseline.get("python", "python")
    output_dir = Path(cfg.get("output_dir", "runs/baselines")) / baseline["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_revision = str(baseline["revision"])
    actual_revision = None
    if repo_dir.exists():
        try:
            actual_revision = verify_revision(repo_dir, expected_revision)
        except RuntimeError:
            if not args.allow_revision_mismatch:
                raise
    elif not args.dry_run:
        raise FileNotFoundError(
            f"{repo_dir} does not exist; run scripts/clone_baselines.sh first"
        )

    defaults = dict(cfg.get("arguments", {}))
    experiments = list(cfg.get("experiments", [{"name": "default", "arguments": {}}]))
    commands: list[dict[str, Any]] = []
    for experiment in experiments:
        experiment_name = str(experiment["name"])
        base_arguments = {**defaults, **dict(experiment.get("arguments", {}))}
        sweeps = {**dict(cfg.get("sweeps", {})), **dict(experiment.get("sweeps", {}))}
        for run_args in expand_sweeps(base_arguments, sweeps):
            run_args = resolve_templates(run_args, {"experiment": experiment_name})
            cmd = [python, entrypoint, *to_argv(run_args)]
            record = {
                "experiment": experiment_name,
                "cwd": str(repo_dir),
                "expected_revision": expected_revision,
                "actual_revision": actual_revision,
                "cmd": cmd,
            }
            commands.append(record)
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=repo_dir, check=True)

    with (output_dir / "commands.json").open("w", encoding="utf-8") as f:
        json.dump(commands, f, indent=2)


if __name__ == "__main__":
    main()
