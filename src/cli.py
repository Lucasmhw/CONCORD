from __future__ import annotations

import argparse

from concord.config import apply_overrides, load_config
from concord.data.preprocess import preprocess_dataset
from concord.training.evaluate import evaluate_main
from concord.training.train import train_main


def main() -> None:
    parser = argparse.ArgumentParser(description="CONCORD reproducible CLI")
    parser.add_argument("command", choices=["preprocess", "train", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    if args.command == "preprocess":
        path = preprocess_dataset(cfg)
        print(f"Processed dataset written to: {path}")
        return
    if args.command == "train":
        metrics = train_main(cfg)
        print(metrics)
        return
    if args.command == "evaluate":
        metrics = evaluate_main(cfg)
        print(metrics)
        return


if __name__ == "__main__":
    main()
