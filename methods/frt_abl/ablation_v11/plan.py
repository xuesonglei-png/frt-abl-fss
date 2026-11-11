"""Print commands for independent entry points; this planner cannot launch training."""

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

ENTRIES = (
    "full",
    "confidence_only",
    "fixed_threshold",
    "no_curriculum",
    "no_weight",
    "no_boundary",
    "standard_abl",
    "anchor_only",
)


def make_plan(tasks=("mnist-add2", "hwf"), seeds=(0, 1, 2, 3, 4), include_factorial=False):
    """Read each experiment's own definitions, never construct shared defaults."""
    if len(set(tasks)) != len(tasks) or any(t not in {"mnist-add2", "hwf"} for t in tasks):
        raise ValueError("Tasks must be unique registered task names")
    if len(set(seeds)) != len(seeds) or any(s not in range(5) for s in seeds):
        raise ValueError("Seeds must be unique registered seeds 0..4")
    entries = ENTRIES + (("fixed_no_curriculum",) if include_factorial else ())
    rows = []
    for entry in entries:
        settings = importlib.import_module(f"{__package__}.{entry}.settings")
        protocol = importlib.import_module(f"{__package__}.{entry}.protocol")
        for task in tasks:
            for seed in seeds:
                cfg = settings.build_config(task, seed)
                command = [
                    sys.executable,
                    "-m",
                    f"{__package__}.{entry}.train",
                    "--task",
                    task,
                    "--seed",
                    str(seed),
                    "--execute-training",
                ]
                rows.append(
                    {
                        "task": task,
                        "variant": entry,
                        "seed": seed,
                        "entry_point": f"{__package__}.{entry}.train",
                        "settings_file": str(Path(settings.__file__).resolve()),
                        "config": cfg,
                        "implementation": protocol.experiment_record(cfg),
                        "command": command,
                        "powershell_command": "& "
                        + " ".join("'" + part.replace("'", "''") + "'" for part in command),
                        "windows_command": subprocess.list2cmdline(command),
                    }
                )
    return {
        "status": "PLAN_ONLY",
        "n_runs": len(rows),
        "training_started": False,
        "model_constructed": False,
        "formal_test_read": False,
        "runs": rows,
        "note": "Each command invokes independent source and settings. Nothing is launched.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=("mnist-add2", "hwf"), default=["mnist-add2", "hwf"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument("--include-factorial", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            make_plan(args.tasks, args.seeds, args.include_factorial), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
