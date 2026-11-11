"""Plan or execute the V10 × annotation-budget matrix; dry-run by default.

The method code is frozen at V10. Each run changes only the prepared annotation
budget (default 1/3/5/10% of training images) and is selected on validation.
Formal test data is never read by this runner.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

from .budget import (DEFAULT_CONFIG_OUT, EXPERIMENT, project_relative,
                     validate_budget_config, verify_budget_prepared)
from .prepare import PROJECT, load_config, verify_prepared
from .policies import VARIANTS

SEEDS = (0, 1, 2, 3, 4, 5)


def discover_configs():
    return [str(path) for path in sorted(Path(DEFAULT_CONFIG_OUT).glob("*.json"))]


def config_status(cfg):
    root = Path(cfg["prepared_root"])
    present = (root / "manifest.json").is_file()
    if not present:
        return False, None
    receipt = verify_prepared(cfg)
    verify_budget_prepared(cfg, receipt)
    return True, receipt


def build_plan(config_paths, variants, seeds):
    plan, statuses = [], []
    for path in config_paths:
        cfg = load_config(path)
        validate_budget_config(cfg)
        prepared, receipt = config_status(cfg)
        statuses.append({"config": project_relative(path), "task": cfg["task"],
                         "level": cfg["budget"]["level"],
                         "level_label": cfg["budget"]["level_label"],
                         "prepared": prepared,
                         "actual_total_unique_labeled_images":
                             cfg["budget"]["actual_total_unique_labeled_images"]})
        for variant in variants:
            for seed in seeds:
                command = [sys.executable, "-m", "methods.frt_abl.train",
                           "--config", str(Path(path).resolve()),
                           "--variant", variant, "--seed", str(seed),
                           "--execute-training"]
                plan.append({"task": cfg["task"],
                             "level": cfg["budget"]["level"],
                             "level_label": cfg["budget"]["level_label"],
                             "actual_total_unique_labeled_images":
                                 cfg["budget"]["actual_total_unique_labeled_images"],
                             "variant": variant, "seed": seed,
                             "prepared": prepared,
                             "config": project_relative(path),
                             "command": command})
    plan.sort(key=lambda entry: (entry["task"], entry["level"],
                                 entry["variant"], entry["seed"]))
    return plan, statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Budget configs; defaults to configs/budget/*.json")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["full"],
                        help="Default is the frozen V10 full method only")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--execute", action="store_true",
                        help="Actually launch the runs")
    parser.add_argument("--confirm-budget-runs", action="store_true")
    parser.add_argument("--confirm-formal-runs", action="store_true",
                        help="Accepted alias for the repository's existing confirmation flag")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Concurrent training processes; default serial")
    parser.add_argument("--threads", type=int, default=None,
                        help="Per-run torch CPU threads override")
    parser.add_argument("--search-workers", type=int, default=None,
                        help="Per-run search worker process override")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--report", default=None,
                        help="Optional path for the plan/execution report JSON")
    args = parser.parse_args()
    paths = args.configs or discover_configs()
    if not paths:
        raise SystemExit("No budget configs found; run `methods.frt_abl.budget prepare --execute` first")
    if len(set(str(Path(path).resolve()) for path in paths)) != len(paths):
        raise SystemExit("Duplicate budget configs are not independent runs")
    if any(seed not in SEEDS for seed in args.seeds):
        raise SystemExit("Registered seeds are 0..5")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Duplicate seeds are not independent repetitions")
    if len(set(args.variants)) != len(args.variants):
        raise SystemExit("Duplicate variants are not independent comparisons")
    if args.execute and not (args.confirm_budget_runs or args.confirm_formal_runs):
        raise SystemExit("Refusing to launch the budget matrix without --confirm-budget-runs")
    if args.parallel < 1:
        raise SystemExit("--parallel must be positive")
    if args.execute and args.device == "cuda":
        probe = subprocess.run([args.python, "-c",
                               "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                              capture_output=True, text=True)
        if probe.returncode != 0:
            raise SystemExit(f"--python does not provide CUDA: {args.python}")
    plan, statuses = build_plan(paths, args.variants, args.seeds)
    missing = [entry for entry in plan if not entry["prepared"]]
    if args.execute and missing:
        examples = ", ".join(f'{entry["task"]}/{entry["level_label"]}' for entry in missing[:4])
        raise SystemExit(f"Prepared budget manifests are missing ({examples}); run prepare first")
    report = {"status": "PLAN_ONLY" if not args.execute else "EXECUTING",
              "experiment": EXPERIMENT,
              "n_runs": len(plan), "parallel": args.parallel, "device": args.device,
              "python": args.python, "seeds": args.seeds, "variants": args.variants,
              "configs": statuses, "runs": plan,
              "note": ("Validation-only selection. Formal test is not read by this runner; "
                       "evaluate separately only after freezing the budget curve protocol.")}
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return

    def launch(item):
        print(f'START {item["task"]} {item["level_label"]} {item["variant"]} '
              f'seed-{item["seed"]}', flush=True)
        subprocess.run(item["command"], cwd=str(PROJECT), check=True)
        print(f'DONE  {item["task"]} {item["level_label"]} {item["variant"]} '
              f'seed-{item["seed"]}', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(launch, item) for item in plan]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
