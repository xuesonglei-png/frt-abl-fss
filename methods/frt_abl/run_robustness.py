"""Plan or execute the post-hoc robustness matrix for frozen v10 `full` runs.

Default is a plan-only report: it verifies configs/manifests/checkpoints but
builds no model and reads no sample split.  Execution is double-gated
(`--execute --confirm-robustness-runs`); a formal-test split additionally
requires `--approve-formal-test`, exactly like evaluate.py.

The baseline is the frozen v10 `full` checkpoint per task and seed.  The three
pre-registered perturbations and the metric definitions live in
`methods/frt_abl/robustness.py` and `configs/robustness.json`.
"""
import argparse
import json
from pathlib import Path

from .prepare import PROJECT, digest, load_config, verify_prepared, write_json
from .robustness import (COMPLETE_TRAINING_STATUS, PROTOCOL_VERSION, build_conditions,
                         describe_checkpoint, discover_checkpoint, evaluate_condition,
                         load_frozen_model, load_robustness_config, load_split,
                         write_summary)

DEFAULT_CONFIGS = tuple(str(PROJECT / "methods" / "frt_abl" / "configs" / name)
                        for name in ("mnist.json", "hwf.json"))
DEFAULT_ROBUSTNESS_CONFIG = str(PROJECT / "methods" / "frt_abl" / "configs"
                                / "robustness.json")


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else PROJECT / value


def _checkpoint_record(info):
    return {"seed": info.get("seed"),
            "checkpoint": info.get("checkpoint"), "run_id": info.get("run_id"),
            "status": info.get("status"),
            "best_validation_accuracy": info.get("best_validation_accuracy"),
            "checkpoint_sha256": info.get("checkpoint_sha256")}


def plan_task(config_path, spec, seeds, stage, split, runs_root, explicit_checkpoint=None):
    cfg = load_config(config_path)
    receipt = verify_prepared(cfg)
    conditions = build_conditions(spec, cfg["task"], stage)
    checkpoints = []
    for seed in seeds:
        if explicit_checkpoint:
            info = describe_checkpoint(explicit_checkpoint)
            info["seed"] = seed
            if info.get("status") not in (None, COMPLETE_TRAINING_STATUS):
                raise ValueError(f"Explicit checkpoint is not a completed training run: "
                                 f"{explicit_checkpoint} status={info.get('status')}")
        else:
            info = discover_checkpoint(runs_root, cfg["task"], seed,
                                       method_version=spec["baseline"]["method_version"],
                                       variant=spec["baseline"]["variant"])
            info["seed"] = seed
        checkpoints.append(_checkpoint_record(info))
    return {"config": str(Path(config_path)), "task": cfg["task"], "cfg": cfg,
            "n_samples": receipt["counts"].get(split), "conditions": conditions,
            "checkpoints": checkpoints}


def plan_report(tasks, seeds, stage, split):
    return {
        "status": "PLAN_ONLY",
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "split": split,
        "seeds": list(seeds),
        "formal_test_read": False,
        "n_conditions_per_task": max(len(task["conditions"]) for task in tasks),
        "n_runs": sum(len(task["conditions"]) * len(seeds) for task in tasks),
        "tasks": [{"config": task["config"], "task": task["task"],
                   "n_samples": task["n_samples"],
                   "conditions": [condition.as_dict() for condition in task["conditions"]],
                   "checkpoints": task["checkpoints"]} for task in tasks],
        "note": ("Plan only. No model was built and no split file was read. "
                 "Execution requires --execute --confirm-robustness-runs."),
    }


def execute_plan(spec, tasks, seeds, stage, split, output_root, summary_root,
                 approve_formal_test=False, resume=False):
    if split == "test" and not approve_formal_test:
        raise SystemExit("Formal test robustness evaluation requires --approve-formal-test")
    import torch

    output_root = Path(output_root)
    summary_root = Path(summary_root)
    if summary_root.exists() and not resume:
        raise SystemExit(f"Summary directory already exists: {summary_root}. "
                         "Use --resume to regenerate it.")
    output_root.mkdir(parents=True, exist_ok=True)
    evaluations = []
    for task in tasks:
        cfg = task["cfg"]
        torch.set_num_threads(cfg["threads"])
        samples = load_split(cfg["prepared_root"], split)
        for seed, checkpoint in zip(seeds, task["checkpoints"]):
            checkpoint_path = Path(checkpoint["checkpoint"])
            model, _ = load_frozen_model(cfg, checkpoint_path,
                                         require_variant=spec["baseline"]["variant"])
            checkpoint_sha256 = digest(checkpoint_path)
            provenance = {
                "protocol_version": PROTOCOL_VERSION,
                "method_version": spec["baseline"]["method_version"],
                "variant": spec["baseline"]["variant"],
                "task": cfg["task"], "seed": seed, "split": split,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "run_id": checkpoint.get("run_id"),
                "status": checkpoint.get("status"),
                "best_validation_accuracy": checkpoint.get("best_validation_accuracy"),
            }
            evaluations.append(dict(provenance))
            for condition in task["conditions"]:
                condition_dir = (output_root / cfg["task"] / f"seed-{seed}"
                                 / condition.condition_id)
                if condition_dir.exists():
                    status_path = condition_dir / "status.json"
                    status = {}
                    if status_path.is_file():
                        try:
                            status = json.loads(status_path.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            status = {}
                    if resume and status.get("status") == "ROBUSTNESS_EVAL_COMPLETED":
                        print(json.dumps({"status": "SKIP_COMPLETED", "task": cfg["task"],
                                          "seed": seed,
                                          "condition": condition.condition_id}), flush=True)
                        continue
                    raise SystemExit(
                        f"Refusing to overwrite existing condition: {condition_dir}. "
                        "Use --resume to keep completed conditions.")
                metrics = evaluate_condition(
                    cfg, model, condition, samples, condition_dir, seed=seed, split=split,
                    noise_salt=spec["noise"]["seed_salt"], provenance=provenance,
                    batch_size=cfg["batch_size"], device="cpu")
                print(json.dumps({"status": "CONDITION_DONE", "task": cfg["task"],
                                  "seed": seed, "condition": condition.condition_id,
                                  "task_accuracy": metrics["task_accuracy"],
                                  "symbol_accuracy": metrics["symbol_accuracy"],
                                  "error_correction_rate": metrics["error_correction_rate"],
                                  "undecided_rate": metrics["undecided_rate"]}), flush=True)
            del model
    write_json(output_root / "baseline.json", {
        "protocol_version": PROTOCOL_VERSION, "stage": stage, "split": split,
        "seeds": list(seeds), "evaluations": evaluations,
        "note": "Frozen v10 full checkpoints; robustness perturbations are evaluation-only."})
    summary = write_summary(output_root, summary_root, overwrite=resume)
    print(json.dumps({"status": "ROBUSTNESS_MATRIX_COMPLETED",
                      "output_root": str(output_root),
                      "summary_root": str(summary_root),
                      "n_conditions": summary["n_conditions"],
                      "n_seeds": summary["n_seeds"]}, ensure_ascii=False, indent=2),
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--robustness-config", default=DEFAULT_ROBUSTNESS_CONFIG)
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="One explicit best.pt per --config (otherwise discovered)")
    parser.add_argument("--runs-root", default=None,
                        help="Completed v10 run directories (default: robustness config)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override the robustness config seeds (0..4)")
    parser.add_argument("--stage", choices=("single", "combined", "all"), default="single",
                        help="Single-factor perturbations first, then combinations")
    parser.add_argument("--split", choices=("validation", "test"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--summary-root", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-robustness-runs", action="store_true")
    parser.add_argument("--approve-formal-test", action="store_true",
                        help="Allow reading the formal test manifest for this run")
    parser.add_argument("--resume", action="store_true",
                        help="Keep completed conditions and regenerate the summary")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Aggregate an existing robustness run directory")
    args = parser.parse_args()
    if args.execute and args.summarize_only:
        raise SystemExit("--execute and --summarize-only are mutually exclusive")
    if args.execute and not args.confirm_robustness_runs:
        raise SystemExit("Refusing to launch robustness runs without --confirm-robustness-runs")
    if args.checkpoints is not None and len(args.checkpoints) != len(args.configs):
        raise SystemExit("--checkpoints must provide exactly one path per --config")

    spec = load_robustness_config(args.robustness_config)
    seeds = args.seeds if args.seeds is not None else spec["seeds"]
    if any(seed not in (0, 1, 2, 3, 4) for seed in seeds):
        raise SystemExit("Registered seeds are 0..4")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("Duplicate seeds are not independent repetitions")
    stage = args.stage
    split = args.split or spec["split"]
    if args.approve_formal_test and split != "test":
        raise SystemExit("--approve-formal-test is only valid together with --split test")
    runs_root = _resolve(args.runs_root or spec["baseline"]["runs_root"])
    output_root = _resolve(args.output_root or spec["output_root"])
    summary_root = _resolve(args.summary_root or spec["summary_root"])

    if args.summarize_only:
        try:
            summary = write_summary(output_root, summary_root, overwrite=args.resume)
        except FileExistsError as error:
            raise SystemExit(f"{error}. Use --resume to regenerate it.") from error
        print(json.dumps({"status": "ROBUSTNESS_SUMMARY_WRITTEN",
                          "output_root": str(output_root),
                          "summary_root": str(summary_root),
                          "n_conditions": summary["n_conditions"],
                          "n_seeds": summary["n_seeds"]}, ensure_ascii=False, indent=2))
        return

    if args.checkpoints is not None and len(seeds) > 1:
        raise SystemExit("Explicit --checkpoints are ambiguous with multiple seeds; "
                         "pass a single seed or omit --checkpoints")

    tasks = []
    for config_path, checkpoint in zip(args.configs,
                                       args.checkpoints or [None] * len(args.configs)):
        tasks.append(plan_task(config_path, spec, seeds, stage, split, runs_root, checkpoint))
    plan = plan_report(tasks, seeds, stage, split)
    if split == "test":
        plan["warning"] = ("Formal test is planned but not read; execution requires "
                           "--approve-formal-test.")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    execute_plan(spec, tasks, seeds, stage, split, output_root, summary_root,
                 approve_formal_test=args.approve_formal_test, resume=args.resume)


if __name__ == "__main__":
    main()
