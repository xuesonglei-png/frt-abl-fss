"""Plan or execute the one-factor-at-a-time sensitivity campaign (frozen v10 full).

Planning is the default: the frozen baseline configs and prepared manifests are
verified, ``plan.json`` plus one config per arm are written, and nothing is
trained.  Execution is double-gated (``--execute --confirm-sensitivity-runs``).
Every training run only trains and selects on validation; the formal test split
is never read by this protocol.

Typical stages:

    # stage A: plan (design seed 4) and inspect the arm grid
    python -m methods.frt_abl.run_sensitivity

    # stage A: execute the screening
    python -m methods.frt_abl.run_sensitivity --execute --confirm-sensitivity-runs --parallel 4

    # stage B: add seeds 0..4 to the same campaign (completed runs are skipped)
    python -m methods.frt_abl.run_sensitivity --seeds 0 1 2 3 4 \
        --execute --confirm-sensitivity-runs --parallel 4

    # aggregate (also: --summarize-only after the fact)
    python -m methods.frt_abl.run_sensitivity --summarize-only

The protocol (grids, margin, seeds, roots) lives in ``configs/sensitivity.json``;
the aggregation and decision rules live in ``methods/frt_abl/sensitivity.py``.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .prepare import PROJECT, load_config, verify_prepared, write_json
from .sensitivity import (COMPLETE_TRAINING_STATUS, DEFAULT_SENSITIVITY_CONFIG,
                          REGISTERED_SEEDS, campaign_identity, declared_factors,
                          default_factor_keys, load_sensitivity_config,
                          normalise_factor_keys, prepare_campaign, write_summary)


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A run that is still starting writes config.json non-atomically.
        return None


def load_baselines(config_paths, spec):
    """Load + verify the frozen baseline configs; no model or split is touched."""
    baselines = {}
    for path in config_paths:
        cfg = load_config(path)
        if cfg["method_version"] != spec["baseline"]["method_version"]:
            raise SystemExit(f"{path}: method_version differs from the sensitivity baseline")
        if cfg["variant"] != spec["baseline"]["variant"]:
            raise SystemExit(f"{path}: sensitivity campaigns require variant "
                             f"{spec['baseline']['variant']!r}")
        if cfg["task"] in baselines:
            raise SystemExit(f"Duplicate task config: {cfg['task']}")
        verify_prepared(cfg)
        baselines[cfg["task"]] = cfg
    if not baselines:
        raise SystemExit("No baseline configs")
    return baselines


def completed_runs(campaign_dir, campaign):
    """Map (task, arm, seed) -> completed run dirs already inside the campaign."""
    completed = {}
    root = Path(campaign_dir) / "runs"
    if not root.is_dir():
        return completed
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        cfg = _load_json(run_dir / "config.json")
        status = _load_json(run_dir / "status.json") or {}
        if not isinstance(cfg, dict):
            continue
        marker = cfg.get("sensitivity") or {}
        if marker.get("campaign") != campaign:
            continue
        if status.get("status") != COMPLETE_TRAINING_STATUS or status.get("stopped_reason"):
            continue
        key = (cfg.get("task"), marker.get("arm"), cfg.get("seed"))
        completed.setdefault(key, []).append(run_dir.name)
    return completed


def command_for(args, config_path, seed, variant):
    command = [args.python, "-m", "methods.frt_abl.train",
               "--config", str(config_path), "--variant", variant,
               "--seed", str(seed), "--execute-training"]
    if args.threads is not None:
        command += ["--threads", str(args.threads)]
    if args.search_workers is not None:
        command += ["--search-workers", str(args.search_workers)]
    if args.device is not None:
        command += ["--device", args.device]
    return command


def _plan_report(plan, pending, skipped):
    return {
        "status": "PLAN_ONLY",
        "protocol_version": plan["protocol_version"],
        "campaign": plan["campaign"],
        "campaign_dir": plan["campaign_dir"],
        "runs_root": plan["runs_root"],
        "seeds": plan["seeds"],
        "factors": [item["key"] for item in plan["factors"]],
        "n_arms": plan["n_arms"],
        "n_runs_in_plan": plan["n_runs"],
        "n_to_run": len(pending),
        "n_skipped_completed": len(skipped),
        "analysis": plan["analysis"],
        "formal_test_read": False,
        "note": ("Planning only: configs and prepared manifests were verified, no model was "
                 "built. Execution requires --execute --confirm-sensitivity-runs."),
    }


CLAIM_TTL_SECONDS = 24 * 3600


def _claim_path(campaign_dir, item):
    name = f"{item['task']}__{item['arm'].replace('=', '-')}__seed-{item['seed']}.claim"
    return Path(campaign_dir) / "claims" / name


def _claim(campaign_dir, item):
    """Exclusive claim so two concurrent executes cannot queue the same run.

    Returns the claim path, or None when another process holds a fresh claim.
    Claims older than CLAIM_TTL_SECONDS are treated as crashed and taken over.
    """
    path = _claim_path(campaign_dir, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({"task": item["task"], "arm": item["arm"],
                                         "seed": item["seed"], "pid": os.getpid()}) + "\n")
            return path
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                return None
            if age <= CLAIM_TTL_SECONDS:
                return None
            try:
                path.unlink()
            except OSError:
                return None
    return None


def execute(plan, pending, args):
    results = []
    claimed = []
    for item in pending:
        claim = _claim(plan["campaign_dir"], item)
        if claim is None:
            print(f"SKIP  {item['task']} {item['arm']} seed-{item['seed']} (claimed)",
                  flush=True)
            results.append({"task": item["task"], "arm": item["arm"], "seed": item["seed"],
                            "command": item["command"], "status": "SKIPPED_CLAIMED",
                            "returncode": None})
        else:
            claimed.append((item, claim))

    def launch(item, claim):
        try:
            print(f"START {item['task']} {item['arm']} seed-{item['seed']}", flush=True)
            try:
                subprocess.run(item["command"], cwd=str(PROJECT), check=True)
            except subprocess.CalledProcessError as error:
                print(f"FAIL  {item['task']} {item['arm']} seed-{item['seed']} "
                      f"rc={error.returncode}", flush=True)
                return {"task": item["task"], "arm": item["arm"], "seed": item["seed"],
                        "command": item["command"], "status": "FAILED",
                        "returncode": error.returncode}
            except OSError as error:
                print(f"FAIL  {item['task']} {item['arm']} seed-{item['seed']} "
                      f"{error}", flush=True)
                return {"task": item["task"], "arm": item["arm"], "seed": item["seed"],
                        "command": item["command"], "status": "FAILED",
                        "returncode": None, "error": repr(error)}
            print(f"DONE  {item['task']} {item['arm']} seed-{item['seed']}", flush=True)
            return {"task": item["task"], "arm": item["arm"], "seed": item["seed"],
                    "command": item["command"], "status": "COMPLETED", "returncode": 0}
        finally:
            try:
                claim.unlink()
            except OSError:
                pass

    if claimed:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(launch, item, claim) for item, claim in claimed]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Frozen baseline configs (default: sensitivity config)")
    parser.add_argument("--sensitivity-config", default=DEFAULT_SENSITIVITY_CONFIG)
    parser.add_argument("--factors", nargs="+", default=None,
                        help="Subset of declared factors (optional knobs need explicit opt-in)")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Only queue these arm ids (e.g. full_boundary_weight=2x)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seed set for this planning call (union with the existing plan)")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--summary-root", default=None)
    parser.add_argument("--campaign", default=None,
                        help="Explicit campaign directory for --summarize-only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-sensitivity-runs", action="store_true")
    parser.add_argument("--allow-duplicate-seed", action="store_true",
                        help="Queue an arm/seed even if a completed run already exists")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--search-workers", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--summarize-only", action="store_true",
                        help="Aggregate an existing campaign without planning or training "
                             "(derives the campaign from --factors; pass --campaign to be explicit)")
    parser.add_argument("--overwrite", action="store_true",
                        help="With --summarize-only: overwrite an existing summary directory")
    parser.add_argument("--print-plan", action="store_true",
                        help="Print the full plan JSON (arm grid and every run)")
    args = parser.parse_args()
    if args.execute and args.summarize_only:
        raise SystemExit("--execute and --summarize-only are mutually exclusive")
    if args.execute and not args.confirm_sensitivity_runs:
        raise SystemExit("Refusing to launch sensitivity runs without --confirm-sensitivity-runs")
    if args.parallel < 1:
        raise SystemExit("--parallel must be positive")
    for name in ("threads", "search_workers"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    spec = load_sensitivity_config(args.sensitivity_config)
    if args.output_root:
        spec["output_root"] = str(_resolve(args.output_root))
    if args.summary_root:
        spec["summary_root"] = str(_resolve(args.summary_root))
    factors = declared_factors(spec)
    declared = {factor.key: factor for factor in factors}
    factor_keys = list(args.factors) if args.factors else default_factor_keys(spec)
    try:
        factor_keys = normalise_factor_keys(spec, factor_keys)
    except ValueError as error:
        raise SystemExit(str(error))
    optional = [key for key in factor_keys if declared[key].optional]
    if optional and not args.factors:
        raise SystemExit("Optional factors require an explicit --factors list")
    seeds = args.seeds if args.seeds is not None else list(spec["seeds"])
    if len(set(seeds)) != len(seeds) or any(seed not in REGISTERED_SEEDS for seed in seeds):
        raise SystemExit("Registered seeds are 0..5 and must be unique")
    config_paths = args.configs or list(spec["baseline"]["configs"])

    if args.summarize_only:
        if args.campaign:
            campaign_dir = _resolve(args.campaign)
        else:
            baselines = load_baselines(config_paths, spec)
            campaign_dir = Path(spec["output_root"]) / campaign_identity(
                spec, baselines, factor_keys, spec["baseline"]["method_version"])
        try:
            summary, output = write_summary(campaign_dir, spec["summary_root"],
                                            overwrite=args.overwrite)
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            raise SystemExit(f"Cannot summarize {campaign_dir}: {error}")
        print(json.dumps({"status": "SENSITIVITY_SUMMARY_WRITTEN",
                          "campaign": summary["campaign"],
                          "campaign_dir": str(campaign_dir),
                          "summary_root": str(output),
                          "n_completed_runs": summary["n_completed"],
                          "verdicts": {task: [item["verdict"] for item in payload["factors"]]
                                       for task, payload in summary["tasks"].items()}},
                         ensure_ascii=False, indent=2))
        return

    baselines = load_baselines(config_paths, spec)
    plan = prepare_campaign(spec, baselines, factor_keys, seeds,
                            spec["baseline"]["method_version"])
    completed = completed_runs(plan["campaign_dir"], plan["campaign"])
    if args.arms:
        known = {record["arm"] for task in plan["tasks"] for record in plan["arms"][task]}
        unknown_arms = [arm for arm in args.arms if arm not in known]
        if unknown_arms:
            raise SystemExit(f"Unknown arm(s): {sorted(unknown_arms)}")
    pending, skipped = [], []
    for item in plan["runs"]:
        if args.arms and item["arm"] not in args.arms:
            continue
        key = (item["task"], item["arm"], item["seed"])
        if completed.get(key) and not args.allow_duplicate_seed:
            skipped.append({"task": item["task"], "arm": item["arm"],
                            "seed": item["seed"], "runs": completed[key]})
            continue
        pending.append(dict(item, command=command_for(args, item["config"], item["seed"],
                                                      plan["variant"])))
    report = _plan_report(plan, pending, skipped)
    if args.print_plan:
        report["plan"] = plan
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    plan["status"] = "EXECUTING"
    write_json(Path(plan["campaign_dir"]) / "plan.json", plan)
    results = execute(plan, pending, args)
    failures = [item for item in results if item["status"] == "FAILED"]
    skipped_claims = [item for item in results if item["status"] == "SKIPPED_CLAIMED"]
    plan["status"] = "EXECUTED_WITH_FAILURES" if failures else "EXECUTED"
    plan.setdefault("executions", []).extend(results)
    write_json(Path(plan["campaign_dir"]) / "plan.json", plan)
    print(json.dumps({"status": plan["status"],
                      "campaign": plan["campaign"],
                      "n_completed": sum(1 for item in results
                                         if item["status"] == "COMPLETED"),
                      "n_skipped_claimed": len(skipped_claims),
                      "n_failed": len(failures),
                      "failed": [{key: item[key] for key in ("task", "arm", "seed")}
                                 for item in failures]},
                     ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
