"""Frozen, resumable small confirmation campaign; never sweeps the old grid.

Budget and sensitivity train/validate only. Generalization evaluations wait for
the already running length/combo matrix and use frozen checkpoints once each.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .controlled_budget import verify_controlled
from .prepare import PROJECT, digest, load_config, verify_prepared, write_json

ROOT = PROJECT / "methods/frt_abl/confirmation-20261104"
SEEDS = (0, 1, 2)
SOURCE_FILES = ("train.py", "runtime.py", "core.py", "policies.py", "prepare.py")
MODULE_ROOT = Path(__file__).parent


def sources():
    return {name: digest(MODULE_ROOT / name) for name in SOURCE_FILES}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stat(values):
    n = len(values)
    mean = statistics.fmean(values) if values else None
    sd = statistics.stdev(values) if n > 1 else None
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n)
    radius = t * sd / math.sqrt(n) if t is not None else None
    return dict(n=n, mean=mean, std=sd, ci95_t=[mean-radius, mean+radius] if radius is not None else None)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    cfgroot = ROOT / "configs"
    cfgroot.mkdir(exist_ok=True)
    jobs = []
    base = load_config("methods/frt_abl/configs/hwf.json")
    verify_prepared(base)
    # Two candidates were selected on validation seed4; seeds0-2 confirm only.
    for seed in SEEDS:
        for arm, override in (("baseline", {}), ("graph_sigma=0.25", {"graph_sigma": 0.25}),
                              ("full_boundary_weight=3x", {"full_boundary_weight": 0.6})):
            cfg = deepcopy(base)
            cfg.update(override)
            cfg.update(seed=seed, output_root=str(ROOT / "sensitivity-runs"),
                       run_tag=arm, experiment="sensitivity-shortlist-confirmation-v2",
                       confirmation=dict(screening_seed=4, confirmation_seeds=list(SEEDS), arm=arm))
            path = cfgroot / f"{arm}__seed-{seed}.json"
            save_config(path, cfg)
            jobs.append(dict(kind="sensitivity", arm=arm, seed=seed, config=str(path)))
    campaign = read(MODULE_ROOT / "prepared/budget-fixed-pool-v2/campaign.json")
    for seed in SEEDS:
        for entry in campaign["entries"]:
            cfg = load_config(entry["config"])
            verify_controlled(cfg)
            cfg["seed"] = seed
            path = cfgroot / f"budget-{entry['level_label']}__seed-{seed}.json"
            save_config(path, cfg)
            jobs.append(dict(kind="budget", arm=entry["level_label"], seed=seed, config=str(path)))
    plan = dict(protocol="three-experiment-confirmation-20261104-v2", sources=sources(),
                jobs=jobs, sensitivity_seeds=list(SEEDS), budget_seeds=list(SEEDS),
                generalization_seeds=[0, 1, 2, 3],
                generalization_primary="argmax_task_accuracy",
                sensitivity_test_disabled=True, budget_test_disabled=True,
                seed4_use="exploratory only; excluded from confirmation intervals",
                decision_rule="Report every preselected arm. Improvement requires positive paired "
                              "delta CI; otherwise no improvement claim. No additional candidates "
                              "are selected using confirmation/test outcomes.")
    target = ROOT / "plan.json"
    if target.exists() and read(target) != plan:
        raise ValueError("Frozen campaign plan changed")
    write_json(target, plan)
    return plan


def save_config(path, cfg):
    if path.exists() and read(path) != cfg:
        raise ValueError(f"Frozen configuration changed: {path}")
    write_json(path, cfg)


def completed(cfg, expected_sources):
    matches = []
    for run in Path(cfg["output_root"]).glob("*"):
        if not run.is_dir() or not (run / "status.json").is_file():
            continue
        status = read(run / "status.json")
        if status.get("status") != "TRAINING_COMPLETED_TEST_NOT_RUN":
            continue
        if read(run / "config.json") != cfg:
            continue
        saved_sources = read(run / "source_manifest.json")
        if any(saved_sources.get(name) != checksum for name, checksum in expected_sources.items()):
            raise ValueError(f"Training source mismatch: {run}")
        env = read(run / "environment.json")
        if env["prepared_manifest_sha256"] != digest(Path(cfg["prepared_root"]) / "manifest.json"):
            raise ValueError(f"Prepared manifest mismatch: {run}")
        if status.get("stopped_reason") or status.get("smoke") or not (run / "best.pt").is_file():
            continue
        matches.append(run)
    if len(matches) > 1:
        raise ValueError(f"Duplicate completed runs: {matches}")
    return matches[0] if matches else None


def run_job(job, plan):
    cfg = load_config(job["config"])
    found = completed(cfg, plan["sources"])
    if found:
        return dict(job, status="COMPLETED", run=str(found), reused=True)
    if sources() != plan["sources"]:
        raise ValueError("Frozen training source changed during campaign")
    logroot = ROOT / "logs"
    logroot.mkdir(exist_ok=True)
    log = logroot / f"{job['kind']}__{job['arm']}__seed-{job['seed']}.log"
    command = [sys.executable, "-m", "methods.frt_abl.train", "--config", job["config"],
               "--execute-training"]
    print(f"START {job['kind']} {job['arm']} seed-{job['seed']}", flush=True)
    with log.open("a", encoding="utf-8") as output:
        process = subprocess.run(command, cwd=PROJECT, stdout=output, stderr=subprocess.STDOUT)
    found = completed(cfg, plan["sources"])
    if process.returncode or found is None:
        return dict(job, status="FAILED", returncode=process.returncode, log=str(log))
    status = read(found / "status.json")
    if job["kind"] == "budget" and status["optimizer_steps"] != cfg["controlled_budget"]["expected_total_optimizer_steps"]:
        return dict(job, status="FAILED_EXPOSURE_AUDIT", run=str(found))
    print(f"DONE {job['kind']} {job['arm']} seed-{job['seed']}", flush=True)
    return dict(job, status="COMPLETED", run=str(found), reused=False)


def summarize(results):
    groups = {}
    values = {}
    for item in results:
        if item["status"] != "COMPLETED":
            continue
        status = read(Path(item["run"]) / "status.json")
        key = f"{item['kind']}/{item['arm']}"
        values.setdefault(key, {})[item["seed"]] = status["best_validation_accuracy"]
    for key, rows in values.items():
        groups[key] = dict(stat(list(rows.values())), by_seed=rows)
    baseline = values.get("sensitivity/baseline", {})
    paired = {}
    for arm in ("graph_sigma=0.25", "full_boundary_weight=3x"):
        rows = values.get(f"sensitivity/{arm}", {})
        seeds = sorted(set(rows) & set(baseline))
        deltas = [rows[seed] - baseline[seed] for seed in seeds]
        paired[arm] = dict(stat(deltas), seeds=seeds, deltas=deltas,
                           note="Seed4 excluded. Descriptive CIs; two candidate comparisons, "
                                "no multiplicity-adjusted significance claim.")
    # All budget pairs use the same seeds, weak data and number of updates.
    budget_paired = {}
    high = values.get("budget/10pct", {})
    for arm in ("1pct", "3pct"):
        rows = values.get(f"budget/{arm}", {})
        seeds = sorted(set(rows) & set(high))
        deltas = [rows[seed] - high[seed] for seed in seeds]
        budget_paired[arm + "-10pct"] = dict(stat(deltas), seeds=seeds, deltas=deltas)
    report = dict(groups=groups, sensitivity_paired=paired, budget_paired=budget_paired,
                  results=results, formal_test_read=False)
    write_json(ROOT / "training_summary.json", report)
    return report


def evaluate_generalization_after_training(timeout_hours=8):
    deadline = time.monotonic() + timeout_hours * 3600
    jobs = []
    for kind in ("length", "combo"):
        cfg = load_config(f"methods/frt_abl/configs/hwf-{kind}-gen.json")
        for seed in (0, 1, 2, 3):
            jobs.append((kind, cfg, seed))
    while True:
        ready = []
        for kind, cfg, seed in jobs:
            expected = dict(cfg, seed=seed)
            run = completed(expected, sources())
            if run:
                ready.append((kind, cfg, seed, run))
        if len(ready) == len(jobs):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Generalization training matrix did not finish; no duplicate training launched")
        time.sleep(30)
    frozen = []
    for kind, cfg, seed, run in ready:
        frozen.append(dict(kind=kind, seed=seed, run=str(run),
                           checkpoint_sha256=digest(run / "best.pt"),
                           prepared_manifest_sha256=digest(Path(cfg["prepared_root"]) / "manifest.json")))
    write_json(ROOT / "generalization_checkpoint_freeze.json", frozen)
    outputs = []
    for kind, cfg, seed, run in ready:
        dest = MODULE_ROOT / f"eval-v11-{kind}-gen" / f"seed{seed}-frozen-test-v1"
        if not (dest / "metrics_by_group.json").is_file():
            command = [sys.executable, "-m", "methods.frt_abl.evaluate_generalization",
                       "--config", f"methods/frt_abl/configs/hwf-{kind}-gen.json",
                       "--checkpoint", str(run / "best.pt"), "--output", str(dest),
                       "--execute-evaluation", "--approve-formal-test"]
            with (ROOT / "logs" / f"generalization-{kind}-seed{seed}.log").open("w", encoding="utf-8") as handle:
                subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        report = read(dest / "metrics_by_group.json")
        if report["checkpoint_sha256"] != digest(run / "best.pt"):
            raise ValueError("Existing evaluation belongs to a different checkpoint")
        outputs.append(dict(kind=kind, seed=seed, output=str(dest), primary=report["primary_eval"],
                            groups=report["groups"], overall=report["overall"]))
        print(f"EVAL {kind} seed-{seed}: OOD argmax={report['primary_eval']['argmax_task_accuracy']:.4f}", flush=True)
    groups = {}
    for kind in ("length", "combo"):
        records = [item for item in outputs if item["kind"] == kind]
        for key in ("argmax_task_accuracy", "argmax_sequence_accuracy", "argmax_symbol_accuracy",
                    "native_sequence_accuracy", "native_symbol_accuracy", "native_coverage"):
            groups[f"{kind}/length7/{key}"] = stat([item["primary"][key] for item in records])
    # The combination-specific primary OOD group has length5 (110 samples),
    # separating unseen compositions from the length7 shift.
    combo = [item for item in outputs if item["kind"] == "combo"]
    for token in ("5|False", "5|True", "7|False", "7|True"):
        selected = [next(row for row in item["groups"]["length_x_contains_holdout_bigram"]["groups"]
                         if row["group"] == token) for item in combo]
        for key in ("argmax_task_accuracy", "argmax_symbol_accuracy", "argmax_sequence_accuracy",
                    "native_sequence_accuracy", "native_coverage"):
            groups[f"combo/{token}/{key}"] = stat([item[key] for item in selected])
    write_json(ROOT / "generalization_summary.json", dict(
        primary_outcome="argmax_task_accuracy", confirmation_seeds=[0, 1, 2, 3],
        seed4_excluded=True, native_requires_true_result=True,
        summary=groups, per_seed=outputs,
        limitation="Same fixed test examples across seeds; intervals describe training-seed "
                   "variability. Expansion was decided after exploratory seed4 test. "
                   "No same-split external-method comparator was trained in this campaign."))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--evaluate-generalization", action="store_true")
    args = parser.parse_args()
    plan = prepare()
    print(json.dumps(dict(status="PLAN_ONLY" if not args.execute else "EXECUTING",
                          jobs=len(plan["jobs"]), root=str(ROOT), parallel=args.parallel)), flush=True)
    if not args.execute:
        return
    state = dict(status="RUNNING", completed=[], started_at=time.time())
    write_json(ROOT / "status.json", state)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_job, job, plan) for job in plan["jobs"]]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = dict(status="FAILED", error=repr(error))
            state["completed"].append(result)
            write_json(ROOT / "status.json", state)
            summarize(state["completed"])
    failures = [item for item in state["completed"] if item["status"] != "COMPLETED"]
    if args.evaluate_generalization:
        try:
            evaluate_generalization_after_training()
            state["generalization"] = "EVALUATED"
        except Exception as error:
            state["generalization"] = dict(status="FAILED", error=repr(error))
            failures.append(state["generalization"])
    state["status"] = "FAILED" if failures else "COMPLETED"
    state["finished_at"] = time.time()
    write_json(ROOT / "status.json", state)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
