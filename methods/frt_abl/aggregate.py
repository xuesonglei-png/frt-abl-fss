"""Paired completed runs only; no mixing versions, failures or same-seed retries."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from .prepare import PROJECT, write_json

COMPLETE = "TRAINING_COMPLETED_TEST_NOT_RUN"


def comparison_id(cfg, environment, source_manifest):
    common = {k: v for k, v in cfg.items() if k not in {"variant", "seed", "output_root"}}
    identity = {"config": common, "code_revision": environment.get("code_revision"),
                "data": environment.get("prepared_manifest_sha256"),
                "runtime": {k: environment.get(k) for k in ("python", "torch", "numpy", "device", "threads")},
                "sources": source_manifest}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def usable(runs):
    # Do not maximize over same-seed retries or treat a failed/stopped run as
    # completed. Select the intended campaign directory explicitly instead.
    counts = Counter(r["seed"] for r in runs)
    return [r for r in runs if counts[r["seed"]] == 1 and r["status"] == COMPLETE
            and r.get("stopped_reason") is None
            and isinstance(r["best_validation_accuracy"], (int, float))
            and math.isfinite(r["best_validation_accuracy"])
            and 0 <= r["best_validation_accuracy"] <= 1]


def collect(runs_root):
    runs_root = Path(runs_root)
    groups = {}
    if not runs_root.is_dir():
        return groups, 0
    seen = 0
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        try:
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
            sources = json.loads((run_dir / "source_manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        seen += 1
        key = (cfg["task"], cfg["variant"], comparison_id(cfg, environment, sources))
        groups.setdefault(key, []).append({
            "run_id": run_dir.name, "seed": cfg["seed"],
            "method_version": cfg.get("method_version", "legacy"),
            "status": status.get("status"),
            "best_validation_accuracy": status.get("best_validation_accuracy"),
            "optimizer_steps": status.get("optimizer_steps"),
            "stopped_reason": status.get("stopped_reason"),
        })
    return groups, seen


def summarize(groups):
    rows = []
    for (task, variant, identity), runs in sorted(groups.items()):
        admitted = usable(runs)
        accuracies = [r["best_validation_accuracy"] for r in admitted]
        counts = Counter(r["seed"] for r in runs)
        rows.append({
            "task": task, "variant": variant, "comparison_id": identity,
            "method_version": runs[0]["method_version"], "n": len(runs), "n_with_accuracy": len(accuracies),
            "seeds": sorted(r["seed"] for r in admitted),
            "duplicate_seeds_excluded": sorted(s for s, count in counts.items() if count > 1),
            "mean_validation_accuracy": statistics.fmean(accuracies) if accuracies else None,
            "std_validation_accuracy": (statistics.stdev(accuracies) if len(accuracies) > 1 else None),
            "min_validation_accuracy": min(accuracies) if accuracies else None,
            "max_validation_accuracy": max(accuracies) if accuracies else None,
            "statuses": sorted({str(r["status"]) for r in runs}),
            "run_ids": [r["run_id"] for r in sorted(admitted, key=lambda x: x["seed"])],
            "excluded_run_ids": [r["run_id"] for r in runs if r not in admitted],
        })
    return rows


def paired_deltas(groups):
    results = []
    for (task, variant, identity), runs in sorted(groups.items()):
        if variant == "full":
            continue
        full = {r["seed"]: r for r in usable(groups.get((task, "full", identity), []))}
        other = {r["seed"]: r for r in usable(runs)}
        seeds = sorted(set(full) & set(other))
        pairs = [{"seed": s, "full": full[s]["best_validation_accuracy"],
                  "ablation": other[s]["best_validation_accuracy"],
                  "delta": full[s]["best_validation_accuracy"]-other[s]["best_validation_accuracy"],
                  "same_optimizer_steps": full[s]["optimizer_steps"] == other[s]["optimizer_steps"]}
                 for s in seeds]
        deltas = [p["delta"] for p in pairs]
        results.append({"task": task, "ablation": variant, "comparison_id": identity,
                        "paired_seeds": seeds, "missing_full_seeds": sorted(set(other)-set(full)),
                        "missing_ablation_seeds": sorted(set(full)-set(other)), "pairs": pairs,
                        "mean_delta": statistics.fmean(deltas) if deltas else None,
                        "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
                        "strictly_positive_on_all_paired_seeds": all(d > 0 for d in deltas) if deltas else None,
                        "same_optimizer_budget": all(p["same_optimizer_steps"] for p in pairs) if pairs else None,
                        "interpretation": "Descriptive validation only; ties/losses retained; no efficacy guarantee."})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(PROJECT / "methods" / "frt_abl" / "runs-v10"))
    parser.add_argument("--output", default=str(PROJECT / "methods" / "frt_abl" / "runs-summary-v10"))
    args = parser.parse_args()
    groups, seen = collect(args.runs)
    rows = summarize(groups)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", {"status": "AGGREGATED_VALIDATION_ONLY",
                                        "runs_root": str(args.runs), "n_run_dirs": seen,
                                        "groups": rows,
                                        "paired_deltas": paired_deltas(groups),
                                        "note": "Complete unique seeds only. Incompatible data/source/configs separated; formal test not read."})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["task", "variant", "method_version", "comparison_id", "n", "n_with_accuracy", "seeds", "duplicate_seeds_excluded",
                  "mean_validation_accuracy", "std_validation_accuracy",
                  "min_validation_accuracy", "max_validation_accuracy", "statuses"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    print(json.dumps({"status": "AGGREGATED_VALIDATION_ONLY", "n_run_dirs": seen,
                      "n_groups": len(rows), "output": str(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
