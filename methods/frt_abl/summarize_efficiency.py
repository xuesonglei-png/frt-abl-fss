"""Summarize completed efficiency runs: curve, time-to-target, budget accuracy.

Validation only.  No formal test split is read.  The summary keeps censoring
explicit: a target not reached within a run is reported with its observed
lower bound, never imputed as a success or silently dropped from the count.

Definitions
-----------
* `first_hit_seconds`: first validation round whose accuracy is >= target.
  Because validation happens once per round, this is an upper bound on the
  true crossing time.
* `interpolated_seconds`: linear interpolation between the last point below
  and the first point at/above the target.  Descriptive only.
* `last_accuracy` at a budget: accuracy of the last validation at or before
  the budget, i.e. what a fixed-budget run would have produced without
  peeking after the budget.
* `best_accuracy` at a budget: best validation accuracy among checkpoints at
  or before the budget (validation-based checkpoint selection is allowed by
  the protocol); reported alongside, never instead of, `last_accuracy`.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .aggregate import comparison_id
from .efficiency import DEFAULT_BUDGETS_SECONDS, DEFAULT_TARGETS, now_iso, parse_seconds
from .prepare import PROJECT, write_json

ANALYZABLE_STATUSES = {
    "TRAINING_COMPLETED_TEST_NOT_RUN",
    "STOPPED_TIME_BUDGET",
    "STOPPED_GATE_CLOSED",
}


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _std(values):
    values = [value for value in values if value is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def _min(values):
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _max(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _run_identity(cfg, environment, sources):
    try:
        if environment and sources:
            return comparison_id(cfg, environment, sources)
    except Exception:
        pass
    identity = {"config": {key: value for key, value in cfg.items()
                           if key not in {"variant", "seed", "output_root"}},
                "code_revision": cfg.get("method_version")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _points_from_efficiency(eff):
    points = []
    for item in eff.get("rounds", []):
        counters = item.get("counters") or {}
        points.append({
            "round": item.get("round"),
            "elapsed_seconds": item.get("elapsed_seconds"),
            "accuracy": item.get("accuracy"),
            "correct": item.get("correct"),
            "total": item.get("total"),
            "optimizer_steps": counters.get("optimizer_steps"),
            "candidate_checks": counters.get("candidate_checks"),
            "search_calls": counters.get("search_calls"),
            "inference_forwards": counters.get("inference_forwards"),
            "partial_round": bool(item.get("partial_round")),
        })
    return _clean_points(points)


def _points_from_history(history):
    """Fallback for old runs; their clock starts after warm-up."""
    points, checks, forwards = [], 0, 0
    for item in history:
        validation = item.get("validation_task") or {}
        checks += int(item.get("candidate_checks") or 0)
        forwards += int(item.get("proposal_forwards") or 0)
        forwards += int(item.get("training_forwards") or 0)
        forwards += int(validation.get("forwards") or 0)
        points.append({
            "round": item.get("round"),
            "elapsed_seconds": item.get("elapsed_seconds"),
            "accuracy": validation.get("accuracy"),
            "correct": validation.get("correct"),
            "total": validation.get("total"),
            "optimizer_steps": item.get("optimizer_steps"),
            "candidate_checks": checks,
            "search_calls": None,
            "inference_forwards": forwards,
            "partial_round": False,
        })
    return _clean_points(points)


def _clean_points(points):
    cleaned = []
    for point in points:
        if not (_finite(point.get("elapsed_seconds")) and _finite(point.get("accuracy"))):
            continue
        cleaned.append(point)
    cleaned.sort(key=lambda point: point["elapsed_seconds"])
    deduped = []
    for point in cleaned:
        if deduped and point["elapsed_seconds"] == deduped[-1]["elapsed_seconds"]:
            deduped[-1] = point
        else:
            deduped.append(point)
    return deduped


def load_run(run_dir, allow_legacy=False):
    run_dir = Path(run_dir)
    cfg = _read_json(run_dir / "config.json")
    status = _read_json(run_dir / "status.json")
    efficiency = _read_json(run_dir / "efficiency.json")
    if cfg is None or status is None:
        return None
    if efficiency is None and not allow_legacy:
        return {"run_id": run_dir.name, "excluded_reason": "no_efficiency_json",
                "status": status.get("status"), "task": cfg.get("task"),
                "variant": cfg.get("variant"), "seed": cfg.get("seed")}
    environment = _read_json(run_dir / "environment.json", {})
    sources = _read_json(run_dir / "source_manifest.json", {})
    history = []
    history_path = run_dir / "history.jsonl"
    if history_path.is_file():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    points = _points_from_efficiency(efficiency) if efficiency else _points_from_history(history)
    total_seconds = None
    if efficiency:
        total_seconds = (efficiency.get("clock") or {}).get("elapsed_seconds")
    if not _finite(total_seconds) and points:
        total_seconds = points[-1]["elapsed_seconds"]
    counters = dict((efficiency or {}).get("counters") or {})
    if not counters:
        counters = {
            "optimizer_steps": status.get("optimizer_steps"),
            "candidate_checks": status.get("candidate_checks"),
            "search_calls": status.get("search_calls"),
            "inference_forwards": (status.get("proposal_forwards", 0)
                                   + status.get("training_forwards", 0)
                                   + status.get("validation_forwards", 0)),
        }
    peak = (efficiency or {}).get("peak") or {}
    gpu = peak.get("gpu") or {}
    hardware = (efficiency or {}).get("hardware") or {}
    best_accuracy = status.get("best_validation_accuracy")
    if not _finite(best_accuracy) and points:
        best_accuracy = max(point["accuracy"] for point in points)
    return {
        "run_id": run_dir.name,
        "task": cfg.get("task"),
        "variant": cfg.get("variant"),
        "seed": cfg.get("seed"),
        "method_version": cfg.get("method_version"),
        "status": status.get("status"),
        "stopped_reason": status.get("stopped_reason"),
        "best_accuracy": best_accuracy,
        "total_seconds": total_seconds,
        "points": points,
        "counters": counters,
        "phase_seconds": (efficiency or {}).get("phase_seconds") or {},
        "peak_process_tree_rss_mb": peak.get("process_tree_rss_peak_mb"),
        "process_tree_peak_working_set_mb": peak.get("process_tree_peak_working_set_mb"),
        "parent_peak_working_set_mb": peak.get("parent_peak_working_set_mb"),
        "peak_gpu_allocated_mb": gpu.get("peak_allocated_mb"),
        "peak_gpu_reserved_mb": gpu.get("peak_reserved_mb"),
        "hardware_fingerprint": hardware.get("fingerprint") or "unknown",
        "hardware": hardware,
        "comparison_id": _run_identity(cfg, environment, sources),
        "time_budget_seconds": ((efficiency or {}).get("clock") or {}).get("time_budget_seconds"),
        "targets": (efficiency or {}).get("targets") or [],
        "clock_origin": "run_start" if efficiency else "post_warmup_legacy",
        "excluded_reason": None,
    }


def load_runs(root, allow_legacy=False):
    root = Path(root)
    runs, excluded = [], []
    if not root.is_dir():
        return runs, excluded
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            run = load_run(run_dir, allow_legacy=allow_legacy)
        except Exception as error:
            excluded.append({"run_id": run_dir.name, "reason": f"load_error: {error!r}"})
            continue
        if run is None:
            continue
        if run.get("excluded_reason"):
            excluded.append({"run_id": run["run_id"], "reason": run["excluded_reason"],
                             "status": run.get("status"), "task": run.get("task"),
                             "variant": run.get("variant"), "seed": run.get("seed")})
        elif run["status"] not in ANALYZABLE_STATUSES:
            excluded.append({"run_id": run["run_id"], "reason": f"status:{run['status']}",
                             "task": run["task"], "variant": run["variant"],
                             "seed": run["seed"]})
        elif not run["points"]:
            excluded.append({"run_id": run["run_id"], "reason": "no_validation_points",
                             "task": run["task"], "variant": run["variant"],
                             "seed": run["seed"]})
        elif not _finite(run["best_accuracy"]):
            excluded.append({"run_id": run["run_id"], "reason": "no_best_accuracy",
                             "task": run["task"], "variant": run["variant"],
                             "seed": run["seed"]})
        else:
            runs.append(run)
    return runs, excluded


def group_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        key = (run["task"], run["variant"], run["comparison_id"],
               run["hardware_fingerprint"])
        groups[key].append(run)
    return groups


def dedupe_seeds(groups):
    """Same-seed retries are not independent and are not maximized over."""
    kept, excluded = {}, []
    for key, group in groups.items():
        counts = Counter(run["seed"] for run in group)
        unique = [run for run in group if counts[run["seed"]] == 1]
        for run in group:
            if counts[run["seed"]] > 1:
                excluded.append({"run_id": run["run_id"], "reason": "duplicate_seed",
                                 "task": run["task"], "variant": run["variant"],
                                 "seed": run["seed"]})
        if unique:
            kept[key] = sorted(unique, key=lambda run: run["seed"])
    return kept, excluded


def time_to_target(points, target, interpolate=True):
    if not points:
        return {"target": target, "reached": False, "first_hit_seconds": None,
                "interpolated_seconds": None, "censored": True,
                "observed_seconds": None, "lower_bound_seconds": None,
                "accuracy_at_hit": None, "candidate_checks_at_hit": None,
                "search_calls_at_hit": None, "inference_forwards_at_hit": None,
                "optimizer_steps_at_hit": None, "partial_round": None}
    previous = None
    for point in points:
        if point["accuracy"] >= target:
            interpolated = None
            if interpolate and previous is not None:
                span = point["accuracy"] - previous["accuracy"]
                if span > 0:
                    fraction = (target - previous["accuracy"]) / span
                    interpolated = (previous["elapsed_seconds"]
                                    + fraction * (point["elapsed_seconds"]
                                                  - previous["elapsed_seconds"]))
            return {"target": target, "reached": True,
                    "first_hit_seconds": point["elapsed_seconds"],
                    "interpolated_seconds": interpolated, "censored": False,
                    "observed_seconds": point["elapsed_seconds"],
                    "lower_bound_seconds": 0.0,
                    "accuracy_at_hit": point["accuracy"],
                    "candidate_checks_at_hit": point.get("candidate_checks"),
                    "search_calls_at_hit": point.get("search_calls"),
                    "inference_forwards_at_hit": point.get("inference_forwards"),
                    "optimizer_steps_at_hit": point.get("optimizer_steps"),
                    "partial_round": point.get("partial_round")}
        previous = point
    last = points[-1]
    return {"target": target, "reached": False, "first_hit_seconds": None,
            "interpolated_seconds": None, "censored": True,
            "observed_seconds": last["elapsed_seconds"],
            "lower_bound_seconds": last["elapsed_seconds"],
            "accuracy_at_hit": None, "candidate_checks_at_hit": None,
            "search_calls_at_hit": None, "inference_forwards_at_hit": None,
            "optimizer_steps_at_hit": None, "partial_round": None}


def accuracy_at_budget(points, budget):
    eligible = [point for point in points if point["elapsed_seconds"] <= budget]
    if not eligible:
        return {"budget_seconds": budget, "available": False,
                "last_accuracy": None, "best_accuracy": None,
                "time_of_best_seconds": None, "last_time_seconds": None,
                "candidate_checks_at_budget": None,
                "run_ended_before_budget": bool(points)
                and points[-1]["elapsed_seconds"] <= budget}
    best = max(eligible, key=lambda point: (point["accuracy"], -point["elapsed_seconds"]))
    last = eligible[-1]
    return {"budget_seconds": budget, "available": True,
            "last_accuracy": last["accuracy"], "best_accuracy": best["accuracy"],
            "time_of_best_seconds": best["elapsed_seconds"],
            "last_time_seconds": last["elapsed_seconds"],
            "candidate_checks_at_budget": last.get("candidate_checks"),
            "run_ended_before_budget": points[-1]["elapsed_seconds"] <= budget}


def step_value(points, elapsed):
    value = None
    for point in points:
        if point["elapsed_seconds"] <= elapsed:
            value = point["accuracy"]
        else:
            break
    return value


def resample_curve(points, grid):
    return [{"elapsed_seconds": elapsed, "accuracy": step_value(points, elapsed)}
            for elapsed in grid]


def run_cost(run):
    counters = run.get("counters") or {}
    total_seconds = run.get("total_seconds")
    checks = counters.get("candidate_checks")
    return {
        "task": run["task"], "variant": run["variant"],
        "hardware_fingerprint": run["hardware_fingerprint"], "seed": run["seed"],
        "run_id": run["run_id"], "status": run["status"],
        "best_accuracy": run["best_accuracy"], "total_seconds": total_seconds,
        "optimizer_steps": counters.get("optimizer_steps"),
        "candidate_checks": checks,
        "search_calls": counters.get("search_calls"),
        "inference_forwards": counters.get("inference_forwards"),
        "inference_images": counters.get("inference_images"),
        "peak_process_tree_rss_mb": run.get("peak_process_tree_rss_mb"),
        "process_tree_peak_working_set_mb": run.get("process_tree_peak_working_set_mb"),
        "peak_gpu_allocated_mb": run.get("peak_gpu_allocated_mb"),
        "candidate_checks_per_second": (checks / total_seconds
                                        if _finite(checks) and _finite(total_seconds)
                                        and total_seconds > 0 else None),
        "seconds_per_optimizer_step": (total_seconds / counters["optimizer_steps"]
                                       if _finite(total_seconds)
                                       and _finite(counters.get("optimizer_steps"))
                                       and counters["optimizer_steps"] else None),
        # Kept in memory for paired comparisons and the JSON summary; CSV
        # writers select explicit fields and ignore this key.
        "points": run["points"],
    }


def summarize_target(rows, target):
    reached = [row for row in rows if row["reached"]]
    censored = [row for row in rows if not row["reached"]]
    first = [row["first_hit_seconds"] for row in reached]
    interpolated = [row["interpolated_seconds"] for row in reached
                    if row["interpolated_seconds"] is not None]
    return {
        "target": target, "n_runs": len(rows), "n_reached": len(reached),
        "n_censored": len(censored), "censored_seeds": [row["seed"] for row in censored],
        "mean_first_hit_seconds": _mean(first), "median_first_hit_seconds": (
            statistics.median(first) if first else None),
        "min_first_hit_seconds": _min(first), "max_first_hit_seconds": _max(first),
        "mean_interpolated_seconds": _mean(interpolated),
        "mean_candidate_checks_at_hit": _mean([row["candidate_checks_at_hit"]
                                               for row in reached]),
        "mean_search_calls_at_hit": _mean([row["search_calls_at_hit"] for row in reached]),
        "mean_inference_forwards_at_hit": _mean([row["inference_forwards_at_hit"]
                                                 for row in reached]),
        "mean_observed_seconds_censored": _mean([row["observed_seconds"]
                                                 for row in censored]),
    }


def summarize_budget(rows, budget):
    available = [row for row in rows if row["available"]]
    return {
        "budget_seconds": budget, "n_runs": len(rows), "n_available": len(available),
        "n_unavailable": len(rows) - len(available),
        "mean_last_accuracy": _mean([row["last_accuracy"] for row in available]),
        "mean_best_accuracy": _mean([row["best_accuracy"] for row in available]),
        "min_last_accuracy": _min([row["last_accuracy"] for row in available]),
        "max_last_accuracy": _max([row["last_accuracy"] for row in available]),
        "mean_time_of_best_seconds": _mean([row["time_of_best_seconds"]
                                            for row in available]),
        "mean_candidate_checks_at_budget": _mean([row["candidate_checks_at_budget"]
                                                  for row in available]),
        "n_ended_before_budget": sum(row["run_ended_before_budget"] for row in available),
    }


def summarize_group(task, variant, runs, targets, budgets, grid_points):
    runs = sorted(runs, key=lambda run: run["seed"])
    max_total = max((run["total_seconds"] for run in runs
                     if _finite(run["total_seconds"])), default=0)
    grid = ([max_total * index / grid_points for index in range(grid_points + 1)]
            if max_total > 0 else [])
    curve = []
    if grid:
        per_run = [resample_curve(run["points"], grid) for run in runs]
        for index, elapsed in enumerate(grid):
            values = [curves[index]["accuracy"] for curves in per_run
                      if curves[index]["accuracy"] is not None]
            curve.append({"elapsed_seconds": elapsed, "n": len(values),
                          "mean_accuracy": _mean(values), "sd_accuracy": _std(values)})

    target_rows = []
    for run in runs:
        for target in targets:
            row = time_to_target(run["points"], target)
            row.update({"task": task, "variant": variant, "seed": run["seed"],
                        "run_id": run["run_id"]})
            target_rows.append(row)
    budget_rows = []
    for run in runs:
        for budget in budgets:
            row = accuracy_at_budget(run["points"], budget)
            row.update({"task": task, "variant": variant, "seed": run["seed"],
                        "run_id": run["run_id"]})
            budget_rows.append(row)

    costs = [run_cost(run) for run in runs]
    phase_keys = sorted({key for run in runs for key in (run.get("phase_seconds") or {})})
    phase_mean = {key: _mean([(run.get("phase_seconds") or {}).get(key) for run in runs])
                  for key in phase_keys}
    return {
        "task": task, "variant": variant,
        "comparison_id": runs[0]["comparison_id"],
        "hardware_fingerprint": runs[0]["hardware_fingerprint"],
        "n_runs": len(runs), "seeds": [run["seed"] for run in runs],
        "total": {
            "mean_seconds": _mean([cost["total_seconds"] for cost in costs]),
            "std_seconds": _std([cost["total_seconds"] for cost in costs]),
            "min_seconds": _min([cost["total_seconds"] for cost in costs]),
            "max_seconds": _max([cost["total_seconds"] for cost in costs]),
            "mean_optimizer_steps": _mean([cost["optimizer_steps"] for cost in costs]),
            "mean_candidate_checks": _mean([cost["candidate_checks"] for cost in costs]),
            "mean_search_calls": _mean([cost["search_calls"] for cost in costs]),
            "mean_inference_forwards": _mean([cost["inference_forwards"] for cost in costs]),
            "mean_inference_images": _mean([cost["inference_images"] for cost in costs]),
            "mean_peak_process_tree_rss_mb": _mean([cost["peak_process_tree_rss_mb"]
                                                    for cost in costs]),
            "max_peak_process_tree_rss_mb": _max([cost["peak_process_tree_rss_mb"]
                                                  for cost in costs]),
            "mean_process_tree_peak_working_set_mb": _mean(
                [cost["process_tree_peak_working_set_mb"] for cost in costs]),
            "max_process_tree_peak_working_set_mb": _max(
                [cost["process_tree_peak_working_set_mb"] for cost in costs]),
            "mean_peak_gpu_allocated_mb": _mean([cost["peak_gpu_allocated_mb"]
                                                 for cost in costs]),
            "mean_best_accuracy": _mean([cost["best_accuracy"] for cost in costs]),
            "mean_candidate_checks_per_second": _mean([cost["candidate_checks_per_second"]
                                                       for cost in costs]),
            "mean_seconds_per_optimizer_step": _mean([cost["seconds_per_optimizer_step"]
                                                      for cost in costs]),
            "mean_phase_seconds": phase_mean,
        },
        "time_to_target": [summarize_target(
            [row for row in target_rows if row["target"] == target], target)
            for target in targets],
        "budget_accuracy": [summarize_budget(
            [row for row in budget_rows if row["budget_seconds"] == budget], budget)
            for budget in budgets],
        "curve": curve,
        "curve_raw": [{"seed": run["seed"], "elapsed_seconds": point["elapsed_seconds"],
                       "accuracy": point["accuracy"],
                       "candidate_checks": point.get("candidate_checks")}
                      for run in runs for point in run["points"]],
        "runs": costs,
        "target_rows": target_rows,
        "budget_rows": budget_rows,
    }


def _fmt(value, digits=2):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(summary):
    lines = ["# FRT-ABL 计算效率实验汇总", ""]
    lines.append(f"- 生成时间：{summary['generated_at']}")
    lines.append(f"- 运行目录：`{summary['runs_root']}`")
    lines.append(f"- 可用 run：{summary['n_usable_runs']} / 扫描 {summary['n_run_dirs']}")
    lines.append("- 指标：validation only；未读取正式 test。")
    lines.append("- 达标用时按每轮 validation 首次达到目标计，属上界；"
                 "插值为两点线性估计，仅描述性。")
    lines.append("")
    if not summary["groups"]:
        lines.append("没有可用 run。请先执行 `run_efficiency.py --execute --confirm-efficiency-runs`。")
        return "\n".join(lines) + "\n"

    lines += ["## 运行与硬件", "",
              "| task | variant | n | seeds | hardware |",
              "|---|---|---:|---|---|"]
    for group in summary["groups"]:
        lines.append(f"| {group['task']} | {group['variant']} | {group['n_runs']} | "
                     f"{','.join(map(str, group['seeds']))} | "
                     f"{group['hardware_fingerprint']} |")
    lines.append("")

    lines += ["## 总成本（均值）", "",
              "| task | variant | 秒 | 优化步 | 推理检查 | 搜索调用 | 前向次数 | 峰值 RSS MiB | 峰值 GPU MiB |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        total = group["total"]
        lines.append(f"| {group['task']} | {group['variant']} | {_fmt(total['mean_seconds'])} | "
                     f"{_fmt(total['mean_optimizer_steps'], 0)} | "
                     f"{_fmt(total['mean_candidate_checks'], 0)} | "
                     f"{_fmt(total['mean_search_calls'], 0)} | "
                     f"{_fmt(total['mean_inference_forwards'], 0)} | "
                     f"{_fmt(total['mean_peak_process_tree_rss_mb'])} | "
                     f"{_fmt(total['mean_peak_gpu_allocated_mb'])} |")
    lines.append("")

    lines += ["## 达到指定准确率所需时间", "",
              "| task | variant | target | 达到/总数 | 首次达标秒（均值） | 插值秒（均值） | 未达标 seed |",
              "|---|---|---:|---:|---:|---:|---|"]
    for group in summary["groups"]:
        for item in group["time_to_target"]:
            lines.append(f"| {group['task']} | {group['variant']} | "
                         f"{_fmt(item['target'], 3)} | {item['n_reached']}/{item['n_runs']} | "
                         f"{_fmt(item['mean_first_hit_seconds'])} | "
                         f"{_fmt(item['mean_interpolated_seconds'])} | "
                         f"{','.join(map(str, item['censored_seeds'])) or '-'} |")
    lines.append("")

    lines += ["## 固定时间预算下的准确率", "",
              "| task | variant | 预算 | 末次 acc | 最佳 acc | 可用 run | 预算前已结束 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        for item in group["budget_accuracy"]:
            lines.append(f"| {group['task']} | {group['variant']} | "
                         f"{_fmt(item['budget_seconds'] / 60)} min | "
                         f"{_fmt(item['mean_last_accuracy'], 4)} | "
                         f"{_fmt(item['mean_best_accuracy'], 4)} | "
                         f"{item['n_available']}/{item['n_runs']} | "
                         f"{item['n_ended_before_budget']} |")
    lines.append("")

    if summary["paired"]:
        lines += ["## 相对 full 的配对差", "",
                  "| task | variant | 总耗时比 | 总耗时差(s) |",
                  "|---|---|---:|---:|"]
        for item in summary["paired"]:
            lines.append(f"| {item['task']} | {item['variant']} | "
                         f"{_fmt(item['mean_total_seconds_ratio'])} | "
                         f"{_fmt(item['mean_total_seconds_delta'])} |")
        lines.append("")

    lines += ["## 读数注意事项", "",
              "1. 单种子或少数种子只能作为描述性成本，不可据此声称显著效率优势；",
              "2. 未达标目标保留为删失，并报告观察到的下界，不补零、不删除；",
              "3. 所有比较要求相同数据清单、代码、配置、线程数、设备与非并发负载；",
              "4. 进程树 RSS 为父进程与搜索 worker 的 RSS 之和，共享页按进程重复计入；",
              "5. 正式 test 仍未读取，不能把 validation 结论写成 test 结论。"]
    return "\n".join(lines) + "\n"


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def build_summary(runs_root, output_root, targets_override, budgets,
                  grid_points=40, allow_legacy=False, markdown=True):
    runs_root = Path(runs_root)
    runs, excluded = load_runs(runs_root, allow_legacy=allow_legacy)
    groups, duplicates = dedupe_seeds(group_runs(runs))
    excluded.extend(duplicates)
    protocol = _read_json(runs_root / "efficiency_protocol.json")

    def targets_for(task):
        if targets_override:
            return sorted(set(targets_override))
        if protocol and task in (protocol.get("targets") or {}):
            return list(protocol["targets"][task])
        return list(DEFAULT_TARGETS.get(task, ()))

    group_reports = []
    for (task, variant, comparison, hardware), members in sorted(groups.items()):
        group_reports.append(summarize_group(task, variant, members,
                                             targets_for(task), budgets, grid_points))
    # Paired comparisons use each task's declared targets so that all
    # ablations are covered even when a group has a different target list.
    paired = _paired_with_task_targets(group_reports, targets_for, budgets)

    n_dirs = (len([path for path in runs_root.iterdir()
                   if path.is_dir() and (path / "config.json").is_file()])
              if runs_root.is_dir() else 0)
    summary = {
        "status": "EFFICIENCY_AGGREGATED_VALIDATION_ONLY",
        "generated_at": now_iso(), "runs_root": str(runs_root),
        "n_run_dirs": n_dirs,
        "n_usable_runs": sum(group["n_runs"] for group in group_reports),
        "excluded": excluded,
        "targets": {task: targets_for(task) for task in
                    sorted({group["task"] for group in group_reports})},
        "budgets_seconds": list(budgets),
        "protocol": protocol,
        "groups": group_reports,
        "paired": paired,
        "notes": [
            "Validation only; formal test was not read.",
            "first_hit_seconds is an upper bound at validation granularity.",
            "Censored targets are reported with observed lower bounds, never imputed.",
            "Pairing is by seed within identical task/config/code/data/hardware groups.",
        ],
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "summary.json", summary)

    cost_fields = ["task", "variant", "hardware_fingerprint", "seed", "run_id", "status",
                   "best_accuracy", "total_seconds", "optimizer_steps", "candidate_checks",
                   "search_calls", "inference_forwards", "inference_images",
                   "peak_process_tree_rss_mb", "process_tree_peak_working_set_mb",
                   "peak_gpu_allocated_mb",
                   "candidate_checks_per_second", "seconds_per_optimizer_step"]
    cost_rows = [run for group in group_reports for run in group["runs"]]
    write_csv(output_root / "cost_runs.csv", cost_rows, cost_fields)

    group_cost_rows = []
    for group in group_reports:
        total = group["total"]
        group_cost_rows.append({
            "task": group["task"], "variant": group["variant"],
            "hardware_fingerprint": group["hardware_fingerprint"],
            "n_runs": group["n_runs"], "seeds": ",".join(map(str, group["seeds"])),
            **{key: total[key] for key in total if key != "mean_phase_seconds"},
        })
    write_csv(output_root / "cost_groups.csv", group_cost_rows,
              ["task", "variant", "hardware_fingerprint", "n_runs", "seeds",
               "mean_seconds", "std_seconds", "min_seconds", "max_seconds",
               "mean_optimizer_steps", "mean_candidate_checks", "mean_search_calls",
               "mean_inference_forwards", "mean_inference_images",
               "mean_peak_process_tree_rss_mb", "max_peak_process_tree_rss_mb",
               "mean_process_tree_peak_working_set_mb",
               "max_process_tree_peak_working_set_mb",
               "mean_peak_gpu_allocated_mb", "mean_best_accuracy",
               "mean_candidate_checks_per_second", "mean_seconds_per_optimizer_step"])

    target_run_rows = [row for group in group_reports for row in group["target_rows"]]
    write_csv(output_root / "time_to_target.csv", target_run_rows,
              ["task", "variant", "seed", "run_id", "target", "reached",
               "first_hit_seconds", "interpolated_seconds", "censored",
               "observed_seconds", "lower_bound_seconds", "accuracy_at_hit",
               "candidate_checks_at_hit", "search_calls_at_hit",
               "inference_forwards_at_hit", "optimizer_steps_at_hit", "partial_round"])
    target_group_rows = []
    for group in group_reports:
        for item in group["time_to_target"]:
            target_group_rows.append({"task": group["task"], "variant": group["variant"],
                                      **item})
    write_csv(output_root / "time_to_target_summary.csv", target_group_rows,
              ["task", "variant", "target", "n_runs", "n_reached", "n_censored",
               "censored_seeds", "mean_first_hit_seconds", "median_first_hit_seconds",
               "min_first_hit_seconds", "max_first_hit_seconds",
               "mean_interpolated_seconds", "mean_candidate_checks_at_hit",
               "mean_search_calls_at_hit", "mean_inference_forwards_at_hit",
               "mean_observed_seconds_censored"])

    budget_run_rows = [row for group in group_reports for row in group["budget_rows"]]
    write_csv(output_root / "budget_accuracy.csv", budget_run_rows,
              ["task", "variant", "seed", "run_id", "budget_seconds", "available",
               "last_accuracy", "best_accuracy", "time_of_best_seconds",
               "last_time_seconds", "candidate_checks_at_budget",
               "run_ended_before_budget"])
    budget_group_rows = []
    for group in group_reports:
        for item in group["budget_accuracy"]:
            budget_group_rows.append({"task": group["task"], "variant": group["variant"],
                                      **item})
    write_csv(output_root / "budget_summary.csv", budget_group_rows,
              ["task", "variant", "budget_seconds", "n_runs", "n_available",
               "n_unavailable", "mean_last_accuracy", "mean_best_accuracy",
               "min_last_accuracy", "max_last_accuracy",
               "mean_time_of_best_seconds", "mean_candidate_checks_at_budget",
               "n_ended_before_budget"])

    curve_rows, curve_mean_rows = [], []
    for group in group_reports:
        for point in group["curve"]:
            curve_mean_rows.append({"task": group["task"], "variant": group["variant"],
                                    **point})
        for point in group["curve_raw"]:
            curve_rows.append({"task": group["task"], "variant": group["variant"],
                               **point})
    write_csv(output_root / "curves_raw.csv", curve_rows,
              ["task", "variant", "seed", "elapsed_seconds", "accuracy",
               "candidate_checks"])
    write_csv(output_root / "curves_mean.csv", curve_mean_rows,
              ["task", "variant", "elapsed_seconds", "n", "mean_accuracy", "sd_accuracy"])

    if markdown:
        (output_root / "EFFICIENCY_REPORT.md").write_text(markdown_report(summary),
                                                          encoding="utf-8")
    return summary


def _paired_with_task_targets(group_reports, targets_for, budgets):
    by_key = {(group["task"], group["variant"], group["comparison_id"],
               group["hardware_fingerprint"]): group for group in group_reports}
    output = []
    for group in group_reports:
        if group["variant"] == "full":
            continue
        full = by_key.get((group["task"], "full", group["comparison_id"],
                           group["hardware_fingerprint"]))
        if full is None:
            continue
        full_runs = {run["seed"]: run for run in full["runs"]}
        pairs = [(run, full_runs[run["seed"]]) for run in group["runs"]
                 if run["seed"] in full_runs]
        if not pairs:
            continue
        targets = targets_for(group["task"])
        target_results = []
        for target in targets:
            deltas, ratios, both = [], [], 0
            for run, baseline in pairs:
                left = time_to_target(run["points"], target)
                right = time_to_target(baseline["points"], target)
                if left["reached"] and right["reached"]:
                    both += 1
                    deltas.append(left["first_hit_seconds"] - right["first_hit_seconds"])
                    if right["first_hit_seconds"] > 0:
                        ratios.append(left["first_hit_seconds"] / right["first_hit_seconds"])
            target_results.append({"target": target, "n_pairs": len(pairs),
                                   "n_both_reached": both,
                                   "mean_delta_seconds": _mean(deltas),
                                   "mean_ratio": _mean(ratios)})
        budget_results = []
        for budget in budgets:
            delta_last, delta_best = [], []
            for run, baseline in pairs:
                left = accuracy_at_budget(run["points"], budget)
                right = accuracy_at_budget(baseline["points"], budget)
                if left["available"] and right["available"]:
                    delta_last.append(left["last_accuracy"] - right["last_accuracy"])
                    delta_best.append(left["best_accuracy"] - right["best_accuracy"])
            budget_results.append({"budget_seconds": budget, "n_pairs": len(pairs),
                                   "mean_delta_last_accuracy": _mean(delta_last),
                                   "mean_delta_best_accuracy": _mean(delta_best)})
        total_deltas = [run["total_seconds"] - baseline["total_seconds"]
                        for run, baseline in pairs
                        if _finite(run["total_seconds"])
                        and _finite(baseline["total_seconds"])]
        total_ratios = [run["total_seconds"] / baseline["total_seconds"]
                        for run, baseline in pairs
                        if _finite(run["total_seconds"])
                        and _finite(baseline["total_seconds"])
                        and baseline["total_seconds"] > 0]
        output.append({"task": group["task"], "variant": group["variant"],
                       "hardware_fingerprint": group["hardware_fingerprint"],
                       "paired_seeds": [run["seed"] for run, _ in pairs],
                       "mean_total_seconds_delta": _mean(total_deltas),
                       "mean_total_seconds_ratio": _mean(total_ratios),
                       "targets": target_results, "budgets": budget_results})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(PROJECT / "methods" / "frt_abl" /
                                               "runs-efficiency-v10"))
    parser.add_argument("--output", default=str(PROJECT / "methods" / "frt_abl" /
                                                 "runs-summary-efficiency-v10"))
    parser.add_argument("--targets", nargs="+", type=float, default=None,
                        help="Override accuracy targets; default from protocol or task defaults")
    parser.add_argument("--budgets", nargs="+", type=parse_seconds, default=None,
                        help="Budget points in seconds; default from protocol or pre-declared")
    parser.add_argument("--grid-points", type=int, default=40)
    parser.add_argument("--allow-legacy", action="store_true",
                        help="Also load old runs without efficiency.json (post-warmup clock)")
    parser.add_argument("--no-markdown", action="store_true")
    args = parser.parse_args()
    if args.grid_points < 2:
        raise SystemExit("--grid-points must be at least 2")
    if args.targets is not None and any(not 0 < target <= 1 for target in args.targets):
        raise SystemExit("--targets must lie in (0, 1]")
    protocol = _read_json(Path(args.runs) / "efficiency_protocol.json")
    budgets = args.budgets
    if budgets is None:
        budgets = ((protocol or {}).get("budgets_seconds")
                   or list(DEFAULT_BUDGETS_SECONDS))
    budgets = sorted(float(budget) for budget in budgets)
    summary = build_summary(args.runs, args.output, args.targets, budgets,
                            grid_points=args.grid_points,
                            allow_legacy=args.allow_legacy,
                            markdown=not args.no_markdown)
    print(json.dumps({"status": summary["status"], "output": args.output,
                      "n_usable_runs": summary["n_usable_runs"],
                      "n_groups": len(summary["groups"]),
                      "n_excluded": len(summary["excluded"])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
