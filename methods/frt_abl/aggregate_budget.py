"""Aggregate annotation-budget runs into curves and low-label stability tables.

Validation metrics only. The formal test split is never read. The aggregator
refuses to mix annotation budgets, code revisions or prepared manifests because
each group keeps the same comparison id used by `aggregate.py`.
"""
import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from .aggregate import comparison_id, usable
from .budget import EXPERIMENT, project_relative
from .policies import resolve_policy
from .prepare import PROJECT, write_json

# Two-sided 95% Student-t critical values; df > 30 falls back to the normal value.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
       27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t_critical_95(df):
    return T95.get(int(df), 1.96)


def mean_ci(values):
    values = [float(value) for value in values]
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    std = statistics.stdev(values)
    ci = t_critical_95(len(values) - 1) * std / math.sqrt(len(values))
    return mean, std, ci


def _ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        for position in range(start, end + 1):
            ranks[order[position]] = rank
        start = end + 1
    return ranks


def _pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def slope_per_log10(fractions, values):
    """Accuracy change per tenfold increase in annotation budget."""
    pairs = [(math.log10(float(f)), float(v)) for f, v in zip(fractions, values)
             if f and f > 0 and v is not None]
    if len(pairs) < 2 or len({x for x, _ in pairs}) < 2:
        return None
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    denominator = sum((x - mx) ** 2 for x, _ in pairs)
    if not denominator:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / denominator


def _mean_or_none(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def history_summary(run_dir):
    path = Path(run_dir) / "history.jsonl"
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    accuracies = []
    alphas, coverage, unsupported = [], [], []
    accepted, calibration_total, round_steps = [], [], []
    for row in rows:
        validation = row.get("validation_task") or {}
        if "accuracy" in validation:
            accuracies.append(float(validation["accuracy"]))
        alphas.append(row.get("alpha"))
        for target, key in ((coverage, "coverage"), (unsupported, "unsupported_weak"),
                            (accepted, "accepted_candidates"),
                            (calibration_total, "calibration_fit_total"),
                            (round_steps, "round_steps")):
            value = row.get(key)
            if isinstance(value, (int, float)):
                target.append(float(value))
    if not accuracies:
        return None
    return {
        "rounds": len(rows),
        "final_accuracy": accuracies[-1],
        "best_round_accuracy": max(accuracies),
        "round_accuracy_std": (statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0),
        "last5_accuracy_std": (statistics.stdev(accuracies[-5:]) if len(accuracies[-5:]) > 1 else 0.0),
        "gate_open_rate": sum(alpha is not None for alpha in alphas) / len(alphas) if alphas else None,
        "mean_coverage": _mean_or_none(coverage),
        "mean_unsupported_weak": _mean_or_none(unsupported),
        "mean_accepted_candidates": _mean_or_none(accepted),
        "mean_calibration_fit_total": _mean_or_none(calibration_total),
        "mean_round_steps": _mean_or_none(round_steps),
    }


def collect(runs_root):
    groups = defaultdict(list)
    seen = skipped = 0
    warnings = []
    root = Path(runs_root)
    if not root.is_dir():
        return groups, 0, 0, [f"Runs root does not exist: {root}"]
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
            sources = json.loads((run_dir / "source_manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        seen += 1
        if cfg.get("experiment") != EXPERIMENT:
            skipped += 1
            continue
        budget = cfg.get("budget") or {}
        missing = [key for key in ("level", "level_label", "actual_fraction",
                                   "actual_total_unique_labeled_images")
                   if key not in budget]
        if missing:
            skipped += 1
            warnings.append(f"{run_dir.name}: budget metadata missing {missing}")
            continue
        try:
            gate_enabled = resolve_policy(cfg).gate
        except (ValueError, KeyError):
            gate_enabled = None
        identity = comparison_id(cfg, environment, sources)
        key = (cfg["task"], cfg["variant"], float(budget["level"]), identity)
        groups[key].append({
            "run_id": run_dir.name,
            "seed": cfg["seed"],
            "status": status.get("status"),
            "best_validation_accuracy": status.get("best_validation_accuracy"),
            "stopped_reason": status.get("stopped_reason"),
            "optimizer_steps": status.get("optimizer_steps"),
            "level": float(budget["level"]),
            "level_label": budget["level_label"],
            "actual_fraction": float(budget["actual_fraction"]),
            "actual_total": budget["actual_total_unique_labeled_images"],
            "train_image_universe": budget.get("train_image_universe"),
            "prepared_root": cfg.get("prepared_root"),
            "gate_enabled": gate_enabled,
            "history": history_summary(run_dir),
        })
    return groups, seen, skipped, warnings


def summarize(groups, expected_seeds):
    rows = []
    for (task, variant, level, identity), records in sorted(groups.items()):
        admitted = usable(records)
        accuracies = [float(record["best_validation_accuracy"]) for record in admitted]
        seeds = sorted(record["seed"] for record in admitted)
        counts = Counter(record["seed"] for record in records)
        mean, std, ci = mean_ci(accuracies)
        missing = sorted(set(expected_seeds) - set(seeds))
        rows.append({
            "task": task, "variant": variant, "level": level,
            "level_label": records[0]["level_label"],
            "actual_fraction": records[0]["actual_fraction"],
            "actual_total": records[0]["actual_total"],
            "train_image_universe": records[0]["train_image_universe"],
            "prepared_root": records[0]["prepared_root"],
            "comparison_id": identity,
            "n_run_dirs": len(records), "n": len(accuracies),
            "seeds": seeds, "missing_seeds": missing,
            "status": ("COMPLETE" if not missing and len(accuracies) >= 2 else "PARTIAL"),
            "mean_validation_accuracy": mean,
            "std_validation_accuracy": std,
            "ci95_validation_accuracy": ci,
            "min_validation_accuracy": min(accuracies) if accuracies else None,
            "max_validation_accuracy": max(accuracies) if accuracies else None,
            "range_validation_accuracy": (max(accuracies) - min(accuracies)) if accuracies else None,
            "cv_validation_accuracy": (std / mean) if std is not None and mean else None,
            "gate_enabled": records[0]["gate_enabled"],
            "duplicate_seeds_excluded": sorted(seed for seed, count in counts.items() if count > 1),
            "run_ids": [record["run_id"] for record in sorted(admitted, key=lambda x: x["seed"])],
            "excluded_run_ids": [record["run_id"] for record in records if record not in admitted],
        })
    return rows


def curve_rows(rows, reference_level=None):
    by_level = defaultdict(list)
    for row in rows:
        by_level[(row["task"], row["variant"], row["level"])].append(row)
    selected, ambiguous = {}, set()
    for (task, variant, level), candidates in by_level.items():
        # One prepared manifest per task/level is the normal design. If a
        # directory contains two campaigns for the same level, use the most
        # complete unique-seed group and flag the ambiguity instead of mixing.
        chosen = max(candidates, key=lambda row: (
            row["status"] == "COMPLETE", row["n"],
            row["mean_validation_accuracy"] is not None, row["comparison_id"]))
        selected[(task, variant, level)] = chosen
        if len(candidates) > 1:
            ambiguous.add((task, variant))
    grouped = defaultdict(list)
    for (task, variant, _level), row in selected.items():
        grouped[(task, variant)].append(row)
    curves, enriched = [], []
    for (task, variant), points in sorted(grouped.items()):
        points = sorted(points, key=lambda row: row["level"])
        usable_points = [row for row in points
                         if row["mean_validation_accuracy"] is not None]
        if not usable_points:
            continue
        if reference_level is None:
            reference = max(usable_points, key=lambda row: row["level"])
        else:
            reference = min(usable_points, key=lambda row: abs(row["level"] - reference_level))
        reference_mean = reference["mean_validation_accuracy"]
        fractions = [row["actual_fraction"] for row in usable_points]
        means = [row["mean_validation_accuracy"] for row in usable_points]
        curve = {
            "task": task, "variant": variant,
            "comparison_ids": {str(row["level"]): row["comparison_id"]
                               for row in usable_points},
            "ambiguous_level_identities": (task, variant) in ambiguous,
            "reference_level": reference["level"],
            "reference_level_label": reference["level_label"],
            "points": [{"level": row["level"], "level_label": row["level_label"],
                        "actual_fraction": row["actual_fraction"],
                        "actual_total": row["actual_total"], "n": row["n"],
                        "mean": row["mean_validation_accuracy"],
                        "std": row["std_validation_accuracy"],
                        "ci95": row["ci95_validation_accuracy"],
                        "cv": row["cv_validation_accuracy"]}
                       for row in usable_points],
            "spearman_rho": spearman(fractions, means),
            "slope_per_log10_budget": slope_per_log10(fractions, means),
            "lowest_level": min(row["level"] for row in usable_points),
            "highest_level": max(row["level"] for row in usable_points),
            "lowest_over_reference_ratio": (
                usable_points[0]["mean_validation_accuracy"] / reference_mean
                if reference_mean else None),
        }
        curves.append(curve)
        for row in points:
            item = dict(row)
            if row["mean_validation_accuracy"] is not None and reference_mean is not None:
                item["reference_level"] = reference["level"]
                item["delta_vs_reference"] = (row["mean_validation_accuracy"] - reference_mean)
                item["ratio_vs_reference"] = (row["mean_validation_accuracy"] / reference_mean
                                              if reference_mean else None)
            enriched.append(item)
    return curves, enriched


def paired_deltas(groups, reference_level=None, expected_seeds=()):
    levels_by_group = defaultdict(list)
    for (task, variant, level, identity), records in groups.items():
        levels_by_group[(task, variant, level)].append((identity, records))
    reference_by_task = {}
    for task, variant, _level in levels_by_group:
        levels = sorted(level for (other_task, other_variant, level) in levels_by_group
                        if (other_task, other_variant) == (task, variant))
        if reference_level is None:
            reference_by_task[(task, variant)] = max(levels)
        else:
            reference_by_task[(task, variant)] = min(levels,
                                                     key=lambda value: abs(value - reference_level))
    results = []
    for (task, variant, level), candidates in sorted(levels_by_group.items()):
        reference_level_value = reference_by_task.get((task, variant))
        if reference_level_value is None or math.isclose(level, reference_level_value):
            continue
        reference_candidates = levels_by_group.get((task, variant, reference_level_value), [])
        if not reference_candidates:
            continue
        identity, records = max(candidates, key=lambda item: len(usable(item[1])))
        reference_identity, reference_records = max(
            reference_candidates, key=lambda item: len(usable(item[1])))
        other = {record["seed"]: record for record in usable(records)}
        reference_map = {record["seed"]: record for record in usable(reference_records)}
        paired = [seed for seed in sorted(set(other) & set(reference_map))]
        deltas, pairs = [], []
        for seed in paired:
            delta = (other[seed]["best_validation_accuracy"]
                     - reference_map[seed]["best_validation_accuracy"])
            deltas.append(delta)
            pairs.append({"seed": seed,
                          "level_accuracy": other[seed]["best_validation_accuracy"],
                          "reference_accuracy": reference_map[seed]["best_validation_accuracy"],
                          "delta": delta})
        if not deltas:
            continue
        mean, std, ci = mean_ci(deltas)
        results.append({
            "task": task, "variant": variant, "level": level,
            "level_label": other[paired[0]]["level_label"],
            "reference_level": reference_level_value,
            "comparison_id": identity,
            "reference_comparison_id": reference_identity,
            "ambiguous_level_identities": len(candidates) > 1,
            "ambiguous_reference_identities": len(reference_candidates) > 1,
            "n": len(deltas), "paired_seeds": paired,
            "missing_level_seeds": sorted(set(expected_seeds) - set(other)),
            "missing_reference_seeds": sorted(set(expected_seeds) - set(reference_map)),
            "mean_delta": mean, "std_delta": std, "ci95_delta": ci,
            "wins": sum(delta > 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "pairs": pairs,
        })
    return results


def stability_rows(groups, rows):
    lowest = defaultdict(lambda: float("inf"))
    for row in rows:
        key = (row["task"], row["variant"])
        lowest[key] = min(lowest[key], row["level"])
    results = []
    for row in rows:
        key = (row["task"], row["variant"], row["level"], row["comparison_id"])
        admitted = usable(groups.get(key, []))
        histories = [record["history"] for record in admitted if record["history"]]
        gate_open = [history["gate_open_rate"] for history in histories
                     if row["gate_enabled"] and history["gate_open_rate"] is not None]
        results.append({
            "task": row["task"], "variant": row["variant"], "level": row["level"],
            "level_label": row["level_label"],
            "is_lowest": math.isclose(row["level"], lowest[(row["task"], row["variant"])]),
            "n": row["n"],
            "seed_std": row["std_validation_accuracy"],
            "seed_cv": row["cv_validation_accuracy"],
            "seed_range": row["range_validation_accuracy"],
            "seed_min": row["min_validation_accuracy"],
            "seed_max": row["max_validation_accuracy"],
            "round_std_mean": _mean_or_none([history["round_accuracy_std"]
                                             for history in histories]),
            "last5_round_std_mean": _mean_or_none([history["last5_accuracy_std"]
                                                   for history in histories]),
            "gate_open_rate": _mean_or_none(gate_open),
            "coverage_mean": _mean_or_none([history["mean_coverage"] for history in histories]),
            "unsupported_weak_mean": _mean_or_none([history["mean_unsupported_weak"]
                                                    for history in histories]),
            "accepted_candidates_mean": _mean_or_none([history["mean_accepted_candidates"]
                                                       for history in histories]),
            "calibration_fit_total_mean": _mean_or_none([history["mean_calibration_fit_total"]
                                                          for history in histories]),
        })
    return results


def load_v10_reference(summary_path=None, campaign_path=None):
    reference = {}
    if summary_path:
        data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        for group in data.get("groups", []):
            if group.get("variant") != "full":
                continue
            reference.setdefault(group["task"], {}).update({
                "mean_validation_accuracy": group.get("mean_validation_accuracy"),
                "std_validation_accuracy": group.get("std_validation_accuracy"),
                "n": group.get("n")})
    if campaign_path:
        data = json.loads(Path(campaign_path).read_text(encoding="utf-8"))
        for task, item in (data.get("v10_reference") or {}).items():
            reference.setdefault(task, {}).update({
                "labeled_images": item.get("labeled_images"),
                "fraction": item.get("fraction")})
    return reference


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value, digits=4):
    return "" if value is None else f"{value:.{digits}f}"


def _pct(value, digits=2):
    return "" if value is None else f"{value * 100:.{digits}f}%"


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return lines


def write_report(path, runs_root, rows, curves, paired, stability, reference,
                 expected_seeds, seen, skipped, warnings):
    lines = ["# 标注预算实验汇总（validation only）", "",
             "- 实验：`%s`" % EXPERIMENT,
             "- 扫描目录：`%s`" % project_relative(runs_root),
             "- 期望 seeds：%s" % ", ".join(str(seed) for seed in expected_seeds),
             "- 扫描 run 目录：%d；跳过的非预算 run：%d" % (seen, skipped),
             "- 正式 test：**未读取**；以下均为 best validation accuracy。", ""]
    if warnings:
        lines += ["## 数据警告", ""] + [f"- {warning}" for warning in warnings] + [""]
    if reference:
        lines += ["## V10 冻结基线参照", ""]
        for task, item in sorted(reference.items()):
            fraction = (f"budget={_pct(item.get('fraction'), 2)} "
                        f"({item.get('labeled_images')} images)"
                        if item.get("fraction") is not None else "budget fraction unknown")
            lines.append(f"- `{task}`: mean={_fmt(item.get('mean_validation_accuracy'))}, "
                         f"{fraction}, n={item.get('n')}")
        lines.append("")
    for curve in curves:
        task, variant = curve["task"], curve["variant"]
        lines += [f"## {task} / {variant}：性能—标注量曲线", ""]
        table = []
        for point in curve["points"]:
            table.append([_pct(point["level"], 0), _pct(point["actual_fraction"], 2),
                          point["actual_total"], point["n"], _fmt(point["mean"]),
                          ("±" + _fmt(point["ci95"])) if point["ci95"] is not None else "-",
                          _fmt(point["std"]), _fmt(point["cv"])])
        lines += _markdown_table(["名义预算", "实际预算", "实际图像", "n", "mean",
                                  "95% CI", "std", "CV"], table)
        lines += ["", f"- Spearman ρ（预算—均值）：{_fmt(curve['spearman_rho'], 3)}",
                  f"- 每 10× 预算的均值变化：{_fmt(curve['slope_per_log10_budget'], 4)}",
                  f"- 最低预算/参考预算：{_fmt(curve['lowest_over_reference_ratio'], 4)}", ""]
    if paired:
        lines += ["## 配对差值（同 seed，相对参考预算）", ""]
        table = [[item["task"], item["variant"], _pct(item["level"], 0),
                  _pct(item["reference_level"], 0), item["n"], _fmt(item["mean_delta"]),
                  ("±" + _fmt(item["ci95_delta"])) if item["ci95_delta"] is not None else "-",
                  f'{item["wins"]}/{item["losses"]}/{item["ties"]}'] for item in paired]
        lines += _markdown_table(["任务", "variant", "预算", "参考", "n", "Δmean",
                                  "95% CI", "胜/负/平"], table) + [""]
    if stability:
        lines += ["## 低标注稳定性", ""]
        table = [[item["task"], item["variant"], _pct(item["level"], 0),
                  "是" if item["is_lowest"] else "", item["n"], _fmt(item["seed_std"]),
                  _fmt(item["seed_range"]), _fmt(item["round_std_mean"]),
                  _fmt(item["last5_round_std_mean"]),
                  _fmt(item["gate_open_rate"], 3) if item["gate_open_rate"] is not None else "-",
                  _fmt(item["coverage_mean"], 3)] for item in stability]
        lines += _markdown_table(["任务", "variant", "预算", "最低档", "n", "seed std",
                                  "seed range", "轮内 std", "末 5 轮 std", "gate 开率",
                                  "coverage"], table) + [""]
    lines += ["## 口径与限制", "",
              "- 预算按 **互异标注图像数** 计算，覆盖 anchor + calibration_fit + calibration_gate；"
              "validation/test 不计入标注预算。",
              "- HWF 同一图像会被多条表达式复用，因此表达式数不是预算单位；表中同时给出实际图像数与实际比例。",
              "- 角色选择沿用 V10 的确定性顺序与图像隔离；3% MNIST 应复现 V10 的 270/270/270 表达式预算。",
              "- 低预算下 gate 需要 `calibration_min_count` 个校准候选；不足时 threshold 会关闭，"
              "这是方法的 fail-closed 行为，不应被解释为“没有标注”。",
              "- 训练轮数固定，但每轮优化步数由 weak 集合规模决定；低预算保留更多 weak 表达式，"
              "步数可能略高，报告中保留 `optimizer_steps` 供核对。",
              "- 以上为 validation 曲线；正式 test 必须在预算协议/配置冻结后单独批准。", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _svg_escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write_svg(path, task, curves, reference=None):
    width, height = 980, 600
    left, right, top, bottom = 100, 40, 70, 85
    plot_width, plot_height = width - left - right, height - top - bottom
    colors = ["#1f5fa8", "#c0392b", "#1e8449", "#8e44ad", "#d68910"]
    all_points = [point for curve in curves for point in curve["points"]]
    fractions = [point["actual_fraction"] for point in all_points if point["actual_fraction"] > 0]
    means = [point["mean"] for point in all_points]
    if reference and reference.get("mean_validation_accuracy") is not None:
        means.append(reference["mean_validation_accuracy"])
    if not fractions or not means:
        return
    log_x = len(set(fractions)) > 1 and min(fractions) > 0
    minimum, maximum = min(fractions), max(fractions)
    y_min = max(0.0, min(means) - 0.02)
    y_max = min(1.0, max(means) + 0.02)
    if y_max <= y_min:
        y_min, y_max = 0.0, 1.0

    def x_position(fraction):
        if log_x:
            low, high = math.log10(minimum), math.log10(maximum)
            return left + ((math.log10(fraction) - low) / (high - low)) * plot_width
        if maximum == minimum:
            return left + plot_width / 2
        return left + ((fraction - minimum) / (maximum - minimum)) * plot_width

    def y_position(value):
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Microsoft YaHei, sans-serif">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="{left}" y="34" font-size="22" font-weight="600" fill="#222">'
             f'{_svg_escape(task)} · 标注预算曲线</text>',
             f'<text x="{left}" y="56" font-size="13" fill="#555">'
             f'mean best validation accuracy ± 95% CI; n per point in table / report</text>']
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = y_position(value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                     f'stroke="#e5e5e5" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                     f'font-size="12" fill="#555">{value:.3f}</text>')
    lines.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
                 f'y2="{top + plot_height}" stroke="#333" stroke-width="1.2"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
                 f'stroke="#333" stroke-width="1.2"/>')
    for curve_index, curve in enumerate(curves):
        color = colors[curve_index % len(colors)]
        points = sorted(curve["points"], key=lambda point: point["actual_fraction"])
        coordinates = [(x_position(point["actual_fraction"]), y_position(point["mean"]),
                        point) for point in points]
        if len(coordinates) > 1:
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in coordinates)
            lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" '
                         f'stroke-width="2.4"/>')
        for x, y, point in coordinates:
            if point["ci95"] is not None:
                high = y_position(point["mean"] + point["ci95"])
                low = y_position(point["mean"] - point["ci95"])
                lines.append(f'<line x1="{x:.2f}" y1="{high:.2f}" x2="{x:.2f}" y2="{low:.2f}" '
                             f'stroke="{color}" stroke-width="1.4"/>')
                lines.append(f'<line x1="{x - 6:.2f}" y1="{high:.2f}" x2="{x + 6:.2f}" '
                             f'y2="{high:.2f}" stroke="{color}" stroke-width="1.4"/>')
                lines.append(f'<line x1="{x - 6:.2f}" y1="{low:.2f}" x2="{x + 6:.2f}" '
                             f'y2="{low:.2f}" stroke="{color}" stroke-width="1.4"/>')
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')
            lines.append(f'<text x="{x:.2f}" y="{y - 12:.2f}" text-anchor="middle" '
                         f'font-size="12" fill="{color}">{point["mean"]:.4f}</text>')
        for point in points:
            x = x_position(point["actual_fraction"])
            lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 22}" '
                         f'text-anchor="middle" font-size="12" fill="#333">'
                         f'{point["level"] * 100:g}%</text>')
        if len(curves) > 1:
            legend_y = top + 8 + curve_index * 22
            lines.append(f'<line x1="{left + plot_width - 170}" y1="{legend_y}" '
                         f'x2="{left + plot_width - 140}" y2="{legend_y}" '
                         f'stroke="{color}" stroke-width="2.4"/>')
            lines.append(f'<text x="{left + plot_width - 132}" y="{legend_y + 4}" '
                         f'font-size="12" fill="#333">{_svg_escape(curve["variant"])}</text>')
    if reference and reference.get("mean_validation_accuracy") is not None:
        value = reference["mean_validation_accuracy"]
        y = y_position(value)
        fraction = reference.get("fraction")
        if fraction and minimum <= float(fraction) <= maximum:
            x = x_position(float(fraction))
            size = 7
            lines.append(f'<polygon points="{x:.2f},{y - size:.2f} {x + size:.2f},{y:.2f} '
                         f'{x:.2f},{y + size:.2f} {x - size:.2f},{y:.2f}" '
                         f'fill="#ffffff" stroke="#555" stroke-width="2"/>')
            label = f'V10 full @ {float(fraction) * 100:.2f}% ({value:.4f})'
            lines.append(f'<text x="{x + 10:.2f}" y="{y - 8:.2f}" font-size="12" '
                         f'fill="#555">{_svg_escape(label)}</text>')
        else:
            lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                         f'stroke="#777" stroke-width="1.4" stroke-dasharray="6 5"/>')
            label = "V10 full"
            if fraction:
                label += f' @ {float(fraction) * 100:.2f}%'
            lines.append(f'<text x="{left + 8}" y="{y - 8:.2f}" font-size="12" fill="#555">'
                         f'{_svg_escape(label)} ({value:.4f})</text>')
    lines.append(f'<text x="{left + plot_width / 2}" y="{height - 30}" text-anchor="middle" '
                 f'font-size="14" fill="#333">标注预算（占训练图像比例，'
                 f'{"对数轴" if log_x else "线性轴"}）</text>')
    lines.append(f'<text transform="translate(28 {top + plot_height / 2}) rotate(-90)" '
                 f'text-anchor="middle" font-size="14" fill="#333">验证任务准确率</text>')
    lines.append('</svg>')
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(PROJECT / "methods" / "frt_abl" / "runs-budget-v1"))
    parser.add_argument("--output", default=str(PROJECT / "methods" / "frt_abl" /
                                                "runs-summary-budget-v1"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--reference-level", type=float, default=None,
                        help="Reference budget; defaults to the largest observed level")
    parser.add_argument("--v10-summary", default=None,
                        help="Optional runs-summary-v10-5seed/summary.json for a reference line")
    parser.add_argument("--campaign-manifest", default=None,
                        help="Optional prepared/budget-v1/manifest.json with V10 fractions")
    parser.add_argument("--no-svg", action="store_true")
    args = parser.parse_args()
    for label, value in (("--v10-summary", args.v10_summary),
                         ("--campaign-manifest", args.campaign_manifest)):
        if value and not Path(value).is_file():
            raise SystemExit(f"{label} does not exist: {value}")
    groups, seen, skipped, warnings = collect(args.runs)
    if not groups:
        raise SystemExit(f"No annotation-budget runs found under {args.runs}")
    rows = summarize(groups, tuple(args.seeds))
    curves, enriched = curve_rows(rows, reference_level=args.reference_level)
    paired = paired_deltas(groups, reference_level=args.reference_level,
                           expected_seeds=tuple(args.seeds))
    stability = stability_rows(groups, rows)
    reference = {}
    if args.v10_summary or args.campaign_manifest:
        reference = load_v10_reference(args.v10_summary, args.campaign_manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "AGGREGATED_VALIDATION_ONLY",
        "experiment": EXPERIMENT,
        "runs_root": str(args.runs),
        "n_run_dirs": seen, "n_non_budget_runs_skipped": skipped,
        "warnings": warnings,
        "expected_seeds": args.seeds,
        "groups": rows,
        "curves": curves,
        "paired_deltas": paired,
        "stability": stability,
        "v10_reference": reference,
        "note": ("Complete unique seeds only; annotation budgets/prepared manifests are never "
                 "mixed. Formal test not read."),
    }
    write_json(output / "summary.json", report)
    write_csv(output / "curves.csv",
              ["task", "variant", "level", "level_label", "actual_fraction", "actual_total",
               "n", "seeds", "missing_seeds", "status", "mean_validation_accuracy",
               "std_validation_accuracy", "ci95_validation_accuracy", "min_validation_accuracy",
               "max_validation_accuracy", "cv_validation_accuracy",
               "delta_vs_reference", "ratio_vs_reference", "comparison_id"], enriched)
    write_csv(output / "paired_deltas.csv",
              ["task", "variant", "level", "level_label", "reference_level", "n",
               "mean_delta", "std_delta", "ci95_delta", "wins", "losses", "ties",
               "paired_seeds", "missing_level_seeds", "missing_reference_seeds"], paired)
    write_csv(output / "stability.csv",
              ["task", "variant", "level", "level_label", "is_lowest", "n", "seed_std",
               "seed_cv", "seed_range", "seed_min", "seed_max", "round_std_mean",
               "last5_round_std_mean", "gate_open_rate", "coverage_mean",
               "unsupported_weak_mean", "accepted_candidates_mean",
               "calibration_fit_total_mean"], stability)
    write_report(output / "report.md", args.runs, rows, curves, paired, stability, reference,
                 tuple(args.seeds), seen, skipped, warnings)
    if not args.no_svg:
        for task in sorted({curve["task"] for curve in curves}):
            task_curves = [curve for curve in curves if curve["task"] == task]
            write_svg(output / f"curve-{task}.svg", task, task_curves, reference.get(task))
    print(json.dumps({"status": "AGGREGATED_VALIDATION_ONLY", "n_run_dirs": seen,
                      "n_groups": len(rows), "n_curves": len(curves),
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
