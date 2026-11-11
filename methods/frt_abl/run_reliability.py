"""Correction-reliability experiment for the frozen benchmark runs.

Default is a file-only preflight.  ``--execute-reliability`` performs the offline
analysis only: it reads the per-round decision logs written by ``train.py`` plus
the hidden weak labels (scoring only), then writes matched-coverage tables,
per-run JSON records and SVG risk--coverage figures.  It never constructs or
trains a model, never reads the formal test split, and never writes anything that
training or model selection may consume.

Example (from the project root):

    py -m methods.frt_abl.run_reliability
    py -m methods.frt_abl.run_reliability --execute-reliability
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .core import ArithmeticKB
from .prepare import PROJECT, digest, load_config, write_json
from .reliability import (DEFAULT_CURVE_VARIANTS, OPERATING_ONLY_VARIANTS,
                          coverage_grid, curve_from_round, line_chart_svg,
                          load_weak_role, operating_metrics, paired_summary,
                          read_round_decisions, risk_auc, summarize, write_csv)
from .policies import METHOD_VERSION, VARIANTS

COMPLETE = "TRAINING_COMPLETED_TEST_NOT_RUN"
SCHEMA_VERSION = 1

DEFINITIONS = {
    "weak_pool": "prepared weak role; N is the coverage denominator for every run of a task",
    "candidate": "abduced explanation z* logged per round; null = no supported explanation",
    "correction": "an accepted candidate (x, z*) used as hard supervision; the log has no raw argmax, so changed/unchanged cannot be split",
    "exact_correct": "z* equals the hidden weak symbol sequence (primary correctness)",
    "task_correct": "value(z*) equals the observed target (logical consistency only)",
    "coverage_operating": "positive-region corrections / N",
    "coverage_curve": "top-k supported candidates ranked by the score, k = floor(coverage_target * N)",
    "error_rate": "errors / supervised, i.e. the erroneous supervision / wrong-correction ratio",
    "erroneous_supervision_fraction_of_all": "errors / N, i.e. the share of the whole weak pool receiving a wrong correction",
    "matched_coverage": "same k (hence same coverage) for every variant and seed of a task",
    "delta": "variant - full; negative risk delta means the variant is safer at that coverage",
}


def decision_rounds(directory):
    """Round indices that have a decisions CSV in one run directory."""
    rounds = []
    for path in Path(directory).glob("decisions_round_*.csv"):
        stem = path.stem
        try:
            rounds.append(int(stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(rounds))


def parse_int_list(text):
    values = [piece.strip() for piece in str(text).split(",") if piece.strip()]
    if not values:
        raise ValueError("Expected a comma-separated list of round indices")
    return sorted({int(value) for value in values})


def parse_seed_list(values):
    """Accept ``0 1 2`` or ``0,1,2`` (PowerShell passes comma text as one token)."""
    seeds = sorted({int(piece) for value in values
                    for piece in str(value).split(",") if piece.strip()})
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def selected_rounds(available, specification):
    if not available:
        raise ValueError("Run has no decision logs")
    if specification == "all":
        return list(available)
    if specification == "last":
        return [max(available)]
    wanted = parse_int_list(specification)
    missing = [round_index for round_index in wanted if round_index not in available]
    if missing:
        raise ValueError(f"Requested rounds are unavailable: {missing}")
    return wanted


def discover(runs_root, tasks, variants, seeds, require_complete=True):
    """Read run configs/statuses; return (entries, skipped) without any data pass."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")
    entries, skipped = [], []
    for directory in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        try:
            cfg = load_config(directory / "config.json")
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            skipped.append({"run": directory.name, "reason": f"unreadable: {error}"})
            continue
        if cfg["task"] not in tasks or cfg["variant"] not in variants or cfg["seed"] not in seeds:
            continue
        if require_complete and status.get("status") != COMPLETE:
            skipped.append({"run": directory.name,
                            "reason": f"status={status.get('status')!r}, not a completed run"})
            continue
        rounds = decision_rounds(directory)
        if not rounds:
            skipped.append({"run": directory.name, "reason": "no decisions_round_*.csv"})
            continue
        entries.append({"run_id": directory.name, "directory": str(directory),
                        "cfg": cfg, "status": status.get("status"), "rounds": rounds})
    keys = Counter((entry["cfg"]["task"], entry["cfg"]["variant"], entry["cfg"]["seed"])
                   for entry in entries)
    duplicates = {key: count for key, count in keys.items() if count > 1}
    if duplicates:
        detail = "; ".join(f"{key}={count}" for key, count in sorted(duplicates.items()))
        raise ValueError(f"Duplicate task/variant/seed runs are not independent repetitions: {detail}")
    return entries, skipped


def decision_header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            raise ValueError(f"Empty decision log: {path}")


def plan(entries, skipped, runs_root, grid, scores, rounds_specification, output, variants):
    """File-only preflight report; validates headers and weak-pool sizes."""
    required = {"sample_id", "candidate", "region", "trained", "lower", "q"}
    counts = Counter()
    rounds_by_task = {}
    n_total_by_task = {}
    for entry in entries:
        cfg = entry["cfg"]
        task = cfg["task"]
        counts[(task, cfg["variant"])] += 1
        rounds = selected_rounds(entry["rounds"], rounds_specification)
        rounds_by_task.setdefault(task, set()).add(tuple(rounds))
        path = Path(entry["directory"]) / f"decisions_round_{rounds[0]:03}.csv"
        header = set(decision_header(path))
        missing = sorted(required - header)
        if missing:
            raise ValueError(f"{entry['run_id']} decision log lacks columns: {missing}")
        if task not in n_total_by_task:
            weak, _ = load_weak_role(cfg)
            n_total_by_task[task] = len(weak)
        elif len(json.loads((Path(cfg["prepared_root"]) / "weak.json").read_text(
                encoding="utf-8"))) != n_total_by_task[task]:
            raise ValueError("Prepared weak-pool size changed between runs")
    curve_variants = [variant for variant in variants if variant in DEFAULT_CURVE_VARIANTS]
    return {
        "status": "PLAN_ONLY", "analysis": "offline correction reliability from decision logs",
        "runs_root": str(runs_root),
        "output": str(output), "n_runs": len(entries), "n_skipped": len(skipped),
        "runs_by_task_variant": [{"task": task, "variant": variant, "n": count}
                                 for (task, variant), count in sorted(counts.items())],
        "curve_variants": curve_variants,
        "operating_only_variants": [variant for variant in variants
                                    if variant in OPERATING_ONLY_VARIANTS],
        "n_total_by_task": n_total_by_task,
        "rounds_by_task": {task: sorted({index for group in groups for index in group})
                           for task, groups in rounds_by_task.items()},
        "coverage_grid": grid, "scores": scores,
        "skipped": skipped,
        "note": "Preflight only. Re-run with --execute-reliability to analyse the logs.",
    }


def _seed_metric(run, round_index, score, coverage, name):
    for point in run["rounds"][str(round_index)]["scores"][score]["points"]:
        if abs(point["coverage_target"] - coverage) < 1e-12:
            return point.get(name)
    raise KeyError(f"coverage {coverage} missing from the run curve")


def _point(run, round_index, score, coverage):
    for point in run["rounds"][str(round_index)]["scores"][score]["points"]:
        if abs(point["coverage_target"] - coverage) < 1e-12:
            return point
    raise KeyError(f"coverage {coverage} missing from the run curve")


def matched_coverage_rows(results, tasks, seeds, curve_variants, scores, grid):
    """Matched-coverage summary rows plus infeasibility bookkeeping."""
    if "full" not in curve_variants:
        raise ValueError("Matched-coverage comparison requires the full reference")
    index = {(run["task"], run["variant"], run["seed"]): run for run in results.values()}
    rounds_by_task = {}
    for run in results.values():
        rounds_by_task.setdefault(run["task"], set()).update(int(r) for r in run["rounds"])
    rows, auc_rows = [], []
    skipped = Counter()
    for task in tasks:
        for score in scores:
            for round_index in sorted(rounds_by_task.get(task, ())):
                runs = {}
                complete = True
                for variant in curve_variants:
                    for seed in seeds:
                        run = index.get((task, variant, seed))
                        if run is None or str(round_index) not in run["rounds"]:
                            complete = False
                            break
                        runs[(variant, seed)] = run
                    if not complete:
                        break
                if not complete:
                    skipped["incomplete_round"] += 1
                    continue
                reference = {seed: runs[("full", seed)] for seed in seeds} if "full" in curve_variants else {}
                feasible_coverages = []
                for coverage in grid:
                    points = {key: _point(run, round_index, score, coverage)
                              for key, run in runs.items()}
                    if any(point["feasible"] is not True for point in points.values()):
                        skipped["infeasible_coverage"] += 1
                        continue
                    feasible_coverages.append(coverage)
                    for variant in curve_variants:
                        metrics = {name: [_seed_metric(runs[(variant, seed)], round_index,
                                                       score, coverage, name)
                                          for seed in seeds]
                                   for name in ("correction_accuracy", "error_rate",
                                                "erroneous_supervision_fraction_of_all",
                                                "task_error_rate")}
                        full_risk = ([_seed_metric(reference[seed], round_index, score,
                                                   coverage, "error_rate") for seed in seeds]
                                     if variant != "full" else [None] * len(seeds))
                        paired = (paired_summary(metrics["error_rate"], full_risk)
                                  if variant != "full" else
                                  {"n": 0, "mean_delta": None, "std_delta": None, "ci95": None,
                                   "n_positive": 0, "n_negative": 0, "n_zero": 0})
                        summary = {name: summarize(values) for name, values in metrics.items()}
                        rows.append({
                            "task": task, "score": score, "round": round_index,
                            "coverage": coverage, "supervised": points[(variant, seeds[0])]["supervised"],
                            "variant": variant, "reference": variant == "full",
                            "n_seeds": len(seeds),
                            "mean_correction_accuracy": summary["correction_accuracy"]["mean"],
                            "std_correction_accuracy": summary["correction_accuracy"]["std"],
                            "mean_error_rate": summary["error_rate"]["mean"],
                            "std_error_rate": summary["error_rate"]["std"],
                            "mean_erroneous_supervision_fraction_of_all":
                                summary["erroneous_supervision_fraction_of_all"]["mean"],
                            "mean_task_error_rate": summary["task_error_rate"]["mean"],
                            "mean_delta_error_rate_vs_full": paired["mean_delta"],
                            "std_delta_error_rate_vs_full": paired["std_delta"],
                            "ci95_delta_error_rate_vs_full": paired["ci95"],
                            "n_variant_better": paired["n_negative"] if variant != "full" else None,
                            "n_full_better": paired["n_positive"] if variant != "full" else None,
                            "n_tied": paired["n_zero"] if variant != "full" else None,
                        })
                # Risk AUC is evaluated on the same feasible coverage range for
                # every variant and seed, otherwise different support rates would
                # integrate over different x-ranges and the comparison would be
                # meaningless.
                for variant in curve_variants:
                    if len(feasible_coverages) >= 2:
                        values = [risk_auc([_point(runs[(variant, seed)], round_index,
                                                   score, coverage)
                                            for coverage in feasible_coverages])
                                  for seed in seeds]
                        full_values = ([risk_auc([_point(reference[seed], round_index,
                                                         score, coverage)
                                                  for coverage in feasible_coverages])
                                        for seed in seeds] if variant != "full"
                                       else [None] * len(seeds))
                    else:
                        values, full_values = [None] * len(seeds), [None] * len(seeds)
                    paired = (paired_summary(values, full_values) if variant != "full" else
                              {"mean_delta": None, "std_delta": None, "ci95": None,
                               "n_negative": None, "n_positive": None, "n_zero": None})
                    summary = summarize(values)
                    auc_rows.append({
                        "task": task, "score": score, "round": round_index, "variant": variant,
                        "reference": variant == "full", "n_seeds": len(seeds),
                        "auc_points": len(feasible_coverages),
                        "auc_coverage_min": min(feasible_coverages) if len(feasible_coverages) >= 2 else None,
                        "auc_coverage_max": max(feasible_coverages) if len(feasible_coverages) >= 2 else None,
                        "mean_risk_auc": summary["mean"], "std_risk_auc": summary["std"],
                        "mean_delta_risk_auc_vs_full": paired["mean_delta"],
                        "ci95_delta_risk_auc_vs_full": paired["ci95"],
                        "n_variant_better": paired["n_negative"] if variant != "full" else None,
                        "n_full_better": paired["n_positive"] if variant != "full" else None,
                    })
    return rows, auc_rows, dict(skipped)


def operating_rows(results, tasks, seeds, variants):
    """Per-seed operating point rows and seed summaries."""
    index = {(run["task"], run["variant"], run["seed"]): run for run in results.values()}
    per_seed, summaries = [], []
    rounds_by_task = {}
    for run in results.values():
        rounds_by_task.setdefault(run["task"], set()).update(int(r) for r in run["rounds"])
    for task in tasks:
        for variant in variants:
            for round_index in sorted(rounds_by_task.get(task, ())):
                runs = [index.get((task, variant, seed)) for seed in seeds]
                if any(run is None or str(round_index) not in run["rounds"] for run in runs):
                    continue
                for seed, run in zip(seeds, runs):
                    payload = run["rounds"][str(round_index)]["operating"]
                    per_seed.append({"task": task, "variant": variant, "seed": seed,
                                     "round": round_index, **payload})
                fields = ("coverage", "error_rate", "correction_accuracy",
                          "erroneous_supervision_fraction_of_all", "task_error_rate",
                          "trained_coverage", "supported_coverage", "candidate_error_rate")
                summary = {name: summarize(
                    [run["rounds"][str(round_index)]["operating"].get(name) for run in runs])
                    for name in fields}
                summaries.append({
                    "task": task, "variant": variant, "round": round_index,
                    "n_seeds": len(seeds),
                    "pseudo_supervision": runs[0]["rounds"][str(round_index)]["operating"]
                    .get("pseudo_supervision"),
                    **{f"mean_{name}": summary[name]["mean"] for name in fields},
                    **{f"std_{name}": summary[name]["std"] for name in fields},
                })
    return per_seed, summaries


def plot_rounds(available_by_task, specification):
    if specification == "all":
        return {task: sorted(rounds) for task, rounds in available_by_task.items()}
    if specification == "last":
        return {task: [max(rounds)] for task, rounds in available_by_task.items() if rounds}
    wanted = parse_int_list(specification)
    return {task: [round_index for round_index in wanted if round_index in rounds]
            for task, rounds in available_by_task.items()}


def render_plots(rows, tasks, scores, curve_variants, plot_round_spec, panel_coverage,
                 output, rounds_by_task):
    """Write SVG risk--coverage and risk-vs-round figures for the chosen rounds."""
    plots = []
    chosen = plot_rounds(rounds_by_task, plot_round_spec)
    for task in tasks:
        for score in scores:
            for round_index in chosen.get(task, []):
                series = []
                for variant in curve_variants:
                    selected = [row for row in rows
                                if row["task"] == task and row["score"] == score
                                and row["round"] == round_index and row["variant"] == variant]
                    if not selected:
                        continue
                    selected.sort(key=lambda row: row["coverage"])
                    series.append({
                        "label": variant,
                        "points": [(row["coverage"], row["mean_error_rate"]) for row in selected],
                        "std": [(row["coverage"], row["std_error_rate"]) for row in selected],
                    })
                if not series:
                    continue
                path = Path(output) / "plots" / (
                    f"risk_coverage__{task}__round-{round_index:03d}__{score}.svg")
                line_chart_svg(
                    path,
                    title=f"Risk--coverage: {task} (round {round_index}, score={score})",
                    subtitle="Mean over design seeds; error bars = 1 SD across seeds; lower is better",
                    series=series, x_label="supervision coverage (fraction of weak pool)",
                    y_label="erroneous supervision ratio",
                    x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    note=("Risk = wrong exact-label corrections / supervised; matched coverage = "
                          "same k per run; offline diagnostic, not a test result."))
                plots.append(str(path))
            # Risk at one matched coverage versus round.
            all_rounds = sorted({row["round"] for row in rows
                                 if row["task"] == task and row["score"] == score
                                 and abs(row["coverage"] - panel_coverage) < 1e-12})
            if not all_rounds:
                continue
            series = []
            for variant in curve_variants:
                selected = {row["round"]: row for row in rows
                            if row["task"] == task and row["score"] == score
                            and row["variant"] == variant
                            and abs(row["coverage"] - panel_coverage) < 1e-12}
                if not selected:
                    continue
                series.append({
                    "label": variant,
                    "points": [(round_index,
                                selected[round_index]["mean_error_rate"]
                                if round_index in selected else None)
                               for round_index in all_rounds],
                    "std": [(round_index, selected[round_index]["std_error_rate"])
                            for round_index in all_rounds if round_index in selected],
                })
            if not series:
                continue
            ticks = all_rounds if len(all_rounds) <= 8 else [
                all_rounds[round(index * (len(all_rounds) - 1) / 7)]
                for index in range(8)]
            path = Path(output) / "plots" / (
                f"risk_vs_round__{task}__coverage-{panel_coverage:g}__{score}.svg")
            line_chart_svg(
                path,
                title=f"Risk at matched coverage {panel_coverage:.2f} vs round: {task} "
                      f"(score={score})",
                subtitle="Mean over design seeds; error bars = 1 SD; rounds without feasible "
                         "coverage are omitted",
                series=series, x_label="training round", y_label="erroneous supervision ratio",
                x_min=min(all_rounds), x_max=max(all_rounds), x_ticks=ticks,
                note="Matched coverage = floor(coverage * N) supervised samples ranked by the "
                     "score; offline diagnostic.")
            plots.append(str(path))
    return plots


def run(entries, skipped, grid, scores, rounds_specification, plot_round_spec,
        panel_coverage, output, curve_variants, tasks, seeds, variants):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reliability output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    prepared = {}
    for task in tasks:
        entry = next(entry for entry in entries if entry["cfg"]["task"] == task)
        weak, truth = load_weak_role(entry["cfg"])
        prepared[task] = {
            "n_total": len(weak),
            "by_id": {sample["id"]: sample for sample in weak},
            "truth": truth,
            "oracle": ArithmeticKB(entry["cfg"]["task"], entry["cfg"]["symbols"]),
            "manifest_sha256": digest(Path(entry["cfg"]["prepared_root"]) / "manifest.json"),
        }
    # Load results, write one compact JSON per run.
    results = {}
    run_dir = output / "runs"
    run_dir.mkdir()
    for position, entry in enumerate(entries, 1):
        cfg = entry["cfg"]
        task = cfg["task"]
        rounds = selected_rounds(entry["rounds"], rounds_specification)
        payload_rounds = {}
        for round_index in rounds:
            path = Path(entry["directory"]) / f"decisions_round_{round_index:03}.csv"
            data = read_round_decisions(path, prepared[task]["by_id"],
                                        prepared[task]["truth"], prepared[task]["oracle"],
                                        round_index=round_index)
            if data.n_total != prepared[task]["n_total"]:
                raise ValueError(f"{entry['run_id']} round {round_index} has {data.n_total} "
                                 f"decision rows, expected {prepared[task]['n_total']}")
            operating = operating_metrics(data)
            operating["pseudo_supervision"] = cfg["variant"] != "anchor_only"
            score_payload = {score: curve_from_round(data, score, grid) for score in scores}
            payload_rounds[str(round_index)] = {
                "supported": data.n_supported, "accepted": data.n_accepted,
                "trained": data.n_trained, "operating": operating, "scores": score_payload,
            }
        results[entry["run_id"]] = {
            "task": task, "variant": cfg["variant"], "seed": cfg["seed"],
            "rounds": payload_rounds,
        }
        record = {
            "schema_version": SCHEMA_VERSION, "run_id": entry["run_id"],
            "task": task, "variant": cfg["variant"], "seed": cfg["seed"],
            "method_version": cfg.get("method_version"),
            "n_total": prepared[task]["n_total"],
            "config_sha256": digest(Path(entry["directory"]) / "config.json"),
            "prepared_manifest_sha256": prepared[task]["manifest_sha256"],
            "rounds": payload_rounds,
        }
        write_json(run_dir / f"{entry['run_id']}.json", record)
        print(json.dumps({"progress": position, "n_runs": len(entries),
                          "run_id": entry["run_id"], "rounds": len(rounds)}), flush=True)

    matched, auc, infeasible = matched_coverage_rows(
        results, tasks, seeds, curve_variants, scores, grid)
    per_seed_operating, operating_summary = operating_rows(
        results, tasks, seeds, variants)
    write_csv(output / "matched_coverage_summary.csv",
              ["task", "score", "round", "coverage", "supervised", "variant", "reference",
               "n_seeds", "mean_correction_accuracy", "std_correction_accuracy",
               "mean_error_rate", "std_error_rate",
               "mean_erroneous_supervision_fraction_of_all", "mean_task_error_rate",
               "mean_delta_error_rate_vs_full", "std_delta_error_rate_vs_full",
               "ci95_delta_error_rate_vs_full", "n_variant_better", "n_full_better", "n_tied"],
              matched)
    write_csv(output / "matched_coverage_auc_summary.csv",
              ["task", "score", "round", "variant", "reference", "n_seeds",
               "auc_points", "auc_coverage_min", "auc_coverage_max",
               "mean_risk_auc", "std_risk_auc", "mean_delta_risk_auc_vs_full",
               "ci95_delta_risk_auc_vs_full", "n_variant_better", "n_full_better"], auc)
    write_csv(output / "operating_points.csv",
              ["task", "variant", "seed", "round", "n_total", "supported_coverage",
               "candidate_error_rate", "task_candidate_error_rate", "coverage", "supervised",
               "errors", "error_rate", "correction_accuracy",
               "erroneous_supervision_fraction_of_all", "task_error_rate",
               "task_correction_accuracy", "task_erroneous_fraction_of_all",
               "trained_coverage", "trained_error_rate", "pseudo_supervision"],
              per_seed_operating)
    write_csv(output / "operating_points_summary.csv",
              ["task", "variant", "round", "n_seeds", "pseudo_supervision",
               "mean_coverage", "std_coverage", "mean_error_rate", "std_error_rate",
               "mean_correction_accuracy", "std_correction_accuracy",
               "mean_erroneous_supervision_fraction_of_all",
               "mean_task_error_rate", "mean_trained_coverage",
               "mean_supported_coverage", "mean_candidate_error_rate"],
              operating_summary)
    rounds_by_task = {}
    for run_payload in results.values():
        rounds_by_task.setdefault(run_payload["task"], set()).update(
            int(index) for index in run_payload["rounds"])
    plots = render_plots(matched, tasks, scores, curve_variants, plot_round_spec,
                         panel_coverage, output, rounds_by_task)
    source_root = Path(__file__).parent
    write_json(output / "aggregate.json", {
        "status": "RELIABILITY_COMPLETED", "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "method_version": METHOD_VERSION, "analysis": "offline correction reliability",
        "runs_root": str(Path(entries[0]["directory"]).parent) if entries else None,
        "output": str(output),
        "tasks": tasks, "variants": variants, "curve_variants": curve_variants,
        "operating_only_variants": [variant for variant in variants
                                    if variant in OPERATING_ONLY_VARIANTS],
        "scores": scores, "coverage_grid": grid, "panel_coverage": panel_coverage,
        "n_runs": len(entries), "n_total_by_task": {task: prepared[task]["n_total"]
                                                    for task in tasks},
        "rounds_by_task": {task: sorted(rounds) for task, rounds in rounds_by_task.items()},
        "n_infeasible": infeasible, "skipped_runs": skipped,
        "prepared_manifests": {task: prepared[task]["manifest_sha256"] for task in tasks},
        "source_hashes": {
            "reliability.py": digest(source_root / "reliability.py"),
            "run_reliability.py": digest(source_root / "run_reliability.py"),
        },
        "files": {
            "matched_coverage_summary": "matched_coverage_summary.csv",
            "matched_coverage_auc_summary": "matched_coverage_auc_summary.csv",
            "operating_points": "operating_points.csv",
            "operating_points_summary": "operating_points_summary.csv",
            "per_run": "runs/<run_id>.json",
            "plots": [str(Path(path).relative_to(output)) for path in plots],
        },
        "definitions": DEFINITIONS,
        "note": ("Offline diagnostic over the prepared weak role and hidden weak labels; "
                 "never feed these outputs into training or model selection. The formal "
                 f"test split is not read. {len(seeds)} design seed(s); paired intervals "
                 "are descriptive and per-round curves within a run are correlated."),
    })
    return {"status": "RELIABILITY_COMPLETED", "n_runs": len(entries),
            "n_matched_rows": len(matched), "n_auc_rows": len(auc),
            "n_operating_rows": len(per_seed_operating), "n_plots": len(plots),
            "n_infeasible": infeasible, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(PROJECT / "methods" / "frt_abl" / "runs-v10"))
    parser.add_argument("--output",
                        default=str(PROJECT / "methods" / "frt_abl" /
                                    "runs-reliability-v10"))
    parser.add_argument("--tasks", nargs="+", choices=("mnist-add2", "hwf"),
                        default=["mnist-add2", "hwf"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=list(DEFAULT_CURVE_VARIANTS + OPERATING_ONLY_VARIANTS))
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"],
                        help="Design seeds; accepts '0 1 2' or '0,1,2'")
    parser.add_argument("--rounds", default="all",
                        help="'all', 'last', or a comma-separated list of round indices")
    parser.add_argument("--scores", nargs="+", choices=("lower", "q"),
                        default=["lower", "q"])
    parser.add_argument("--coverage-start", type=float, default=0.05)
    parser.add_argument("--coverage-stop", type=float, default=1.0)
    parser.add_argument("--coverage-step", type=float, default=0.05)
    parser.add_argument("--plot-rounds", default="last",
                        help="'last', 'all', or a comma-separated list of round indices")
    parser.add_argument("--panel-coverage", type=float, default=0.5,
                        help="Matched coverage used by the risk-vs-round figures")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Include runs whose status is not TRAINING_COMPLETED_TEST_NOT_RUN")
    parser.add_argument("--execute-reliability", action="store_true",
                        help="Run the offline analysis; omitting it only validates the plan")
    args = parser.parse_args()
    try:
        seeds = parse_seed_list(args.seeds)
    except ValueError as error:
        raise SystemExit(str(error))
    if any(seed not in (0, 1, 2, 3, 4) for seed in seeds):
        raise SystemExit("Registered seeds are 0..4")
    if len(set(args.variants)) != len(args.variants):
        raise SystemExit("Variants must be unique")
    if len(set(args.scores)) != len(args.scores):
        raise SystemExit("Scores must be unique")
    grid = coverage_grid(args.coverage_start, args.coverage_stop, args.coverage_step)
    if not any(abs(point - args.panel_coverage) < 1e-12 for point in grid):
        raise SystemExit("--panel-coverage must be a point of the coverage grid")
    entries, skipped = discover(args.runs, set(args.tasks), set(args.variants),
                                set(seeds), require_complete=not args.allow_incomplete)
    if not entries:
        raise SystemExit("No matching complete runs with decision logs were found")
    tasks = [task for task in args.tasks
             if any(entry["cfg"]["task"] == task for entry in entries)]
    if not tasks:
        raise SystemExit("No requested task has matching runs")
    curve_variants = [variant for variant in args.variants
                      if variant in DEFAULT_CURVE_VARIANTS]
    if not curve_variants:
        raise SystemExit("At least one gated variant is required for matched-coverage curves")
    if "full" not in curve_variants:
        raise SystemExit("The paired comparison requires the full variant")
    report = plan(entries, skipped, args.runs, grid, args.scores, args.rounds,
                  args.output, args.variants)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute_reliability:
        return
    summary = run(entries, skipped, grid, args.scores, args.rounds, args.plot_rounds,
                  args.panel_coverage, args.output, curve_variants, tasks,
                  seeds, args.variants)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
