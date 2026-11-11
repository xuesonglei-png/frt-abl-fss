"""Three-region evolution analysis for FRT-ABL training logs.

This module is post-hoc and read-only with respect to experiments: it consumes the
per-round artifacts that ``train.py`` already writes (``decisions_round_*.csv``,
``history.jsonl``, ``config.json``, ``status.json``) and never imports torch,
loads data, builds a model, or writes into a run directory.  It is intended for
the V10 ``full`` baseline, but the same parser works for any variant whose
decision log has the V10 schema.

Definitions used in the outputs
-------------------------------
* Region: the current best candidate's three-way decision at a round
  (``positive`` / ``negative`` / ``boundary``; ``conflict`` is kept separate and
  is never silently merged into negative).
* Transition: a sample's region at round ``t-1`` -> region at round ``t``.
  The first observed round uses ``initial`` as the source.
* Boundary spell: a maximal run of consecutive observed rounds in ``boundary``.
  A spell ends when the sample leaves boundary, or is right-censored at the last
  logged round.
* Waiting rounds: ``boundary_rounds``, the number of consecutive observed
  boundary rounds in a spell.  For a converted spell this is the number of
  rounds spent in boundary before the first positive round.
* Reliable correction: a boundary -> positive conversion whose row passes the
  method's own gate-level reliability test: ``evidence == supported``,
  ``eligible`` (stable/margin/unknown/completeness as recorded), and
  ``lower >= alpha`` for the round's selected alpha.  Two variants are reported:
  *immediate* (passes on the conversion round) and *eventual* (passes on any
  later positive round while still positive).  In V10 ``full``'s soft-acceptance
  regime a freshly found candidate has ``stable_count=1``, so immediate is often
  false while eventual is the informative measure.  This is a
  *decision-policy* reliability proxy, not ground-truth correction accuracy.
  Offline truth can be attached with ``audit.py``; this module never reads weak
  labels.
* Sustained correction: the conversion round plus the next
  ``--persist-rounds - 1`` observed rounds are all positive.  If the run ends
  before the window is complete the status is ``censored`` and it is not counted
  as sustained.  *Eventual reliable sustained* requires that some later
  gate-reliable round also starts such a positive window.
* Actual training weight: ``effective_weight`` on rows with ``trained == True``
  (i.e. the weight actually multiplied into the pseudo-label loss).  Boundary
  consistency loss is a separate fixed-weight term and is not represented by the
  per-candidate ``weight`` column.

Outputs (under ``--output``)
----------------------------
``region_rounds.csv``
    One row per run-round: region counts/ratios, entries/exits, conversions,
    reliability counts, actual training weights, validation accuracy/loss.
``transitions.csv``
    One row per run-round-source-destination transition, with conditional rate.
``boundary_spells.csv``
    One row per boundary spell, including converted/negative/censored exits,
    wait length, reliability, sustained status and post-conversion weights.
``run_summary.json``
    Scalar per-run summary metrics.
``summary.json``
    Seed-level aggregation by task/variant/method version, with mean/std/95% CI.
``curve_summary.csv``
    Round-level means/stds across seeds (learning curves + region ratios).
``transition_summary.csv``
    Transition counts and conditional rates pooled across seeds.
``analysis_manifest.json``
    Input run list, counts, code revision(s), and the "no test / no training"
    status note.
``REPORT.md``
    Human-readable summary with the definitions and limitations above.

Usage (from the project root)::

    py -m methods.frt_abl.region_evolution \
        --runs methods/frt_abl/runs-v10 \
        --output methods/frt_abl/runs-region-evolution-v10 \
        --execute-analysis

Without ``--execute-analysis`` the command only discovers and lists runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

from .policies import VARIANTS
from .prepare import PROJECT, digest, write_json

REGIONS = ("positive", "negative", "boundary", "conflict")
INITIAL = "initial"
MISSING = "missing"
CENSORED = "censored"
COMPLETE_STATUS = "TRAINING_COMPLETED_TEST_NOT_RUN"

DEFAULT_CONFIG = {
    "stable_rounds": 2,
    "allow_truncated": False,
    "min_margin": 0.02,
    "alpha_start": 0.95,
    "beta": 0.2,
}

T_CRIT_95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
}

ROUND_FIELDS = [
    "run_id", "task", "variant", "seed", "round", "round_index", "alpha", "gate_open", "total",
    "candidate_found", "candidate_found_ratio",
    "positive", "negative", "boundary", "conflict",
    "positive_ratio", "negative_ratio", "boundary_ratio", "conflict_ratio",
    "boundary_supported", "boundary_unresolved",
    "entered_positive", "entered_negative", "entered_boundary", "entered_conflict",
    "exited_positive", "exited_negative", "exited_boundary", "exited_conflict",
    "converted_boundary_to_positive", "reliable_conversions",
    "sustained_conversions", "reliable_sustained_conversions",
    "trained_count", "trained_ratio", "weight_sum", "weight_mean", "weight_median", "weight_p90",
    "new_conversion_weight_mean", "continuing_weight_mean", "effective_sample_size",
    "val_accuracy", "loss", "history_coverage", "history_accepted_candidates",
    "history_regions_match",
]

TRANSITION_FIELDS = [
    "run_id", "task", "variant", "seed", "round", "round_index",
    "from_region", "to_region", "count", "rate_given_from",
]

SPELL_FIELDS = [
    "run_id", "task", "variant", "seed", "sample_id", "spell_index",
    "start_round", "end_round", "conversion_t", "boundary_rounds", "waited_span",
    "exit_region", "converted", "censored", "reliable", "reliable_eventual",
    "first_reliable_offset", "reliable_within_3", "reliable_within_5", "reliable_within_10",
    "sustained", "reliable_sustained",
    "start_evidence", "start_q", "start_lower", "start_upper", "start_alpha",
    "exit_evidence", "exit_q", "exit_lower", "exit_upper", "exit_alpha",
    "exit_eligible", "exit_stable_count", "exit_unknown_count", "exit_n_supported", "exit_complete",
    "exit_weight", "exit_effective_weight", "exit_trained",
    "post_conversion_weight_sum", "post_conversion_trained_rounds", "post_conversion_mean_weight",
]

CURVE_FIELDS = [
    "positive_ratio", "negative_ratio", "boundary_ratio", "conflict_ratio",
    "candidate_found_ratio", "boundary_unresolved_ratio",
    "converted_boundary_to_positive", "reliable_conversions",
    "trained_ratio", "weight_mean", "effective_sample_size",
    "val_accuracy", "loss", "history_coverage",
]

# Scalar per-run metrics included in summary.json.  Every name is guaranteed to
# exist in a run summary (possibly with a None value) so downstream readers can
# rely on the schema.
SUMMARY_METRICS = [
    "n_rounds", "n_samples", "n_decision_rows",
    "mean_positive_ratio", "mean_negative_ratio", "mean_boundary_ratio", "mean_conflict_ratio",
    "mean_candidate_found_ratio", "mean_boundary_unresolved_ratio", "mean_boundary_supported_ratio",
    "pooled_positive_ratio", "pooled_negative_ratio", "pooled_boundary_ratio", "pooled_conflict_ratio",
    "final_positive_ratio", "final_negative_ratio", "final_boundary_ratio", "final_conflict_ratio",
    "mean_training_coverage", "final_training_coverage",
    "mean_val_accuracy", "final_val_accuracy", "best_val_accuracy", "best_val_round", "val_auc",
    "spell_count", "converted_spells", "censored_spells", "exited_negative_spells",
    "exited_conflict_spells", "missing_spells", "conversion_rate_per_spell",
    "reliable_conversions", "reliable_conversion_share",
    "eventual_reliable_conversions", "eventual_reliable_share",
    "eventual_reliable_sustained_conversions", "eventual_reliable_sustained_share",
    "sustained_conversions", "sustained_share",
    "reliable_sustained_conversions", "reliable_sustained_share",
    "spells_started_unresolved", "spells_started_supported",
    "conversions_from_unresolved", "reliable_conversions_from_unresolved",
    "eventual_reliable_conversions_from_unresolved",
    "eventual_reliable_sustained_conversions_from_unresolved",
    "conversion_rate_from_unresolved", "reliable_conversion_rate_from_unresolved",
    "eventual_reliable_conversion_rate_from_unresolved",
    "ever_boundary_samples", "ever_boundary_converted_samples", "ever_boundary_conversion_rate",
    "ever_boundary_reliable_samples", "ever_boundary_reliable_rate",
    "ever_boundary_eventual_reliable_samples", "ever_boundary_eventual_reliable_rate",
    "ever_boundary_reliable_sustained_samples", "ever_boundary_reliable_sustained_rate",
    "end_in_boundary_samples", "end_boundary_share_of_ever_boundary",
    "total_boundary_observations", "mean_boundary_rounds_per_ever_boundary_sample",
    "boundary_round_share", "mean_spells_per_ever_boundary_sample",
    "wait_converted_mean", "wait_converted_median", "wait_converted_p25",
    "wait_converted_p90", "wait_converted_max",
    "wait_all_mean", "wait_censored_mean", "wait_unresolved_converted_mean",
    "first_reliable_offset_mean", "first_reliable_offset_median", "first_reliable_offset_p90",
    "eventual_reliable_within_1_share", "eventual_reliable_within_2_share",
    "eventual_reliable_within_3_share", "eventual_reliable_within_5_share",
    "eventual_reliable_within_10_share",
    "converted_within_1_share", "converted_within_2_share", "converted_within_3_share",
    "converted_within_5_share", "converted_within_10_share",
    "conversion_rate_within_1_of_spells", "conversion_rate_within_2_of_spells",
    "conversion_rate_within_3_of_spells", "conversion_rate_within_5_of_spells",
    "conversion_rate_within_10_of_spells",
    "trained_count_total", "trained_weight_sum",
    "trained_weight_pooled_mean", "trained_weight_pooled_median", "trained_weight_pooled_p90",
    "trained_weight_max", "new_conversion_weight_mean", "new_conversion_weight_median",
    "continuing_weight_mean", "mean_effective_sample_size",
    "post_conversion_weight_sum_mean", "post_conversion_weight_sum_median",
    "post_conversion_trained_rounds_mean",
    "region_count_mismatches", "history_missing_rounds",
]


@dataclass
class RunMeta:
    run_id: str
    path: Path
    task: str
    variant: str
    seed: int
    method_version: str
    status: str | None
    smoke: bool
    config: dict


@dataclass
class RunAnalysis:
    meta: RunMeta
    round_rows: list
    transition_rows: list
    spell_rows: list
    summary: dict
    warnings: list


def _round_number(path):
    match = re.fullmatch(r"decisions_round_(\d+)\.csv", Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognized decision log name: {path}")
    return int(match.group(1))


def _as_float(value, field, where, default=None):
    text = "" if value is None else str(value).strip()
    if text == "":
        if default is not None:
            return default
        raise ValueError(f"{where}: missing numeric field {field}")
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"{where}: invalid numeric field {field}={value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{where}: non-finite numeric field {field}={value!r}")
    return number


def _optional_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _optional_int(value):
    number = _optional_float(value)
    if number is None:
        return None
    return int(round(number))


def _as_bool(value, field, where, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{where}: invalid boolean field {field}={value!r}")


def _optional_bool(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_region(value, where):
    text = "" if value is None else str(value).strip().lower()
    aliases = {
        "positive": "positive", "pos": "positive",
        "negative": "negative", "neg": "negative",
        "boundary": "boundary", "bound": "boundary",
        "conflict": "conflict",
    }
    if text not in aliases:
        raise ValueError(f"{where}: unknown region {value!r}; expected one of {REGIONS}")
    return aliases[text]


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _percentile(values, fraction):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _ci95(values):
    values = [v for v in values if v is not None]
    n = len(values)
    if n < 2:
        return None
    critical = T_CRIT_95.get(n, 1.96 if n > 10 else None)
    if critical is None:
        return None
    return critical * statistics.stdev(values) / math.sqrt(n)


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _write_csv(path, fieldnames, rows):
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def load_run_meta(run_dir):
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config.json")
    status = _load_json(run_dir / "status.json")
    run_id = run_dir.name
    parts = run_id.split("__")
    task = config.get("task") or (parts[1] if len(parts) > 1 else "")
    variant = config.get("variant") or (parts[2] if len(parts) > 2 else "")
    seed = config.get("seed")
    if seed is None and len(parts) > 3 and parts[3].startswith("seed-"):
        try:
            seed = int(parts[3].split("-", 1)[1])
        except (IndexError, ValueError):
            seed = None
    if not task or not variant or seed is None:
        raise ValueError(f"Cannot infer task/variant/seed for run directory {run_dir}")
    method_version = config.get("method_version") or (parts[0] if parts else "legacy")
    merged = dict(DEFAULT_CONFIG)
    for key in DEFAULT_CONFIG:
        if key in config:
            merged[key] = config[key]
    merged["stable_rounds"] = int(merged["stable_rounds"])
    merged["allow_truncated"] = bool(merged["allow_truncated"])
    merged["min_margin"] = float(merged["min_margin"])
    merged["alpha_start"] = float(merged["alpha_start"])
    merged["beta"] = float(merged["beta"])
    return RunMeta(
        run_id=run_id, path=run_dir, task=task, variant=variant, seed=int(seed),
        method_version=str(method_version), status=status.get("status"),
        smoke=bool(status.get("smoke", False)) or "__smoke" in run_id,
        config=merged,
    )


def discover_runs(paths, variant="full", task="all", seeds=None, include_incomplete=False):
    """Return (run metadata list, skipped-run records)."""
    candidates = {}
    for raw in paths:
        root = Path(raw)
        if not root.exists():
            raise FileNotFoundError(f"Run path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Run path is not a directory: {root}")
        if list(root.glob("decisions_round_*.csv")):
            candidates[root.resolve()] = root
            continue
        direct_children = [child for child in sorted(root.iterdir()) if child.is_dir()]
        direct_runs = [child for child in direct_children
                       if list(child.glob("decisions_round_*.csv"))]
        if direct_runs:
            for child in direct_runs:
                candidates[child.resolve()] = child
            continue
        # Fallback for an arbitrary nesting level; the direct-child fast path
        # above is what the normal runs-v10 layout uses.
        for decision_file in root.rglob("decisions_round_*.csv"):
            candidates[decision_file.parent.resolve()] = decision_file.parent

    runs, skipped = [], []
    for path in sorted(candidates.values(), key=lambda item: item.name):
        try:
            meta = load_run_meta(path)
        except ValueError as error:
            skipped.append({"path": str(path), "reason": str(error)})
            continue
        if task != "all" and meta.task != task:
            skipped.append({"path": str(path), "reason": f"task={meta.task} filtered out"})
            continue
        if variant != "all" and meta.variant != variant:
            skipped.append({"path": str(path), "reason": f"variant={meta.variant} filtered out"})
            continue
        if seeds is not None and meta.seed not in seeds:
            skipped.append({"path": str(path), "reason": f"seed={meta.seed} filtered out"})
            continue
        if not include_incomplete:
            if meta.smoke:
                skipped.append({"path": str(path), "reason": "smoke run excluded"})
                continue
            if meta.status is not None and meta.status != COMPLETE_STATUS:
                skipped.append({"path": str(path), "reason": f"status={meta.status} excluded"})
                continue
        runs.append(meta)
    return runs, skipped


def check_unique_seeds(runs, allow_duplicates=False):
    groups = defaultdict(list)
    for meta in runs:
        groups[(meta.task, meta.variant, meta.method_version, meta.seed)].append(meta)
    duplicates = {
        key: [meta.run_id for meta in members]
        for key, members in groups.items() if len(members) > 1
    }
    if duplicates and not allow_duplicates:
        detail = "; ".join(
            f"{task}/{variant}/{version}/seed-{seed}: {ids}"
            for (task, variant, version, seed), ids in sorted(duplicates.items())
        )
        raise ValueError(
            "Duplicate seed runs detected. Re-run with --allow-duplicate-seeds only if "
            "the duplicates were intentionally retained: " + detail
        )
    return duplicates


def load_history(run_dir):
    path = Path(run_dir) / "history.jsonl"
    history = {}
    if not path.exists():
        return history
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                report = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid history JSON at {path}:{line_number}: {error}") from error
            round_number = report.get("round")
            if not isinstance(round_number, int):
                raise ValueError(f"{path}:{line_number}: missing integer round")
            if round_number in history:
                raise ValueError(f"{path}:{line_number}: duplicate round {round_number}")
            history[round_number] = report
    return history


DECISION_REQUIRED = {
    "sample_id", "evidence", "region", "q", "lower", "upper", "weight", "trained",
    "age", "complete", "margin", "stable_count", "unknown_count", "n_supported",
    "eligible", "curriculum_weight", "effective_weight",
}


def load_decisions(path):
    """Load one decisions CSV into a sample-id keyed dictionary.

    The parser is deliberately strict about required columns and region values.
    Silently dropping malformed rows would make the evolution counts unauditable.
    """
    import csv

    path = Path(path)
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = DECISION_REQUIRED - fields
        if missing:
            raise ValueError(f"{path.name}: missing decision columns {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            where = f"{path.name}:{line_number}"
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"{where}: empty sample_id")
            if sample_id in rows:
                raise ValueError(f"{where}: duplicate sample_id {sample_id}")
            region = _normalize_region(row.get("region"), where)
            q = _as_float(row.get("q"), "q", where)
            lower = _as_float(row.get("lower"), "lower", where)
            upper = _as_float(row.get("upper"), "upper", where)
            for name, value in (("q", q), ("lower", lower), ("upper", upper)):
                if not -1e-12 <= value <= 1 + 1e-12:
                    raise ValueError(f"{where}: {name} outside [0,1]: {value}")
            if lower > upper + 1e-9:
                raise ValueError(f"{where}: lower > upper ({lower} > {upper})")
            weight = _as_float(row.get("weight"), "weight", where, default=0.0)
            curriculum_weight = _as_float(row.get("curriculum_weight"), "curriculum_weight",
                                          where, default=1.0)
            effective = _optional_float(row.get("effective_weight"))
            if effective is None:
                effective = weight * curriculum_weight
            trained = _as_bool(row.get("trained"), "trained", where, default=False)
            rows[sample_id] = {
                "region": region,
                "evidence": (row.get("evidence") or "").strip().lower(),
                "q": q, "lower": lower, "upper": upper,
                "weight": weight, "curriculum_weight": curriculum_weight,
                "effective": effective,
                "applied_weight": effective if trained else 0.0,
                "trained": trained,
                "age": _optional_int(row.get("age")),
                "complete": _optional_bool(row.get("complete")),
                "margin": _optional_float(row.get("margin")),
                "stable_count": _optional_int(row.get("stable_count")),
                "unknown_count": _optional_int(row.get("unknown_count")),
                "n_supported": _optional_int(row.get("n_supported")),
                "eligible": _optional_bool(row.get("eligible")),
            }
    if not rows:
        raise ValueError(f"{path.name}: no decision rows")
    return rows


def _eligible_from_fields(row, config):
    if row.get("evidence") != "supported":
        return False
    stable = row.get("stable_count")
    if stable is None or stable < int(config["stable_rounds"]):
        return False
    unknown = row.get("unknown_count")
    if unknown is None or unknown != 0:
        return False
    complete = row.get("complete")
    if not complete and not bool(config["allow_truncated"]):
        return False
    n_supported = row.get("n_supported")
    margin = row.get("margin")
    margin = 0.0 if margin is None else margin
    if n_supported == 1 and complete and unknown == 0:
        margin_ok = True
    else:
        margin_ok = margin >= float(config["min_margin"])
    return margin_ok


def _is_reliable(row, alpha, config):
    """Gate-level reliability of a conversion-round positive decision."""
    if alpha is None or row.get("region") != "positive":
        return False
    if row.get("evidence") != "supported":
        return False
    eligible = row.get("eligible")
    if eligible is None:
        eligible = _eligible_from_fields(row, config)
    if not eligible:
        return False
    lower = row.get("lower")
    if lower is None:
        return False
    return lower + 1e-12 >= float(alpha)


def _spell_event(meta, sample_id, spell, end_t, end_round, exit_region, row, alpha, censored):
    start_row = spell["start_row"]
    converted = exit_region == "positive"
    reliable = converted and row is not None and _is_reliable(row, alpha, meta.config)
    return {
        "run_id": meta.run_id, "task": meta.task, "variant": meta.variant, "seed": meta.seed,
        "sample_id": sample_id, "spell_index": spell["spell_index"],
        "start_round": spell["start_round"], "end_round": end_round,
        "conversion_t": end_t if converted else None,
        "boundary_rounds": spell["boundary_rounds"],
        "waited_span": end_t - spell["start_t"],
        "exit_region": exit_region, "converted": converted, "censored": censored,
        "reliable": reliable, "reliable_eventual": False, "first_reliable_offset": None,
        "reliable_within_3": False, "reliable_within_5": False, "reliable_within_10": False,
        "sustained": None, "reliable_sustained": None,
        "start_evidence": start_row.get("evidence"),
        "start_q": start_row.get("q"), "start_lower": start_row.get("lower"),
        "start_upper": start_row.get("upper"), "start_alpha": spell.get("start_alpha"),
        "exit_evidence": row.get("evidence") if row else None,
        "exit_q": row.get("q") if row else None,
        "exit_lower": row.get("lower") if row else None,
        "exit_upper": row.get("upper") if row else None,
        "exit_alpha": alpha,
        "exit_eligible": row.get("eligible") if row else None,
        "exit_stable_count": row.get("stable_count") if row else None,
        "exit_unknown_count": row.get("unknown_count") if row else None,
        "exit_n_supported": row.get("n_supported") if row else None,
        "exit_complete": row.get("complete") if row else None,
        "exit_weight": row.get("weight") if row else None,
        "exit_effective_weight": row.get("effective") if row else None,
        "exit_trained": row.get("trained") if row else None,
        "post_conversion_weight_sum": None,
        "post_conversion_trained_rounds": None,
        "post_conversion_mean_weight": None,
    }


def analyze_run(meta, persist_rounds=2):
    """Analyze one run's per-round decision logs.

    The function streams the round CSVs and retains only per-sample region and
    applied-weight histories, so a 26k-sample / 30-round run does not require
    holding all raw rows at once.
    """
    if persist_rounds < 1:
        raise ValueError("persist_rounds must be >= 1")
    files = sorted(meta.path.glob("decisions_round_*.csv"), key=_round_number)
    if not files:
        raise ValueError(f"No decisions_round_*.csv in {meta.path}")
    round_numbers = [_round_number(path) for path in files]
    if len(set(round_numbers)) != len(round_numbers):
        raise ValueError(f"Duplicate round numbers in {meta.path}")
    history = load_history(meta.path)
    warnings = []
    if round_numbers != list(range(round_numbers[0], round_numbers[0] + len(round_numbers))):
        warnings.append("decision rounds are not contiguous; observed sequence is used as time")
    missing_history = sorted(set(round_numbers) - set(history))
    if missing_history:
        warnings.append(f"history.jsonl missing rounds {missing_history}; alpha/accuracy unavailable")

    prev_regions = {}
    spells = {}
    spell_counter = Counter()
    region_history = defaultdict(list)
    reliable_history = defaultdict(list)
    applied_weight_history = defaultdict(list)
    trained_history = defaultdict(list)
    all_samples = set()
    round_rows = []
    transition_rows = []
    spell_rows = []
    conversion_events = []
    region_mismatches = 0
    trained_weights_all = []
    new_conversion_weights = []
    continuing_weights = []

    for round_index, (round_number, path) in enumerate(zip(round_numbers, files)):
        rows = load_decisions(path)
        report = history.get(round_number, {})
        alpha = _optional_float(report.get("alpha"))
        current_regions = {sample_id: row["region"] for sample_id, row in rows.items()}
        previous_regions = prev_regions

        # Keep per-sample histories aligned with the observed round index.
        new_samples = set(rows) - all_samples
        for sample_id in new_samples:
            while len(region_history[sample_id]) < round_index:
                region_history[sample_id].append(None)
            while len(reliable_history[sample_id]) < round_index:
                reliable_history[sample_id].append(False)
            while len(applied_weight_history[sample_id]) < round_index:
                applied_weight_history[sample_id].append(0.0)
            while len(trained_history[sample_id]) < round_index:
                trained_history[sample_id].append(False)
        for sample_id in all_samples - set(rows):
            region_history[sample_id].append(None)
            reliable_history[sample_id].append(False)
            applied_weight_history[sample_id].append(0.0)
            trained_history[sample_id].append(False)
        for sample_id, row in rows.items():
            region_history[sample_id].append(row["region"])
            reliable_history[sample_id].append(_is_reliable(row, alpha, meta.config))
            applied_weight_history[sample_id].append(row["applied_weight"])
            trained_history[sample_id].append(row["trained"])
        all_samples.update(rows)

        counts = Counter(row["region"] for row in rows.values())
        total = len(rows)
        supported = sum(1 for row in rows.values() if row["evidence"] == "supported")
        boundary_supported = sum(
            1 for row in rows.values()
            if row["region"] == "boundary" and row["evidence"] == "supported"
        )
        boundary_unresolved = sum(
            1 for row in rows.values()
            if row["region"] == "boundary" and row["evidence"] != "supported"
        )

        transitions = Counter()
        entered = Counter()
        exited = Counter()

        for sample_id, row in rows.items():
            current = row["region"]
            previous = previous_regions.get(sample_id)
            if previous is None:
                transitions[(INITIAL, current)] += 1
                entered[current] += 1
            else:
                transitions[(previous, current)] += 1
                if previous != current:
                    entered[current] += 1
                    exited[previous] += 1
            if current == "boundary":
                if previous != "boundary":
                    spell_counter[sample_id] += 1
                    spells[sample_id] = {
                        "spell_index": spell_counter[sample_id],
                        "start_t": round_index,
                        "start_round": round_number,
                        "boundary_rounds": 0,
                        "start_row": row,
                        "start_alpha": alpha,
                        "last_row": row,
                        "last_alpha": alpha,
                    }
                spell = spells[sample_id]
                spell["boundary_rounds"] += 1
                spell["last_row"] = row
                spell["last_alpha"] = alpha
            else:
                if previous == "boundary" and sample_id in spells:
                    spell = spells.pop(sample_id)
                    event = _spell_event(meta, sample_id, spell, round_index, round_number,
                                         current, row, alpha, censored=False)
                    spell_rows.append(event)
                    if event["converted"]:
                        conversion_events.append(event)

        # Samples absent from a later log are an anomalous but possible state.
        missing_samples = set(previous_regions) - set(rows)
        for sample_id in sorted(missing_samples):
            previous = previous_regions[sample_id]
            transitions[(previous, MISSING)] += 1
            exited[previous] += 1
            if previous == "boundary" and sample_id in spells:
                spell = spells.pop(sample_id)
                event = _spell_event(meta, sample_id, spell, round_index, round_number,
                                     MISSING, None, None, censored=False)
                spell_rows.append(event)
        if missing_samples:
            warnings.append(f"round {round_number}: {len(missing_samples)} samples missing from log")

        # Round-level aggregated quantities.
        trained_rows = [row for row in rows.values() if row["trained"]]
        trained_weights = [row["applied_weight"] for row in trained_rows]
        trained_weights_all.extend(trained_weights)
        weight_sum = sum(trained_weights)
        weight_mean = _mean(trained_weights)
        weight_median = _median(trained_weights)
        weight_p90 = _percentile(trained_weights, 0.90)
        weight_square_sum = sum(weight * weight for weight in trained_weights)
        effective_sample_size = (
            weight_sum * weight_sum / weight_square_sum if weight_square_sum > 0 else None
        )

        new_conversion_ids = {
            sample_id for sample_id, row in rows.items()
            if previous_regions.get(sample_id) == "boundary" and row["region"] == "positive"
        }
        reliable_conversions = sum(
            1 for sample_id in new_conversion_ids
            if _is_reliable(rows[sample_id], alpha, meta.config)
        )
        new_conversion_weights.extend(
            rows[sample_id]["applied_weight"] for sample_id in new_conversion_ids
            if rows[sample_id]["trained"]
        )
        continuing_weights.extend(
            row["applied_weight"] for sample_id, row in rows.items()
            if row["region"] == "positive" and sample_id not in new_conversion_ids and row["trained"]
        )

        history_regions = report.get("regions") if isinstance(report.get("regions"), dict) else None
        history_match = None
        if history_regions is not None:
            history_match = all(
                int(history_regions.get(region, 0) or 0) == counts.get(region, 0)
                for region in REGIONS
            )
            if not history_match:
                region_mismatches += 1
        validation = report.get("validation_task") if isinstance(report.get("validation_task"), dict) else {}
        val_accuracy = _optional_float(validation.get("accuracy"))
        loss = _optional_float(report.get("loss"))
        history_coverage = _optional_float(report.get("coverage"))
        history_accepted = _optional_int(report.get("accepted_candidates"))

        round_rows.append({
            "run_id": meta.run_id, "task": meta.task, "variant": meta.variant,
            "seed": meta.seed, "round": round_number, "round_index": round_index,
            "alpha": alpha, "gate_open": alpha is not None, "total": total,
            "candidate_found": supported,
            "candidate_found_ratio": _safe_ratio(supported, total),
            "positive": counts.get("positive", 0),
            "negative": counts.get("negative", 0),
            "boundary": counts.get("boundary", 0),
            "conflict": counts.get("conflict", 0),
            "positive_ratio": _safe_ratio(counts.get("positive", 0), total),
            "negative_ratio": _safe_ratio(counts.get("negative", 0), total),
            "boundary_ratio": _safe_ratio(counts.get("boundary", 0), total),
            "conflict_ratio": _safe_ratio(counts.get("conflict", 0), total),
            "boundary_supported": boundary_supported,
            "boundary_unresolved": boundary_unresolved,
            "entered_positive": entered.get("positive", 0),
            "entered_negative": entered.get("negative", 0),
            "entered_boundary": entered.get("boundary", 0),
            "entered_conflict": entered.get("conflict", 0),
            "exited_positive": exited.get("positive", 0),
            "exited_negative": exited.get("negative", 0),
            "exited_boundary": exited.get("boundary", 0),
            "exited_conflict": exited.get("conflict", 0),
            "converted_boundary_to_positive": len(new_conversion_ids),
            "reliable_conversions": reliable_conversions,
            "sustained_conversions": 0,
            "reliable_sustained_conversions": 0,
            "trained_count": len(trained_rows),
            "trained_ratio": _safe_ratio(len(trained_rows), total),
            "weight_sum": weight_sum,
            "weight_mean": weight_mean,
            "weight_median": weight_median,
            "weight_p90": weight_p90,
            "new_conversion_weight_mean": _mean(
                [rows[sample_id]["applied_weight"] for sample_id in new_conversion_ids
                 if rows[sample_id]["trained"]]
            ),
            "continuing_weight_mean": _mean([
                row["applied_weight"] for sample_id, row in rows.items()
                if row["region"] == "positive" and sample_id not in new_conversion_ids
                and row["trained"]
            ]),
            "effective_sample_size": effective_sample_size,
            "val_accuracy": val_accuracy,
            "loss": loss,
            "history_coverage": history_coverage,
            "history_accepted_candidates": history_accepted,
            "history_regions_match": history_match,
        })

        from_totals = Counter()
        for (source, _destination), count in transitions.items():
            from_totals[source] += count
        for (source, destination), count in sorted(transitions.items()):
            transition_rows.append({
                "run_id": meta.run_id, "task": meta.task, "variant": meta.variant,
                "seed": meta.seed, "round": round_number, "round_index": round_index,
                "from_region": source, "to_region": destination, "count": count,
                "rate_given_from": _safe_ratio(count, from_totals[source]),
            })
        prev_regions = current_regions

    # Close spells still open at the end of the run as right-censored.
    if round_numbers:
        last_t = len(round_numbers) - 1
        last_round = round_numbers[-1]
        for sample_id, spell in sorted(spells.items()):
            event = _spell_event(meta, sample_id, spell, last_t, last_round, CENSORED,
                                 spell.get("last_row"), spell.get("last_alpha"), censored=True)
            spell_rows.append(event)
    round_index_of = {row["round"]: row["round_index"] for row in round_rows}
    round_by_index = {row["round_index"]: row for row in round_rows}

    # Persistence, eventual reliability and post-conversion exposure are only
    # known after the run.  "Reliable" at the conversion round is often false in
    # full's soft-acceptance regime (a freshly found candidate has stable_count=1),
    # so the analysis also asks whether the candidate becomes gate-certified later
    # while staying positive.
    for event in conversion_events:
        if not event["converted"]:
            continue
        conversion_t = round_index_of[event["end_round"]]
        sample_id = event["sample_id"]
        sequence = region_history[sample_id]
        reliable_flags = reliable_history[sample_id]
        end_t = len(sequence)
        window = sequence[conversion_t:conversion_t + persist_rounds]
        if len(window) < persist_rounds:
            sustained = "censored"
        elif all(region == "positive" for region in window):
            sustained = "yes"
        else:
            sustained = "no"
        event["sustained"] = sustained

        reliable_offsets = [
            offset for offset in range(0, end_t - conversion_t)
            if reliable_flags[conversion_t + offset]
        ]
        event["reliable_eventual"] = bool(reliable_offsets)
        event["first_reliable_offset"] = min(reliable_offsets) if reliable_offsets else None
        event["reliable_within_3"] = any(offset <= 2 for offset in reliable_offsets)
        event["reliable_within_5"] = any(offset <= 4 for offset in reliable_offsets)
        event["reliable_within_10"] = any(offset <= 9 for offset in reliable_offsets)
        eventual_sustained = False
        for offset in reliable_offsets:
            start = conversion_t + offset
            reliable_window = sequence[start:start + persist_rounds]
            if (len(reliable_window) == persist_rounds
                    and all(region == "positive" for region in reliable_window)):
                eventual_sustained = True
                break
        event["reliable_sustained"] = eventual_sustained

        weights = applied_weight_history[sample_id][conversion_t:]
        trained = trained_history[sample_id][conversion_t:]
        applied = [weight for weight, was_trained in zip(weights, trained) if was_trained]
        event["post_conversion_weight_sum"] = sum(applied)
        event["post_conversion_trained_rounds"] = len(applied)
        event["post_conversion_mean_weight"] = _mean(applied)
        row = round_by_index.get(conversion_t)
        if row is not None:
            if sustained == "yes":
                row["sustained_conversions"] += 1
            if eventual_sustained:
                row["reliable_sustained_conversions"] += 1

    for event in spell_rows:
        if event["converted"] and event["sustained"] is None:
            event["sustained"] = "censored"
            event["reliable_sustained"] = False

    run_summary = _summarize_run(meta, round_rows, spell_rows, all_samples,
                                 trained_weights_all, new_conversion_weights, continuing_weights,
                                 region_mismatches, missing_history, warnings)
    return RunAnalysis(meta=meta, round_rows=round_rows, transition_rows=transition_rows,
                       spell_rows=spell_rows, summary=run_summary, warnings=warnings)


def _summarize_run(meta, round_rows, spell_rows, all_samples,
                   trained_weights, new_conversion_weights, continuing_weights,
                   region_mismatches, missing_history, warnings):
    summary = {
        "run_id": meta.run_id, "task": meta.task, "variant": meta.variant,
        "seed": meta.seed, "method_version": meta.method_version,
        "status": meta.status, "smoke": meta.smoke,
        "n_rounds": len(round_rows), "n_samples": len(all_samples),
        "n_decision_rows": sum(row["total"] for row in round_rows),
        "region_count_mismatches": region_mismatches,
        "history_missing_rounds": len(missing_history),
        "warnings": list(warnings),
    }
    if not round_rows:
        for metric in SUMMARY_METRICS:
            summary.setdefault(metric, None)
        return summary

    def mean_metric(key):
        return _mean([row.get(key) for row in round_rows])

    summary["mean_positive_ratio"] = mean_metric("positive_ratio")
    summary["mean_negative_ratio"] = mean_metric("negative_ratio")
    summary["mean_boundary_ratio"] = mean_metric("boundary_ratio")
    summary["mean_conflict_ratio"] = mean_metric("conflict_ratio")
    summary["mean_candidate_found_ratio"] = mean_metric("candidate_found_ratio")
    summary["mean_boundary_supported_ratio"] = _mean([
        _safe_ratio(row["boundary_supported"], row["total"]) for row in round_rows
    ])
    summary["mean_boundary_unresolved_ratio"] = _mean([
        _safe_ratio(row["boundary_unresolved"], row["total"]) for row in round_rows
    ])
    pooled_total = sum(row["total"] for row in round_rows)
    pooled = {region: sum(row[region] for row in round_rows) for region in REGIONS}
    for region in REGIONS:
        summary[f"pooled_{region}_ratio"] = _safe_ratio(pooled[region], pooled_total)
    final = round_rows[-1]
    for region in REGIONS:
        summary[f"final_{region}_ratio"] = _safe_ratio(final[region], final["total"])
    summary["mean_training_coverage"] = mean_metric("trained_ratio")
    summary["final_training_coverage"] = final.get("trained_ratio")

    val_rows = [row for row in round_rows if row.get("val_accuracy") is not None]
    summary["mean_val_accuracy"] = _mean([row["val_accuracy"] for row in val_rows])
    summary["final_val_accuracy"] = val_rows[-1]["val_accuracy"] if val_rows else None
    if val_rows:
        best = max(val_rows, key=lambda row: row["val_accuracy"])
        summary["best_val_accuracy"] = best["val_accuracy"]
        summary["best_val_round"] = best["round"]
    else:
        summary["best_val_accuracy"] = None
        summary["best_val_round"] = None
    if len(val_rows) >= 2:
        area = 0.0
        for left, right in zip(val_rows, val_rows[1:]):
            area += ((left["val_accuracy"] + right["val_accuracy"]) / 2.0
                     * (right["round_index"] - left["round_index"]))
        summary["val_auc"] = area / max(1, val_rows[-1]["round_index"] - val_rows[0]["round_index"])
    else:
        summary["val_auc"] = None

    converted = [event for event in spell_rows if event["converted"]]
    censored = [event for event in spell_rows if event["censored"]]
    exited_negative = [event for event in spell_rows if event["exit_region"] == "negative"]
    exited_conflict = [event for event in spell_rows if event["exit_region"] == "conflict"]
    missing = [event for event in spell_rows if event["exit_region"] == MISSING]
    reliable = [event for event in converted if event["reliable"]]
    eventual_reliable = [event for event in converted if event["reliable_eventual"]]
    sustained = [event for event in converted if event["sustained"] == "yes"]
    reliable_sustained = [event for event in converted if event["reliable_sustained"]]
    started_unresolved = [event for event in spell_rows if event["start_evidence"] != "supported"]
    started_supported = [event for event in spell_rows if event["start_evidence"] == "supported"]
    conversions_unresolved = [event for event in converted if event["start_evidence"] != "supported"]
    reliable_unresolved = [event for event in conversions_unresolved if event["reliable"]]
    eventual_reliable_unresolved = [
        event for event in conversions_unresolved if event["reliable_eventual"]
    ]
    reliable_sustained_unresolved = [
        event for event in conversions_unresolved if event["reliable_sustained"]
    ]
    reliable_offsets = [
        event["first_reliable_offset"] for event in eventual_reliable
        if event["first_reliable_offset"] is not None
    ]

    summary.update({
        "spell_count": len(spell_rows),
        "converted_spells": len(converted),
        "censored_spells": len(censored),
        "exited_negative_spells": len(exited_negative),
        "exited_conflict_spells": len(exited_conflict),
        "missing_spells": len(missing),
        "conversion_rate_per_spell": _safe_ratio(len(converted), len(spell_rows)),
        "reliable_conversions": len(reliable),
        "reliable_conversion_share": _safe_ratio(len(reliable), len(converted)),
        "eventual_reliable_conversions": len(eventual_reliable),
        "eventual_reliable_share": _safe_ratio(len(eventual_reliable), len(converted)),
        "eventual_reliable_sustained_conversions": len(reliable_sustained),
        "eventual_reliable_sustained_share": _safe_ratio(len(reliable_sustained), len(converted)),
        "sustained_conversions": len(sustained),
        "sustained_share": _safe_ratio(len(sustained), len(converted)),
        "reliable_sustained_conversions": len(reliable_sustained),
        "reliable_sustained_share": _safe_ratio(len(reliable_sustained), len(converted)),
        "spells_started_unresolved": len(started_unresolved),
        "spells_started_supported": len(started_supported),
        "conversions_from_unresolved": len(conversions_unresolved),
        "reliable_conversions_from_unresolved": len(reliable_unresolved),
        "eventual_reliable_conversions_from_unresolved": len(eventual_reliable_unresolved),
        "eventual_reliable_sustained_conversions_from_unresolved": len(reliable_sustained_unresolved),
        "conversion_rate_from_unresolved": _safe_ratio(len(conversions_unresolved),
                                                        len(started_unresolved)),
        "reliable_conversion_rate_from_unresolved": _safe_ratio(len(reliable_unresolved),
                                                                 len(started_unresolved)),
        "eventual_reliable_conversion_rate_from_unresolved": _safe_ratio(
            len(eventual_reliable_unresolved), len(started_unresolved)
        ),
        "ever_boundary_samples": len({event["sample_id"] for event in spell_rows}),
        "ever_boundary_converted_samples": len({event["sample_id"] for event in converted}),
        "ever_boundary_reliable_samples": len({event["sample_id"] for event in reliable}),
        "ever_boundary_eventual_reliable_samples": len({
            event["sample_id"] for event in eventual_reliable
        }),
        "ever_boundary_reliable_sustained_samples": len({
            event["sample_id"] for event in reliable_sustained
        }),
        "end_in_boundary_samples": final["boundary"],
        "total_boundary_observations": pooled["boundary"],
        "boundary_round_share": _safe_ratio(pooled["boundary"], pooled_total),
    })
    ever_boundary = summary["ever_boundary_samples"]
    for metric in ("ever_boundary_conversion_rate", "ever_boundary_reliable_rate",
                   "ever_boundary_eventual_reliable_rate",
                   "ever_boundary_reliable_sustained_rate"):
        denominator = ever_boundary
        numerator = {
            "ever_boundary_conversion_rate": summary["ever_boundary_converted_samples"],
            "ever_boundary_reliable_rate": summary["ever_boundary_reliable_samples"],
            "ever_boundary_eventual_reliable_rate": summary[
                "ever_boundary_eventual_reliable_samples"
            ],
            "ever_boundary_reliable_sustained_rate": summary[
                "ever_boundary_reliable_sustained_samples"
            ],
        }[metric]
        summary[metric] = _safe_ratio(numerator, denominator)
    summary["end_boundary_share_of_ever_boundary"] = _safe_ratio(
        summary["end_in_boundary_samples"], ever_boundary
    )
    summary["mean_boundary_rounds_per_ever_boundary_sample"] = _safe_ratio(
        pooled["boundary"], ever_boundary
    )
    summary["mean_spells_per_ever_boundary_sample"] = _safe_ratio(len(spell_rows), ever_boundary)

    converted_waits = [event["boundary_rounds"] for event in converted]
    all_waits = [event["boundary_rounds"] for event in spell_rows]
    censored_waits = [event["boundary_rounds"] for event in censored]
    unresolved_waits = [event["boundary_rounds"] for event in conversions_unresolved]
    summary.update({
        "wait_converted_mean": _mean(converted_waits),
        "wait_converted_median": _median(converted_waits),
        "wait_converted_p25": _percentile(converted_waits, 0.25),
        "wait_converted_p90": _percentile(converted_waits, 0.90),
        "wait_converted_max": max(converted_waits) if converted_waits else None,
        "wait_all_mean": _mean(all_waits),
        "wait_censored_mean": _mean(censored_waits),
        "wait_unresolved_converted_mean": _mean(unresolved_waits),
    })
    for horizon in (1, 2, 3, 5, 10):
        within = sum(1 for wait in converted_waits if wait <= horizon)
        summary[f"converted_within_{horizon}_share"] = _safe_ratio(within, len(converted))
        summary[f"conversion_rate_within_{horizon}_of_spells"] = _safe_ratio(
            within, len(spell_rows)
        )
    summary["first_reliable_offset_mean"] = _mean(reliable_offsets)
    summary["first_reliable_offset_median"] = _median(reliable_offsets)
    summary["first_reliable_offset_p90"] = _percentile(reliable_offsets, 0.90)
    for horizon in (1, 2, 3, 5, 10):
        within = sum(1 for offset in reliable_offsets if offset <= horizon - 1)
        summary[f"eventual_reliable_within_{horizon}_share"] = _safe_ratio(
            within, len(converted)
        )

    summary["trained_count_total"] = len(trained_weights)
    summary["trained_weight_sum"] = sum(trained_weights)
    summary["trained_weight_pooled_mean"] = _mean(trained_weights)
    summary["trained_weight_pooled_median"] = _median(trained_weights)
    summary["trained_weight_pooled_p90"] = _percentile(trained_weights, 0.90)
    summary["trained_weight_max"] = max(trained_weights) if trained_weights else None
    summary["new_conversion_weight_mean"] = _mean(new_conversion_weights)
    summary["new_conversion_weight_median"] = _median(new_conversion_weights)
    summary["continuing_weight_mean"] = _mean(continuing_weights)
    summary["mean_effective_sample_size"] = mean_metric("effective_sample_size")
    post_sums = [event["post_conversion_weight_sum"] for event in converted
                 if event["post_conversion_weight_sum"] is not None]
    post_rounds = [event["post_conversion_trained_rounds"] for event in converted
                   if event["post_conversion_trained_rounds"] is not None]
    summary["post_conversion_weight_sum_mean"] = _mean(post_sums)
    summary["post_conversion_weight_sum_median"] = _median(post_sums)
    summary["post_conversion_trained_rounds_mean"] = _mean(post_rounds)

    for metric in SUMMARY_METRICS:
        summary.setdefault(metric, None)
    if summary.get("spell_count") == 0:
        message = "no boundary spells observed; three-region evolution is degenerate for this run"
        summary["warnings"].append(message)
        if message not in warnings:
            warnings.append(message)
    return summary


def _group_analyses(analyses):
    groups = defaultdict(list)
    for analysis in analyses:
        key = (analysis.meta.task, analysis.meta.variant, analysis.meta.method_version)
        groups[key].append(analysis)
    return groups


def summarize_metrics(analyses):
    groups = []
    for (task, variant, method_version), members in sorted(_group_analyses(analyses).items()):
        seeds = sorted(member.meta.seed for member in members)
        metrics = {}
        for name in SUMMARY_METRICS:
            values = [member.summary.get(name) for member in members]
            numeric = [value for value in values if isinstance(value, (int, float))
                       and not isinstance(value, bool)]
            metrics[name] = {
                "mean": _mean(numeric),
                "std": statistics.stdev(numeric) if len(numeric) > 1 else None,
                "ci95": _ci95(numeric),
                "min": min(numeric) if numeric else None,
                "max": max(numeric) if numeric else None,
                "n": len(numeric),
                "values": values,
            }
        groups.append({
            "task": task, "variant": variant, "method_version": method_version,
            "n_runs": len(members), "seeds": seeds,
            "run_ids": sorted(member.meta.run_id for member in members),
            "metrics": metrics,
        })
    return {
        "status": "REGION_EVOLUTION_SUMMARY_VALIDATION_ONLY",
        "groups": groups,
        "note": ("Validation-log-only descriptive summary. Reliable means gate-level "
                 "eligibility and lower >= alpha, not ground-truth correctness. "
                 "Formal test was not read."),
    }


def summarize_curves(analyses):
    groups = defaultdict(list)
    for analysis in analyses:
        for row in analysis.round_rows:
            key = (analysis.meta.task, analysis.meta.variant, analysis.meta.method_version,
                   row["round"])
            groups[key].append(row)
    rows = []
    for (task, variant, method_version, round_number), members in sorted(groups.items()):
        output = {
            "task": task, "variant": variant, "method_version": method_version,
            "round": round_number, "n_runs": len(members),
        }
        for field in CURVE_FIELDS:
            values = [member.get(field) for member in members
                      if member.get(field) is not None]
            output[f"{field}_mean"] = _mean(values)
            output[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(output)
    fields = ["task", "variant", "method_version", "round", "n_runs"]
    for field in CURVE_FIELDS:
        fields.extend([f"{field}_mean", f"{field}_std"])
    return fields, rows


def summarize_transitions(analyses):
    groups = defaultdict(lambda: {"count": 0, "runs": set(), "from_total": 0})
    for analysis in analyses:
        for row in analysis.transition_rows:
            key = (analysis.meta.task, analysis.meta.variant, analysis.meta.method_version,
                   row["from_region"], row["to_region"])
            groups[key]["count"] += row["count"]
            groups[key]["runs"].add(analysis.meta.run_id)
    from_totals = Counter()
    for (task, variant, method_version, source, _destination), record in groups.items():
        from_totals[(task, variant, method_version, source)] += record["count"]
    rows = []
    for key, record in sorted(groups.items()):
        task, variant, method_version, source, destination = key
        total = from_totals[(task, variant, method_version, source)]
        rows.append({
            "task": task, "variant": variant, "method_version": method_version,
            "from_region": source, "to_region": destination, "count": record["count"],
            "rate_given_from": _safe_ratio(record["count"], total),
            "n_runs": len(record["runs"]),
        })
    return rows


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def render_report(analyses, summary, manifest):
    groups = summary.get("groups", [])
    lines = [
        "# 三域演化分析报告（V10 full 基准）",
        "",
        f"- 生成时间：{manifest['created_at']}",
        f"- 输入 run：{manifest['n_runs']} 个；任务：{', '.join(manifest['tasks']) or '—'}；"
        f"seeds：{', '.join(str(seed) for seed in manifest['seeds']) or '—'}",
        f"- 方法版本：{', '.join(manifest['method_versions']) or '—'}",
        f"- 数据规模：{manifest['n_rounds']} 个 run-round，"
        f"{manifest['n_decision_rows']} 条逐样本决策",
        "- 正式 test：**未读取**；训练：**未执行**；本报告只分析已有 validation 日志。",
        "",
        "## 指标定义",
        "",
        "- **三域比例**：每轮当前候选落在 positive / negative / boundary 的样本占比；"
        "conflict 单独统计，不并入 negative。",
        "- **转移率**：相邻两轮同一 sample 的区域转移计数，"
        "`rate_given_from = count / 上一轮该区域总数`；第一轮以 `initial` 为源。",
        "- **等待轮数**：一个边界 spell 内连续处于 boundary 的观测轮数；"
        "对 B→P 事件，即转正前等待的轮数。run 结束时仍在边界的 spell 记为右删失。",
        "- **可靠修正**：候选在 positive 轮同时满足 `evidence=supported`、"
        "`eligible=True`、`lower >= alpha`；这是方法自身的门控级可靠性，"
        "不是离线真值正确率。报告分 **immediate**（转正轮即满足）与 "
        "**eventual**（转正后任一 positive 轮才满足）两种口径；"
        "full 的软接纳使 immediate 通常为 0，应主要看 eventual。",
        "- **持续修正**：转正轮及随后 `persist_rounds-1` 个观测轮均为 positive；"
        "若 run 提前结束则记为 censored，不计入持续。",
        "- **实际训练权重**：`trained=True` 行的 `effective_weight`（真正乘入伪标签损失的权重）；"
        "边界一致性损失使用单独的固定权重，不在该列中体现。",
        "- **学习曲线**：`history.jsonl` 的 validation task accuracy；"
        "区域比例与覆盖率按轮对齐。",
        "",
        "## 跨 seed 汇总",
        "",
    ]
    if groups:
        table_rows = []
        for group in groups:
            metrics = group["metrics"]
            def metric(name):
                return metrics.get(name, {}).get("mean")
            table_rows.append([
                group["task"], group["variant"], group["n_runs"],
                ", ".join(str(seed) for seed in group["seeds"]),
                metric("mean_positive_ratio"), metric("mean_boundary_ratio"),
                metric("mean_negative_ratio"), metric("mean_boundary_unresolved_ratio"),
                metric("conversion_rate_per_spell"), metric("reliable_conversion_share"),
                metric("eventual_reliable_share"),
                metric("eventual_reliable_sustained_share"), metric("wait_converted_mean"),
                metric("best_val_accuracy"),
            ])
        lines.append(_markdown_table(
            ["task", "variant", "runs", "seeds", "positive", "boundary", "negative",
             "boundary-unresolved", "B→P/spell", "immediate reliable/conv",
             "eventual reliable/conv", "eventual sustained/conv",
             "mean wait", "best val"],
            table_rows,
        ))
    else:
        lines.append("没有可汇总的 run。")
    degenerate = [analysis.meta.run_id for analysis in analyses
                  if analysis.summary.get("spell_count") == 0]
    if degenerate:
        lines += [
            "",
            f"> 注意：{len(degenerate)} 个 run 没有出现 boundary spell"
            "（可能是软接纳在该任务上从第 0 轮起就覆盖了全部 weak 样本）。"
            "这些 run 的三域演化在该任务上退化，等待/转移指标无定义或为 0；"
            "这不代表日志缺失，而是软接纳机制的结构性结果。",
        ]
    lines += ["", "## 等待与修正", ""]
    if groups:
        wait_rows = []
        for group in groups:
            metrics = group["metrics"]
            def metric(name):
                return metrics.get(name, {}).get("mean")
            wait_rows.append([
                group["task"], group["variant"],
                metric("wait_converted_mean"), metric("wait_converted_median"),
                metric("wait_converted_p90"), metric("wait_converted_max"),
                metric("converted_within_1_share"), metric("converted_within_3_share"),
                metric("converted_within_5_share"), metric("converted_within_10_share"),
                metric("ever_boundary_conversion_rate"),
                metric("eventual_reliable_conversion_rate_from_unresolved"),
            ])
        lines.append(_markdown_table(
            ["task", "variant", "mean", "median", "p90", "max",
             "≤1", "≤3", "≤5", "≤10", "ever-B conversion", "unresolved→eventual reliable"],
            wait_rows,
        ))
    lines += ["", "## 实际训练权重", ""]
    if groups:
        weight_rows = []
        for group in groups:
            metrics = group["metrics"]
            def metric(name):
                return metrics.get(name, {}).get("mean")
            weight_rows.append([
                group["task"], group["variant"],
                metric("trained_weight_pooled_mean"), metric("trained_weight_pooled_median"),
                metric("trained_weight_pooled_p90"), metric("new_conversion_weight_mean"),
                metric("continuing_weight_mean"), metric("mean_effective_sample_size"),
                metric("post_conversion_weight_sum_mean"),
                metric("post_conversion_trained_rounds_mean"),
            ])
        lines.append(_markdown_table(
            ["task", "variant", "mean", "median", "p90", "new B→P", "continuing",
             "ESS", "post-conv weight sum", "post-conv trained rounds"],
            weight_rows,
        ))
    lines += ["", "## 学习曲线", ""]
    if groups:
        curve_rows = []
        for group in groups:
            metrics = group["metrics"]
            def metric(name):
                return metrics.get(name, {}).get("mean")
            curve_rows.append([
                group["task"], group["variant"],
                metric("mean_val_accuracy"), metric("best_val_accuracy"),
                metric("final_val_accuracy"), metric("best_val_round"), metric("val_auc"),
                metric("mean_training_coverage"), metric("final_training_coverage"),
            ])
        lines.append(_markdown_table(
            ["task", "variant", "mean val", "best val", "final val", "best round",
             "val AUC", "mean train cover", "final train cover"],
            curve_rows,
        ))
    lines += ["", "## 逐 run 概览", ""]
    run_rows = []
    for analysis in analyses:
        item = analysis.summary
        run_rows.append([
            analysis.meta.run_id, analysis.meta.seed,
            item.get("mean_positive_ratio"), item.get("mean_boundary_ratio"),
            item.get("mean_negative_ratio"), item.get("spell_count"),
            item.get("converted_spells"), item.get("reliable_conversions"),
            item.get("wait_converted_mean"), item.get("best_val_accuracy"),
        ])
    lines.append(_markdown_table(
        ["run", "seed", "positive", "boundary", "negative", "spells", "conversions",
         "reliable", "mean wait", "best val"],
        run_rows,
    ))
    lines += [
        "",
        "## 解释边界",
        "",
        "1. V10 full 采用软接纳：所有找到支持的候选进入正域，未找到解释的样本留在边界。"
        "因此 full 的 negative 域通常为空，这是决策机制的结构性结果，不是日志缺失；"
        "本报告保留 negative/conflict 统计以检查是否出现异常。",
        "2. “可靠修正”只表示候选通过了方法的门控级可靠性条件"
        "（eligible 且 lower >= alpha），**不等于**离线真值正确。"
        "报告同时给出转正轮即通过的 immediate 口径与转正后若干轮内才通过的 eventual 口径："
        "full 的软接纳会让新找到支持的候选在 stable_count=1 时立即转正，"
        "因此 immediate 口径通常接近 0，应当以 eventual / eventual-sustained 为主要观察。"
        "若需要真值正确率，必须在离线审计中另接审计标签（见 `audit.py`），"
        "且不得回灌训练或选模。",
        "3. 决策日志每样本只记录当前 best candidate，不是全部候选解释；"
        "因此转移/等待统计的单位是“样本的最佳候选”，不是所有候选公式。",
        "4. 等待轮数是观测到的 boundary 轮数；run 结束时仍在边界的 spell 右删失，"
        "均值会低估真实等待，报告同时给出删失数。",
        "5. 权重是伪标签损失中的 effective weight；边界一致性损失使用固定权重，"
        "未包含在逐候选 `weight` 中。",
        "6. 全部结论仅基于 validation 日志，未读取正式 test，不能外推为 test 性能或因果效应。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        "py -m methods.frt_abl.region_evolution --runs methods/frt_abl/runs-v10 "
        "--output methods/frt_abl/runs-region-evolution-v10 --execute-analysis",
        "```",
        "",
        "先不加 `--execute-analysis` 只列出将要分析的 run；"
        "`--variant all` 可扩展到消融，`--allow-duplicate-seeds` 仅在刻意保留重复 run 时使用。",
        "",
    ]
    return "\n".join(lines)


def write_outputs(analyses, output, persist_rounds, overwrite=False,
                  skipped=None, duplicates=None):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    skipped = skipped or []
    duplicates = duplicates or {}

    round_rows = [row for analysis in analyses for row in analysis.round_rows]
    transition_rows = [row for analysis in analyses for row in analysis.transition_rows]
    spell_rows = [row for analysis in analyses for row in analysis.spell_rows]
    _write_csv(output / "region_rounds.csv", ROUND_FIELDS, round_rows)
    _write_csv(output / "transitions.csv", TRANSITION_FIELDS, transition_rows)
    _write_csv(output / "boundary_spells.csv", SPELL_FIELDS, spell_rows)

    summary = summarize_metrics(analyses)
    curve_fields, curve_rows = summarize_curves(analyses)
    _write_csv(output / "curve_summary.csv", curve_fields, curve_rows)
    _write_csv(output / "transition_summary.csv",
               ["task", "variant", "method_version", "from_region", "to_region",
                "count", "rate_given_from", "n_runs"],
               summarize_transitions(analyses))

    now = datetime.now(timezone(timedelta(hours=8)))
    manifest = {
        "status": "REGION_EVOLUTION_COMPLETED",
        "created_at": now.isoformat(timespec="seconds"),
        "code_file": Path(__file__).name,
        "code_sha256": digest(Path(__file__)),
        "source": "post-hoc decisions_round_*.csv + history.jsonl; no training executed",
        "output": str(output),
        "persist_rounds": persist_rounds,
        "n_runs": len(analyses),
        "n_rounds": len(round_rows),
        "n_decision_rows": sum(row["total"] for row in round_rows),
        "n_spells": len(spell_rows),
        "tasks": sorted({analysis.meta.task for analysis in analyses}),
        "seeds": sorted({analysis.meta.seed for analysis in analyses}),
        "method_versions": sorted({analysis.meta.method_version for analysis in analyses}),
        "run_ids": sorted(analysis.meta.run_id for analysis in analyses),
        "skipped_runs": skipped,
        "duplicate_seeds": {f"{key[0]}/{key[1]}/{key[2]}/seed-{key[3]}": value
                            for key, value in duplicates.items()},
        "formal_test_read": False,
        "training_executed": False,
        "note": ("Only existing validation logs were read. Reliable corrections are "
                 "gate-level proxies, not ground-truth accuracy. No formal test was read."),
    }
    warnings = {analysis.meta.run_id: analysis.warnings for analysis in analyses
                if analysis.warnings}
    manifest["warnings"] = warnings

    write_json(output / "summary.json", summary)
    write_json(output / "run_summary.json", {
        "status": "REGION_EVOLUTION_RUN_SUMMARIES",
        "persist_rounds": persist_rounds,
        "runs": [analysis.summary for analysis in analyses],
    })
    write_json(output / "analysis_manifest.json", manifest)
    (output / "REPORT.md").write_text(
        render_report(analyses, summary, manifest), encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+",
                        default=[str(PROJECT / "methods" / "frt_abl" / "runs-v10")],
                        help="Run directory, or a root containing run directories")
    parser.add_argument("--output",
                        default=str(PROJECT / "methods" / "frt_abl" /
                                    "runs-region-evolution-v10"))
    parser.add_argument("--variant", choices=("all",) + VARIANTS, default="full",
                        help="Variant filter; default is the V10 full baseline")
    parser.add_argument("--task", choices=("all", "mnist-add2", "hwf"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--persist-rounds", type=int, default=2,
                        help="Rounds (including conversion round) required for sustained")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Include failed/stopped/smoke runs (default: completed only)")
    parser.add_argument("--allow-duplicate-seeds", action="store_true",
                        help="Do not fail when the same task/variant/version/seed repeats")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing output directory")
    parser.add_argument("--execute-analysis", action="store_true",
                        help="Actually read logs and write outputs")
    args = parser.parse_args(argv)
    if args.persist_rounds < 1:
        raise SystemExit("--persist-rounds must be >= 1")

    runs, skipped = discover_runs(args.runs, variant=args.variant, task=args.task,
                                  seeds=args.seeds,
                                  include_incomplete=args.include_incomplete)
    duplicates = check_unique_seeds(runs, allow_duplicates=args.allow_duplicate_seeds)
    if not runs:
        raise SystemExit("No run directories matched the filters; nothing to analyze")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output directory already exists: {output}; use --overwrite")

    if not args.execute_analysis:
        plan = {
            "status": "PLANNED_NOT_EXECUTED",
            "n_runs": len(runs),
            "runs": [{"run_id": meta.run_id, "task": meta.task, "variant": meta.variant,
                      "seed": meta.seed, "method_version": meta.method_version,
                      "status": meta.status} for meta in runs],
            "skipped": skipped,
            "duplicate_seeds": {f"{key[0]}/{key[1]}/{key[2]}/seed-{key[3]}": value
                                for key, value in duplicates.items()},
            "output": str(output),
            "persist_rounds": args.persist_rounds,
            "note": "No files were written. Re-run with --execute-analysis after reviewing.",
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    started = time.perf_counter()
    analyses = []
    for index, meta in enumerate(runs, 1):
        print(f"[region-evolution] {index}/{len(runs)} {meta.run_id}", file=sys.stderr,
              flush=True)
        analyses.append(analyze_run(meta, persist_rounds=args.persist_rounds))
    manifest = write_outputs(analyses, output, args.persist_rounds,
                             overwrite=args.overwrite, skipped=skipped,
                             duplicates=duplicates)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "analysis_manifest.json", manifest)
    print(json.dumps({
        "status": "REGION_EVOLUTION_COMPLETED",
        "n_runs": len(analyses),
        "n_rounds": manifest["n_rounds"],
        "n_spells": manifest["n_spells"],
        "output": str(output),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
