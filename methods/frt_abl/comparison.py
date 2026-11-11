"""Offline comparison of FRT-ABL V10 with the five reproduced baselines.

Importing this module imports no torch module, launches no process and reads no
formal-test manifest or per-sample prediction.  It normalises already finished
native artifacts into one observation schema, aggregates them per strict
identity and writes descriptive tables with explicit comparability reasons.

Three layers:

protocol    ``configs/comparison.json``: seeds, task/metric vocabulary, the V10
            primary entry, one adoption recipe per baseline and optional
            re-run metadata (never launched implicitly).
adoption    read-only parsers; a missing artifact is reported as a blocker and
            is never replaced by a guessed number.
aggregation strict-identity entries (mean, sample SD, 95% CI, n, seeds),
            comparability reasons, a ranking gate and seed-paired deltas.

The project protocol forbids mixing native tasks, metrics, splits, data
protocols or supervision into one ranking.  The comparability table therefore
names every field that differs instead of silently ordering the rows.
"""
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .aggregate_budget import mean_ci, t_critical_95
from .prepare import PROJECT, write_json

PROTOCOL_VERSION = "frt-abl-comparison-v1"
REGISTERED_SEEDS = (0, 1, 2, 3, 4)
PRIMARY_METHOD = "frt_abl"
DEFAULT_CONFIG = PROJECT / "methods" / "frt_abl" / "configs" / "comparison.json"
DEFAULT_RUNS = PROJECT / "methods" / "frt_abl" / "runs-comparison-v10"
DEFAULT_OUTPUT = PROJECT / "methods" / "frt_abl" / "runs-summary-comparison-v10"
STATUS_CSV = PROJECT / "reproduction_status.csv"

# The six fields that decide whether two numbers may be read side by side.
IDENTITY_FIELDS = ("task", "metric", "split", "data_protocol", "supervision", "selection")
PRIMARY = "primary"
ADOPTED = "adopted"
HARNESS = "harness-run"


class ComparisonError(ValueError):
    """Raised when the comparison protocol or an input artifact is invalid."""


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S%z")


def resolve_path(value, project=PROJECT):
    path = Path(value)
    return path if path.is_absolute() else Path(project) / path


def relative_path(path, project=PROJECT):
    try:
        return str(Path(path).resolve().relative_to(Path(project).resolve())).replace("\\", "/")
    except ValueError:
        return str(Path(path)).replace("\\", "/")


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fraction(value, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{label} is not numeric: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ComparisonError(f"{label} is not finite: {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ComparisonError(f"{label} must be a fraction in [0, 1], got {value!r} "
                              "(percent values must be converted by the source recipe)")
    return value


def dotted_get(payload, dotted):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ComparisonError(f"field {dotted!r} is missing")
        value = value[part]
    return value


def parse_seed_list(text):
    """Parse the status CSV ``training_seeds`` column; UNRECORDED stays empty."""
    if text is None:
        return ()
    text = str(text).strip()
    if not text or text.upper() in {"UNRECORDED", "NA", "N/A", "NONE"}:
        return ()
    seeds = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            return ()
    return tuple(sorted(set(seeds)))


def load_status_rows(path=STATUS_CSV):
    path = Path(path)
    if not path.is_file():
        return []
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_status_row(rows, selector):
    if not selector:
        return None
    for row in rows:
        if (row.get("method") == selector["method"]
                and row.get("variant") == selector["variant"]):
            return row
    return None


# ---------------------------------------------------------------------------
# Native artifact parsers.  Every parser returns a list of
# ``(metric, value, extra)`` tuples; ``value`` may be None (no numeric result).
# ---------------------------------------------------------------------------
def parse_json_fields(spec, path):
    payload = load_json(path)
    parsed = []
    for metric, dotted in spec["fields"].items():
        parsed.append((metric, _fraction(dotted_get(payload, dotted), f"{path}:{dotted}"),
                       {"correct": None, "total": None}))
    return parsed


def parse_json_epoch_series(spec, path):
    payload = load_json(path)
    series = payload.get(spec["list_field"])
    if not isinstance(series, list) or not series:
        raise ComparisonError(f"{path}: empty or missing {spec['list_field']!r}")
    field = spec["value_field"]
    select = spec.get("select", "last")
    if select == "last":
        row = series[-1]
    elif select == "max":
        row = max(series, key=lambda item: item[field])
    else:
        raise ComparisonError(f"{path}: unsupported epoch selection {select!r}")
    extras = {"correct": row.get("correct"), "total": row.get("total"),
              "epoch": row.get("epoch"), "select": select,
              "peak": max(item[field] for item in series)}
    return [(spec["metric"], _fraction(row[field], f"{path}:{field}"), extras)]


def parse_formal_test_aggregate(spec, path):
    payload = load_json(path)
    rows = payload.get(spec.get("list_field", "formal_test"))
    if not isinstance(rows, list) or not rows:
        raise ComparisonError(f"{path}: empty or missing formal_test rows")
    correct_field = spec.get("correct_field", "correct")
    total_field = spec.get("total_field", "total")
    trial_field = spec.get("trial_field", "trial")
    totals = defaultdict(lambda: [0, 0])
    for row in rows:
        trial = row.get(trial_field)
        totals[trial][0] += int(row[correct_field])
        totals[trial][1] += int(row[total_field])
    trial_values = [correct / total for _, (correct, total) in sorted(totals.items()) if total]
    if not trial_values:
        raise ComparisonError(f"{path}: no non-empty formal-test trial")
    value = statistics.fmean(trial_values)
    extras = {"correct": None, "total": None, "trial_values": trial_values,
              "n_trials": len(trial_values),
              "note": "Trial values are decision trials sharing one seed, not independent seeds."}
    return [(spec["metric"], _fraction(value, f"{path}:trials"), extras)]


PARSERS = {
    "json_fields": parse_json_fields,
    "json_epoch_series": parse_json_epoch_series,
    "formal_test_aggregate": parse_formal_test_aggregate,
}


# ---------------------------------------------------------------------------
# Protocol validation
# ---------------------------------------------------------------------------
def validate_comparison_config(cfg):
    problems = []

    def need(condition, message):
        if not condition:
            problems.append(message)

    need(isinstance(cfg, dict), "comparison config must be a JSON object")
    if not isinstance(cfg, dict):
        return problems
    need(cfg.get("protocol_version") == PROTOCOL_VERSION,
         f"protocol_version must be {PROTOCOL_VERSION!r}")
    seeds = cfg.get("seeds")
    need(isinstance(seeds, list) and seeds, "seeds must be a non-empty list")
    if isinstance(seeds, list):
        need(all(isinstance(seed, int) and seed in REGISTERED_SEEDS for seed in seeds),
             "seeds must be integers in 0..4")
        need(len(set(seeds)) == len(seeds), "duplicate seeds are not independent repetitions")

    tasks = cfg.get("tasks")
    need(isinstance(tasks, dict) and tasks, "tasks must be a non-empty object")
    if isinstance(tasks, dict):
        for task, entry in tasks.items():
            need(isinstance(entry, dict), f"task {task!r} must be an object")
            if isinstance(entry, dict):
                for field in ("display", "family", "unit", "target"):
                    need(isinstance(entry.get(field), str) and entry[field],
                         f"task {task!r} needs a non-empty {field}")
                if "order" in entry:
                    need(isinstance(entry["order"], int), f"task {task!r} order must be an integer")

    metrics = cfg.get("metrics")
    need(isinstance(metrics, dict) and metrics, "metrics must be a non-empty object")
    if isinstance(metrics, dict):
        for metric, entry in metrics.items():
            need(isinstance(entry, dict), f"metric {metric!r} must be an object")
            if isinstance(entry, dict):
                need(entry.get("direction") in {"higher", "lower"},
                     f"metric {metric!r} direction must be higher or lower")
                for field in ("display", "definition"):
                    need(isinstance(entry.get(field), str) and entry[field],
                         f"metric {metric!r} needs a non-empty {field}")

    primary = cfg.get("primary")
    need(isinstance(primary, dict), "primary must be an object")
    if isinstance(primary, dict):
        need(primary.get("method") == PRIMARY_METHOD,
             f"primary.method must be {PRIMARY_METHOD!r}")
        for field in ("display", "method_version", "summary"):
            need(isinstance(primary.get(field), str) and primary[field],
                 f"primary needs a non-empty {field}")
        need(isinstance(primary.get("variants"), list) and primary["variants"],
             "primary.variants must be a non-empty list")
        for field in ("split", "data_protocol", "supervision", "selection"):
            need(isinstance(primary.get(field), str) and primary[field],
                 f"primary needs a non-empty {field}")
        overrides = primary.get("identity_by_task", {})
        need(isinstance(overrides, dict), "primary.identity_by_task must be an object")
        if isinstance(overrides, dict):
            for task, override in overrides.items():
                need(task in (tasks or {}), f"primary.identity_by_task task {task!r} is not declared")
                need(isinstance(override, dict), f"primary.identity_by_task[{task!r}] must be an object")
                if isinstance(override, dict):
                    for field, value in override.items():
                        need(field in ("split", "data_protocol", "supervision", "selection"),
                             f"primary.identity_by_task[{task!r}] field {field!r} is not an identity field")
                        need(isinstance(value, str) and value,
                             f"primary.identity_by_task[{task!r}].{field} must be a non-empty string")
        evaluations = primary.get("test_evaluations", [])
        need(isinstance(evaluations, list), "primary.test_evaluations must be a list")
        if isinstance(evaluations, list):
            for index, item in enumerate(evaluations):
                need(isinstance(item, dict), f"primary.test_evaluations[{index}] must be an object")
                if isinstance(item, dict):
                    need(isinstance(item.get("path"), str) and item["path"],
                         f"primary.test_evaluations[{index}] needs a path")
                    need(item.get("task") in (tasks or {}),
                         f"primary.test_evaluations[{index}] task is not declared")
                    need(item.get("metric", "task_accuracy") in (metrics or {}),
                         f"primary.test_evaluations[{index}] metric is not declared")

    baselines = cfg.get("baselines")
    need(isinstance(baselines, list) and baselines, "baselines must be a non-empty list")
    seen_methods = set()
    if isinstance(baselines, list):
        for baseline in baselines:
            need(isinstance(baseline, dict), "each baseline must be an object")
            if not isinstance(baseline, dict):
                continue
            method = baseline.get("method")
            need(isinstance(method, str) and method, "baseline.method must be a non-empty string")
            need(method != PRIMARY_METHOD, f"baseline {method!r} may not reuse the primary id")
            need(method not in seen_methods, f"duplicate baseline method {method!r}")
            seen_methods.add(method)
            for field in ("display", "paper"):
                need(isinstance(baseline.get(field), str) and baseline[field],
                     f"baseline {method!r} needs a non-empty {field}")
            adoptions = baseline.get("adoptions")
            need(isinstance(adoptions, list) and adoptions,
                 f"baseline {method!r} needs at least one adoption recipe")
            if not isinstance(adoptions, list):
                continue
            for index, spec in enumerate(adoptions):
                prefix = f"baseline {method!r} adoption[{index}]"
                if not isinstance(spec, dict):
                    problems.append(f"{prefix} must be an object")
                    continue
                need(spec.get("parser") in PARSERS or spec.get("parser") == "status_row",
                     f"{prefix} parser must be one of {sorted(PARSERS) + ['status_row']}")
                need(spec.get("task") in (tasks or {}), f"{prefix} task is not declared")
                for field in ("split", "data_protocol", "supervision", "selection"):
                    need(isinstance(spec.get(field), str) and spec[field],
                         f"{prefix} needs a non-empty {field}")
                need("path" in spec or "path_template" in spec,
                     f"{prefix} needs path or path_template")
                if "path_template" in spec:
                    need(bool(spec.get("seeds")),
                         f"{prefix} path_template requires a seeds list")
                adoption_seeds = spec.get("seeds")
                if adoption_seeds is not None:
                    need(isinstance(adoption_seeds, list) and adoption_seeds,
                         f"{prefix} seeds must be a non-empty list")
                    if isinstance(adoption_seeds, list):
                        # Native seeds are recorded as they were run (e.g. 1234 for
                        # Scallop); only the project's own repetitions are 0..4.
                        need(all(isinstance(seed, int) and seed >= 0 for seed in adoption_seeds),
                             f"{prefix} seeds must be non-negative integers")
                        need(len(set(adoption_seeds)) == len(adoption_seeds),
                             f"{prefix} duplicate seeds")
                    need("path_template" in spec,
                         f"{prefix} needs path_template when seeds are declared")
                if spec.get("parser") == "json_fields":
                    fields = spec.get("fields")
                    need(isinstance(fields, dict) and fields,
                         f"{prefix} json_fields needs a non-empty fields object")
                    if isinstance(fields, dict):
                        for metric in fields:
                            need(metric in (metrics or {}),
                                 f"{prefix} field metric {metric!r} is not declared")
                else:
                    need(spec.get("metric") in (metrics or {}),
                         f"{prefix} metric is not declared")
                declared = spec.get("seeds") or ([spec["seed"]] if "seed" in spec else [])
                for seed in declared:
                    need(seed is None or (isinstance(seed, int) and seed >= 0),
                         f"{prefix} seed must be None or a non-negative integer")
            status_csv = baseline.get("status_csv")
            if status_csv is not None:
                need(isinstance(status_csv, dict)
                     and isinstance(status_csv.get("method"), str)
                     and isinstance(status_csv.get("variant"), str),
                     f"baseline {method!r} status_csv needs method and variant strings")
    return problems


def load_comparison_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    try:
        cfg = load_json(path)
    except FileNotFoundError as error:
        raise ComparisonError(f"comparison config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ComparisonError(f"comparison config is not valid JSON: {path}: {error}") from error
    problems = validate_comparison_config(cfg)
    if problems:
        raise ComparisonError("invalid comparison protocol:\n- " + "\n- ".join(problems))
    return cfg


# ---------------------------------------------------------------------------
# Adoption of finished native artifacts
# ---------------------------------------------------------------------------
def _observation(baseline, spec, metric, value, seed, extra, source, note=None):
    return {
        "method": baseline["method"],
        "display": baseline["display"],
        "method_version": baseline.get("method_version"),
        "task": spec["task"],
        "metric": metric,
        "split": spec["split"],
        "data_protocol": spec["data_protocol"],
        "supervision": spec["supervision"],
        "selection": spec["selection"],
        "seed": seed,
        "value": value,
        "std": spec.get("std"),
        "n": spec.get("n"),
        "seeds": [seed] if seed is not None else list(spec.get("aggregate_seeds", [])),
        "correct": (extra or {}).get("correct"),
        "total": (extra or {}).get("total"),
        "status": "COMPLETE" if value is not None else "NO_NUMERIC_VALUE",
        "budget": spec.get("budget"),
        "label": spec.get("label"),
        "source": source,
        "provenance": ADOPTED,
        "note": note or spec.get("note"),
        "extra": {key: value_ for key, value_ in (extra or {}).items() if key != "note"},
        "blockers": list(baseline.get("blockers", [])) + list(spec.get("blockers", [])),
    }


def adopt_observations(cfg, project=PROJECT, status_rows=None, methods=None):
    """Read every declared native artifact; return (observations, issues)."""
    status_rows = load_status_rows() if status_rows is None else status_rows
    observations, issues = [], []
    wanted = set(methods) if methods else None
    for baseline in cfg["baselines"]:
        method = baseline["method"]
        if wanted is not None and method not in wanted:
            continue
        status_row = find_status_row(status_rows, baseline.get("status_csv"))
        for spec in baseline["adoptions"]:
            if spec["parser"] == "status_row":
                try:
                    _adopt_status_row(baseline, spec, status_row, observations, issues)
                except (ComparisonError, ValueError) as error:
                    issues.append({"kind": "PARSE_ERROR", "method": method,
                                   "path": relative_path(STATUS_CSV, project),
                                   "seed": spec.get("seed"), "note": str(error)})
                continue
            paths = []
            if "path_template" in spec:
                for seed in spec.get("seeds") or []:
                    paths.append((seed, resolve_path(spec["path_template"].format(seed=seed), project)))
            else:
                paths.append((spec.get("seed"), resolve_path(spec["path"], project)))
            for seed, path in paths:
                if not path.is_file():
                    issues.append({"kind": "MISSING_ARTIFACT", "method": method,
                                   "path": relative_path(path, project), "seed": seed,
                                   "note": spec.get("note")})
                    continue
                try:
                    parsed = PARSERS[spec["parser"]](spec, path)
                except (ComparisonError, OSError, ValueError) as error:
                    issues.append({"kind": "PARSE_ERROR", "method": method,
                                   "path": relative_path(path, project), "seed": seed,
                                   "note": str(error)})
                    continue
                for metric, value, extra in parsed:
                    observations.append(_observation(
                        baseline, spec, metric, value, seed, extra,
                        relative_path(path, project)))
    return observations, issues


def _adopt_status_row(baseline, spec, status_row, observations, issues):
    """Adopt a numeric aggregate straight from ``reproduction_status.csv``."""
    method = baseline["method"]
    if status_row is None:
        issues.append({"kind": "MISSING_STATUS_ROW", "method": method,
                       "path": relative_path(STATUS_CSV), "seed": None,
                       "note": "status_csv selector did not match a row"})
        return
    value = status_row.get("reproduction_mean") or None
    value = _fraction(float(value), f"{method}:reproduction_mean") if value else None
    std = status_row.get("reproduction_std") or None
    std = float(std) if std else None
    completed = status_row.get("completed_repetitions") or None
    extra = {"correct": None, "total": None}
    observation = _observation(
        baseline, spec, spec["metric"], value, spec.get("seed"), extra,
        relative_path(STATUS_CSV), note=status_row.get("current_blocker"))
    observation["std"] = std
    observation["n"] = int(completed) if str(completed).isdigit() else None
    observation["seeds"] = list(parse_seed_list(status_row.get("training_seeds")))
    observation["status"] = ("COMPLETE" if value is not None
                             else "NO_NUMERIC_VALUE")
    observations.append(observation)


def primary_observations(cfg, project=PROJECT):
    """V10 validation aggregates plus any explicitly registered test evaluation."""
    spec = cfg["primary"]
    observations, issues = [], []
    summary_path = resolve_path(spec["summary"], project)
    if not summary_path.is_file():
        issues.append({"kind": "MISSING_ARTIFACT", "method": PRIMARY_METHOD,
                       "path": relative_path(summary_path, project), "seed": None,
                       "note": "V10 aggregate summary not found"})
    else:
        summary = load_json(summary_path)
        for group in summary.get("groups", []):
            if group.get("variant") not in spec["variants"]:
                continue
            value = group.get("mean_validation_accuracy")
            if value is None or not group.get("n_with_accuracy"):
                continue
            identity = dict(spec.get("identity_by_task", {}).get(group["task"], {}))
            observations.append({
                "method": PRIMARY_METHOD, "display": spec["display"],
                "method_version": spec.get("method_version"),
                "task": group["task"], "metric": "task_accuracy",
                "split": identity.get("split", spec["split"]),
                "data_protocol": identity.get("data_protocol", spec["data_protocol"]),
                "supervision": identity.get("supervision", spec["supervision"]),
                "selection": identity.get("selection", spec["selection"]),
                "seed": None, "value": _fraction(value, "mean_validation_accuracy"),
                "std": group.get("std_validation_accuracy"),
                "n": group.get("n_with_accuracy"),
                "seeds": list(group.get("seeds", [])),
                "correct": None, "total": None,
                "status": "COMPLETE", "budget": None,
                "source": relative_path(summary_path, project),
                "provenance": PRIMARY,
                "note": f"variant={group.get('variant')}; validation only; run_ids kept in the summary",
                "extra": {"variant": group.get("variant")}, "blockers": [],
            })
    for index, item in enumerate(spec.get("test_evaluations", [])):
        path = resolve_path(item["path"], project)
        if not path.is_file():
            issues.append({"kind": "MISSING_ARTIFACT", "method": PRIMARY_METHOD,
                           "path": relative_path(path, project), "seed": item.get("seed"),
                           "note": f"registered test evaluation[{index}] not found"})
            continue
        payload = load_json(path)
        field = item.get("field", "task_accuracy")
        observations.append({
            "method": PRIMARY_METHOD, "display": spec["display"],
            "method_version": spec.get("method_version"),
            "task": item["task"], "metric": item.get("metric", "task_accuracy"),
            "split": item.get("split", "test"),
            "data_protocol": item.get("data_protocol", spec["data_protocol"]),
            "supervision": item.get("supervision", spec["supervision"]),
            "selection": item.get("selection", "validation-selected"),
            "seed": item.get("seed"), "value": _fraction(dotted_get(payload, field), field),
            "std": None, "n": 1, "seeds": [item["seed"]] if item.get("seed") is not None else [],
            "correct": payload.get("task_correct"), "total": payload.get("task_total"),
            "status": "COMPLETE", "budget": None,
            "source": relative_path(path, project), "provenance": PRIMARY,
            "note": "registered formal-test evaluation", "extra": {}, "blockers": [],
        })
    return observations, issues


# ---------------------------------------------------------------------------
# Harness-normalised runs
# ---------------------------------------------------------------------------
RUN_REQUIRED = ("method", "task", "metric", "split", "data_protocol", "supervision",
                "selection", "value")


def collect_run_observations(runs_root=DEFAULT_RUNS, cfg=None, project=PROJECT):
    """Read every harness-normalised ``runs/*/metrics.json`` (never adopted copies)."""
    root = Path(runs_root)
    observations, issues = [], []
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
            payload = None
            try:
                payload = load_json(metrics_path)
                missing = [key for key in RUN_REQUIRED if key not in payload]
                if missing:
                    raise ComparisonError(f"missing fields: {', '.join(missing)}")
                payload["value"] = _fraction(payload["value"], str(metrics_path))
            except (ComparisonError, OSError, ValueError, KeyError) as error:
                issues.append({"kind": "INVALID_RUN", "method": payload.get("method")
                               if isinstance(payload, dict) else None,
                               "path": relative_path(metrics_path, project),
                               "seed": None, "note": str(error)})
                continue
            payload.setdefault("display", payload["method"])
            payload.setdefault("method_version", None)
            payload.setdefault("seed", None)
            payload.setdefault("std", None)
            payload.setdefault("n", 1)
            payload.setdefault("seeds", [payload["seed"]] if payload["seed"] is not None else [])
            payload.setdefault("correct", None)
            payload.setdefault("total", None)
            payload.setdefault("status", "COMPLETE")
            payload.setdefault("budget", None)
            payload.setdefault("source", relative_path(metrics_path, project))
            payload.setdefault("provenance", HARNESS)
            payload.setdefault("note", None)
            payload.setdefault("extra", {})
            payload.setdefault("blockers", [])
            observations.append(payload)
    return observations, issues


def write_adopted(runs_root=DEFAULT_RUNS, cfg=None, project=PROJECT, overwrite=False,
                  methods=None):
    """Persist the adopted observations with per-source hashes; no training."""
    observations, issues = adopt_observations(cfg or load_comparison_config(), project,
                                              methods=methods)
    root = Path(runs_root)
    target = root / "adopted" / "observations.json"
    if target.exists() and not overwrite:
        raise ComparisonError(f"{relative_path(target, project)} exists; pass --overwrite to rebuild")
    sources = {}
    for observation in observations:
        source = observation.get("source")
        if not source:
            continue
        path = resolve_path(source, project)
        if path.is_file() and source not in sources:
            sources[source] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
    payload = {
        "status": "ADOPTED_FROM_EXISTING_ARTIFACTS",
        "protocol_version": PROTOCOL_VERSION,
        "created_at": timestamp(),
        "note": ("Read-only adoption of finished native baseline artifacts. "
                 "No model was executed and no formal test manifest was read here."),
        "sources": sources,
        "issues": issues,
        "observations": observations,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, payload)
    return payload


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _entry_key(observation):
    return (observation["method"], observation.get("method_version"),
            *(observation[field] for field in IDENTITY_FIELDS))


def build_entries(observations):
    """Aggregate observations per strict identity; duplicate seeds are excluded."""
    grouped = defaultdict(list)
    for observation in observations:
        grouped[_entry_key(observation)].append(observation)
    entries = []
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(part) for part in pair[0])):
        entry = {field: key[2 + index] for index, field in enumerate(IDENTITY_FIELDS)}
        entry.update({"method": key[0], "method_version": key[1],
                      "display": items[0].get("display", key[0])})
        labels = sorted({item.get("label") for item in items if item.get("label")})
        entry["label"] = labels[0] if len(labels) == 1 else None
        if entry["label"]:
            entry["display"] = f"{entry['display']} · {entry['label']}"
        entry["sources"] = sorted({item.get("source") for item in items if item.get("source")})
        entry["provenance"] = sorted({item.get("provenance", ADOPTED) for item in items})
        entry["budget"] = next((item.get("budget") for item in items if item.get("budget")), None)
        entry["notes"] = sorted({item.get("note") for item in items if item.get("note")})
        entry["blockers"] = sorted({blocker for item in items for blocker in item.get("blockers", [])})
        entry["duplicate_seeds_excluded"] = []
        seeded = [item for item in items if item.get("seed") is not None]
        counts = Counter(item["seed"] for item in seeded)
        duplicates = sorted(seed for seed, count in counts.items() if count > 1)
        if duplicates:
            entry["duplicate_seeds_excluded"] = duplicates
            seeded = [item for item in seeded if counts[item["seed"]] == 1]
        unseeded = [item for item in items if item.get("seed") is None]
        numeric_seeded = [item for item in seeded if item.get("value") is not None]
        numeric_aggregate = [item for item in unseeded if item.get("value") is not None]
        per_seed = {}
        if numeric_seeded:
            per_seed = {item["seed"]: item["value"] for item in numeric_seeded}
        entry["per_seed_values"] = {str(seed): value for seed, value in sorted(per_seed.items())}
        if len(numeric_seeded) >= 2:
            values = [numeric_seeded_item["value"] for numeric_seeded_item in numeric_seeded]
            value, std, ci = mean_ci(values)
            entry.update({"value": value, "std": std, "ci95": ci,
                          "n": len(values), "seeds": sorted(per_seed),
                          "n_mode": "seeds"})
        elif len(numeric_seeded) == 1 and numeric_aggregate:
            item = numeric_aggregate[0]
            entry.update({"value": item.get("value"), "std": item.get("std"),
                          "n": item.get("n"), "seeds": sorted(per_seed),
                          "ci95": _ci_from(item.get("std"), item.get("n")),
                          "n_mode": "mixed"})
        elif numeric_seeded:
            item = numeric_seeded[0]
            entry.update({"value": item["value"], "std": None, "ci95": None, "n": 1,
                          "seeds": list(per_seed), "n_mode": "single-seed"})
        elif len(numeric_aggregate) >= 2:
            # Several complete runs of the same unseeded configuration.  They are
            # repetitions of a run, not registered seeds; the seed list stays empty.
            values = [item["value"] for item in numeric_aggregate]
            value, std, ci = mean_ci(values)
            entry.update({"value": value, "std": std, "ci95": ci, "n": len(values),
                          "seeds": sorted({seed for item in numeric_aggregate
                                           for seed in item.get("seeds", [])}),
                          "n_mode": "unseeded-runs"})
        elif numeric_aggregate:
            item = numeric_aggregate[0]
            entry.update({"value": item.get("value"), "std": item.get("std"),
                          "n": item.get("n") or len(item.get("seeds", [])) or 1,
                          "seeds": item.get("seeds", []),
                          "ci95": _ci_from(item.get("std"), item.get("n")),
                          "n_mode": "aggregate"})
        else:
            entry.update({"value": None, "std": None, "ci95": None, "n": None,
                          "seeds": [], "n_mode": "no-numeric-value"})
        trials = next((item["extra"].get("trial_values") for item in items
                       if (item.get("extra") or {}).get("trial_values")), None)
        entry["trial_values"] = trials
        entry["status"] = ("COMPLETE" if entry["value"] is not None else "NO_NUMERIC_VALUE")
        entries.append(entry)
    return entries


def _ci_from(std, n):
    if std is None or not n or n < 2:
        return None
    return t_critical_95(n - 1) * std / math.sqrt(n)


def compare(entries, task_order=None, metric_order=None):
    """Group by (task, metric), compare each member with the primary reference."""
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["task"], entry["metric"])].append(entry)
    task_order = task_order or {}
    metric_order = metric_order or {}

    def group_key(item):
        # ``sorted(dict.items(), key=...)`` receives (key, value) pairs.
        (task, metric), _members = item
        return (task_order.get(task, 1_000_000), metric_order.get(metric, 1_000_000),
                task, metric)

    groups = []
    for (task, metric), members in sorted(grouped.items(), key=group_key):
        reference = next((item for item in members if item["method"] == PRIMARY_METHOD), None)
        rows = []
        for member in sorted(members, key=lambda item: (item["method"] != PRIMARY_METHOD,
                                                        item["method"],
                                                        -(item["value"] or 0.0))):
            if reference is None:
                differences = []
                comparability = "NO_REFERENCE"
            elif member is reference:
                differences = []
                comparability = "REFERENCE"
            else:
                differences = [field for field in IDENTITY_FIELDS[2:]
                               if member[field] != reference[field]]
                comparability = "IDENTICAL" if not differences else "PROTOCOL_DIFFERS"
            member = dict(member)
            member["comparability"] = comparability
            member["differences"] = differences
            member["rankable"] = comparability in {"IDENTICAL", "REFERENCE"} and member["value"] is not None
            member["delta_vs_reference"] = (None if comparability != "IDENTICAL"
                                            or member["value"] is None or reference is None
                                            or reference["value"] is None
                                            else member["value"] - reference["value"])
            rows.append(member)
        rankable = [row for row in rows if row["rankable"]]
        if reference is None:
            ranking_status = "NO_REFERENCE"
        elif len(rankable) >= 2:
            ranking_status = "RANKABLE"
        elif len(rows) == 1:
            ranking_status = "SINGLE_METHOD"
        elif all(row["comparability"] == "PROTOCOL_DIFFERS" for row in rows
                 if row["method"] != PRIMARY_METHOD):
            ranking_status = "NOT_RANKABLE_PROTOCOL_DIFFERS"
        else:
            ranking_status = "NOT_RANKABLE"
        groups.append({"task": task, "metric": metric, "reference": reference["method"]
                       if reference else None, "members": rows,
                       "ranking_status": ranking_status,
                       "difference_fields": sorted({field for row in rows
                                                    for field in row["differences"]})})
    deltas = paired_deltas(groups)
    return {"groups": groups, "paired_deltas": deltas,
            "n_groups": len(groups),
            "n_entries": len(entries),
            "ranking_status": ("RANKABLE" if any(group["ranking_status"] == "RANKABLE"
                                                 for group in groups)
                               else "NOT_ESTABLISHED")}


def paired_deltas(groups):
    """Seed-paired deltas only inside strict-identity groups with >= 2 seeds."""
    records = []
    for group in groups:
        reference = next((row for row in group["members"]
                          if row["comparability"] == "REFERENCE"), None)
        if reference is None:
            continue
        for row in group["members"]:
            if row is reference or row["comparability"] != "IDENTICAL":
                continue
            common = sorted(set(reference["per_seed_values"]) & set(row["per_seed_values"]))
            if len(common) < 2:
                continue
            pairs = [{"seed": int(seed),
                      "reference": reference["per_seed_values"][seed],
                      "member": row["per_seed_values"][seed],
                      "delta": row["per_seed_values"][seed] - reference["per_seed_values"][seed]}
                     for seed in common]
            deltas = [pair["delta"] for pair in pairs]
            mean = statistics.fmean(deltas)
            std = statistics.stdev(deltas)
            records.append({
                "task": group["task"], "metric": group["metric"],
                "reference": reference["method"], "member": row["method"],
                "paired_seeds": [int(seed) for seed in common],
                "pairs": pairs, "mean_delta": mean, "std_delta": std,
                "ci95": t_critical_95(len(deltas) - 1) * std / math.sqrt(len(deltas)),
                "strictly_positive_on_all_paired_seeds": all(value > 0 for value in deltas),
                "strictly_negative_on_all_paired_seeds": all(value < 0 for value in deltas),
                "note": "Descriptive only; ties and losses retained; n<5 is not a test.",
            })
    return records


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def _pct(value, digits=2):
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _write_csv(path, fieldnames, rows):
    import csv
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_comparison(output=DEFAULT_OUTPUT, cfg=None, project=PROJECT,
                     runs_root=DEFAULT_RUNS, overwrite=False):
    """Collect, aggregate and write the comparison tables and report."""
    cfg = cfg or load_comparison_config()
    output = Path(output)
    if output.exists() and not overwrite:
        raise ComparisonError(f"{relative_path(output, project)} exists; pass --overwrite to rebuild")
    primary, primary_issues = primary_observations(cfg, project)
    adopted, adopted_issues = adopt_observations(cfg, project)
    harness, harness_issues = collect_run_observations(runs_root, cfg, project)
    issues = primary_issues + adopted_issues + harness_issues
    observations = primary + adopted + harness
    entries = build_entries(observations)
    known_tasks = set(cfg["tasks"])
    unknown = sorted({entry["task"] for entry in entries if entry["task"] not in known_tasks})
    if unknown:
        issues.append({"kind": "UNDECLARED_TASK", "method": None, "path": None,
                       "seed": None, "note": f"tasks not in protocol: {unknown}"})
    comparison = compare(entries,
                         {task: entry.get("order", 1_000_000) for task, entry in cfg["tasks"].items()},
                         {metric: index for index, metric in enumerate(cfg["metrics"])})
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "COMPARISON_AGGREGATED_DESCRIPTIVE_ONLY",
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": timestamp(),
        "runs_root": relative_path(runs_root, project),
        "n_observations": len(observations),
        "n_entries": len(entries),
        "ranking_status": comparison["ranking_status"],
        "primary_note": ("V10 rows are validation aggregates; no formal test was read. "
                         "Register test evaluations explicitly after approval."),
        "entries": entries,
        "groups": comparison["groups"],
        "paired_deltas": comparison["paired_deltas"],
        "issues": issues,
    }
    write_json(output / "summary.json", summary)
    fieldnames = ["task", "metric", "method", "display", "method_version", "value", "std",
                  "ci95", "n", "seeds", "split", "data_protocol", "supervision", "selection",
                  "comparability", "rankable", "delta_vs_reference", "budget", "source",
                  "status", "note"]
    rows = []
    for group in comparison["groups"]:
        for member in group["members"]:
            rows.append({**member, "seeds": ";".join(str(seed) for seed in member["seeds"]),
                         "source": ";".join(member["sources"]),
                         "note": "; ".join(member["notes"]),
                         "rankable": member["rankable"]})
    _write_csv(output / "comparison.csv", fieldnames, rows)
    _write_csv(output / "comparability.csv",
               ["task", "metric", "method", "display", "comparability", "differences",
                "ranking_status"],
               [{"task": group["task"], "metric": group["metric"], "method": member["method"],
                 "display": member["display"],
                 "comparability": member["comparability"],
                 "differences": ";".join(member["differences"]),
                 "ranking_status": group["ranking_status"]}
                for group in comparison["groups"] for member in group["members"]])
    write_report(output / "report.md", summary, cfg)
    return summary


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def write_report(path, summary, cfg):
    tasks = cfg["tasks"]
    metrics = cfg["metrics"]
    lines = [
        "# FRT-ABL V10 与五种复现基线的对比实验",
        "",
        f"协议：`{PROTOCOL_VERSION}`　状态：`{summary['status']}`　生成：{summary['generated_at']}",
        "",
        "> 本文件由 `methods.frt_abl.run_comparison --summarize-only` 离线生成。"
        "没有启动训练、没有运行模型、没有读取正式 test 清单或逐样本预测；"
        "所有基线数字都来自各方法窗口已经完成并登记的原生产物。",
        "",
        "## 1. 口径",
        "",
        "- **身份六元组**：任务、指标、划分、数据协议、监督、选模规则。"
        "只有六项全相同才允许进入同一排名；否则只能并排展示并标注差异。",
        "- **主方法（V10）**：当前只有 validation 聚合（`runs-summary-v10-5seed`），"
        "正式 test 需在放行后由 `evaluate.py` 逐个登记到 `primary.test_evaluations`。",
        "- **五种基线**：A³BL、DOLPHIN、DeepProbLog、Scallop、ABL-HED 的原生结果，"
        "按各方法自己的任务/数据/选模口径采纳；不改写原始文件。",
        "- **统计**：n≥2 时给出样本标准差（ddof=1）与 95% t 区间；单 seed 只报点值。",
        "",
        "### 方法与采纳来源",
        "",
    ]
    lines += _markdown_table(
        ["方法", "论文", "任务", "阻断项"],
        [[baseline["display"], baseline["paper"],
          "、".join(sorted({spec["task"] for spec in baseline["adoptions"]})),
          str(len(baseline.get("blockers", [])))]
         for baseline in cfg["baselines"]])
    lines += [
        "",
        "## 2. 主表",
        "",
    ]
    if not summary["groups"]:
        lines.append("_没有可汇总的观测。_")
    for group in summary["groups"]:
        task = tasks.get(group["task"], {"display": group["task"]})
        metric = metrics.get(group["metric"], {"display": group["metric"]})
        lines += ["", f"### {task['display']} · {metric['display']}", ""]
        rows = []
        for member in group["members"]:
            rows.append([
                member["display"],
                _pct(member["value"]),
                _number(member["std"], 4) if member["std"] is not None else "—",
                member["n"] if member["n"] is not None else "—",
                ",".join(str(seed) for seed in member["seeds"]) or "—",
                member["split"],
                member["data_protocol"],
                member["supervision"],
                member["selection"],
                member["comparability"],
            ])
        lines += _markdown_table(
            ["方法", "值", "SD", "n", "seeds", "划分", "数据协议", "监督", "选模", "可比性"],
            rows)
        lines += ["", f"排名状态：`{group['ranking_status']}`　"
                      f"参考方法：`{group['reference'] or '—'}`　"
                      f"差异字段：{', '.join(group['difference_fields']) or '—'}", ""]
        for member in group["members"]:
            details = []
            if member["comparability"] == "PROTOCOL_DIFFERS":
                details.append("差异字段：" + (", ".join(member["differences"]) or "—"))
            if member["budget"]:
                details.append(member["budget"])
            details.extend(member["notes"])
            details.append("证据 `" + "; ".join(member["sources"]) + "`")
            lines.append(f"- {member['display']}：" + "｜".join(details))
        if group["ranking_status"] == "NOT_RANKABLE_PROTOCOL_DIFFERS":
            lines += ["", "> 该组不允许写成名次或“优于/劣于”。差异字段未消除前，"
                          "数值只能作为方向性参考。"]
        elif group["ranking_status"] == "NO_REFERENCE":
            lines += ["", "> 该组没有主方法参照，只列已有原生结果，不构成对比结论。"]
    lines += ["", "## 3. 配对差值", ""]
    if summary["paired_deltas"]:
        rows = [[item["task"], item["metric"], item["reference"], item["member"],
                 _pct(item["mean_delta"]), ",".join(str(seed) for seed in item["paired_seeds"])]
                for item in summary["paired_deltas"]]
        lines += _markdown_table(["任务", "指标", "参考", "对比", "均值差", "配对 seed"], rows)
        lines += ["", "> 配对差值只在六元组完全相同的组内计算；"
                      "单 seed 或协议不同的组不产生差值。"]
    else:
        lines.append("_当前没有任何严格同身份的 ≥2 seed 组，因此没有配对差值。_")
    lines += ["", "## 4. 阻断与缺失", ""]
    if summary["issues"]:
        rows = [[item.get("kind"), item.get("method") or "—", item.get("path") or "—",
                 item.get("seed") if item.get("seed") is not None else "—",
                 (item.get("note") or "")[:120]]
                for item in summary["issues"]]
        lines += _markdown_table(["类型", "方法", "路径", "seed", "说明"], rows)
    else:
        lines.append("_没有阻断项。_")
    lines += [
        "", "## 5. 解释边界", "",
        "1. 五种基线与主方法当前**没有统一任务**的正式结果：A³BL/DeepProbLog/Scallop 是各自原生"
        "数据管线，DOLPHIN 是 5/10/15 图求和，ABL-HED 是四符号方程；主方法使用 COMMON "
        "`mnist-add2-v1` 清单。",
        "2. 主方法使用额外逐数字符号标注（`METHOD_LABEL_BUDGET`），"
        "基线为仅和标签弱监督；两者不是同一监督预算。",
        "3. 主方法当前只有 validation 选模结果；与基线的原生 test 数值不可直接互比。",
        "4. 任何排名声明都需要先冻结同一任务、同一划分、同一指标、同一监督与同一选模规则，"
        "并把五方法适配到 COMMON 清单；本表不提供该结论。",
        "",
        "## 6. 复现入口",
        "",
        "```powershell",
        "py -m methods.frt_abl.run_comparison",
        "py -m methods.frt_abl.run_comparison --execute-adoption",
        "py -m methods.frt_abl.run_comparison --summarize-only",
        "```",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
