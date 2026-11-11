"""One-factor-at-a-time parameter sensitivity around the frozen v10 `full` baseline.

The protocol is declared in ``configs/sensitivity.json``: neighbour count,
graph bandwidth, gate risk ceiling, curriculum ramp (speed) and boundary-loss
weight, plus one optional knob (curriculum start weight).  Every grid contains
the frozen baseline value, and every arm changes exactly one parameter of the
frozen v10 ``full`` configuration.  An arm is therefore only readable as a
paired comparison against the baseline arm trained inside the same campaign
(same source manifest, same environment, same prepared data manifest).

Nothing in this module trains, imports torch, reads a prepared manifest or
touches the formal test split.  ``run_sensitivity.py`` is the plan/execute entry
point; this module validates the protocol, materialises arm configs, and
aggregates completed training runs.

Decision rule for "a broad valid parameter region" (predeclared, not tuned):

* accuracy equality margin ``analysis.accuracy_margin`` (default 0.005);
* a level is ``within_margin`` when its paired mean delta vs baseline is not
  worse than ``-margin``;
* a level is ``non_inferior`` when the lower 95% CI bound of that paired delta
  is above ``-margin`` (needs >= 2 paired seeds);
* ``valid`` = ``within_margin``, coverage not collapsed (missing coverage fails
  closed) and the level has a same-seed baseline pair;
* ``confirmed_valid`` = ``valid`` and ``non_inferior`` (inclusive at the margin,
  float-safe);
* a *plateau* is a contiguous block of levels that contains the baseline level,
  has at least ``analysis.min_block_levels`` levels and covers at least
  ``analysis.min_block_span`` of the tested level range.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .policies import METHOD_VERSION
from .prepare import PROJECT, write_json

PROTOCOL_VERSION = "frt-abl-sensitivity-v1"
COMPLETE_TRAINING_STATUS = "TRAINING_COMPLETED_TEST_NOT_RUN"
BASELINE_ARM = "baseline"
REGISTERED_SEEDS = (0, 1, 2, 3, 4, 5)
DEFAULT_SENSITIVITY_CONFIG = str(PROJECT / "methods" / "frt_abl" / "configs"
                                 / "sensitivity.json")

# Keys a planner may change without turning the arm into a second factor.
# `variant` is deliberately NOT here: flipping the variant would be a second
# factor and must fail `assert_single_factor`.
_SYSTEM_KEYS = frozenset({"output_root", "sensitivity", "seed", "run_tag"})

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
        27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}

# A delta that lands exactly on the predeclared margin (e.g. 0.795 - 0.800) must
# not flip because binary floating point represents it as -0.005000000000000004.
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same_number(left, right):
    return (_is_number(left) and _is_number(right)
            and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12))


def _mean(values):
    clean = [float(value) for value in values if _is_number(value)]
    return statistics.fmean(clean) if clean else None


def _t_critical(count):
    if count < 2:
        return None
    return _T95.get(count - 1, 1.96)


def _ci95(values):
    clean = [float(value) for value in values if _is_number(value)]
    if len(clean) < 2:
        return None
    return _t_critical(len(clean)) * statistics.stdev(clean) / math.sqrt(len(clean))


def _rank(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2.0 + 1.0
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def _spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if _is_number(x) and _is_number(y)]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def _format_level(value):
    return f"{float(value):g}"


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        # A run that is still starting writes config.json non-atomically; a
        # truncated file must not crash the whole summary.
        return None


def _load_history(path):
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


# --------------------------------------------------------------------------- #
# protocol declaration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Factor:
    """One predeclared sensitivity knob."""

    key: str
    levels: tuple
    relative: bool = False
    optional: bool = False
    rationale: str = ""

    def resolve(self, baseline_value, level):
        if self.relative:
            # Round the product so 0.2 * 3.0 is logged as 0.6, not 0.6000000000000001.
            return round(float(baseline_value) * float(level), 12)
        return level

    def token(self, level):
        return f"{_format_level(level)}x" if self.relative else _format_level(level)

    def arm_id(self, level):
        return f"{self.key}={self.token(level)}"

    def is_baseline_level(self, baseline_value, level):
        return _same_number(level, 1.0 if self.relative else baseline_value)


def declared_factors(spec):
    factors = []
    for key, entry in spec["factors"].items():
        factors.append(Factor(key=key, levels=tuple(entry["levels"]),
                              relative=bool(entry.get("relative", False)),
                              optional=bool(entry.get("optional", False)),
                              rationale=entry.get("why", "")))
    return factors


def factor_map(spec):
    return {factor.key: factor for factor in declared_factors(spec)}


def default_factor_keys(spec):
    return [factor.key for factor in declared_factors(spec) if not factor.optional]


def normalise_factor_keys(spec, factor_keys):
    """Declaration order, no duplicates; undeclared keys are an error."""
    declared = [factor.key for factor in declared_factors(spec)]
    requested = list(factor_keys)
    unknown = sorted(set(requested) - set(declared))
    if unknown:
        raise ValueError(f"Undeclared factor(s): {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate factor keys")
    return [key for key in declared if key in set(requested)]


def _check_levels(name, levels):
    if not isinstance(levels, (list, tuple)) or not levels:
        raise ValueError(f"{name} must be a non-empty list")
    if any(not _is_number(value) for value in levels):
        raise ValueError(f"{name} must contain numbers")
    numbers = [float(value) for value in levels]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{name} must be unique")
    if numbers != sorted(numbers):
        raise ValueError(f"{name} must be sorted ascending")
    return numbers


def validate_sensitivity_config(spec):
    """Validate the declared protocol without touching the filesystem or a model."""
    if spec.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
    baseline = spec.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("Missing baseline block")
    if baseline.get("method_version") != METHOD_VERSION:
        raise ValueError(f"The sensitivity baseline must be {METHOD_VERSION}")
    if baseline.get("variant") != "full":
        raise ValueError("The sensitivity baseline is the full method only")
    configs = baseline.get("configs")
    if (not isinstance(configs, (list, tuple)) or not configs
            or any(not isinstance(path, str) or not path for path in configs)):
        raise ValueError("baseline.configs must be a non-empty list of config paths")
    seeds = spec.get("seeds")
    if (not isinstance(seeds, (list, tuple)) or not seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool)
                   or seed not in REGISTERED_SEEDS for seed in seeds)
            or len(set(seeds)) != len(seeds)):
            raise ValueError("seeds must be a non-empty unique subset of 0..5")
    for key in ("output_root", "summary_root"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError(f"Missing {key}")

    factors = spec.get("factors")
    if not isinstance(factors, dict) or not factors:
        raise ValueError("factors must be a non-empty mapping")
    seen_optional = False
    for key, entry in factors.items():
        if not isinstance(entry, dict):
            raise ValueError(f"factors.{key} must be a mapping")
        levels = _check_levels(f"factors.{key}.levels", entry.get("levels"))
        for level in levels:
            if key == "knn_k" and not (float(level).is_integer() and level >= 1):
                raise ValueError("knn_k must use integer levels >= 1")
            if key == "graph_sigma" and not level > 0:
                raise ValueError("graph_sigma must use positive levels")
            if key == "risk_limit" and not 0 < level < 1:
                raise ValueError("risk_limit must use levels in (0,1)")
            if key == "curriculum_ramp_rounds" and not (float(level).is_integer() and level >= 1):
                raise ValueError("curriculum_ramp_rounds must use integer levels >= 1")
            if key == "curriculum_min_weight" and not 0 < level <= 1:
                raise ValueError("curriculum_min_weight must use levels in (0,1]")
            if key == "full_boundary_weight" and level < 0:
                raise ValueError("full_boundary_weight multiplier must be >= 0")
        relative = entry.get("relative", False)
        if not isinstance(relative, bool):
            raise ValueError(f"factors.{key}.relative must be boolean")
        if relative and not any(_same_number(level, 1.0) for level in levels):
            raise ValueError(f"Relative factor {key} must contain the baseline multiplier 1.0")
        optional = entry.get("optional", False)
        if not isinstance(optional, bool):
            raise ValueError(f"factors.{key}.optional must be boolean")
        if optional:
            # Optional knobs are appended after the primary grid in the file.
            seen_optional = True
        elif seen_optional:
            raise ValueError("Primary factors must be declared before optional factors")
        if not isinstance(entry.get("why", ""), str):
            raise ValueError(f"factors.{key}.why must be a string")

    analysis = spec.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Missing analysis block")
    margin = analysis.get("accuracy_margin")
    if not _is_number(margin) or not 0 < float(margin) <= 0.1:
        raise ValueError("analysis.accuracy_margin must lie in (0,0.1]")
    tolerance = analysis.get("coverage_tolerance")
    if not _is_number(tolerance) or not 0 <= float(tolerance) <= 0.5:
        raise ValueError("analysis.coverage_tolerance must lie in [0,0.5]")
    min_levels = analysis.get("min_block_levels")
    if not isinstance(min_levels, int) or isinstance(min_levels, bool) or min_levels < 2:
        raise ValueError("analysis.min_block_levels must be an integer >= 2")
    min_span = analysis.get("min_block_span")
    if not _is_number(min_span) or not 0 < float(min_span) <= 1:
        raise ValueError("analysis.min_block_span must lie in (0,1]")
    notes = spec.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("notes must be a list of strings")
    return spec


def load_sensitivity_config(path):
    spec = validate_sensitivity_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))
    for key in ("output_root", "summary_root"):
        spec[key] = str(_absolute(spec[key]))
    spec["baseline"] = dict(spec["baseline"])
    spec["baseline"]["configs"] = [str(_absolute(item))
                                   for item in spec["baseline"]["configs"]]
    return spec


def _absolute(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


# --------------------------------------------------------------------------- #
# campaign planning
# --------------------------------------------------------------------------- #
def validate_value(key, value):
    """Per-key range checks so an invalid grid fails during planning."""
    if key == "knn_k" and not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
        raise ValueError("knn_k must be an integer >= 1")
    if key == "graph_sigma" and not float(value) > 0:
        raise ValueError("graph_sigma must be positive")
    if key == "risk_limit" and not 0 < float(value) < 1:
        raise ValueError("risk_limit must lie in (0,1)")
    if key == "curriculum_ramp_rounds" and not (isinstance(value, int)
                                                and not isinstance(value, bool)
                                                and value >= 1):
        raise ValueError("curriculum_ramp_rounds must be an integer >= 1")
    if key == "curriculum_min_weight" and not 0 < float(value) <= 1:
        raise ValueError("curriculum_min_weight must lie in (0,1]")
    if key == "full_boundary_weight" and float(value) < 0:
        raise ValueError("full_boundary_weight must be non-negative")


def build_arms(spec, cfg, factor_keys=None):
    """Build the baseline arm plus one arm per non-baseline grid level."""
    if cfg.get("variant") != spec["baseline"]["variant"]:
        raise ValueError("Sensitivity campaigns train the frozen full variant only")
    if cfg.get("method_version") != spec["baseline"]["method_version"]:
        raise ValueError("Sensitivity campaigns reuse the frozen method version only")
    factors = factor_map(spec)
    keys = list(factor_keys) if factor_keys is not None else default_factor_keys(spec)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate factor keys")
    if any(key not in factors for key in keys):
        raise ValueError(f"Undeclared factor(s): {sorted(set(keys) - set(factors))}")
    arms = [{"arm": BASELINE_ARM, "factor": None, "level": None, "overrides": {}}]
    seen = {BASELINE_ARM}
    for key in keys:
        factor = factors[key]
        if key not in cfg:
            raise ValueError(f"Baseline config lacks factor key: {key}")
        baseline_value = cfg[key]
        if not any(factor.is_baseline_level(baseline_value, level) for level in factor.levels):
            raise ValueError(f"{key} grid must contain the frozen baseline value {baseline_value!r}")
        for level in factor.levels:
            if factor.is_baseline_level(baseline_value, level):
                continue
            value = factor.resolve(baseline_value, level)
            validate_value(key, value)
            arm = factor.arm_id(level)
            if arm in seen:
                raise ValueError(f"Duplicate arm id: {arm}")
            seen.add(arm)
            arms.append({"arm": arm, "factor": key, "level": level, "overrides": {key: value}})
    return arms


def assert_single_factor(baseline_cfg, arm_cfg, arm):
    """Refuse any arm that changes more than the one declared parameter."""
    changed = {key for key in set(baseline_cfg) | set(arm_cfg)
               if baseline_cfg.get(key) != arm_cfg.get(key)} - _SYSTEM_KEYS
    if changed != set(arm["overrides"]):
        raise ValueError(f"Arm {arm['arm']} is not a one-factor change: {sorted(changed)}")


def build_arm_config(spec, cfg, arm, campaign, runs_root):
    merged = dict(cfg)
    merged.update(arm["overrides"])
    merged["variant"] = spec["baseline"]["variant"]
    merged["output_root"] = str(runs_root)
    # train.py embeds run_tag in the run directory name, so sensitivity runs are
    # self-describing even before opening config.json.
    merged["run_tag"] = arm["arm"]
    merged["sensitivity"] = {
        "protocol_version": spec["protocol_version"],
        "campaign": campaign,
        "arm": arm["arm"],
        "factor": arm["factor"],
        "level": arm["level"],
        "baseline": not arm["overrides"],
    }
    assert_single_factor(cfg, merged, arm)
    return merged


def _baseline_signature(cfg):
    return {key: value for key, value in sorted(cfg.items()) if key not in _SYSTEM_KEYS}


def campaign_identity(spec, baselines, factor_keys, code_revision):
    """Pin protocol, grids, baseline configs and code revision; the seed set may grow."""
    factors = factor_map(spec)
    identity = {
        "protocol_version": spec["protocol_version"],
        "code_revision": code_revision,
        "baseline": {"method_version": spec["baseline"]["method_version"],
                     "variant": spec["baseline"]["variant"]},
        "factors": {key: {"levels": [float(level) for level in factors[key].levels],
                          "relative": factors[key].relative,
                          "optional": factors[key].optional}
                    for key in factor_keys},
        "configs": {task: _baseline_signature(cfg) for task, cfg in sorted(baselines.items())},
    }
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def materialise_configs(campaign_dir, spec, baselines, arms, campaign):
    config_root = Path(campaign_dir) / "configs"
    runs_root = Path(campaign_dir) / "runs"
    paths = {}
    for task, cfg in baselines.items():
        for arm in arms[task]:
            target = config_root / task / f"{arm['arm']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            merged = build_arm_config(spec, cfg, arm, campaign, runs_root)
            text = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
            if target.is_file() and target.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"Arm config changed inside an existing campaign: {target}")
            if not target.is_file():
                target.write_text(text, encoding="utf-8")
            paths[(task, arm["arm"])] = target
    return paths


def prepare_campaign(spec, baselines, factor_keys, seeds, code_revision, materialise=True):
    """Build (or re-open) the campaign plan; planning never trains anything."""
    validate_sensitivity_config(spec)
    if not baselines:
        raise ValueError("No baseline configs")
    for task, cfg in baselines.items():
        if cfg.get("task") != task:
            raise ValueError(f"Baseline config task mismatch: {cfg.get('task')!r} vs {task!r}")
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds) or any(seed not in REGISTERED_SEEDS for seed in seeds):
        raise ValueError("seeds must be a unique subset of 0..5")
    factor_keys = normalise_factor_keys(spec, factor_keys)
    campaign = campaign_identity(spec, baselines, factor_keys, code_revision)
    campaign_dir = Path(spec["output_root"]) / campaign
    runs_root = campaign_dir / "runs"
    arms = {task: build_arms(spec, cfg, factor_keys) for task, cfg in baselines.items()}
    if materialise:
        campaign_dir.mkdir(parents=True, exist_ok=True)
        runs_root.mkdir(parents=True, exist_ok=True)
        configs = materialise_configs(campaign_dir, spec, baselines, arms, campaign)
    else:
        configs = {}
    existing = _load_json(campaign_dir / "plan.json") if campaign_dir.is_dir() else None
    if existing and existing.get("campaign") != campaign:
        raise ValueError("Existing campaign directory pins a different identity")
    if existing:
        seeds = sorted(set(int(seed) for seed in existing.get("seeds", [])) | set(seeds))
    runs = []
    for task in sorted(baselines):
        for arm in arms[task]:
            for seed in seeds:
                record = {"task": task, "arm": arm["arm"], "seed": seed,
                          "config": str(configs.get((task, arm["arm"]), ""))}
                runs.append(record)
    factors = factor_map(spec)
    plan = {
        "status": "PLAN_ONLY",
        "protocol_version": spec["protocol_version"],
        "campaign": campaign,
        "campaign_dir": str(campaign_dir),
        "runs_root": str(runs_root),
        "output_root": str(Path(spec["output_root"])),
        "code_revision": code_revision,
        "method_version": spec["baseline"]["method_version"],
        "variant": spec["baseline"]["variant"],
        "seeds": seeds,
        "factors": [{"key": key, "levels": list(factors[key].levels),
                     "relative": factors[key].relative,
                     "optional": factors[key].optional,
                     "why": factors[key].rationale} for key in factor_keys],
        "tasks": sorted(baselines),
        "baseline_values": {task: {key: baselines[task][key] for key in factor_keys}
                            for task in sorted(baselines)},
        "arms": {task: arms[task] for task in sorted(baselines)},
        "n_arms": {task: len(arms[task]) for task in sorted(baselines)},
        "n_runs": len(runs),
        "analysis": dict(spec["analysis"]),
        "runs": runs,
        "notes": list(spec.get("notes", [])) + [
            "Every arm changes exactly one frozen-v10 parameter; the baseline arm is re-trained "
            "inside the campaign so source/environment identities match.",
            "Validation only: train.py cannot read the formal test split.",
        ],
    }
    if materialise:
        # A re-plan must not erase the execution audit trail of an earlier stage.
        if existing and existing.get("executions"):
            plan["executions"] = existing["executions"]
            plan["status"] = existing.get("status", "PLAN_ONLY")
        write_json(campaign_dir / "plan.json", plan)
    return plan


# --------------------------------------------------------------------------- #
# aggregation of completed training runs
# --------------------------------------------------------------------------- #
def history_metrics(history):
    """Coverage / performance-curve metrics from one run's history.jsonl."""
    metrics = {"rounds": len(history), "exposure_coverage_mean": None,
               "exposure_coverage_final": None, "acceptance_coverage_final": None,
               "explanation_coverage_final": None, "boundary_share_final": None,
               "positive_share_final": None, "gate_coverage_final": None,
               "gate_accepted_final": None, "gate_candidates_final": None,
               "validation_auc": None, "validation_final": None, "rounds_to_best": None,
               "alpha_median": None, "alpha_none_rounds": None,
               "boundary_loss_mean": None, "pseudo_loss_mean": None,
               "graph_changed_fraction_mean": None}
    if not history:
        return metrics
    exposures = [record.get("coverage") for record in history]
    metrics["exposure_coverage_mean"] = _mean(exposures)
    metrics["exposure_coverage_final"] = history[-1].get("coverage")
    accuracies = []
    for record in history:
        task = record.get("validation_task")
        if isinstance(task, dict) and _is_number(task.get("accuracy")):
            accuracies.append(float(task["accuracy"]))
    if accuracies:
        metrics["validation_auc"] = _mean(accuracies)
        metrics["validation_final"] = accuracies[-1]
        metrics["rounds_to_best"] = accuracies.index(max(accuracies)) + 1
    alphas = [record.get("alpha") for record in history if _is_number(record.get("alpha"))]
    metrics["alpha_median"] = statistics.median(alphas) if alphas else None
    metrics["alpha_none_rounds"] = sum(1 for record in history
                                       if not _is_number(record.get("alpha")))
    last = history[-1]
    regions = last.get("regions") if isinstance(last.get("regions"), dict) else {}
    total = sum(int(count) for count in regions.values()
                if _is_number(count) and count >= 0)
    if total > 0:
        metrics["positive_share_final"] = int(regions.get("positive", 0)) / total
        metrics["boundary_share_final"] = int(regions.get("boundary", 0)) / total
        accepted = last.get("accepted_candidates")
        if _is_number(accepted):
            metrics["acceptance_coverage_final"] = float(accepted) / total
        unsupported = last.get("unsupported_weak")
        if _is_number(unsupported):
            metrics["explanation_coverage_final"] = (total - float(unsupported)) / total
    metrics["boundary_loss_mean"] = _mean(
        [record.get("loss_components_per_update", {}).get("boundary")
         for record in history if isinstance(record.get("loss_components_per_update"), dict)])
    metrics["pseudo_loss_mean"] = _mean(
        [record.get("loss_components_per_update", {}).get("pseudo")
         for record in history if isinstance(record.get("loss_components_per_update"), dict)])
    fractions = []
    for record in history:
        count = record.get("q_stats", {}).get("count") if isinstance(record.get("q_stats"), dict) else None
        changed = record.get("graph_changed_candidates")
        if _is_number(count) and count > 0 and _is_number(changed):
            fractions.append(float(changed) / float(count))
    metrics["graph_changed_fraction_mean"] = _mean(fractions)
    # Gate coverage: the `gate` report lists, per candidate alpha, how many gate
    # candidates the threshold would accept.  For the `full` variant every
    # supported weak sample enters the positive region, so acceptance coverage is
    # pinned near 1; the chosen threshold's share of the loosest tested threshold
    # is the coverage signal that actually varies with sigma/knn/risk_limit.
    gate = last.get("gate")
    if isinstance(gate, list) and gate:
        counts = [(float(item.get("alpha", 0.0)), float(item["count"]))
                  for item in gate if isinstance(item, dict)
                  and _is_number(item.get("alpha")) and _is_number(item.get("count"))]
        if counts:
            loosest = max(count for _, count in counts)
            metrics["gate_candidates_final"] = loosest
            chosen = last.get("alpha")
            chosen_count = None
            if _is_number(chosen):
                for alpha_value, count in counts:
                    if _same_number(alpha_value, float(chosen)):
                        chosen_count = count
                        break
            metrics["gate_accepted_final"] = chosen_count if chosen_count is not None else 0.0
            # loosest == 0 is the total gate-collapse case (no candidate clears even
            # the loosest threshold); coverage is 0, not "unknown".
            metrics["gate_coverage_final"] = (metrics["gate_accepted_final"] / loosest
                                              if loosest else 0.0)
    return metrics


def _fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def collect_runs(runs_root, campaign):
    """Read every run directory of one campaign; foreign runs are reported, not merged."""
    runs, excluded = [], []
    root = Path(runs_root)
    if not root.is_dir():
        return runs, excluded
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        cfg = _load_json(run_dir / "config.json")
        status = _load_json(run_dir / "status.json") or {}
        if cfg is None:
            excluded.append({"run_dir": run_dir.name,
                             "reason": "config.json missing, unreadable or incomplete"})
            continue
        marker = cfg.get("sensitivity")
        if not isinstance(marker, dict) or marker.get("campaign") != campaign:
            excluded.append({"run_dir": run_dir.name, "reason": "not this sensitivity campaign"})
            continue
        env = _load_json(run_dir / "environment.json") or {}
        sources = _load_json(run_dir / "source_manifest.json") or {}
        runs.append({
            "run_dir": run_dir.name,
            "run_id": run_dir.name,
            "task": cfg.get("task"),
            "arm": marker.get("arm"),
            "factor": marker.get("factor"),
            "level": marker.get("level"),
            "seed": cfg.get("seed"),
            "status": status.get("status"),
            "stopped_reason": status.get("stopped_reason"),
            "best_validation_accuracy": status.get("best_validation_accuracy"),
            "optimizer_steps": status.get("optimizer_steps"),
            "code_revision": status.get("code_revision") or env.get("code_revision"),
            "runtime": {key: env.get(key) for key in ("python", "torch", "numpy",
                                                      "device", "threads")},
            "sources_sha": _fingerprint(sources) if sources else None,
            "history": history_metrics(_load_history(run_dir / "history.jsonl")),
        })
    return runs, excluded


def usable_runs(runs):
    """Completed, unique-seed, finite-accuracy runs only.

    A failed/stopped attempt must not veto a completed run for the same seed
    (that would silently delete valid data), but two *completed* runs for one
    seed are an undeclared retry and are both excluded rather than maximised.
    """
    candidates = []
    for run in runs:
        if run["status"] != COMPLETE_TRAINING_STATUS or run["stopped_reason"] is not None:
            continue
        accuracy = run["best_validation_accuracy"]
        if not _is_number(accuracy) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            continue
        candidates.append(run)
    counts = Counter(run["seed"] for run in candidates)
    return [run for run in candidates if counts[run["seed"]] == 1]


def _metric_values(runs, key):
    values = []
    for run in runs:
        value = run["history"].get(key)
        if _is_number(value) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def arm_statistics(runs):
    admitted = usable_runs(runs)
    accuracies = [float(run["best_validation_accuracy"]) for run in admitted]
    completed = [run for run in runs if run["status"] == COMPLETE_TRAINING_STATUS
                 and run["stopped_reason"] is None]
    stats = {
        "n": len(accuracies),
        "seeds": sorted(run["seed"] for run in admitted),
        "run_ids": sorted(run["run_id"] for run in admitted),
        "statuses": sorted({str(run["status"]) for run in runs}),
        "duplicate_seeds": sorted(seed for seed, count in Counter(
            run["seed"] for run in completed).items() if count > 1),
        "accuracy_mean": _mean(accuracies),
        "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
        "accuracy_ci95": _ci95(accuracies),
        "accuracy_min": min(accuracies) if accuracies else None,
        "accuracy_max": max(accuracies) if accuracies else None,
        "accuracy_median": statistics.median(accuracies) if accuracies else None,
        "optimizer_steps": sorted({run["optimizer_steps"] for run in admitted
                                   if _is_number(run["optimizer_steps"])}),
    }
    for key in ("acceptance_coverage_final", "explanation_coverage_final",
                "gate_coverage_final", "gate_accepted_final",
                "exposure_coverage_mean", "boundary_share_final", "validation_auc",
                "validation_final", "rounds_to_best", "alpha_median", "alpha_none_rounds",
                "boundary_loss_mean", "graph_changed_fraction_mean"):
        stats[key] = _mean(_metric_values(admitted, key))
    return stats


def paired_statistics(arm_runs, baseline_by_seed):
    admitted = usable_runs(arm_runs)
    pairs = []
    for run in admitted:
        baseline = baseline_by_seed.get(run["seed"])
        if baseline is None:
            continue
        delta = float(run["best_validation_accuracy"]) - float(baseline["best_validation_accuracy"])
        pairs.append({"seed": run["seed"],
                      "arm": float(run["best_validation_accuracy"]),
                      "baseline": float(baseline["best_validation_accuracy"]),
                      "delta": delta,
                      "same_steps": (run["optimizer_steps"] == baseline["optimizer_steps"]
                                     and _is_number(run["optimizer_steps"]))})
    deltas = [pair["delta"] for pair in pairs]
    return {
        "n_pairs": len(pairs),
        "pairs": pairs,
        "delta_mean": _mean(deltas),
        "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else None,
        "delta_ci95": _ci95(deltas),
        "delta_min": min(deltas) if deltas else None,
        "delta_max": max(deltas) if deltas else None,
        "wins": sum(1 for delta in deltas if delta > 0),
        "losses": sum(1 for delta in deltas if delta < 0),
        "ties": sum(1 for delta in deltas if delta == 0),
        "same_optimizer_budget": all(
            pair.get("same_steps", True) for pair in pairs) if pairs else None,
    }


def _contiguous_blocks(flags):
    blocks, start = [], None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            blocks.append((start, index - 1))
            start = None
    if start is not None:
        blocks.append((start, len(flags) - 1))
    return blocks


def _plateau(rows, key, baseline_index, min_levels, min_span):
    flags = [bool(row.get(key)) for row in rows]
    blocks = _contiguous_blocks(flags)
    empty = {"levels": [], "n_levels": 0, "span": 0.0, "broad": False,
             "contains_baseline": False, "start": None, "end": None}
    if not blocks:
        return empty
    holder = [block for block in blocks if baseline_index is not None
              and block[0] <= baseline_index <= block[1]]
    main = holder[0] if holder else max(blocks, key=lambda block: (block[1] - block[0], -block[0]))
    span = (main[1] - main[0]) / (len(rows) - 1) if len(rows) > 1 else 0.0
    return {"levels": [rows[index]["level"] for index in range(main[0], main[1] + 1)],
            "n_levels": main[1] - main[0] + 1, "span": span,
            "broad": (main[1] - main[0] + 1) >= min_levels and span >= min_span,
            "contains_baseline": baseline_index is not None
            and main[0] <= baseline_index <= main[1],
            "start": main[0], "end": main[1]}


_VERDICT_TEXT = {
    "broad_confirmed": "宽泛有效区间（描述性与非劣性检验均通过）",
    "broad_descriptive_only": "点估计为宽平台，但种子噪声下无法确认等价",
    "narrow": "有效区间较窄",
    "isolated": "仅个别级别可视为等价",
    "no_valid_level": "没有级别可视为等价",
    "incomplete_grid": "网格未跑完或存在未配对级别，无法判定",
    "insufficient_levels": "级别数不足（<3），无法判定",
    "incomparable": "源码/环境指纹不一致，不可比",
    "screening_broad": "单种子筛查：点估计为宽平台，待多种子确认",
    "screening_narrow": "单种子筛查：有效区间较窄",
    "screening_isolated": "单种子筛查：仅个别级别可视为等价",
    "screening_none": "单种子筛查：没有级别可视为等价",
}


def _verdict(rows, incomplete, n_seeds, descriptive, confirmatory):
    if incomplete:
        return "incomplete_grid"
    if len(rows) < 3:
        return "insufficient_levels"
    if n_seeds < 2:
        if descriptive["broad"] and descriptive["contains_baseline"]:
            return "screening_broad"
        if descriptive["n_levels"] >= 2:
            return "screening_narrow"
        if descriptive["n_levels"] == 1:
            return "screening_isolated"
        return "screening_none"
    if (descriptive["broad"] and descriptive["contains_baseline"]
            and confirmatory["broad"] and confirmatory["contains_baseline"]):
        return "broad_confirmed"
    if descriptive["broad"] and descriptive["contains_baseline"]:
        return "broad_descriptive_only"
    if descriptive["n_levels"] >= 2:
        return "narrow"
    if descriptive["n_levels"] == 1:
        return "isolated"
    return "no_valid_level"


def _level_rows(task, factor, plan, summaries, baseline_summary, paired, analysis):
    baseline_value = plan["baseline_values"][task][factor.key]
    margin = float(analysis["accuracy_margin"])
    coverage_tolerance = float(analysis["coverage_tolerance"])
    arm_by_level = {}
    for record in plan["arms"][task]:
        if record["factor"] == factor.key:
            arm_by_level[float(record["level"])] = record["arm"]
    baseline_coverage = baseline_summary.get("acceptance_coverage_final")
    baseline_explanation = baseline_summary.get("explanation_coverage_final")
    baseline_gate = baseline_summary.get("gate_coverage_final")
    rows, missing, unpaired = [], [], []
    baseline_index = None
    for index, level in enumerate(factor.levels):
        is_baseline = factor.is_baseline_level(baseline_value, level)
        arm = BASELINE_ARM if is_baseline else arm_by_level.get(float(level))
        if arm is None:
            missing.append(float(level))
            continue
        summary = baseline_summary if is_baseline else (summaries.get(arm) or {})
        if not summary or summary.get("n", 0) == 0:
            missing.append(float(level))
            continue
        delta = {"n_pairs": 0, "delta_mean": 0.0, "delta_ci95": 0.0,
                 "wins": 0, "losses": 0, "ties": 0}
        if not is_baseline:
            delta = paired.get((task, arm)) or delta
            if delta.get("n_pairs", 0) == 0:
                # No same-seed baseline pair: show the cross-seed difference for
                # context only. Such a row is never "valid": without a pair the
                # comparison is confounded by seed, so the grid is incomplete.
                unpaired.append(float(level))
                delta = dict(delta)
                delta["delta_mean"] = (summary["accuracy_mean"] - baseline_summary["accuracy_mean"]
                                       if _is_number(summary.get("accuracy_mean"))
                                       and _is_number(baseline_summary.get("accuracy_mean"))
                                       else None)
                delta["delta_ci95"] = None
        delta_mean = delta.get("delta_mean")
        delta_ci = delta.get("delta_ci95")
        delta_lower = (delta_mean - delta_ci if _is_number(delta_mean)
                       and _is_number(delta_ci) else None)
        acceptance = summary.get("acceptance_coverage_final")
        explanation = summary.get("explanation_coverage_final")
        gate_coverage = summary.get("gate_coverage_final")
        coverage_checks = []
        if _is_number(acceptance) and _is_number(baseline_coverage):
            coverage_checks.append(acceptance >= baseline_coverage - coverage_tolerance - _EPS)
        if _is_number(explanation) and _is_number(baseline_explanation):
            coverage_checks.append(explanation >= baseline_explanation - coverage_tolerance - _EPS)
        if _is_number(gate_coverage) and _is_number(baseline_gate):
            coverage_checks.append(gate_coverage >= baseline_gate - coverage_tolerance - _EPS)
        coverage_ok = all(coverage_checks) if coverage_checks else None
        if is_baseline:
            baseline_index = index
            coverage_ok = True
            within_margin = True
            non_inferior = True if summary.get("n", 0) >= 2 else None
            confirmed_valid = True if summary.get("n", 0) >= 2 else None
            valid = True
        else:
            # Missing coverage evidence must fail closed: an arm cannot buy
            # accuracy by silently dropping the coverage report.
            coverage_ok = bool(coverage_checks) and all(coverage_checks)
            within_margin = delta_mean is not None and delta_mean >= -margin - _EPS
            non_inferior = (delta_lower is not None and delta_lower > -margin - _EPS)
            paired_row = delta.get("n_pairs", 0) > 0
            valid = bool(within_margin) and coverage_ok and paired_row
            confirmed_valid = (valid and non_inferior) if non_inferior is not None else None
        rows.append({
            "level": float(level), "level_index": index,
            "value": factor.resolve(baseline_value, level),
            "arm": arm, "n": summary.get("n", 0),
            "seeds": summary.get("seeds", []),
            "accuracy_mean": summary.get("accuracy_mean"),
            "accuracy_std": summary.get("accuracy_std"),
            "accuracy_ci95": summary.get("accuracy_ci95"),
            "delta_mean": delta_mean,
            "delta_ci95": delta_ci,
            "delta_lower": delta_lower,
            "delta_upper": ((delta_mean + delta_ci)
                            if _is_number(delta_mean) and _is_number(delta_ci) else None),
            "paired": bool(delta.get("n_pairs", 0) > 0),
            "wins": delta.get("wins", 0), "losses": delta.get("losses", 0),
            "acceptance_coverage": acceptance,
            "explanation_coverage": explanation,
            "gate_coverage": gate_coverage,
            "exposure_coverage": summary.get("exposure_coverage_mean"),
            "boundary_share": summary.get("boundary_share_final"),
            "coverage_delta": (acceptance - baseline_coverage
                               if _is_number(acceptance) and _is_number(baseline_coverage) else None),
            "within_margin": bool(within_margin),
            "non_inferior": non_inferior,
            "coverage_ok": coverage_ok,
            "valid": bool(valid),
            "confirmed_valid": confirmed_valid,
            "statuses": summary.get("statuses", []),
        })
    return rows, baseline_index, missing, unpaired


def _factor_result(task, factor, plan, summaries, baseline_summary, paired, analysis):
    rows, baseline_index, missing, unpaired = _level_rows(
        task, factor, plan, summaries, baseline_summary, paired, analysis)
    min_levels = int(analysis["min_block_levels"])
    min_span = float(analysis["min_block_span"])
    descriptive = _plateau(rows, "valid", baseline_index, min_levels, min_span)
    confirmatory = _plateau(rows, "confirmed_valid", baseline_index, min_levels, min_span)
    available = [row for row in rows if row["accuracy_mean"] is not None]
    # A single available level is not evidence of "no variation".
    accuracy_range = ((max(row["accuracy_mean"] for row in available)
                       - min(row["accuracy_mean"] for row in available))
                      if len(available) >= 2 else None)
    max_abs_delta = max((abs(row["delta_mean"]) for row in available
                         if _is_number(row["delta_mean"])), default=None)
    coverages = [row["acceptance_coverage"] for row in available
                 if _is_number(row["acceptance_coverage"])]
    coverage_range = (max(coverages) - min(coverages)) if len(coverages) > 1 else None
    rho = _spearman([row["level_index"] for row in available],
                    [row["accuracy_mean"] for row in available])
    trend = ("n/a" if rho is None else
             "monotone" if abs(rho) >= 0.8 else
             "weak" if abs(rho) >= 0.4 else "flat")
    n_seeds = min((row["n"] for row in rows), default=0)
    incomplete = bool(missing or unpaired)
    verdict = _verdict(rows, incomplete, n_seeds, descriptive, confirmatory)
    budget_flags = []
    for row in rows:
        if row["arm"] == BASELINE_ARM:
            continue
        info = paired.get((task, row["arm"])) or {}
        if info.get("same_optimizer_budget") is not None:
            budget_flags.append(bool(info["same_optimizer_budget"]))
    return {
        "factor": factor.key,
        "relative": factor.relative,
        "unit": "multiplier of the frozen baseline value" if factor.relative else factor.key,
        "baseline_value": plan["baseline_values"][task][factor.key],
        "levels": rows,
        "missing_levels": missing,
        "unpaired_levels": unpaired,
        "baseline_level_index": baseline_index,
        "descriptive_plateau": descriptive,
        "confirmatory_plateau": confirmatory,
        "accuracy_range": accuracy_range,
        "max_abs_delta": max_abs_delta,
        "coverage_range": coverage_range,
        "spearman_rho": rho,
        "trend": trend,
        "n_seeds": n_seeds,
        "same_optimizer_budget": all(budget_flags) if budget_flags else None,
        "verdict": verdict,
        "verdict_zh": _VERDICT_TEXT.get(verdict, verdict),
    }


def _task_conclusion(factors, baseline):
    broad = [item for item in factors if item["verdict"] in
             {"broad_confirmed", "screening_broad"}]
    narrow = [item for item in factors if item["verdict"] in
              {"narrow", "screening_narrow"}]
    none_valid = [item for item in factors if item["verdict"] in
                  {"no_valid_level", "screening_none"}]
    sentences = []
    if broad:
        sentences.append(f"{len(broad)} 个因素在点估计上存在宽平台："
                         + "、".join(item["factor"] for item in broad) + "。")
    if narrow:
        sentences.append("、".join(item["factor"] for item in narrow) + " 的有效区间较窄。")
    if none_valid:
        sentences.append("、".join(item["factor"] for item in none_valid)
                         + " 没有可视为等价的级别。")
    if not sentences:
        sentences.append("当前数据不足以判定有效区间。")
    if baseline.get("accuracy_std") is None:
        sentences.append("当前只有单种子筛查，未估计种子波动；结论需多种子确认。")
    return "".join(sentences)


def summarize(campaign_dir):
    """Aggregate one campaign into per-arm, per-factor and per-level tables."""
    campaign_dir = Path(campaign_dir)
    plan = _load_json(campaign_dir / "plan.json")
    if plan is None:
        raise FileNotFoundError(f"Missing campaign plan: {campaign_dir / 'plan.json'}")
    analysis = dict(plan.get("analysis") or {})
    runs, excluded = collect_runs(campaign_dir / "runs", plan["campaign"])
    if not runs:
        raise ValueError(f"No sensitivity runs under {campaign_dir / 'runs'}")
    by_arm = {}
    for run in runs:
        by_arm.setdefault((run["task"], run["arm"]), []).append(run)
    tasks = {}
    factors = factor_map_from_plan(plan)
    for task in plan["tasks"]:
        baseline_runs = by_arm.get((task, BASELINE_ARM), [])
        baseline_summary = arm_statistics(baseline_runs)
        baseline_by_seed = {run["seed"]: run for run in usable_runs(baseline_runs)}
        summaries = {}
        paired = {}
        for arm, arm_runs in by_arm.items():
            if arm[0] != task:
                continue
            summaries[arm[1]] = arm_statistics(arm_runs)
            if arm[1] != BASELINE_ARM:
                paired[(task, arm[1])] = paired_statistics(arm_runs, baseline_by_seed)
        consistency = _consistency(task, runs)
        comparable = not (consistency["mixed_runtime"] or consistency["mixed_sources"])
        factor_results = [
            _factor_result(task, factors[item["key"]], plan, summaries,
                           baseline_summary, paired, analysis)
            for item in plan["factors"]
        ]
        if not comparable:
            # Mixed source/runtime fingerprints make every arm-vs-baseline number
            # confounded; say so instead of printing a verdict.
            for item in factor_results:
                item["verdict"] = "incomparable"
                item["verdict_zh"] = _VERDICT_TEXT["incomparable"]
        ranking = sorted((item for item in factor_results
                          if item["accuracy_range"] is not None),
                         key=lambda item: item["accuracy_range"], reverse=True)
        arm_records = {record["arm"]: record for record in plan["arms"][task]}
        duplicate_seeds = {arm: item["duplicate_seeds"] for arm, item in summaries.items()
                           if item.get("duplicate_seeds")}
        tasks[task] = {
            "baseline": baseline_summary,
            "n_completed": sum(item.get("n", 0) for item in summaries.values()),
            "arms": [dict(item, arm=arm, task=task,
                          factor=arm_records.get(arm, {}).get("factor"),
                          level=arm_records.get(arm, {}).get("level"))
                     for arm, item in sorted(summaries.items())],
            "factors": factor_results,
            "ranking": [{"factor": item["factor"], "accuracy_range": item["accuracy_range"],
                         "coverage_range": item["coverage_range"],
                         "max_abs_delta": item["max_abs_delta"],
                         "verdict": item["verdict"]} for item in ranking],
            "consistency": dict(consistency, comparable=comparable),
            "duplicate_seeds": duplicate_seeds,
            "conclusion": _task_conclusion(factor_results, baseline_summary),
        }
    return {
        "status": "SENSITIVITY_SUMMARY",
        "protocol_version": plan.get("protocol_version"),
        "campaign": plan["campaign"],
        "campaign_dir": str(campaign_dir),
        "code_revision": plan.get("code_revision"),
        "method_version": plan.get("method_version"),
        "variant": plan.get("variant"),
        "seeds": plan.get("seeds"),
        "n_run_dirs": len(runs) + len(excluded),
        "n_completed": sum(item["n_completed"] for item in tasks.values()),
        "analysis": analysis,
        "excluded_runs": excluded,
        "tasks": tasks,
        "notes": [
            "Performance is best validation accuracy; the formal test split is never read.",
            "Deltas are arm minus baseline on the same seed (paired).",
            "A 'broad' block needs >= min_block_levels levels covering >= min_block_span "
            "of the tested range and must contain the baseline level.",
            "Non-inferiority is a diagnostic: it uses a t-based 95% CI over seeds, "
            "not a distribution-free guarantee.",
            "A task with mixed source/runtime fingerprints is marked incomparable and gets "
            "no verdict, because arm-vs-baseline differences cannot be attributed.",
        ],
    }


def factor_map_from_plan(plan):
    return {item["key"]: Factor(item["key"], tuple(item["levels"]),
                                bool(item.get("relative", False)),
                                bool(item.get("optional", False)),
                                item.get("why", ""))
            for item in plan["factors"]}


def _consistency(task, runs):
    scoped = [run for run in runs if run["task"] == task]
    runtimes = sorted({_fingerprint(run["runtime"]) for run in scoped})
    sources = sorted({run["sources_sha"] for run in scoped if run["sources_sha"]})
    revisions = sorted({run["code_revision"] for run in scoped if run["code_revision"]})
    return {"n_runs": len(scoped), "runtime_fingerprints": runtimes,
            "source_fingerprints": sources, "code_revisions": revisions,
            "mixed_runtime": len(runtimes) > 1, "mixed_sources": len(sources) > 1}


# --------------------------------------------------------------------------- #
# report rendering and writing
# --------------------------------------------------------------------------- #
def _fmt(value, digits=4):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_signed(value, digits=4):
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def _fmt_pct(value, digits=1):
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _yes_no(value):
    if value is None:
        return "—"
    return "是" if value else "否"


def _fmt_level(value):
    """Compact grid value: 10 stays 10, 0.05 stays 0.05 (no unit padding)."""
    if not _is_number(value):
        return "—"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _csv_value(value):
    """Round report floats for CSV readability while keeping ints/bools/None intact."""
    if isinstance(value, (bool, int)) or not _is_number(value):
        return value
    return round(float(value), 6)


def render_report(summary):
    """Chinese markdown report; every number comes from the summary, no narration."""
    analysis = summary["analysis"]
    lines = [
        "# 参数敏感性实验报告",
        "",
        f"- campaign：`{summary['campaign']}`",
        f"- 方法版本：`{summary['method_version']}`（variant `{summary['variant']}`）",
        f"- code revision：`{summary['code_revision']}`",
        f"- seeds：{summary['seeds']}",
        f"- 判定口径：准确率等价边界 ±{analysis.get('accuracy_margin')}；"
        f"覆盖率容差 {analysis.get('coverage_tolerance')}；"
        f"宽平台要求 ≥{analysis.get('min_block_levels')} 个级别且覆盖 ≥"
        f"{analysis.get('min_block_span')} 的测试范围",
        "- 使用限制：只读 validation，未读正式 test；同比只在本 campaign 内成立",
        "",
        "## 1. 总览",
        "",
        "| 任务 | 因素 | 可用级别 | 准确率极差 | 最大绝对差 | 覆盖率极差 | 判定 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for task, payload in summary["tasks"].items():
        baseline = payload["baseline"]
        for item in payload["factors"]:
            lines.append(
                f"| {task} | `{item['factor']}` | {len(item['levels'])} | "
                f"{_fmt(item['accuracy_range'])} | {_fmt(item['max_abs_delta'])} | "
                f"{_fmt(item['coverage_range'])} | {item['verdict_zh']} |")
        if not payload["factors"]:
            lines.append(f"| {task} | — | 0 | — | — | — | 没有可分析的因素 |")
    lines += ["", "## 2. 基线（campaign 内重跑）", "",
              "| 任务 | n | 准确率均值 | 95% CI | 接纳覆盖率 | 解释覆盖率 | 门控覆盖率 | 优化器步数 |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for task, payload in summary["tasks"].items():
        baseline = payload["baseline"]
        lines.append(
            f"| {task} | {baseline['n']} | {_fmt(baseline.get('accuracy_mean'))} | "
            f"±{_fmt(baseline.get('accuracy_ci95'))} | "
            f"{_fmt_pct(baseline.get('acceptance_coverage_final'))} | "
            f"{_fmt_pct(baseline.get('explanation_coverage_final'))} | "
            f"{_fmt_pct(baseline.get('gate_coverage_final'))} | "
            f"{','.join(str(value) for value in baseline.get('optimizer_steps', [])) or '—'} |")
    lines += ["", "## 3. 各因素明细", ""]
    for task, payload in summary["tasks"].items():
        lines.append(f"### {task}")
        lines.append("")
        lines.append(f"结论：{payload['conclusion']}")
        lines.append("")
        for item in payload["factors"]:
            lines.append(f"#### `{item['factor']}`（baseline={item['baseline_value']}"
                         f"{'，网格为倍数' if item['relative'] else ''}；"
                         f"趋势={item['trend']}，ρ={_fmt(item['spearman_rho'], 3)}；"
                         f"优化器步数一致={_yes_no(item.get('same_optimizer_budget'))}）")
            lines.append("")
            lines.append(f"判定：**{item['verdict_zh']}**"
                         f"（描述性平台 {item['descriptive_plateau']['n_levels']} 级 / "
                         f"span {_fmt(item['descriptive_plateau']['span'], 2)}；"
                         f"确认性平台 {item['confirmatory_plateau']['n_levels']} 级 / "
                         f"span {_fmt(item['confirmatory_plateau']['span'], 2)}）")
            lines.append("")
            lines.append("| 级别 | 参数值 | 臂 | n | 配对 | 准确率 | 95% CI | Δ vs baseline | "
                         "95% CI(Δ) | 接纳覆盖 | 解释覆盖 | 门控覆盖 | 等价 | 确认 |")
            lines.append("|---|---|---|---:|:--:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|")
            for row in item["levels"]:
                valid = "是" if row["valid"] else "否"
                confirmed = ("—" if row["confirmed_valid"] is None
                             else ("是" if row["confirmed_valid"] else "否"))
                lines.append(
                    f"| {_fmt_level(row['level'])}{'×' if item['relative'] else ''} | "
                    f"{_fmt_level(row['value'])} | `{row['arm']}` | "
                    f"{row['n']} | {'是' if row['paired'] else '否'} | "
                    f"{_fmt(row['accuracy_mean'])} | "
                    f"±{_fmt(row['accuracy_ci95'])} | {_fmt_signed(row['delta_mean'])} | "
                    f"±{_fmt(row['delta_ci95'])} | "
                    f"{_fmt_pct(row['acceptance_coverage'])} | "
                    f"{_fmt_pct(row['explanation_coverage'])} | "
                    f"{_fmt_pct(row['gate_coverage'])} | {valid} | {confirmed} |")
            if item["missing_levels"]:
                lines.append("")
                lines.append("缺失级别：" + "、".join(_fmt_level(value)
                                                      for value in item["missing_levels"])
                             + "（未完成或未运行）")
            if item.get("unpaired_levels"):
                lines.append("未配对级别：" + "、".join(_fmt_level(value)
                                                        for value in item["unpaired_levels"])
                             + "（与 baseline 无同 seed 配对，不参与等价判定）")
            lines.append("")
    lines += ["## 4. 排除与一致性", ""]
    excluded = summary.get("excluded_runs") or []
    lines.append(f"- 排除的 run 目录：{len(excluded)}")
    for item in excluded[:20]:
        lines.append(f"  - `{item['run_dir']}`：{item['reason']}")
    for task, payload in summary["tasks"].items():
        consistency = payload["consistency"]
        lines.append(f"- {task}：{consistency['n_runs']} runs；runtime 指纹 "
                     f"{len(consistency['runtime_fingerprints'])} 种；source 指纹 "
                     f"{len(consistency['source_fingerprints'])} 种；code revision "
                     f"{consistency['code_revisions']}；可比={_yes_no(consistency.get('comparable'))}")
        if not consistency.get("comparable", True):
            lines.append(f"  - ⚠ {task} 的 source/runtime 指纹不唯一：判定已标为不可比，"
                         "数值只能作为过程记录。")
        for arm, seeds in (payload.get("duplicate_seeds") or {}).items():
            lines.append(f"  - ⚠ {task}/{arm} 有重复的已完成种子 {seeds}：这些 run 全部被排除"
                         "（同 seed 重试不算独立重复，也不择优）。")
    lines += ["", "## 5. 解读边界", "",
              "- 单种子阶段只能作为筛查：`screening_*` 判定不能当作论文结论。",
              "- 数据/源码/环境指纹必须唯一；出现 `mixed_*` 的任务判定为 `incomparable`。",
              "- 未配对级别（与 baseline 无同 seed）不参与等价判定，网格按未完成处理。",
              "- 零方差配对差（各种子准确率完全相同）会得到零宽 CI 并自动通过非劣性；"
              "这只是一致性重跑的证据，不构成额外的统计强度。",
              "- 同一 (臂, 种子) 的重复 run 属未声明重试：全部排除，不择优；"
              "`--allow-duplicate-seed` 仅用于诊断。",
              "- 正式 test 仍未读取；冻结配置后才能单独评估。", ""]
    return "\n".join(lines)


def write_summary(campaign_dir, summary_root, overwrite=False):
    summary = summarize(campaign_dir)
    output = Path(summary_root) / summary["campaign"]
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite sensitivity summary: {output}")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "summary.csv", "levels.csv", "report.md"):
            stale = output / name
            if stale.is_file():
                stale.unlink()
    else:
        output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "arm", "factor", "level", "n", "accuracy_mean", "accuracy_std",
                  "accuracy_ci95", "acceptance_coverage_final", "explanation_coverage_final",
                  "gate_coverage_final", "exposure_coverage_mean", "boundary_share_final",
                  "validation_auc", "optimizer_steps", "statuses", "duplicate_seeds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task, payload in summary["tasks"].items():
            for row in payload["arms"]:
                record = {key: _csv_value(row.get(key)) for key in fields}
                record["optimizer_steps"] = ",".join(
                    str(value) for value in row.get("optimizer_steps", []))
                record["statuses"] = ",".join(str(value) for value in row.get("statuses", []))
                record["duplicate_seeds"] = ",".join(
                    str(value) for value in row.get("duplicate_seeds", []))
                writer.writerow(record)
    with (output / "levels.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "factor", "level", "level_index", "value", "arm", "n", "paired",
                  "accuracy_mean", "accuracy_std", "accuracy_ci95", "delta_mean",
                  "delta_ci95", "delta_lower", "delta_upper", "acceptance_coverage",
                  "explanation_coverage", "gate_coverage", "exposure_coverage",
                  "boundary_share", "coverage_ok", "within_margin", "non_inferior",
                  "valid", "confirmed_valid", "verdict"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task, payload in summary["tasks"].items():
            for item in payload["factors"]:
                for row in item["levels"]:
                    record = dict(row)
                    record["task"] = task
                    record["factor"] = item["factor"]
                    record["verdict"] = item["verdict"]
                    writer.writerow({key: _csv_value(record.get(key)) for key in fields})
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary, output
