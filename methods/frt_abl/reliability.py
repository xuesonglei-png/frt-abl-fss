"""Correction-reliability analysis over per-round decision logs (offline only).

This is the analysis core of the "correction reliability" experiment: at the
same supervision coverage, compare the false-supervision ratio of the benchmark
variants, and draw the risk--coverage curve.  The module imports no torch, builds
no model, reads only the prepared weak role plus the decision logs written by
``train.py``, and never feeds anything back into training or model selection.

Terminology (frozen for this experiment; see RELIABILITY_EXPERIMENT_2026-10-29.md)
----------------------------------------------------------------------------------
weak pool      the prepared ``weak`` role.  N is identical for every run of a task.
candidate      the abduced explanation z* logged as ``candidate``; null means the
               search found no supported explanation and no supervision is possible.
correction     an accepted candidate (x, z*) used as hard supervision.  A decision
               log does not record the raw argmax, so this experiment cannot split
               "changed" from "unchanged" candidates; every accepted candidate is a
               candidate correction, by the project's definition of (x, z*).
exact correct  z* equals the hidden weak symbol sequence (primary correctness).
task correct   value(z*) equals the observed target; logical consistency only.
coverage       supervised weak samples / N.  Operating coverage uses the positive
               region; the curve uses the top-k supported candidates ranked by a
               reliability score (lower approximation or calibrated q).
risk           erroneous supervision ratio = errors / supervised (primary risk).
all-pool risk  errors / N, i.e. the share of the whole weak pool that would receive
               a wrong correction at that coverage.

Statistics are descriptive: five design seeds, paired differences against ``full``
with a two-sided 95% t interval, no p-value claims and no distribution-free
guarantee.  Per-round curves within a run are correlated; the interval describes
seed variation only.
"""
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import statistics

from .core import ArithmeticKB
from .prepare import idx_bytes

DEFAULT_CURVE_VARIANTS = ("full", "confidence_only", "fixed_threshold",
                          "no_curriculum", "no_weight", "no_boundary")
# These variants have no gated reliability rule, so their operating point is
# reported but they are excluded from matched-coverage curve comparisons:
#   standard_abl accepts every supported candidate (weight 1), and
#   anchor_only issues no pseudo-label supervision at all.
OPERATING_ONLY_VARIANTS = ("standard_abl", "anchor_only")
SCORE_KEYS = ("lower", "q")

# Two-sided 95% critical values for Student's t; five design seeds use 2.776.
_T_CRIT_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
              8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
              14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
              20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
              26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}


def student_t_crit(n):
    """Two-sided 95% t critical value; normal approximation above 30."""
    if n < 2:
        return None
    return _T_CRIT_95.get(n, 1.960)


def load_weak_role(cfg, role="weak"):
    """Return ``(samples, truth)`` for a prepared role, without any model import.

    ``truth[sample_id]`` is the hidden symbol sequence used only for offline
    scoring.  MNIST reads the original TRAIN label file (the weak role is a
    train-only subset); HWF decodes the symbol from each image directory exactly
    as ``prepare.hwf_source`` did.  A formal test manifest is never touched.
    """
    root = Path(cfg["prepared_root"])
    samples = json.loads((root / f"{role}.json").read_text(encoding="utf-8"))
    truth = {}
    if cfg["task"] == "mnist-add2":
        import struct
        blob = idx_bytes(Path(cfg["data_root"]), "train-labels-idx1-ubyte")
        magic, count = struct.unpack(">II", blob[:8])
        if magic != 2049 or len(blob) != 8 + count:
            raise ValueError("Invalid MNIST label IDX")
        labels = blob[8:]
        for sample in samples:
            values = []
            for image in sample["images"]:
                split, index = image.split(":")
                if split != "train":
                    raise ValueError(f"Prepared weak MNIST sample is not a TRAIN image: {image}")
                position = int(index)
                if not 0 <= position < count:
                    raise ValueError(f"MNIST label index out of range: {image}")
                values.append(labels[position])
            truth[sample["id"]] = tuple(values)
    else:
        symbols = cfg["symbols"]
        for sample in samples:
            truth[sample["id"]] = tuple(hwf_symbol_index(image, symbols)
                                        for image in sample["images"])
    return samples, truth


def hwf_symbol_index(image_path, symbols):
    """Map an HWF image directory name to its position in ``symbols``."""
    name = image_path.split("/")[0]
    key = {"times": "*", "div": "/"}.get(name, name)
    if key not in symbols:
        raise ValueError(f"Unknown HWF symbol directory: {image_path}")
    return symbols.index(key)


@dataclass
class RoundData:
    """One round's weak-pool decisions, reduced to what the curve needs."""
    round_index: int
    n_total: int
    n_supported: int
    n_supported_errors: int
    n_supported_task_errors: int
    n_accepted: int
    n_accepted_errors: int
    n_accepted_task_errors: int
    n_trained: int
    n_trained_errors: int
    n_trained_task_errors: int
    scores: dict
    errors: list
    task_errors: list
    sample_ids: list


_DECISION_FIELDS = ("sample_id", "candidate", "region", "trained", "lower", "q")


def read_round_decisions(path, weak_by_id, truth, oracle, round_index=None):
    """Parse one ``decisions_round_*.csv`` and join hidden truth offline.

    Rows with ``candidate`` null are unsupported and cannot be supervised; they
    still count in ``n_total`` (coverage denominator) but not in the ranking.
    The header and the exact weak-id set are validated so a partial or mixed log
    cannot silently change the denominator.
    """
    path = Path(path)
    ids, lower, q = [], [], []
    errors, task_errors = [], []
    n_total = n_supported_errors = n_supported_task_errors = 0
    n_accepted = n_accepted_errors = n_accepted_task_errors = 0
    n_trained = n_trained_errors = n_trained_task_errors = 0
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Empty decision log: {path}") from error
        position = {name: index for index, name in enumerate(header)}
        missing = [name for name in _DECISION_FIELDS if name not in position]
        if missing:
            raise ValueError(f"Decision log {path.name} lacks columns: {missing}")
        for row in reader:
            sample_id = row[position["sample_id"]]
            if sample_id not in weak_by_id:
                raise ValueError(f"Decision log {path.name} has an unknown sample id: {sample_id}")
            if sample_id in seen:
                raise ValueError(f"Decision log {path.name} repeats a sample id: {sample_id}")
            seen.add(sample_id)
            n_total += 1
            raw = row[position["candidate"]].strip()
            candidate = json.loads(raw) if raw not in ("", "null", "None") else None
            if candidate is not None:
                candidate = tuple(int(value) for value in candidate)
                if candidate != truth[sample_id]:
                    n_supported_errors += 1
                    errors.append(1)
                else:
                    errors.append(0)
                value = oracle.value(candidate)
                target = float(weak_by_id[sample_id]["target"])
                task_ok = value is not None and math.isclose(
                    float(value), target, rel_tol=0.0, abs_tol=1e-8)
                if not task_ok:
                    n_supported_task_errors += 1
                    task_errors.append(1)
                else:
                    task_errors.append(0)
                score_lower = float(row[position["lower"]])
                score_q = float(row[position["q"]])
                for name, score in (("lower", score_lower), ("q", score_q)):
                    if not math.isfinite(score) or not -1e-9 <= score <= 1 + 1e-9:
                        raise ValueError(f"Decision log {path.name} has an invalid {name} score")
                lower.append(score_lower)
                q.append(score_q)
                ids.append(sample_id)
                if row[position["region"]] == "positive":
                    n_accepted += 1
                    n_accepted_errors += errors[-1]
                    n_accepted_task_errors += task_errors[-1]
                if row[position["trained"]] == "True":
                    n_trained += 1
                    n_trained_errors += errors[-1]
                    n_trained_task_errors += task_errors[-1]
            if row[position["region"]] == "positive" and candidate is None:
                raise ValueError(f"Decision log {path.name} accepts an unsupported candidate")
            if row[position["trained"]] == "True" and candidate is None:
                raise ValueError(f"Decision log {path.name} trains on an unsupported candidate")
    if seen != set(weak_by_id):
        missing_ids = len(set(weak_by_id) - seen)
        raise ValueError(f"Decision log {path.name} misses {missing_ids} weak samples")
    if n_accepted > len(ids) or n_trained > len(ids):
        raise ValueError(f"Decision log {path.name} supervises unsupported candidates")
    return RoundData(
        round_index=round_index, n_total=n_total, n_supported=len(ids),
        n_supported_errors=n_supported_errors,
        n_supported_task_errors=n_supported_task_errors,
        n_accepted=n_accepted, n_accepted_errors=n_accepted_errors,
        n_accepted_task_errors=n_accepted_task_errors,
        n_trained=n_trained, n_trained_errors=n_trained_errors,
        n_trained_task_errors=n_trained_task_errors,
        scores={"lower": lower, "q": q}, errors=errors, task_errors=task_errors,
        sample_ids=ids)


def cumulative_errors(scores, errors, task_errors, sample_ids):
    """Return cumulative (exact, task) error counts for the score ranking.

    Ties are broken by sample id so the ordering -- and therefore every curve
    point -- is reproducible without RNG or file-order dependence.
    """
    if not (len(scores) == len(errors) == len(task_errors) == len(sample_ids)):
        raise ValueError("Ranking arrays must have equal lengths")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], sample_ids[i]))
    cumulative = [0] * (len(order) + 1)
    cumulative_task = [0] * (len(order) + 1)
    for position, index in enumerate(order, 1):
        cumulative[position] = cumulative[position - 1] + errors[index]
        cumulative_task[position] = cumulative_task[position - 1] + task_errors[index]
    return cumulative, cumulative_task


def _point_metrics(cumulative, cumulative_task, k, n_total):
    errors = cumulative[k]
    task_errors = cumulative_task[k]
    return {
        "coverage": k / n_total,
        "supervised": k,
        "errors": errors,
        "error_rate": errors / k,
        "correction_accuracy": 1.0 - errors / k,
        "erroneous_supervision_fraction_of_all": errors / n_total,
        "task_errors": task_errors,
        "task_error_rate": task_errors / k,
        "task_correction_accuracy": 1.0 - task_errors / k,
        "task_erroneous_fraction_of_all": task_errors / n_total,
    }


def curve_from_round(data, score_key, coverage_grid):
    """Risk--coverage points at fixed coverage targets.

    At target coverage ``c`` the curve supervises ``k = floor(c * N)`` samples
    with the highest ``score_key``.  ``k`` depends only on the pool size, so two
    runs evaluated with the same grid supervise the same number of samples:
    this is the "matched supervision coverage" comparison.  A point is
    infeasible when the run found fewer than ``k`` supported candidates.
    """
    if score_key not in data.scores:
        raise ValueError(f"Unknown score key: {score_key}")
    cumulative, cumulative_task = cumulative_errors(
        data.scores[score_key], data.errors, data.task_errors, data.sample_ids)
    distinct = len({round(score, 12) for score in data.scores[score_key]})
    tie_share = (1.0 - distinct / len(data.scores[score_key])
                 if data.scores[score_key] else None)
    points = []
    for coverage in coverage_grid:
        k = int(math.floor(float(coverage) * data.n_total + 1e-9))
        point = {"coverage_target": float(coverage), "k": k, "feasible": False}
        if 1 <= k <= data.n_supported:
            point["feasible"] = True
            point.update(_point_metrics(cumulative, cumulative_task, k, data.n_total))
        else:
            point.update({name: None for name in (
                "coverage", "supervised", "errors", "error_rate",
                "correction_accuracy", "erroneous_supervision_fraction_of_all",
                "task_errors", "task_error_rate", "task_correction_accuracy",
                "task_erroneous_fraction_of_all")})
        points.append(point)
    return {"score": score_key, "points": points, "risk_auc": risk_auc(points),
            "distinct_scores": distinct, "tie_share": tie_share}


def risk_auc(points):
    """Partial area under the risk--coverage curve over feasible grid points.

    The integral starts at the first feasible coverage, not at zero; compare it
    only between runs evaluated with the same grid.  Lower is better.
    """
    feasible = sorted((p for p in points if p["feasible"]),
                      key=lambda p: p["coverage"])
    if len(feasible) < 2:
        return None
    area = 0.0
    for left, right in zip(feasible, feasible[1:]):
        width = right["coverage"] - left["coverage"]
        area += width * (left["error_rate"] + right["error_rate"]) / 2.0
    return area


def operating_metrics(data):
    """Metrics at the method's own decision point (positive region)."""
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "n_total": data.n_total,
        "supported_coverage": ratio(data.n_supported, data.n_total),
        "candidate_error_rate": ratio(data.n_supported_errors, data.n_supported),
        "task_candidate_error_rate": ratio(data.n_supported_task_errors, data.n_supported),
        "coverage": ratio(data.n_accepted, data.n_total),
        "supervised": data.n_accepted,
        "errors": data.n_accepted_errors,
        "error_rate": ratio(data.n_accepted_errors, data.n_accepted),
        "correction_accuracy": (1.0 - data.n_accepted_errors / data.n_accepted
                                if data.n_accepted else None),
        "erroneous_supervision_fraction_of_all": ratio(data.n_accepted_errors, data.n_total),
        "task_error_rate": ratio(data.n_accepted_task_errors, data.n_accepted),
        "task_correction_accuracy": (1.0 - data.n_accepted_task_errors / data.n_accepted
                                     if data.n_accepted else None),
        "task_erroneous_fraction_of_all": ratio(data.n_accepted_task_errors, data.n_total),
        "trained_coverage": ratio(data.n_trained, data.n_total),
        "trained_error_rate": ratio(data.n_trained_errors, data.n_trained),
    }


def summarize(values):
    """Mean/std/min/max over non-null values (descriptive only)."""
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def paired_summary(values, reference):
    """Paired difference ``values - reference`` with a descriptive 95% t interval."""
    pairs = [(float(a), float(b)) for a, b in zip(values, reference)
             if a is not None and b is not None]
    if not pairs:
        return {"n": 0, "mean_delta": None, "std_delta": None, "ci95": None,
                "n_positive": 0, "n_negative": 0, "n_zero": 0}
    deltas = [a - b for a, b in pairs]
    std = statistics.stdev(deltas) if len(deltas) > 1 else None
    critical = student_t_crit(len(deltas))
    return {"n": len(deltas), "mean_delta": statistics.fmean(deltas),
            "std_delta": std,
            "ci95": (critical * std / math.sqrt(len(deltas))
                     if std is not None and critical is not None else None),
            "n_positive": sum(delta > 0 for delta in deltas),
            "n_negative": sum(delta < 0 for delta in deltas),
            "n_zero": sum(delta == 0 for delta in deltas)}


def coverage_grid(start=0.05, stop=1.0, step=0.05):
    """Pre-declared matched-coverage grid; 1.0 (supervise every candidate) is allowed."""
    for name, value in (("start", start), ("stop", stop), ("step", step)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Coverage {name} must be positive")
    if start > stop or step < 1e-9:
        raise ValueError("Coverage grid requires start <= stop and a positive step")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    if count < 1 or count > 200:
        raise ValueError("Coverage grid must contain between 1 and 200 points")
    grid = [round(start + index * step, 10) for index in range(count)]
    if any(not 0 < value <= 1 for value in grid):
        raise ValueError("Coverage grid values must lie inside (0, 1]")
    return grid


def write_csv(path, fieldnames, rows):
    """Write dictionaries with stable column order; missing keys become empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


_SVG_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
               "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22")


def _xml_escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _nice_top(value):
    """Round a positive axis maximum up to a readable 0.05/0.1/0.2 multiple."""
    if value is None or value <= 0:
        return 0.05
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        top = math.ceil(value / step) * step
        if top / step <= 10:
            return min(round(top, 10), 1.0)
    return 1.0


def line_chart_svg(path, title, subtitle, series, x_label, y_label,
                   x_min=0.0, x_max=1.0, x_ticks=None, y_max=None,
                   note=None, width=960, height=600):
    """Dependency-free SVG line chart used for the experiment figures.

    ``series`` is a list of ``{"label": str, "points": [(x, y), ...]}`` with an
    optional ``"std": [(x, std), ...]`` for vertical error bars.  A ``None`` y
    value breaks the line into segments instead of interpolating across a gap.
    No matplotlib is required, so the figure adds no runtime dependency.
    """
    if x_max <= x_min:
        x_max = x_min + 1.0
    left, right, top, bottom = 95, width - 250, 105, height - 95
    plot_width, plot_height = right - left, bottom - top
    if plot_width <= 0 or plot_height <= 0:
        raise ValueError("Chart canvas is too small for the margins")
    if x_ticks is None:
        x_ticks = [x_min + (x_max - x_min) * index / 5 for index in range(6)]
    observed = 0.0
    for item in series:
        deviations = {x: value for x, value in (item.get("std") or [])
                      if value is not None}
        for x, y in item["points"]:
            if y is not None:
                observed = max(observed, y + max(0.0, deviations.get(x, 0.0)))
    y_top = _nice_top(y_max if y_max is not None else observed)

    def sx(value):
        return left + (float(value) - x_min) / (x_max - x_min) * plot_width

    def sy(value):
        return bottom - float(value) / y_top * plot_height

    parts = [f'<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Arial, sans-serif">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="{left}" y="46" font-size="21" font-weight="600" fill="#111827">'
             f'{_xml_escape(title)}</text>']
    if subtitle:
        parts.append(f'<text x="{left}" y="72" font-size="13" fill="#4b5563">'
                     f'{_xml_escape(subtitle)}</text>')
    # Horizontal grid and y tick labels.
    for index in range(6):
        value = y_top * index / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
                     f'stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="11" '
                     f'text-anchor="end" fill="#374151">{value:.2f}</text>')
    # Vertical grid and x tick labels.
    for value in x_ticks:
        x = sx(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
                     f'stroke="#f3f4f6" stroke-width="1"/>')
        label = f"{value:g}"
        parts.append(f'<text x="{x:.2f}" y="{bottom + 22}" font-size="11" '
                     f'text-anchor="middle" fill="#374151">{_xml_escape(label)}</text>')
    # Axes.
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
                 f'stroke="#111827" stroke-width="1.4"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
                 f'stroke="#111827" stroke-width="1.4"/>')
    parts.append(f'<text x="{(left + right) / 2:.2f}" y="{height - 46}" font-size="13" '
                 f'text-anchor="middle" fill="#111827">{_xml_escape(x_label)}</text>')
    parts.append(f'<text transform="translate(24,{(top + bottom) / 2:.2f}) rotate(-90)" '
                 f'font-size="13" text-anchor="middle" fill="#111827">'
                 f'{_xml_escape(y_label)}</text>')
    # Series.
    for order, item in enumerate(series):
        color = item.get("color") or _SVG_COLORS[order % len(_SVG_COLORS)]
        segments, current = [], []
        for x, y in item["points"]:
            if y is None:
                if len(current) > 1:
                    segments.append(current)
                current = []
                continue
            current.append((x, y))
        if len(current) > 1:
            segments.append(current)
        for segment in segments:
            coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in segment)
            parts.append(f'<polyline points="{coordinates}" fill="none" '
                         f'stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
        points = [(x, y) for x, y in item["points"] if y is not None]
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
        for x, deviation in (item.get("std") or []):
            if deviation is None or not points:
                continue
            mean = next((y for px, y in points if px == x), None)
            if mean is None or deviation <= 0:
                continue
            high, low = sy(mean + deviation), sy(max(0.0, mean - deviation))
            x_pixel = sx(x)
            parts.append(f'<line x1="{x_pixel:.2f}" y1="{high:.2f}" x2="{x_pixel:.2f}" '
                         f'y2="{low:.2f}" stroke="{color}" stroke-width="1.1"/>')
            parts.append(f'<line x1="{x_pixel - 4:.2f}" y1="{high:.2f}" '
                         f'x2="{x_pixel + 4:.2f}" y2="{high:.2f}" stroke="{color}" stroke-width="1.1"/>')
            parts.append(f'<line x1="{x_pixel - 4:.2f}" y1="{low:.2f}" '
                         f'x2="{x_pixel + 4:.2f}" y2="{low:.2f}" stroke="{color}" stroke-width="1.1"/>')
    # Legend.
    legend_x, legend_y = right + 24, top + 6
    for order, item in enumerate(series):
        color = item.get("color") or _SVG_COLORS[order % len(_SVG_COLORS)]
        y = legend_y + order * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" '
                     f'stroke="{color}" stroke-width="2.6"/>')
        parts.append(f'<text x="{legend_x + 34}" y="{y + 4}" font-size="12" fill="#111827">'
                     f'{_xml_escape(item["label"])}</text>')
    if note:
        parts.append(f'<text x="{left}" y="{height - 20}" font-size="11" fill="#6b7280">'
                     f'{_xml_escape(note)}</text>')
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path
