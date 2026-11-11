"""Grouped evaluation for the HWF length/combo generalization split.

The evaluator is intentionally thin: it calls the existing V10 evaluator (same
checkpoint binding, same formal-test approval, same native/argmax predictions)
and then joins ``predictions.csv`` with the prepared test metadata to report
task, symbol and full-sequence accuracy by length and operator group.

Inert unless ``--execute-evaluation`` is passed; ``--report-only`` recomputes the
grouped report from an existing predictions file without touching a model.
"""
import argparse
import csv
import json
from pathlib import Path

from .prepare import digest, load_config, write_json
from .prepare_generalization import (DEFAULT_GROUP_BY, ALLOWED_GROUP_BY,
                                     group_key, operator_combo,
                                     verify_generalization_prepared)

REPORT_FIELDS = (
    "count",
    "native_task_correct", "native_task_accuracy",
    "native_symbol_correct", "native_symbol_total", "native_symbol_accuracy",
    "native_sequence_correct", "native_sequence_accuracy",
    "native_covered", "native_coverage",
    "argmax_task_correct", "argmax_task_accuracy",
    "argmax_symbol_correct", "argmax_symbol_total", "argmax_symbol_accuracy",
    "argmax_sequence_correct", "argmax_sequence_accuracy",
)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Not a boolean prediction value: {value!r}")


def as_int(value):
    return int(str(value).strip())


def metadata_for_report(sample):
    metadata = sample.get("generalization")
    if isinstance(metadata, dict):
        return metadata
    combo = operator_combo(sample["images"])
    return {"length": len(sample["images"]), "combo": "".join(combo) or "()",
            "combo_group": "unknown", "contains_holdout_bigram": "unknown",
            "exact_combo_seen_in_train": "unknown"}


class _Accumulator:
    def __init__(self):
        self.count = 0
        self.native_task = 0
        self.native_symbol = 0
        self.native_symbol_total = 0
        self.native_sequence = 0
        self.native_covered = 0
        self.argmax_task = 0
        self.argmax_symbol = 0
        self.argmax_symbol_total = 0
        self.argmax_sequence = 0

    def add(self, length, row):
        self.count += 1
        self.native_task += as_bool(row["native_task_correct"])
        self.native_symbol += as_int(row["native_symbol_correct"])
        self.native_symbol_total += length
        self.native_sequence += as_bool(row["native_sequence_correct"])
        self.native_covered += as_bool(row["native_covered"])
        self.argmax_task += as_bool(row["argmax_task_correct"])
        self.argmax_symbol += as_int(row["argmax_symbol_correct"])
        self.argmax_symbol_total += length
        self.argmax_sequence += as_bool(row["argmax_sequence_correct"])

    @staticmethod
    def _ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    def row(self):
        return {
            "count": self.count,
            "native_task_correct": self.native_task,
            "native_task_accuracy": self._ratio(self.native_task, self.count),
            "native_symbol_correct": self.native_symbol,
            "native_symbol_total": self.native_symbol_total,
            "native_symbol_accuracy": self._ratio(self.native_symbol, self.native_symbol_total),
            "native_sequence_correct": self.native_sequence,
            "native_sequence_accuracy": self._ratio(self.native_sequence, self.count),
            "native_covered": self.native_covered,
            "native_coverage": self._ratio(self.native_covered, self.count),
            "argmax_task_correct": self.argmax_task,
            "argmax_task_accuracy": self._ratio(self.argmax_task, self.count),
            "argmax_symbol_correct": self.argmax_symbol,
            "argmax_symbol_total": self.argmax_symbol_total,
            "argmax_symbol_accuracy": self._ratio(self.argmax_symbol, self.argmax_symbol_total),
            "argmax_sequence_correct": self.argmax_sequence,
            "argmax_sequence_accuracy": self._ratio(self.argmax_sequence, self.count),
        }


def _sort_group_key(dimension, value):
    if dimension in {"length", "length_x_combo_group",
                     "length_x_contains_holdout_bigram"}:
        parts = value.split("|")
        numeric = int(parts[0]) if parts[0].isdigit() else -1
        return (numeric, value)
    if dimension == "combo_group":
        order = {"seen": 0, "unseen": 1, "unknown": 2}
        return (order.get(value, 3), value)
    if dimension == "contains_holdout_bigram":
        order = {"False": 0, "True": 1, "unknown": 2}
        return (order.get(value, 3), value)
    if dimension == "exact_combo_seen_in_train":
        order = {"True": 0, "False": 1, "unknown": 2}
        return (order.get(value, 3), value)
    return (value,)


def compute_group_report(predictions, samples, dimensions):
    """Purely functional report computation; no model and no file I/O."""
    sample_ids = [sample.get("id") for sample in samples]
    if any(sample_id is None for sample_id in sample_ids):
        raise ValueError("Test samples must all have an id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Test samples contain duplicate ids")
    prediction_ids = [row.get("sample_id") for row in predictions]
    if any(sample_id is None for sample_id in prediction_ids):
        raise ValueError("Every prediction must have a sample_id")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("Predictions contain duplicate sample_id values")
    expected_ids = set(sample_ids)
    actual_ids = set(prediction_ids)
    missing = expected_ids - actual_ids
    unexpected = actual_ids - expected_ids
    if missing or unexpected:
        raise ValueError(
            f"Prediction/test id mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}")
    by_id = {sample["id"]: sample for sample in samples}
    overall = _Accumulator()
    groups = {dimension: {} for dimension in dimensions}
    for row in predictions:
        sample_id = row.get("sample_id")
        sample = by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"Prediction refers to unknown test sample: {sample_id!r}")
        length = len(sample["images"])
        metadata = metadata_for_report(sample)
        overall.add(length, row)
        for dimension in dimensions:
            value = group_key(metadata, dimension)
            groups[dimension].setdefault(value, _Accumulator()).add(length, row)

    grouped = {}
    for dimension, values in groups.items():
        rows = []
        for value in sorted(values, key=lambda item: _sort_group_key(dimension, item)):
            row = {"group": value, **values[value].row()}
            rows.append(row)
        valid = [row["native_task_accuracy"] for row in rows
                 if row["native_task_accuracy"] is not None]
        grouped[dimension] = {
            "macro_native_task_accuracy": (sum(valid) / len(valid)) if valid else None,
            "groups": rows,
        }
    return {
        "overall": overall.row(),
        "group_dimensions": list(dimensions),
        "groups": grouped,
    }


def read_predictions(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_report(cfg, output, split, dimensions):
    root = Path(cfg["prepared_root"])
    test_name = "test.json" if split == "test" else "smoke_test.json"
    test_path = root / test_name
    if not test_path.is_file():
        raise FileNotFoundError(f"{test_name} not found in {root}")
    samples = json.loads(test_path.read_text(encoding="utf-8"))
    output = Path(output)
    predictions_path = output / "predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions.csv not found in {output}")
    predictions = read_predictions(predictions_path)
    report = compute_group_report(predictions, samples, dimensions)
    receipt = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    generalization = receipt.get("generalization") or {}
    primary = generalization.get("primary_eval_length")
    primary_row = None
    if primary is not None and "length" in report["groups"]:
        primary_row = next((row for row in report["groups"]["length"]["groups"]
                            if row["group"] == str(primary)), None)
    report.update({
        "status": "GROUPED_EVALUATION_REPORT",
        "split": split,
        "formal_test": split == "test",
        "prepared_root": str(root),
        "prepared_manifest_sha256": digest(root / "manifest.json"),
        "checkpoint_sha256": metrics.get("checkpoint_sha256"),
        "generalization": receipt.get("generalization"),
        "primary_eval_length": primary,
        "primary_eval": primary_row,
        "primary_outcome": "argmax_task_accuracy",
        "native_uses_ground_truth_result": True,
        "note": ("Argmax task accuracy is the primary answer-free generalization outcome. "
                 "Native consumes the true result: task accuracy mostly measures search "
                 "coverage, while symbol/sequence accuracy measures conditional recovery. "
                 "Task accuracy is result correctness; symbol accuracy is token-level; "
                 "sequence accuracy is exact full-expression correctness. Native is the "
                 "method's abduced prediction; argmax is the perception-only reference. "
                 "primary_eval is the declared OOD length group; overall mixes all "
                 "eval lengths."),
    })
    write_json(output / "metrics_by_group.json", report)
    with (output / "metrics_by_group.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "group", *REPORT_FIELDS])
        writer.writerow(["all", "all", *[report["overall"].get(field) for field in REPORT_FIELDS]])
        for dimension in dimensions:
            for row in report["groups"][dimension]["groups"]:
                writer.writerow([dimension, row["group"],
                                 *[row.get(field) for field in REPORT_FIELDS]])
    return report


def _dimensions(cfg, text):
    if text:
        dimensions = tuple(item.strip() for item in text.split(",") if item.strip())
    else:
        dimensions = tuple(cfg.get("generalization", {}).get("group_by", DEFAULT_GROUP_BY))
    unknown = [item for item in dimensions if item not in ALLOWED_GROUP_BY]
    if unknown:
        raise SystemExit(f"Unknown --group-by dimensions: {unknown}")
    if len(set(dimensions)) != len(dimensions):
        raise SystemExit("Duplicate --group-by dimensions are not allowed")
    return dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("test", "smoke"), default="test")
    parser.add_argument("--execute-evaluation", action="store_true")
    parser.add_argument("--approve-formal-test", action="store_true",
                        help="Explicitly allow reading the formal test manifest (after freezing)")
    parser.add_argument("--report-only", action="store_true",
                        help="Recompute grouped metrics from an existing predictions.csv")
    parser.add_argument("--group-by", default=None,
                        help="Comma-separated dimensions; defaults to the config's group_by")
    args = parser.parse_args()

    cfg = load_config(args.config)
    verify_generalization_prepared(cfg)
    dimensions = _dimensions(cfg, args.group_by)
    output = Path(args.output)

    if args.split == "test" and not args.approve_formal_test:
        print(json.dumps({"status": "FORMAL_TEST_REQUIRES_APPROVAL",
                          "split": "test", "test_read": False,
                          "note": "Re-run with --approve-formal-test only after the model and "
                                  "config are frozen."}))
        raise SystemExit(2)

    if args.report_only:
        report = build_report(cfg, output, args.split, dimensions)
        print(json.dumps({"status": report["status"], "overall": report["overall"],
                          "primary_eval": report.get("primary_eval"),
                          "dimensions": list(dimensions), "output": str(output)},
                         ensure_ascii=False, indent=2))
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --report-only is used")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.execute_evaluation:
        print(json.dumps({"status": "CHECKPOINT_PRESENT_NOT_EVALUATED",
                          "sha256": digest(args.checkpoint), "split": args.split},
                         ensure_ascii=False, indent=2))
        return

    from .evaluate import evaluate
    metrics = evaluate(cfg, args.checkpoint, output, split=args.split,
                       approved_formal_test=args.approve_formal_test)
    report = build_report(cfg, output, args.split, dimensions)
    print(json.dumps({"status": report["status"],
                      "overall": report["overall"],
                      "primary_eval": report.get("primary_eval"),
                      "group_dimensions": list(dimensions),
                      "dimensions": {dimension: report["groups"][dimension]["macro_native_task_accuracy"]
                                     for dimension in dimensions},
                      "output": str(output),
                      "metrics": {"task_accuracy": metrics["task_accuracy"],
                                  "symbol_accuracy": metrics["symbol_accuracy"],
                                  "sequence_accuracy": metrics["sequence_accuracy"],
                                  "coverage": metrics["coverage"]}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
