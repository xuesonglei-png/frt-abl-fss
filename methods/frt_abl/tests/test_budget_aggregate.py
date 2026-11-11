"""Unit tests for budget-curve aggregation and low-label stability summaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from methods.frt_abl.aggregate import COMPLETE
from methods.frt_abl.aggregate_budget import (collect, curve_rows, history_summary,
                                              load_v10_reference, main as aggregate_main,
                                              mean_ci, paired_deltas, spearman,
                                              stability_rows, summarize, write_svg)
from methods.frt_abl.budget import BUDGET_VERSION, EXPERIMENT


def write_run(root, task, level, seed, accuracy, gate_open=True):
    level_label = f"{level * 100:g}pct"
    run = Path(root) / f"run-{task}-{level_label}-{seed}"
    run.mkdir(parents=True)
    config = {
        "experiment": EXPERIMENT, "experiment_version": BUDGET_VERSION,
        "task": task, "variant": "full", "seed": seed,
        "prepared_root": f"prepared/{task}-{level_label}",
        "budget": {"level": level, "level_label": level_label,
                   "actual_fraction": level + 0.0002,
                   "actual_total_unique_labeled_images": int(level * 100000),
                   "train_image_universe": 100000},
    }
    status = {"status": COMPLETE, "best_validation_accuracy": accuracy,
              "stopped_reason": None, "optimizer_steps": 100}
    environment = {"code_revision": "rev", "prepared_manifest_sha256": f"data-{level}",
                   "python": "3.11", "torch": "2.5", "numpy": "1.26",
                   "device": "cpu", "threads": 2}
    sources = {"train.py": "sha"}
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (run / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    (run / "source_manifest.json").write_text(json.dumps(sources), encoding="utf-8")
    rows = []
    for round_index in range(4):
        rows.append({"round": round_index, "alpha": 0.95 if gate_open else None,
                     "validation_task": {"accuracy": accuracy - 0.01 * (3 - round_index)},
                     "coverage": 0.5, "unsupported_weak": 10,
                     "accepted_candidates": 20, "calibration_fit_total": 30,
                     "round_steps": 5})
    (run / "history.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return run


class HistoryTests(unittest.TestCase):
    def test_history_summary_records_gate_and_round_instability(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(temporary, "mnist-add2", 0.01, 0, 0.80, gate_open=False)
            summary = history_summary(run)
            self.assertEqual(summary["rounds"], 4)
            self.assertEqual(summary["gate_open_rate"], 0.0)
            self.assertAlmostEqual(summary["final_accuracy"], 0.80)
            self.assertGreater(summary["round_accuracy_std"], 0)

    def test_mean_ci_uses_student_t_for_five_seeds(self):
        mean, std, ci = mean_ci([0.80, 0.82, 0.84, 0.86, 0.88])
        self.assertAlmostEqual(mean, 0.84)
        self.assertAlmostEqual(ci, 2.776 * std / 5 ** 0.5)

    def test_spearman_handles_monotone_and_tied_values(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [0.7, 0.8, 0.9]), 1.0)
        self.assertIsNone(spearman([1, 2], [0.7, 0.8]))


class AggregationTests(unittest.TestCase):
    def build_groups(self, root):
        for seed, accuracy in enumerate((0.80, 0.82, 0.84)):
            write_run(root, "mnist-add2", 0.01, seed, accuracy, gate_open=False)
        for seed, accuracy in enumerate((0.90, 0.91, 0.92)):
            write_run(root, "mnist-add2", 0.10, seed, accuracy, gate_open=True)
        return collect(root)

    def test_collect_and_summarize_keep_levels_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            groups, seen, skipped, warnings = self.build_groups(temporary)
            self.assertEqual(seen, 6)
            self.assertEqual(skipped, 0)
            self.assertEqual(warnings, [])
            rows = summarize(groups, (0, 1, 2))
            self.assertEqual(len(rows), 2)
            low = min(rows, key=lambda row: row["level"])
            high = max(rows, key=lambda row: row["level"])
            self.assertEqual(low["n"], 3)
            self.assertAlmostEqual(low["mean_validation_accuracy"], 0.82)
            self.assertAlmostEqual(high["mean_validation_accuracy"], 0.91)
            self.assertAlmostEqual(low["ci95_validation_accuracy"],
                                   4.303 * 0.02 / 3 ** 0.5)

    def test_curves_and_paired_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            groups, _seen, _skipped, _warnings = self.build_groups(temporary)
            rows = summarize(groups, (0, 1, 2))
            curves, enriched = curve_rows(rows)
            self.assertEqual(len(curves), 1)
            self.assertAlmostEqual(curves[0]["reference_level"], 0.10)
            self.assertEqual(len(curves[0]["points"]), 2)
            low = [row for row in enriched if row["level"] == 0.01][0]
            self.assertAlmostEqual(low["delta_vs_reference"], -0.09)
            paired = paired_deltas(groups, expected_seeds=(0, 1, 2))
            self.assertEqual(len(paired), 1)
            self.assertEqual(paired[0]["n"], 3)
            self.assertEqual((paired[0]["wins"], paired[0]["losses"]), (0, 3))
            self.assertAlmostEqual(paired[0]["mean_delta"], -0.09)

    def test_stability_marks_lowest_budget_and_gate_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            groups, _seen, _skipped, _warnings = self.build_groups(temporary)
            rows = summarize(groups, (0, 1, 2))
            stability = stability_rows(groups, rows)
            low = [row for row in stability if row["level"] == 0.01][0]
            high = [row for row in stability if row["level"] == 0.10][0]
            self.assertTrue(low["is_lowest"])
            self.assertFalse(high["is_lowest"])
            self.assertEqual(low["gate_open_rate"], 0.0)
            self.assertEqual(high["gate_open_rate"], 1.0)
            self.assertIsNotNone(low["round_std_mean"])

    def test_svg_writer_produces_valid_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            groups, _seen, _skipped, _warnings = self.build_groups(temporary)
            rows = summarize(groups, (0, 1, 2))
            curves, _enriched = curve_rows(rows)
            destination = Path(temporary) / "curve.svg"
            write_svg(destination, "mnist-add2", curves)
            content = destination.read_text(encoding="utf-8")
            self.assertIn("<svg", content)
            self.assertIn("</svg>", content)
            self.assertIn("标注预算", content)

    def test_v10_reference_merges_accuracy_and_budget_fraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = Path(temporary) / "summary.json"
            summary.write_text(json.dumps({
                "groups": [{"task": "mnist-add2", "variant": "full",
                            "mean_validation_accuracy": 0.9812,
                            "std_validation_accuracy": 0.0016,
                            "n": 5}]}), encoding="utf-8")
            campaign = Path(temporary) / "manifest.json"
            campaign.write_text(json.dumps({
                "v10_reference": {"mnist-add2": {
                    "labeled_images": 1620, "fraction": 0.03}}}), encoding="utf-8")
            reference = load_v10_reference(summary, campaign)
            self.assertAlmostEqual(reference["mnist-add2"]["mean_validation_accuracy"], 0.9812)
            self.assertAlmostEqual(reference["mnist-add2"]["fraction"], 0.03)
            self.assertEqual(reference["mnist-add2"]["labeled_images"], 1620)

    def test_main_writes_summary_curve_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            runs.mkdir()
            for seed, accuracy in enumerate((0.80, 0.82, 0.84)):
                write_run(runs, "mnist-add2", 0.01, seed, accuracy, gate_open=False)
            for seed, accuracy in enumerate((0.90, 0.91, 0.92)):
                write_run(runs, "mnist-add2", 0.10, seed, accuracy, gate_open=True)
            output = Path(temporary) / "summary"
            argv = ["aggregate_budget", "--runs", str(runs), "--output", str(output),
                    "--seeds", "0", "1", "2"]
            with mock.patch.object(sys, "argv", argv):
                aggregate_main()
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "curves.csv").is_file())
            self.assertTrue((output / "stability.csv").is_file())
            self.assertTrue((output / "paired_deltas.csv").is_file())
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "curve-mnist-add2.svg").is_file())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "AGGREGATED_VALIDATION_ONLY")
            self.assertEqual(len(summary["curves"]), 1)
            self.assertEqual(len(summary["curves"][0]["points"]), 2)
            self.assertEqual(len(summary["paired_deltas"]), 1)
            self.assertIn("性能—标注量曲线", (output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
