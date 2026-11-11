"""Synthetic checks for the correction-reliability analysis; no dataset or training."""
import csv
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

from methods.frt_abl.core import ArithmeticKB
from methods.frt_abl.reliability import (coverage_grid, cumulative_errors,
                                         curve_from_round, hwf_symbol_index,
                                         line_chart_svg, load_weak_role,
                                         operating_metrics, paired_summary,
                                         read_round_decisions, risk_auc,
                                         RoundData, summarize, student_t_crit)
from methods.frt_abl.run_reliability import (decision_rounds, matched_coverage_rows,
                                             parse_int_list, parse_seed_list,
                                             selected_rounds)

DIGITS = list("0123456789")
SYMBOLS = list("0123456789+-*/")


def round_data(scores, errors, task_errors=None, accepted=2, accepted_errors=1,
               trained=1, trained_errors=1):
    task_errors = errors if task_errors is None else task_errors
    return RoundData(
        round_index=0, n_total=3, n_supported=len(scores),
        n_supported_errors=sum(errors),
        n_supported_task_errors=sum(task_errors),
        n_accepted=accepted, n_accepted_errors=accepted_errors,
        n_accepted_task_errors=accepted_errors,
        n_trained=trained, n_trained_errors=trained_errors,
        n_trained_task_errors=trained_errors,
        scores={"lower": scores, "q": [0.5] * len(scores)}, errors=errors,
        task_errors=task_errors, sample_ids=[f"s{index}" for index in range(len(scores))])


def decision_text(rows, fields=("sample_id", "candidate", "region", "trained", "lower", "q")):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([row[name] for name in fields])
    return buffer.getvalue()


class MetricTests(unittest.TestCase):
    def test_coverage_grid_is_predeclared_and_inside_unit_interval(self):
        grid = coverage_grid(0.05, 1.0, 0.05)
        self.assertEqual(len(grid), 20)
        self.assertAlmostEqual(grid[0], 0.05)
        self.assertAlmostEqual(grid[-1], 1.0)
        with self.assertRaises(ValueError):
            coverage_grid(0.0, 1.0, 0.1)
        with self.assertRaises(ValueError):
            coverage_grid(0.5, 0.2, 0.1)

    def test_student_t_critical_values(self):
        self.assertEqual(student_t_crit(5), 2.776)
        self.assertEqual(student_t_crit(31), 1.960)
        self.assertIsNone(student_t_crit(1))

    def test_ranked_cumulative_errors_use_score_then_sample_id(self):
        cumulative, cumulative_task = cumulative_errors(
            [0.5, 0.9, 0.5], [1, 0, 1], [1, 0, 0], ["b", "a", "a"])
        # Order: (0.9, a), then the 0.5 tie by sample id: a before b.
        self.assertEqual(cumulative, [0, 0, 1, 2])
        self.assertEqual(cumulative_task, [0, 0, 0, 1])

    def test_curve_point_is_matched_by_k_and_marks_infeasible_coverages(self):
        data = round_data([0.9, 0.2], [0, 1])
        curve = curve_from_round(data, "lower", [0.5, 1.0])
        first, second = curve["points"]
        # N=3, floor(0.5*3)=1 supervised sample -> one correct -> zero risk.
        self.assertTrue(first["feasible"])
        self.assertEqual(first["supervised"], 1)
        self.assertEqual(first["error_rate"], 0.0)
        # k=3 exceeds the two supported candidates.
        self.assertFalse(second["feasible"])
        self.assertIsNone(second["error_rate"])
        self.assertIsNone(curve["risk_auc"])
        self.assertEqual(curve["distinct_scores"], 2)
        self.assertEqual(curve["tie_share"], 0.0)
        tied = curve_from_round(round_data([0.5, 0.5], [0, 1]), "lower", [0.5])
        self.assertEqual(tied["distinct_scores"], 1)
        self.assertAlmostEqual(tied["tie_share"], 0.5)

    def test_curve_risk_decomposition(self):
        data = round_data([0.9, 0.8, 0.7], [1, 1, 0], [0, 1, 0])
        curve = curve_from_round(data, "lower", [1.0])
        point = curve["points"][0]
        self.assertEqual(point["supervised"], 3)
        self.assertAlmostEqual(point["error_rate"], 2 / 3)
        self.assertAlmostEqual(point["correction_accuracy"], 1 / 3)
        self.assertAlmostEqual(point["erroneous_supervision_fraction_of_all"], 2 / 3)
        self.assertAlmostEqual(point["task_error_rate"], 1 / 3)
        self.assertIsNone(curve["risk_auc"])

    def test_risk_auc_trapezoid(self):
        points = [
            {"feasible": True, "coverage": 0.2, "error_rate": 0.2},
            {"feasible": True, "coverage": 0.4, "error_rate": 0.0},
            {"feasible": False, "coverage": 0.6, "error_rate": None},
        ]
        self.assertAlmostEqual(risk_auc(points), 0.2 * (0.2 + 0.0) / 2)

    def test_operating_metrics(self):
        data = round_data([0.9, 0.2], [1, 0], accepted=1, accepted_errors=1,
                          trained=1, trained_errors=1)
        metrics = operating_metrics(data)
        self.assertAlmostEqual(metrics["coverage"], 1 / 3)
        self.assertAlmostEqual(metrics["supported_coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["error_rate"], 1.0)
        self.assertAlmostEqual(metrics["erroneous_supervision_fraction_of_all"], 1 / 3)
        self.assertAlmostEqual(metrics["candidate_error_rate"], 1 / 2)
        self.assertAlmostEqual(metrics["trained_coverage"], 1 / 3)

    def test_summarize_and_paired_interval(self):
        summary = summarize([0.2, None, 0.4])
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["mean"], 0.3)
        self.assertIsNone(summarize([])["mean"])
        values = [0.3, 0.1, 0.2, 0.4, 0.0]
        reference = [0.1, 0.1, 0.1, 0.1, 0.1]
        paired = paired_summary(values, reference)
        deltas = [value - base for value, base in zip(values, reference)]
        self.assertEqual(paired["n"], 5)
        self.assertAlmostEqual(paired["mean_delta"], statistics.fmean(deltas))
        self.assertAlmostEqual(
            paired["ci95"], 2.776 * statistics.stdev(deltas) / math.sqrt(5))
        self.assertEqual(paired["n_negative"], 1)
        self.assertEqual(paired["n_positive"], 3)
        self.assertEqual(paired["n_zero"], 1)


class DecisionLogTests(unittest.TestCase):
    def setUp(self):
        self.weak = {
            "a": {"id": "a", "images": ["train:0", "train:1"], "target": 3},
            "b": {"id": "b", "images": ["train:2", "train:3"], "target": 4},
            "c": {"id": "c", "images": ["train:4", "train:5"], "target": 5},
        }
        self.truth = {"a": (1, 2), "b": (1, 3), "c": (0, 0)}
        self.oracle = ArithmeticKB("mnist-add2", DIGITS)
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, text, name="decisions_round_000.csv"):
        path = Path(self.temp.name) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_supported_accepted_and_trained_subsets(self):
        path = self.write(decision_text([
            {"sample_id": "a", "candidate": json.dumps([1, 2]), "region": "positive",
             "trained": "True", "lower": 0.9, "q": 0.8},
            {"sample_id": "b", "candidate": json.dumps([1, 2]), "region": "positive",
             "trained": "False", "lower": 0.95, "q": 0.9},
            {"sample_id": "c", "candidate": "null", "region": "boundary",
             "trained": "False", "lower": 0.0, "q": 0.0},
        ]))
        data = read_round_decisions(path, self.weak, self.truth, self.oracle)
        self.assertEqual(data.n_total, 3)
        self.assertEqual(data.n_supported, 2)
        self.assertEqual(data.n_supported_errors, 1)
        self.assertEqual(data.n_accepted, 2)
        self.assertEqual(data.n_accepted_errors, 1)
        self.assertEqual(data.n_trained, 1)
        self.assertEqual(data.errors, [0, 1])
        self.assertEqual(data.task_errors, [0, 1])

    def test_rejects_partial_logs(self):
        path = self.write(decision_text([
            {"sample_id": "a", "candidate": json.dumps([1, 2]), "region": "positive",
             "trained": "True", "lower": 0.9, "q": 0.8},
        ]))
        with self.assertRaises(ValueError):
            read_round_decisions(path, self.weak, self.truth, self.oracle)

    def test_rejects_unsupported_acceptance(self):
        path = self.write(decision_text([
            {"sample_id": "a", "candidate": "null", "region": "positive",
             "trained": "False", "lower": 0.0, "q": 0.0},
            {"sample_id": "b", "candidate": json.dumps([1, 3]), "region": "positive",
             "trained": "True", "lower": 0.9, "q": 0.9},
            {"sample_id": "c", "candidate": "null", "region": "boundary",
             "trained": "False", "lower": 0.0, "q": 0.0},
        ]))
        with self.assertRaises(ValueError):
            read_round_decisions(path, self.weak, self.truth, self.oracle)

    def test_rejects_missing_columns(self):
        path = self.write(decision_text([
            {"sample_id": "a", "candidate": "null", "region": "boundary",
             "trained": "False", "lower": 0.0},
        ], fields=("sample_id", "candidate", "region", "trained", "lower")))
        with self.assertRaises(ValueError):
            read_round_decisions(path, self.weak, self.truth, self.oracle)

    def test_round_discovery_and_selection(self):
        self.write("", "decisions_round_000.csv")
        self.write("", "decisions_round_002.csv")
        self.assertEqual(decision_rounds(self.temp.name), [0, 2])
        self.assertEqual(selected_rounds([0, 2], "last"), [2])
        self.assertEqual(selected_rounds([0, 2], "0,2"), [0, 2])
        with self.assertRaises(ValueError):
            selected_rounds([0, 2], "1")
        self.assertEqual(parse_int_list("2, 0"), [0, 2])
        self.assertEqual(parse_seed_list(["0,1", "3"]), [0, 1, 3])
        with self.assertRaises(ValueError):
            parse_seed_list([])


class TruthAndPlotTests(unittest.TestCase):
    def test_hwf_symbol_index(self):
        self.assertEqual(hwf_symbol_index("2/a.png", SYMBOLS), 2)
        self.assertEqual(hwf_symbol_index("times/a.png", SYMBOLS), 12)
        self.assertEqual(hwf_symbol_index("div/a.png", SYMBOLS), 13)
        with self.assertRaises(ValueError):
            hwf_symbol_index("unknown/a.png", SYMBOLS)

    def test_load_weak_role_for_hwf(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared"
            prepared.mkdir()
            weak = [{"id": "expr-1", "images": ["2/a.png", "times/b.png", "3/c.png"],
                     "target": 6}]
            (prepared / "weak.json").write_text(json.dumps(weak), encoding="utf-8")
            cfg = {"task": "hwf", "prepared_root": str(prepared), "symbols": SYMBOLS}
            samples, truth = load_weak_role(cfg)
            self.assertEqual(len(samples), 1)
            self.assertEqual(truth["expr-1"], (2, 12, 3))

    def test_svg_chart_is_written_with_series_and_tolerates_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.svg"
            line_chart_svg(
                path, "Risk--coverage", "synthetic",
                series=[{"label": "full", "points": [(0.2, 0.4), (0.4, None), (0.6, 0.1)],
                         "std": [(0.2, 0.05), (0.6, 0.02)]},
                        {"label": "confidence_only", "points": [(0.2, 0.3), (0.6, 0.2)]}],
                x_label="coverage", y_label="risk", x_ticks=[0.2, 0.4, 0.6],
                note="synthetic")
            svg = path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("full", svg)
            self.assertIn("confidence_only", svg)
            self.assertIn("<polyline", svg)
            self.assertIn("Risk--coverage", svg)


class MatchedCoverageTests(unittest.TestCase):
    @staticmethod
    def run_payload(task, variant, seed, error_rate, auc, feasible=True):
        return {"task": task, "variant": variant, "seed": seed, "rounds": {"0": {
            "operating": {"pseudo_supervision": True}, "scores": {"lower": {
                "risk_auc": auc, "points": [{
                    "coverage_target": 0.5, "k": 3, "feasible": feasible,
                    "coverage": 0.5, "supervised": 3,
                    "correction_accuracy": 1 - error_rate if feasible else None,
                    "error_rate": error_rate if feasible else None,
                    "erroneous_supervision_fraction_of_all":
                        error_rate / 2 if feasible else None,
                    "task_error_rate": error_rate if feasible else None,
                }]}}}}}

    def test_paired_delta_and_seed_wins(self):
        results = {
            "full-0": self.run_payload("t", "full", 0, 0.4, 0.3),
            "full-1": self.run_payload("t", "full", 1, 0.2, 0.2),
            "conf-0": self.run_payload("t", "confidence_only", 0, 0.3, 0.25),
            "conf-1": self.run_payload("t", "confidence_only", 1, 0.1, 0.15),
        }
        rows, auc, skipped = matched_coverage_rows(
            results, ["t"], [0, 1], ["full", "confidence_only"], ["lower"], [0.5])
        self.assertEqual(len(rows), 2)
        self.assertEqual(skipped.get("infeasible_coverage", 0), 0)
        conf = next(row for row in rows if row["variant"] == "confidence_only")
        self.assertAlmostEqual(conf["mean_error_rate"], 0.2)
        self.assertAlmostEqual(conf["mean_delta_error_rate_vs_full"], -0.1)
        self.assertEqual(conf["n_variant_better"], 2)
        self.assertEqual(conf["n_full_better"], 0)
        full = next(row for row in rows if row["variant"] == "full")
        self.assertIsNone(full["mean_delta_error_rate_vs_full"])
        self.assertEqual(len(auc), 2)

    def test_infeasible_point_is_skipped_not_mixed(self):
        results = {
            "full-0": self.run_payload("t", "full", 0, 0.4, 0.3),
            "full-1": self.run_payload("t", "full", 1, 0.2, 0.2),
            "conf-0": self.run_payload("t", "confidence_only", 0, 0.3, 0.25),
            "conf-1": self.run_payload("t", "confidence_only", 1, 0.1, 0.15,
                                       feasible=False),
        }
        rows, _, skipped = matched_coverage_rows(
            results, ["t"], [0, 1], ["full", "confidence_only"], ["lower"], [0.5])
        self.assertEqual(rows, [])
        self.assertEqual(skipped.get("infeasible_coverage"), 1)


    @staticmethod
    def run_payload_range(task, variant, seed, rates):
        points = []
        for coverage, error_rate in rates.items():
            points.append({
                "coverage_target": coverage, "k": int(coverage * 6),
                "feasible": True, "coverage": coverage, "supervised": int(coverage * 6),
                "correction_accuracy": 1 - error_rate, "error_rate": error_rate,
                "erroneous_supervision_fraction_of_all": error_rate / 2,
                "task_error_rate": error_rate,
            })
        return {"task": task, "variant": variant, "seed": seed, "rounds": {"0": {
            "operating": {"pseudo_supervision": True},
            "scores": {"lower": {"risk_auc": None, "points": points}}}}}

    def test_auc_uses_the_common_feasible_coverage_range(self):
        results = {
            "full-0": self.run_payload_range("t", "full", 0, {0.5: 0.4, 1.0: 0.3}),
            "full-1": self.run_payload_range("t", "full", 1, {0.5: 0.2, 1.0: 0.1}),
            "conf-0": self.run_payload_range("t", "confidence_only", 0,
                                             {0.5: 0.3, 1.0: 0.2}),
            "conf-1": self.run_payload_range("t", "confidence_only", 1,
                                             {0.5: 0.1, 1.0: 0.0}),
        }
        rows, auc, skipped = matched_coverage_rows(
            results, ["t"], [0, 1], ["full", "confidence_only"], ["lower"], [0.5, 1.0])
        self.assertEqual(skipped.get("infeasible_coverage", 0), 0)
        self.assertEqual(len(rows), 4)
        conf = next(row for row in auc if row["variant"] == "confidence_only")
        self.assertEqual(conf["auc_points"], 2)
        self.assertAlmostEqual(conf["auc_coverage_min"], 0.5)
        self.assertAlmostEqual(conf["auc_coverage_max"], 1.0)
        self.assertAlmostEqual(conf["mean_risk_auc"], (0.125 + 0.025) / 2)
        self.assertAlmostEqual(conf["mean_delta_risk_auc_vs_full"], -0.05)


if __name__ == "__main__":
    unittest.main()
