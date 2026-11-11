import json
import tempfile
import time
import unittest
from pathlib import Path

from methods.frt_abl.efficiency import (EfficiencyRecorder, MemoryMonitor,
                                        parse_seconds, process_tree_memory)
from methods.frt_abl.summarize_efficiency import (accuracy_at_budget, build_summary,
                                                  resample_curve, time_to_target)


def _points():
    return [
        {"round": 0, "elapsed_seconds": 0.0, "accuracy": 0.1,
         "candidate_checks": 0, "search_calls": 0, "inference_forwards": 0,
         "optimizer_steps": 0},
        {"round": 1, "elapsed_seconds": 100.0, "accuracy": 0.7,
         "candidate_checks": 1000, "search_calls": 10, "inference_forwards": 20,
         "optimizer_steps": 10},
        {"round": 2, "elapsed_seconds": 200.0, "accuracy": 0.9,
         "candidate_checks": 2000, "search_calls": 20, "inference_forwards": 40,
         "optimizer_steps": 20},
    ]


class DurationTests(unittest.TestCase):
    def test_parse_seconds(self):
        self.assertEqual(parse_seconds("90"), 90.0)
        self.assertEqual(parse_seconds("5m"), 300.0)
        self.assertEqual(parse_seconds("1h30m"), 5400.0)
        self.assertEqual(parse_seconds(1.5), 1.5)
        with self.assertRaises(ValueError):
            parse_seconds("0")
        with self.assertRaises(ValueError):
            parse_seconds("-1m")


class CurveTests(unittest.TestCase):
    def test_time_to_target_upper_bound_and_interpolation(self):
        result = time_to_target(_points(), 0.8)
        self.assertTrue(result["reached"])
        self.assertEqual(result["first_hit_seconds"], 200.0)
        self.assertAlmostEqual(result["interpolated_seconds"], 150.0)
        self.assertEqual(result["candidate_checks_at_hit"], 2000)

    def test_time_to_target_censoring(self):
        result = time_to_target(_points(), 0.99)
        self.assertFalse(result["reached"])
        self.assertTrue(result["censored"])
        self.assertEqual(result["lower_bound_seconds"], 200.0)
        self.assertIsNone(result["first_hit_seconds"])

    def test_accuracy_at_budget_last_and_best(self):
        result = accuracy_at_budget(_points(), 150.0)
        self.assertTrue(result["available"])
        self.assertEqual(result["last_accuracy"], 0.7)
        self.assertEqual(result["best_accuracy"], 0.7)
        self.assertEqual(result["candidate_checks_at_budget"], 1000)
        late = accuracy_at_budget(_points(), 1000.0)
        self.assertEqual(late["last_accuracy"], 0.9)
        self.assertEqual(late["best_accuracy"], 0.9)
        early = accuracy_at_budget(_points(), -1.0)
        self.assertFalse(early["available"])

    def test_step_resample_carries_last_value(self):
        curve = resample_curve(_points(), [0.0, 50.0, 150.0, 500.0])
        self.assertEqual([item["accuracy"] for item in curve], [0.1, 0.1, 0.7, 0.9])


class RecorderTests(unittest.TestCase):
    def test_recorder_writes_atomic_json_and_crossings(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {"task": "mnist-add2", "variant": "full", "seed": 4,
                   "method_version": "test", "track": "METHOD_LABEL_BUDGET",
                   "device": "cpu", "threads": 2, "search_workers": 2}
            recorder = EfficiencyRecorder(Path(directory), cfg, "run-test",
                                          targets=[0.8, 0.9],
                                          time_budget_seconds=1000.0,
                                          sample_interval_seconds=0.05)
            recorder.start()
            for index, accuracy in enumerate([0.5, 0.85, 0.91]):
                recorder.begin_round(index)
                with recorder.phase("update"):
                    time.sleep(0.005)
                recorder.set_counters(optimizer_steps=(index + 1) * 10,
                                      candidate_checks=(index + 1) * 1000,
                                      search_calls=(index + 1) * 10,
                                      proposal_forwards=(index + 1) * 20,
                                      training_forwards=(index + 1) * 5,
                                      validation_forwards=(index + 1) * 2)
                recorder.record_round({"round": index, "accuracy": accuracy})
            summary = recorder.finish("TRAINING_COMPLETED_TEST_NOT_RUN")
            self.assertEqual(summary["target_crossings"]["0.800000"]["round"], 1)
            self.assertEqual(summary["target_crossings"]["0.900000"]["round"], 2)
            payload = json.loads((Path(directory) / "efficiency.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(payload["rounds"][2]["counters"]["optimizer_steps"], 30)
            self.assertIn("update", payload["rounds"][0]["round_phase_seconds"])
            self.assertTrue((Path(directory) / "efficiency_curve.csv").is_file())
            self.assertGreaterEqual(summary["elapsed_seconds"], 0.0)

    def test_memory_monitor_returns_structured_peak(self):
        monitor = MemoryMonitor(interval_seconds=0.05)
        monitor.start()
        time.sleep(0.1)
        result = monitor.stop()
        self.assertIn("process_tree_rss_peak_mb", result)
        self.assertIn("gpu", result)
        snapshot = process_tree_memory()
        self.assertIn("processes_observed", snapshot)


def _make_run(root, task, variant, seed, points, hardware="hw-test"):
    run_dir = Path(root) / f"run-{task}-{variant}-{seed}"
    run_dir.mkdir()
    cfg = {"task": task, "variant": variant, "seed": seed,
           "method_version": "test", "output_root": "x"}
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    rounds = []
    for index, (elapsed, accuracy) in enumerate(points):
        counters = {"optimizer_steps": (index + 1) * 10,
                    "candidate_checks": (index + 1) * 1000,
                    "search_calls": (index + 1) * 10,
                    "inference_forwards": (index + 1) * 20,
                    "inference_images": (index + 1) * 200}
        rounds.append({"round": index, "elapsed_seconds": elapsed,
                       "accuracy": accuracy, "correct": int(accuracy * 1000),
                       "total": 1000, "partial_round": False, "counters": counters})
    efficiency = {
        "schema_version": 1, "status": "COMPLETED", "run_id": run_dir.name,
        "method_version": "test", "task": task, "variant": variant, "seed": seed,
        "clock": {"elapsed_seconds": points[-1][0], "time_budget_seconds": None},
        "targets": [0.8, 0.9], "target_crossings": {},
        "counters": rounds[-1]["counters"],
        "phase_seconds": {"setup": 1.0, "proposal": 100.0},
        "peak": {"process_tree_rss_peak_mb": 500.0,
                 "parent_peak_working_set_mb": 400.0,
                 "gpu": {"available": False, "peak_allocated_mb": None}},
        "hardware": {"fingerprint": hardware}, "rounds": rounds,
    }
    (run_dir / "efficiency.json").write_text(json.dumps(efficiency), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "TRAINING_COMPLETED_TEST_NOT_RUN",
        "best_validation_accuracy": max(item[1] for item in points)}),
        encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({
        "python": "3", "torch": "2", "numpy": "2", "device": "cpu", "threads": 2}),
        encoding="utf-8")
    (run_dir / "source_manifest.json").write_text(json.dumps({"efficiency.py": "hash"}),
                                                  encoding="utf-8")


class SummaryTests(unittest.TestCase):
    def test_end_to_end_summary_and_censoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            root.mkdir()
            full = [(0, 0.1), (100, 0.7), (200, 0.9)]
            standard = [(0, 0.1), (200, 0.6), (400, 0.8)]
            for seed in (4, 5):
                _make_run(root, "mnist-add2", "full", seed, full)
                _make_run(root, "mnist-add2", "standard_abl", seed, standard)
            (root / "efficiency_protocol.json").write_text(json.dumps({
                "targets": {"mnist-add2": [0.8, 0.9]},
                "budgets_seconds": [150, 300]}), encoding="utf-8")
            output = Path(directory) / "summary"
            summary = build_summary(root, output, None, [150, 300], grid_points=8)
            self.assertEqual(summary["n_usable_runs"], 4)
            self.assertEqual(len(summary["groups"]), 2)
            full_group = next(group for group in summary["groups"]
                              if group["variant"] == "full")
            target_09 = next(item for item in full_group["time_to_target"]
                             if item["target"] == 0.9)
            self.assertEqual(target_09["n_reached"], 2)
            self.assertEqual(target_09["n_runs"], 2)
            standard_group = next(group for group in summary["groups"]
                                  if group["variant"] == "standard_abl")
            target_09_std = next(item for item in standard_group["time_to_target"]
                                 if item["target"] == 0.9)
            self.assertEqual(target_09_std["n_reached"], 0)
            self.assertEqual(target_09_std["n_censored"], 2)
            budget_300 = next(item for item in standard_group["budget_accuracy"]
                              if item["budget_seconds"] == 300)
            self.assertEqual(budget_300["mean_last_accuracy"], 0.6)
            self.assertTrue((output / "EFFICIENCY_REPORT.md").is_file())
            self.assertTrue((output / "curves_mean.csv").is_file())
            self.assertTrue((output / "time_to_target.csv").is_file())
            self.assertEqual(len(summary["paired"]), 1)


if __name__ == "__main__":
    unittest.main()
