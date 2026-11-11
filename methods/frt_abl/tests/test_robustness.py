"""Robustness harness tests: grids, KB masking, noise, metrics, discovery.

No dataset, checkpoint or experiment is used; synthetic tensors and temporary
directories only.
"""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from methods.frt_abl.core import ArithmeticKB, Evidence
from methods.frt_abl.robustness import (Condition, ImageNoise, MaskedKnowledge,
                                        MetricAccumulator, PROTOCOL_VERSION,
                                        build_conditions, discover_checkpoint,
                                        knowledge_units, masked_units, summarize,
                                        validate_robustness_config, write_summary)


def make_spec():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "baseline": {"method_version": "frt-abl-v10-20261028", "variant": "full",
                     "runs_root": "runs"},
        "seeds": [4],
        "split": "validation",
        "output_root": "out",
        "summary_root": "summary",
        "noise": {"kind": "gaussian", "seed_salt": "unit-test",
                  "levels": {"gaussian": [0.0, 0.1, 0.2],
                             "salt_pepper": [0.0, 0.05]}},
        "knowledge": {"levels": [0.0, 0.2, 0.5], "units": {}},
        "search_budget": {
            "levels": [1.0, 0.5],
            "mnist-add2": {"beam_size": [256, 128], "max_checks": [100, 50]},
            "hwf": {"beam_size": [256, 128], "max_checks": [512, 256]},
        },
        "combined": {"noise_sigma": [0.1, 0.2], "knowledge_ratio": [0.2, 0.5],
                     "budget_scale": [0.5]},
    }


def make_condition(**overrides):
    values = {"condition_id": "clean", "stage": "clean", "noise_kind": "none",
              "noise_sigma": 0.0, "knowledge_ratio": 0.0, "budget_scale": 1.0,
              "beam_size": 256, "max_checks": 512}
    values.update(overrides)
    return Condition(**values)


def make_record(**overrides):
    record = {"length": 2, "raw_task_correct": True, "native_task_correct": True,
              "raw_symbol_matches": 2, "native_symbol_matches": 2,
              "raw_sequence_correct": True, "native_sequence_correct": True,
              "covered": True, "raw_syntax_valid": True, "raw_unresolved": False,
              "candidates_checked": 100, "search_complete": True,
              "unknown_candidates": 0}
    record.update(overrides)
    return record


class GridTests(unittest.TestCase):
    def test_single_stage_is_one_factor_at_a_time(self):
        conditions = build_conditions(make_spec(), "mnist-add2", "single")
        self.assertEqual([c.condition_id for c in conditions],
                         ["clean", "noise-gaussian-0.1", "noise-gaussian-0.2",
                          "knowledge-0.2", "knowledge-0.5", "budget-0.5"])
        self.assertTrue(all(c.stage in {"clean", "single"} for c in conditions))

    def test_combined_stage_is_a_factorial_with_clean_anchor(self):
        conditions = build_conditions(make_spec(), "mnist-add2", "combined")
        self.assertEqual(len(conditions), 1 + 2 * 2 * 1)
        self.assertEqual(conditions[0].condition_id, "clean")
        self.assertEqual(len({c.condition_id for c in conditions}), len(conditions))
        combo = [c for c in conditions if c.stage == "combined"]
        self.assertTrue(all(c.noise_sigma and c.knowledge_ratio and c.budget_scale < 1
                            for c in combo))

    def test_all_stage_deduplicates_the_clean_condition(self):
        single = build_conditions(make_spec(), "hwf", "single")
        combined = build_conditions(make_spec(), "hwf", "combined")
        both = build_conditions(make_spec(), "hwf", "all")
        self.assertEqual(len(both), len(single) + len(combined) - 1)
        self.assertEqual(sum(c.condition_id == "clean" for c in both), 1)

    def test_declared_budgets_are_carried_into_conditions(self):
        conditions = {c.condition_id: c for c in build_conditions(make_spec(), "mnist-add2",
                                                                  "single")}
        self.assertEqual((conditions["clean"].beam_size, conditions["clean"].max_checks),
                         (256, 100))
        self.assertEqual((conditions["budget-0.5"].beam_size,
                          conditions["budget-0.5"].max_checks), (128, 50))

    def test_invalid_configs_are_rejected(self):
        cases = []
        spec = make_spec()
        spec["baseline"]["variant"] = "standard_abl"
        cases.append(spec)
        spec = make_spec()
        spec["noise"]["levels"]["gaussian"] = [0.0, 0.2, 0.1]
        cases.append(spec)
        spec = make_spec()
        spec["knowledge"]["levels"] = [0.0, 1.0]
        cases.append(spec)
        spec = make_spec()
        spec["search_budget"]["mnist-add2"]["max_checks"] = [100]
        cases.append(spec)
        spec = make_spec()
        spec["combined"]["budget_scale"] = [0.75]
        cases.append(spec)
        spec = make_spec()
        spec["noise"]["kind"] = "uniform"
        spec["combined"]["noise_sigma"] = [0.1]
        cases.append(spec)
        for case in cases:
            with self.assertRaises(ValueError):
                validate_robustness_config(case)

    def test_invalid_condition_values_are_rejected(self):
        with self.assertRaises(ValueError):
            make_condition(noise_kind="gaussian", noise_sigma=0.0)
        with self.assertRaises(ValueError):
            make_condition(noise_kind="none", noise_sigma=0.2)
        with self.assertRaises(ValueError):
            make_condition(knowledge_ratio=1.0)
        with self.assertRaises(ValueError):
            make_condition(budget_scale=0.0)
        with self.assertRaises(ValueError):
            make_condition(stage="unknown")


class KnowledgeMaskTests(unittest.TestCase):
    def test_units_and_nested_ratio_selection(self):
        self.assertEqual(len(knowledge_units("mnist-add2")), 100)
        self.assertEqual(len(knowledge_units("hwf")), 36)
        self.assertEqual(len(masked_units("mnist-add2", 0.0)), 0)
        low = masked_units("mnist-add2", 0.1)
        high = masked_units("mnist-add2", 0.2)
        self.assertEqual(len(low), 10)
        self.assertEqual(len(high), 20)
        self.assertTrue(low < high)
        self.assertEqual(low, masked_units("mnist-add2", 0.1))
        self.assertEqual(len(masked_units("hwf", 0.25)), 9)

    def test_addition_fact_removal_stays_unresolved(self):
        kb = MaskedKnowledge("mnist-add2", list("0123456789"), 0.1)
        masked = sorted(kb.masked)[0]
        unmasked = sorted(set(knowledge_units("mnist-add2")) - kb.masked)[0]
        self.assertEqual(kb.check(masked, masked[0] + masked[1]), Evidence.UNRESOLVED)
        self.assertEqual(kb.check(unmasked, unmasked[0] + unmasked[1]), Evidence.SUPPORTED)
        self.assertEqual(kb.check(unmasked, unmasked[0] + unmasked[1] + 1), Evidence.REFUTED)
        self.assertEqual(kb.syntax_valid((0, 0)), ArithmeticKB("mnist-add2",
                                                               list("0123456789")).syntax_valid((0, 0)))

    def test_hwf_masked_operator_cell_is_unresolved(self):
        symbols = list("123456789") + ["+", "-", "*", "/"]
        kb = MaskedKnowledge("hwf", symbols, 0.5)
        operator, digit = sorted(kb.masked)[0]
        labels = (symbols.index("1"), symbols.index(operator), symbols.index(str(digit)))
        self.assertEqual(kb.check(labels, 0), Evidence.UNRESOLVED)
        self.assertIsNone(kb.value(labels))
        # Precedence is unchanged when nothing is masked.
        full = MaskedKnowledge("hwf", symbols, 0.0)
        self.assertEqual(full.value((symbols.index("2"), symbols.index("+"),
                                     symbols.index("3"), symbols.index("*"),
                                     symbols.index("4"))), 14)

    def test_masking_never_changes_syntax_validity(self):
        symbols = list("123456789") + ["+", "-", "*", "/"]
        base, masked = ArithmeticKB("hwf", symbols), MaskedKnowledge("hwf", symbols, 0.3)
        for labels in ((1, 9, 2), (1, 2, 2), (1, 9), (5,)):
            self.assertEqual(base.syntax_valid(labels), masked.syntax_valid(labels))


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.linspace(0, 1, 28 * 28).reshape(1, 28, 28)

    def test_gaussian_is_deterministic_per_image_and_nested_in_sigma(self):
        low = ImageNoise("gaussian", 0.1)
        high = ImageNoise("gaussian", 0.2)
        first = low.apply(self.image, "image-a")
        self.assertTrue(torch.equal(first, low.apply(self.image, "image-a")))
        self.assertFalse(torch.equal(first, low.apply(self.image, "image-b")))
        larger = high.apply(self.image, "image-a")
        self.assertGreater(float((larger - self.image).abs().mean()),
                           float((first - self.image).abs().mean()))
        self.assertTrue(torch.all(first >= 0) and torch.all(first <= 1))

    def test_salt_and_pepper_only_uses_extremes(self):
        noisy = ImageNoise("salt_pepper", 0.2).apply(self.image, "image-a")
        changed = noisy != self.image
        self.assertTrue(torch.all((noisy[changed] == 0) | (noisy[changed] == 1)))
        self.assertTrue(torch.equal(noisy, ImageNoise("salt_pepper", 0.2)
                                    .apply(self.image, "image-a")))

    def test_noise_free_condition_is_identity(self):
        self.assertTrue(torch.equal(self.image,
                                    ImageNoise("none", 0.0).apply(self.image, "x")))


class MetricTests(unittest.TestCase):
    def test_confusion_based_rates(self):
        accumulator = MetricAccumulator()
        accumulator.add(make_record(raw_task_correct=False, native_task_correct=True,
                                    raw_symbol_matches=1))
        accumulator.add(make_record())
        accumulator.add(make_record(native_task_correct=False, native_symbol_matches=1,
                                    covered=False, raw_syntax_valid=False,
                                    raw_unresolved=True))
        accumulator.add(make_record(length=3, covered=False, raw_symbol_matches=3,
                                    native_symbol_matches=3))
        metrics = accumulator.finalize()
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["task_correct"], 3)
        self.assertAlmostEqual(metrics["task_accuracy"], 0.75)
        self.assertEqual(metrics["symbol_total"], 9)
        self.assertEqual(metrics["symbol_correct"], 8)
        self.assertAlmostEqual(metrics["symbol_accuracy"], 8 / 9)
        self.assertEqual(metrics["raw_error_count"], 1)
        self.assertEqual(metrics["error_correction_count"], 1)
        self.assertAlmostEqual(metrics["error_correction_rate"], 1.0)
        self.assertEqual(metrics["corrupted_count"], 1)
        self.assertAlmostEqual(metrics["corruption_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["correction_precision"], 0.5)
        self.assertEqual(metrics["covered"], 2)
        self.assertAlmostEqual(metrics["undecided_rate"], 0.5)
        self.assertEqual(metrics["raw_argmax_unresolved_count"], 1)
        self.assertAlmostEqual(metrics["raw_argmax_unresolved_rate"], 0.25)
        self.assertEqual(metrics["raw_argmax_syntax_invalid_count"], 1)
        self.assertAlmostEqual(metrics["raw_argmax_syntax_invalid_rate"], 0.25)
        self.assertEqual(metrics["mean_candidates_checked"], 100)
        self.assertEqual(metrics["by_length"]["3"]["task_accuracy"], 1.0)

    def test_undefined_rates_stay_none(self):
        metrics = MetricAccumulator().finalize()
        self.assertIsNone(metrics["task_accuracy"])
        self.assertIsNone(metrics["error_correction_rate"])
        self.assertIsNone(metrics["correction_precision"])


def base_metrics():
    return {"n": 10, "task_accuracy": 0.9, "symbol_accuracy": 0.85,
            "sequence_accuracy": 0.8, "raw_argmax_task_accuracy": 0.8,
            "raw_argmax_symbol_accuracy": 0.75, "error_correction_rate": 0.6,
            "correction_precision": 0.7, "corruption_rate": 0.1, "coverage": 0.95,
            "undecided_rate": 0.05, "raw_argmax_unresolved_rate": 0.0,
            "mean_candidates_checked": 100.0, "search_complete_rate": 1.0}


def write_condition_tree(root, task, seed, condition_id, condition, metrics,
                         directory_name=None):
    directory = Path(root) / task / f"seed-{seed}" / (directory_name or condition_id)
    directory.mkdir(parents=True)
    (directory / "condition.json").write_text(json.dumps({
        "protocol_version": PROTOCOL_VERSION, "task": task, "seed": seed,
        "split": "validation", "condition": condition, "n_samples": 10,
        "provenance": {}, "note": "test"}), encoding="utf-8")
    metrics = dict(metrics, status="ROBUSTNESS_EVAL_COMPLETED", task=task,
                   seed=seed, split="validation", condition=condition)
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


class SummaryTests(unittest.TestCase):
    def test_deltas_are_condition_minus_clean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            clean = make_condition().as_dict()
            noise = make_condition(condition_id="noise-gaussian-0.1", stage="single",
                                   noise_kind="gaussian", noise_sigma=0.1).as_dict()
            base = base_metrics()
            write_condition_tree(root, "mnist-add2", 4, "clean", clean, base)
            degraded = dict(base, task_accuracy=0.7, symbol_accuracy=0.6,
                            undecided_rate=0.3, coverage=0.7, error_correction_rate=0.4)
            write_condition_tree(root, "mnist-add2", 4, "noise-gaussian-0.1", noise, degraded)
            summary = summarize(root)
            rows = {row["condition_id"]: row for row in summary["conditions"]}
            self.assertAlmostEqual(rows["noise-gaussian-0.1"]["deltas"]["task_accuracy"]["mean"],
                                   -0.2)
            self.assertAlmostEqual(rows["noise-gaussian-0.1"]["deltas"]["symbol_accuracy"]["mean"],
                                   -0.25)
            self.assertAlmostEqual(rows["noise-gaussian-0.1"]["deltas"]["undecided_rate"]["mean"],
                                   0.25)
            self.assertEqual(rows["clean"]["deltas"]["task_accuracy"]["mean"], 0.0)
            with tempfile.TemporaryDirectory() as output:
                target = Path(output) / "summary"
                written = write_summary(root, target)
                self.assertTrue((target / "summary.json").is_file())
                self.assertTrue((target / "summary.csv").is_file())
                self.assertEqual(written["n_conditions"], 2)
                with self.assertRaises(FileExistsError):
                    write_summary(root, target)
                regenerated = write_summary(root, target, overwrite=True)
                self.assertEqual(regenerated["n_conditions"], 2)

    def test_duplicate_completed_condition_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            clean = make_condition().as_dict()
            write_condition_tree(root, "mnist-add2", 4, "clean", clean, base_metrics())
            write_condition_tree(root, "mnist-add2", 4, "clean", clean, base_metrics(),
                                 directory_name="clean-copy")
            with self.assertRaises(ValueError):
                summarize(root)

    def test_baseline_consistency_uses_the_recorded_validation_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            write_condition_tree(root, "mnist-add2", 4, "clean",
                                 make_condition().as_dict(), base_metrics())
            (root / "baseline.json").write_text(json.dumps({"evaluations": [
                {"task": "mnist-add2", "seed": 4, "best_validation_accuracy": 0.8}]}),
                encoding="utf-8")
            summary = summarize(root)
            self.assertEqual(summary["baseline_consistency"][0]["delta"], 0.0)
            self.assertTrue(summary["baseline_consistency"][0]["comparable"])

    def test_empty_run_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                summarize(temporary)


class CheckpointDiscoveryTests(unittest.TestCase):
    def _run_dir(self, root, seed=4, status="TRAINING_COMPLETED_TEST_NOT_RUN",
                 stopped=None, best=True):
        run = Path(root) / f"frt-abl-v10-20261028__mnist-add2__full__seed-{seed}__id-1"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({
            "method_version": "frt-abl-v10-20261028", "task": "mnist-add2",
            "variant": "full", "seed": seed}), encoding="utf-8")
        (run / "status.json").write_text(json.dumps({
            "status": status, "stopped_reason": stopped,
            "best_validation_accuracy": 0.98}), encoding="utf-8")
        if best:
            (run / "best.pt").write_bytes(b"checkpoint")
        return run

    def test_single_completed_run_is_found(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._run_dir(temporary)
            info = discover_checkpoint(temporary, "mnist-add2", 4)
            self.assertEqual(Path(info["checkpoint"]), run / "best.pt")
            self.assertEqual(info["best_validation_accuracy"], 0.98)

    def test_missing_or_stopped_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            self._run_dir(temporary, status="STOPPED_MAX_UPDATES", stopped="MAX_UPDATES")
            with self.assertRaises(FileNotFoundError):
                discover_checkpoint(temporary, "mnist-add2", 4)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                discover_checkpoint(temporary, "mnist-add2", 4)

    def test_ambiguous_completed_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = self._run_dir(temporary)
            second = Path(temporary) / "frt-abl-v10-20261028__mnist-add2__full__seed-4__id-2"
            second.mkdir()
            for name in ("config.json", "status.json", "best.pt"):
                source = first / name
                (second / name).write_bytes(source.read_bytes())
            with self.assertRaises(ValueError):
                discover_checkpoint(temporary, "mnist-add2", 4)


if __name__ == "__main__":
    unittest.main()
