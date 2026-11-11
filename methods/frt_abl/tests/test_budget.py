"""Unit tests for image-level annotation-budget allocation and config validation."""
import json
from pathlib import Path
import tempfile
import unittest

from methods.frt_abl.budget import (BUDGET_ROLES, BUDGET_VERSION, EXPERIMENT,
                                    allocate_budget_roles, budget_plan, level_label,
                                    role_budget_record, split_integer_budget,
                                    universe_from_samples, validate_budget_config,
                                    verify_budget_prepared)


def mnist_samples(count=40):
    rows = [{"id": f"s{i}", "images": [f"i{2*i}", f"i{2*i+1}"], "target": i % 10}
            for i in range(count)]
    return {"train": rows,
            "validation": [{"id": "val", "images": ["val1", "val2"], "target": 1}],
            "test": [{"id": "test", "images": ["test1", "test2"], "target": 2}]}


def hwf_samples(count=40):
    rows = []
    for i in range(count):
        length = 1 + 2 * (i % 3)
        rows.append({"id": f"s{i}", "images": [f"i{i}_{j}" for j in range(length)],
                     "target": 1})
    return {"train": rows,
            "validation": [{"id": "val", "images": ["val"], "target": 1}],
            "test": [{"id": "test", "images": ["test"], "target": 1}]}


def budget_cfg(task="mnist-add2"):
    return {"task": task, "validation_count": 5}


class BudgetSplitTests(unittest.TestCase):
    def test_largest_remainder_conserves_total_and_keeps_roles_nonempty(self):
        split = split_integer_budget(354, {"anchor": 1, "calibration_fit": 3,
                                           "calibration_gate": 3})
        self.assertEqual(sum(split.values()), 354)
        self.assertTrue(all(value >= 1 for value in split.values()))
        self.assertEqual(split, {"anchor": 50, "calibration_fit": 152,
                                 "calibration_gate": 152})

    def test_equal_split_is_exact_for_mnist(self):
        split = split_integer_budget(540, {"anchor": 1, "calibration_fit": 1,
                                           "calibration_gate": 1})
        self.assertEqual(split, {"anchor": 180, "calibration_fit": 180,
                                 "calibration_gate": 180})

    def test_level_labels(self):
        self.assertEqual(level_label(0.01), "1pct")
        self.assertEqual(level_label(0.03), "3pct")
        self.assertEqual(level_label(0.10), "10pct")
        self.assertEqual(level_label(0.025), "2p5pct")

    def test_universe_is_unique_images_not_expressions(self):
        samples = mnist_samples(10)
        universe = universe_from_samples(samples)
        self.assertEqual(universe["expressions"], 10)
        self.assertEqual(universe["unique_images"], 20)

    def test_budget_plan_uses_image_universe(self):
        cfg = budget_cfg("hwf")
        plan = budget_plan(cfg, 0.01, {"expressions": 100, "unique_images": 35411})
        self.assertEqual(plan["target_total_unique_labeled_images"], 354)
        self.assertEqual(sum(plan["target_role_unique_images"].values()), 354)


class AllocationTests(unittest.TestCase):
    def test_mnist_roles_are_image_disjoint_and_exact(self):
        samples = mnist_samples(60)
        targets = {"anchor": 4, "calibration_fit": 4, "calibration_gate": 4}
        roles, purged = allocate_budget_roles(budget_cfg(), samples, targets)
        image_sets = {role: {i for s in roles[role] for i in s["images"]}
                      for role in BUDGET_ROLES}
        for first in image_sets:
            for second in image_sets:
                if first >= second:
                    continue
                self.assertFalse(image_sets[first] & image_sets[second])
        # Each MNIST expression contributes exactly two new images.
        self.assertEqual([role_budget_record(roles)[role]["expressions"]
                          for role in BUDGET_ROLES], [2, 2, 2])
        self.assertEqual(role_budget_record(roles)["anchor"]["unique_labeled_images"], 4)
        self.assertEqual(len(roles["weak"]), 54)
        self.assertEqual(purged, 0)

    def test_hwf_validation_is_reserved_before_budget_roles(self):
        samples = hwf_samples(80)
        targets = {"anchor": 3, "calibration_fit": 6, "calibration_gate": 6}
        roles, _purged = allocate_budget_roles(budget_cfg("hwf"), samples, targets)
        self.assertEqual(len(roles["validation"]), 5)
        budget_sets = [{i for s in roles[role] for i in s["images"]}
                       for role in BUDGET_ROLES]
        validation = {i for s in roles["validation"] for i in s["images"]}
        for images in budget_sets:
            self.assertFalse(images & validation)
        record = role_budget_record(roles)
        for role, target in targets.items():
            self.assertGreaterEqual(record[role]["unique_labeled_images"], target)
            self.assertGreaterEqual(record[role]["expressions"], 1)

    def test_too_small_budget_fails_closed(self):
        samples = hwf_samples(20)
        targets = {"anchor": 1000, "calibration_fit": 1000, "calibration_gate": 1000}
        with self.assertRaises(ValueError):
            allocate_budget_roles(budget_cfg("hwf"), samples, targets)


def valid_record():
    role_images = {"anchor": 4, "calibration_fit": 4, "calibration_gate": 4}
    expressions = {"anchor": 2, "calibration_fit": 2, "calibration_gate": 2}
    return {
        "experiment": EXPERIMENT, "version": BUDGET_VERSION,
        "level": 0.12, "level_label": "12pct",
        "train_image_universe": 100, "train_expressions": 50,
        "target_total_unique_labeled_images": 12,
        "actual_total_unique_labeled_images": 12,
        "actual_fraction": 0.12,
        "role_share": {"anchor": 1, "calibration_fit": 1, "calibration_gate": 1},
        "target_role_unique_images": dict(role_images),
        "actual_role_unique_images": dict(role_images),
        "role_expressions": dict(expressions),
    }


def valid_cfg(prepared_root):
    return {"task": "mnist-add2", "prepared_root": str(prepared_root),
            "anchor_count": 2, "calibration_fit_count": 2, "calibration_gate_count": 2,
            "experiment": EXPERIMENT, "experiment_version": BUDGET_VERSION,
            "budget": valid_record()}


class ConfigValidationTests(unittest.TestCase):
    def test_valid_budget_config_and_prepared_receipt_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = valid_cfg(root)
            receipt = {"experiment": EXPERIMENT, "experiment_version": BUDGET_VERSION,
                       "budget": dict(cfg["budget"]),
                       "counts": {"anchor": 2, "calibration_fit": 2,
                                  "calibration_gate": 2, "weak": 10}}
            (root / "roles.json").write_text("[]", encoding="utf-8")
            for role in ("anchor", "calibration_fit", "calibration_gate", "weak"):
                (root / f"{role}.json").write_text("[]", encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps(receipt), encoding="utf-8")
            validate_budget_config(cfg)
            saved = verify_budget_prepared(cfg)
            self.assertEqual(saved["counts"]["anchor"], 2)

    def test_mismatched_total_is_rejected(self):
        cfg = {"task": "mnist-add2", "anchor_count": 2, "calibration_fit_count": 2,
               "calibration_gate_count": 2, "experiment": EXPERIMENT,
               "experiment_version": BUDGET_VERSION, "budget": valid_record()}
        cfg["budget"]["actual_total_unique_labeled_images"] = 13
        with self.assertRaises(ValueError):
            validate_budget_config(cfg)

    def test_mismatched_role_count_is_rejected(self):
        cfg = {"task": "mnist-add2", "anchor_count": 3, "calibration_fit_count": 2,
               "calibration_gate_count": 2, "experiment": EXPERIMENT,
               "experiment_version": BUDGET_VERSION, "budget": valid_record()}
        with self.assertRaises(ValueError):
            validate_budget_config(cfg)

    def test_non_budget_experiment_is_rejected(self):
        cfg = {"task": "mnist-add2", "anchor_count": 2, "calibration_fit_count": 2,
               "calibration_gate_count": 2, "experiment": "other",
               "experiment_version": BUDGET_VERSION, "budget": valid_record()}
        with self.assertRaises(ValueError):
            validate_budget_config(cfg)


if __name__ == "__main__":
    unittest.main()
