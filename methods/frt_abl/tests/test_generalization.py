"""Specifications for the HWF length/operator-combination generalization split.

These tests use a tiny synthetic HWF raw-data tree and do not import torch, build
a model, read the real dataset or start an experiment.
"""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from methods.frt_abl.evaluate_generalization import build_report, compute_group_report
from methods.frt_abl.prepare_generalization import (
    _normalise_holdout, expression_operators, operator_combo, sample_metadata,
    validate_generalization_config)
from methods.frt_abl.prepare_generalization import (prepare_generalization,
                                                   verify_generalization_prepared)


SYMBOLS = list("123456789") + ["+", "-", "*", "/"]
DIGITS = "123456789"
OPS = "+-*/"


def make_row(split, index, expression, result=7, prefix=None):
    """Create one raw HWF row plus the image files it references."""
    tag = prefix or split
    images = []
    for position, character in enumerate(expression):
        if position % 2 == 0:
            directory, name = character, f"{tag}_d{index}_{position}.jpg"
        else:
            directory = {"*": "times", "/": "div"}.get(character, character)
            name = f"{tag}_o{index}_{position}.jpg"
        images.append(f"{directory}/{name}")
    identifier = f"{prefix or split}_{index:04d}"
    for image in images:
        path = IMAGE_ROOT / image
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return {"id": identifier, "img_paths": images, "expr": expression, "res": result}


def write_raw_data(root, train_expressions, test_expressions):
    global IMAGE_ROOT
    IMAGE_ROOT = root / "Handwritten_Math_Symbols"
    train = [make_row("train", i, expression)
             for i, expression in enumerate(train_expressions)]
    test = [make_row("test", i, expression, prefix="test")
            for i, expression in enumerate(test_expressions)]
    (root / "expr_train.json").write_text(json.dumps(train), encoding="utf-8")
    (root / "expr_test.json").write_text(json.dumps(test), encoding="utf-8")
    return train, test


def config(root, prepared, holdout_enabled=False, holdout_count=1):
    return {
        "task": "hwf", "track": "METHOD_LABEL_BUDGET", "symbols": list(SYMBOLS),
        "data_root": str(root), "prepared_root": str(prepared),
        "validation_count": 3, "anchor_count": 2, "calibration_fit_count": 2,
        "calibration_gate_count": 2,
        "generalization": {
            "name": "synthetic-length-gen",
            "train_lengths": [1, 3, 5],
            "eval_lengths": [1, 3, 5, 7],
            "validation_lengths": [1, 3, 5],
            "primary_eval_length": 7,
            "group_by": ["length", "combo", "combo_group", "contains_holdout_bigram",
                         "exact_combo_seen_in_train", "length_x_combo_group",
                         "length_x_contains_holdout_bigram"],
            "combo_holdout": {
                "enabled": holdout_enabled,
                "lengths": [5],
                "counts": {"5": holdout_count},
                "salt": "synthetic-holdout-v1",
            },
        },
    }


def synthetic_expressions():
    train = []
    train += ["1"] * 6
    for operator in OPS:
        train += [f"1{operator}2"] * 3
    for combo in ("+-", "+*", "-+", "-*"):
        train += [f"1{combo[0]}2{combo[1]}3"] * 3
    # Length-7 rows must be excluded from every training role.
    train += ["1+2+3+4", "1*2*3*4", "1-2-3-4", "1/2/3/4"]
    test = ["5", "6", "5+6", "5*6",
            "5+6-7", "5+6*7", "5-6+7", "5-6*7",
            "5+6-7*8", "5+6*7+8", "5-6+7-8", "5-6*7-8", "5*6*7*8", "5/6/7/8"]
    return train, test


class GeneralizationMetadataTests(unittest.TestCase):
    def test_operator_combo_maps_times_and_div(self):
        self.assertEqual(operator_combo(["1/a.jpg", "times/b.jpg", "2/c.jpg"]), ("*",))
        self.assertEqual(operator_combo(["1/a.jpg", "div/b.jpg", "2/c.jpg"]), ("/",))
        self.assertEqual(operator_combo(["1/a.jpg"]), ())
        self.assertEqual(expression_operators("1*2/3-4"), ("*", "/", "-"))
        with self.assertRaises(ValueError):
            operator_combo(["unknown/a.jpg"])

    def test_holdout_is_deterministic_and_marks_contiguous_windows(self):
        directories = {"+": "+", "-": "-", "*": "times", "/": "div"}
        rows = [{"images": [f"1/a{i}.jpg", f"{directories[op]}/b{i}.jpg", f"2/c{i}.jpg"]}
                for i, op in enumerate(OPS * 2)]
        block = {"enabled": True, "lengths": [3], "counts": {"3": 1},
                 "salt": "synthetic-holdout-v1"}
        first = _normalise_holdout(block, rows)
        second = _normalise_holdout(block, rows)
        self.assertEqual(first, second)
        chosen = tuple(first["heldout_by_length"]["3"][0])
        self.assertEqual(first["heldout_units"][0]["combo"], list(chosen))
        train_combos = {("+",), ("-",), ("*",), ("/",)}
        chosen_directory = directories[chosen[0]]
        hit = sample_metadata({"images": [f"1/a.jpg", f"{chosen_directory}/b.jpg", "2/c.jpg"]},
                              train_combos, first)
        self.assertTrue(hit["contains_holdout_bigram"])
        self.assertEqual(hit["combo_group"], "unseen")
        other = sample_metadata({"images": ["1/a.jpg", "times/b.jpg", "2/c.jpg"]},
                                train_combos, first)
        self.assertFalse(other["contains_holdout_bigram"])

    def test_length_seven_combo_is_unseen_even_without_holdout(self):
        metadata = sample_metadata(
            {"images": ["1/a.jpg", "+/b.jpg", "2/c.jpg", "*/d.jpg", "3/e.jpg"]},
            {("+",), ("+", "*")}, _normalise_holdout({"enabled": False}, []))
        self.assertEqual(metadata["length"], 5)
        self.assertTrue(metadata["exact_combo_seen_in_train"])
        metadata7 = sample_metadata(
            {"images": ["1/a.jpg", "+/b.jpg", "2/c.jpg", "*/d.jpg", "3/e.jpg",
                        "-/f.jpg", "4/g.jpg"]},
            {("+",), ("+", "*")}, _normalise_holdout({"enabled": False}, []))
        self.assertEqual(metadata7["length"], 7)
        self.assertFalse(metadata7["exact_combo_seen_in_train"])
        self.assertEqual(metadata7["combo_group"], "unseen")


class PreparationTests(unittest.TestCase):
    def test_dry_run_plans_image_disjoint_roles_and_length_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            train, test = synthetic_expressions()
            write_raw_data(root, train, test)
            dest = Path(directory) / "prepared"
            cfg = config(root, dest)
            receipt = prepare_generalization(cfg, dry_run=True)
            self.assertEqual(receipt["status"], "DRY_RUN_ONLY")
            self.assertFalse(dest.exists())
            self.assertEqual(receipt["counts"]["validation"], 3)
            self.assertEqual(receipt["counts"]["anchor"], 2)
            self.assertEqual(receipt["counts"]["test"], len(test))
            self.assertGreater(receipt["counts"]["weak"], 0)
            self.assertEqual(receipt["excluded_train_expressions"]["not_in_train_lengths"], 4)
            self.assertEqual(receipt["generalization"]["primary_test_count"], 6)
            self.assertEqual(receipt["generalization"]["test_group_counts"]["length"]["7"], 6)
            self.assertFalse(receipt["generalization"]["combo_holdout"]["enabled"])

    def test_materialised_root_verifies_and_respects_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            train, test = synthetic_expressions()
            write_raw_data(root, train, test)
            dest = Path(directory) / "prepared"
            cfg = config(root, dest, holdout_enabled=True, holdout_count=1)
            receipt = prepare_generalization(cfg, dry_run=False)
            self.assertTrue(dest.is_dir())
            self.assertEqual(verify_generalization_prepared(cfg)["counts"], receipt["counts"])
            chosen = tuple(receipt["generalization"]["combo_holdout"]["heldout_by_length"]["5"][0])
            self.assertNotIn(chosen, {tuple(row["generalization"]["combo_tuple"])
                                      for role in ("weak", "anchor", "calibration_fit",
                                                   "calibration_gate", "validation")
                                      for row in json.loads((dest / f"{role}.json").read_text(encoding="utf-8"))})
            heldout_test = [row for row in json.loads((dest / "test.json").read_text(encoding="utf-8"))
                            if tuple(row["generalization"]["combo_tuple"]) == chosen]
            self.assertTrue(heldout_test)
            self.assertTrue(all(row["generalization"]["combo_group"] == "unseen"
                                for row in heldout_test))

    def test_grouped_report_reads_prepared_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            train, test = synthetic_expressions()
            write_raw_data(root, train, test)
            dest = Path(directory) / "prepared"
            cfg = config(root, dest)
            prepare_generalization(cfg, dry_run=False)
            test_rows = json.loads((dest / "test.json").read_text(encoding="utf-8"))
            output = Path(directory) / "eval"
            output.mkdir()
            fields = ["sample_id", "native_task_correct", "native_symbol_correct",
                      "native_sequence_correct", "native_covered",
                      "argmax_task_correct", "argmax_symbol_correct",
                      "argmax_sequence_correct"]
            with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                for index, row in enumerate(test_rows):
                    length = len(row["images"])
                    correct = index % 2 == 0
                    writer.writerow([row["id"], correct, length if correct else 0,
                                     correct, correct, False, 0, False])
            report = build_report(cfg, output, "test", ("length", "combo_group"))
            self.assertEqual(report["overall"]["count"], len(test_rows))
            self.assertEqual(report["overall"]["native_task_correct"], len(test_rows) // 2)
            self.assertTrue((output / "metrics_by_group.json").is_file())
            self.assertTrue((output / "metrics_by_group.csv").is_file())
            self.assertEqual(report["generalization"]["name"], "synthetic-length-gen")

    def test_config_validation_rejects_bad_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            cfg = config(root, Path(directory) / "prepared")
            validate_generalization_config(cfg)
            broken = dict(cfg)
            broken["generalization"] = dict(cfg["generalization"], train_lengths=[1, 3, 4])
            with self.assertRaises(ValueError):
                validate_generalization_config(broken)
            broken = dict(cfg)
            broken["generalization"] = dict(cfg["generalization"], primary_eval_length=9)
            with self.assertRaises(ValueError):
                validate_generalization_config(broken)
            broken = dict(cfg)
            broken["generalization"] = dict(cfg["generalization"],
                                            validation_lengths=[1, 3])
            with self.assertRaises(ValueError):
                validate_generalization_config(broken)
            broken = dict(cfg)
            broken["generalization"] = dict(
                cfg["generalization"],
                combo_holdout={"enabled": True, "lengths": [5],
                               "counts": {"5": 0}, "salt": "x"})
            with self.assertRaises(ValueError):
                validate_generalization_config(broken)


class GroupReportTests(unittest.TestCase):
    @staticmethod
    def prediction(sample_id, length, task, symbol, sequence, covered):
        return {
            "sample_id": sample_id,
            "native_task_correct": task, "native_symbol_correct": symbol,
            "native_sequence_correct": sequence, "native_covered": covered,
            "argmax_task_correct": False, "argmax_symbol_correct": 0,
            "argmax_sequence_correct": False,
        }

    def test_grouped_task_symbol_sequence_denominators(self):
        samples = [
            {"id": "a", "images": ["1/a.jpg", "+/b.jpg", "2/c.jpg"],
             "generalization": {"length": 3, "combo": "+", "combo_group": "seen",
                                "contains_holdout_bigram": False,
                                "exact_combo_seen_in_train": True}},
            {"id": "b", "images": ["1/d.jpg", "+/e.jpg", "2/f.jpg", "-/g.jpg", "3/h.jpg"],
             "generalization": {"length": 5, "combo": "+-", "combo_group": "unseen",
                                "contains_holdout_bigram": True,
                                "exact_combo_seen_in_train": False}},
            {"id": "c", "images": ["1/i.jpg", "+/j.jpg", "2/k.jpg", "+/l.jpg", "3/m.jpg",
                                   "+/n.jpg", "4/o.jpg"],
             "generalization": {"length": 7, "combo": "+++", "combo_group": "unseen",
                                "contains_holdout_bigram": False,
                                "exact_combo_seen_in_train": False}},
        ]
        predictions = [
            self.prediction("a", 3, True, 3, True, True),
            self.prediction("b", 5, True, 4, False, True),
            self.prediction("c", 7, False, 5, False, False),
        ]
        report = compute_group_report(predictions, samples, ("length", "combo_group"))
        overall = report["overall"]
        self.assertEqual(overall["count"], 3)
        self.assertEqual(overall["native_task_accuracy"], 2 / 3)
        self.assertEqual(overall["native_symbol_correct"], 12)
        self.assertEqual(overall["native_symbol_total"], 15)
        self.assertAlmostEqual(overall["native_symbol_accuracy"], 12 / 15)
        self.assertEqual(overall["native_sequence_accuracy"], 1 / 3)
        self.assertEqual(overall["native_coverage"], 2 / 3)

        by_length = {row["group"]: row for row in report["groups"]["length"]["groups"]}
        self.assertEqual(by_length["3"]["native_task_accuracy"], 1.0)
        self.assertEqual(by_length["5"]["native_symbol_accuracy"], 4 / 5)
        self.assertEqual(by_length["7"]["native_sequence_accuracy"], 0.0)
        self.assertEqual(by_length["7"]["native_coverage"], 0.0)

        by_group = {row["group"]: row for row in report["groups"]["combo_group"]["groups"]}
        self.assertEqual(by_group["seen"]["count"], 1)
        self.assertEqual(by_group["unseen"]["count"], 2)
        self.assertEqual(by_group["unseen"]["native_task_accuracy"], 0.5)

    def test_grouped_report_rejects_prediction_id_mismatch(self):
        samples = [
            {"id": "a", "images": ["1/a.jpg"]},
            {"id": "b", "images": ["2/b.jpg"]},
        ]
        prediction = self.prediction("a", 1, True, 1, True, True)
        with self.assertRaisesRegex(ValueError, "missing=1"):
            compute_group_report([prediction], samples, ("length",))
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            compute_group_report([prediction, prediction], samples[:1], ("length",))

    def test_grouped_report_rejects_duplicate_test_ids(self):
        sample = {"id": "a", "images": ["1/a.jpg"]}
        prediction = self.prediction("a", 1, True, 1, True, True)
        with self.assertRaisesRegex(ValueError, "Test samples contain duplicate ids"):
            compute_group_report([prediction], [sample, sample], ("length",))


if __name__ == "__main__":
    unittest.main()
