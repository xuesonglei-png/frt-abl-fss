"""Offline comparison protocol: parsing, aggregation, comparability and report.

Every test is hermetic: temporary directories stand in for the project, no model
is executed, no baseline is re-run and no formal-test manifest is read.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from methods.frt_abl.comparison import (ADOPTED, ComparisonError, DEFAULT_CONFIG, PRIMARY,
                                        PRIMARY_METHOD, PROTOCOL_VERSION, build_entries,
                                        compare, load_comparison_config, mean_ci,
                                        parse_formal_test_aggregate, parse_json_epoch_series,
                                        parse_json_fields, parse_seed_list,
                                        validate_comparison_config, write_adopted,
                                        write_comparison)


def base_config():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "track": "UNIT_TEST",
        "seeds": [0, 1],
        "primary": {
            "method": PRIMARY_METHOD,
            "display": "FRT-ABL (V10)",
            "method_version": "frt-abl-v10-20261028",
            "summary": "summary.json",
            "variants": ["full"],
            "split": "validation",
            "data_protocol": "common-toy",
            "supervision": "sum+labels",
            "selection": "best-validation-epoch",
            "test_evaluations": []
        },
        "tasks": {"toy": {"display": "Toy task", "family": "toy", "unit": "pair",
                          "target": "0..1", "order": 1}},
        "metrics": {"task_accuracy": {"display": "Accuracy", "direction": "higher",
                                      "definition": "correct / total"}},
        "baselines": [{
            "method": "base", "display": "BASE", "paper": "Test paper",
            "adoptions": [{
                "label": "run", "parser": "json_fields", "path": "base.json",
                "fields": {"task_accuracy": "acc"}, "task": "toy",
                "split": "native-test", "data_protocol": "native-base",
                "supervision": "sum-only", "selection": "last-epoch", "seed": 0,
            }],
        }],
    }


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def observation(**overrides):
    record = {
        "method": "base", "display": "BASE", "method_version": None,
        "task": "toy", "metric": "task_accuracy", "split": "native-test",
        "data_protocol": "native-base", "supervision": "sum-only",
        "selection": "last-epoch", "seed": 0, "value": 0.9, "std": None, "n": None,
        "seeds": [0], "correct": None, "total": None, "status": "COMPLETE",
        "budget": None, "label": None, "source": "base.json",
        "provenance": ADOPTED, "note": None, "extra": {}, "blockers": [],
    }
    record.update(overrides)
    return record


class ProtocolTests(unittest.TestCase):
    def test_repository_protocol_is_valid(self):
        cfg = load_comparison_config(DEFAULT_CONFIG)
        self.assertEqual(cfg["protocol_version"], PROTOCOL_VERSION)
        methods = [baseline["method"] for baseline in cfg["baselines"]]
        self.assertEqual(methods, ["a3bl", "dolphin", "deepproblog", "scallop", "abl_hed"])
        self.assertEqual(cfg["primary"]["method"], PRIMARY_METHOD)
        self.assertEqual(cfg["seeds"], [0, 1, 2, 3, 4])

    def test_duplicate_seed_and_unknown_metric_are_rejected(self):
        cfg = base_config()
        cfg["seeds"] = [0, 0]
        cfg["baselines"][0]["adoptions"][0]["fields"] = {"not_a_metric": "acc"}
        problems = validate_comparison_config(cfg)
        self.assertTrue(any("duplicate seeds" in problem for problem in problems))
        self.assertTrue(any("not declared" in problem for problem in problems))

    def test_primary_may_not_double_as_baseline(self):
        cfg = base_config()
        cfg["baselines"][0]["method"] = PRIMARY_METHOD
        problems = validate_comparison_config(cfg)
        self.assertTrue(any("may not reuse the primary id" in problem for problem in problems))

    def test_seed_list_parser_keeps_unrecorded_empty(self):
        self.assertEqual(parse_seed_list("0;1;2;3;4"), (0, 1, 2, 3, 4))
        self.assertEqual(parse_seed_list("1234"), (1234,))
        self.assertEqual(parse_seed_list("UNRECORDED"), ())
        self.assertEqual(parse_seed_list(""), ())


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_json_fields_reads_dotted_path_and_rejects_percent(self):
        path = self.root / "result.json"
        write_json(path, {"testing": {"accuracy": 0.9608}})
        parsed = parse_json_fields({"fields": {"task_accuracy": "testing.accuracy"}}, path)
        self.assertEqual(parsed[0][0], "task_accuracy")
        self.assertAlmostEqual(parsed[0][1], 0.9608)
        write_json(path, {"testing": {"accuracy": 96.08}})
        with self.assertRaises(ComparisonError):
            parse_json_fields({"fields": {"task_accuracy": "testing.accuracy"}}, path)

    def test_json_fields_rejects_bool_and_missing_field(self):
        path = self.root / "result.json"
        write_json(path, {"acc": True})
        with self.assertRaises(ComparisonError):
            parse_json_fields({"fields": {"task_accuracy": "acc"}}, path)
        write_json(path, {"accuracy": 0.5})
        with self.assertRaises(ComparisonError):
            parse_json_fields({"fields": {"task_accuracy": "acc"}}, path)

    def test_epoch_series_selects_last_or_max(self):
        path = self.root / "result.json"
        write_json(path, {"metrics": [{"epoch": 1, "accuracy": 0.9, "correct": 9, "total": 10},
                                      {"epoch": 2, "accuracy": 0.8, "correct": 8, "total": 10}]})
        last = parse_json_epoch_series({"list_field": "metrics", "value_field": "accuracy",
                                        "select": "last", "metric": "task_accuracy"}, path)
        self.assertAlmostEqual(last[0][1], 0.8)
        best = parse_json_epoch_series({"list_field": "metrics", "value_field": "accuracy",
                                        "select": "max", "metric": "task_accuracy"}, path)
        self.assertAlmostEqual(best[0][1], 0.9)
        self.assertAlmostEqual(best[0][2]["peak"], 0.9)

    def test_formal_test_aggregate_means_local_trials(self):
        path = self.root / "metrics.json"
        rows = [{"trial": 0, "length": 5, "correct": 9, "total": 10},
                {"trial": 0, "length": 6, "correct": 10, "total": 10},
                {"trial": 1, "length": 5, "correct": 8, "total": 10},
                {"trial": 1, "length": 6, "correct": 10, "total": 10}]
        write_json(path, {"formal_test": rows})
        parsed = parse_formal_test_aggregate({"metric": "task_accuracy"}, path)
        self.assertAlmostEqual(parsed[0][1], (19 / 20 + 18 / 20) / 2)
        self.assertEqual(parsed[0][2]["n_trials"], 2)


class AggregationTests(unittest.TestCase):
    def test_per_seed_mean_std_and_ci(self):
        entries = build_entries([observation(seed=seed, value=value, seeds=[seed])
                                 for seed, value in [(0, 0.8), (1, 0.9), (2, 1.0)]])
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertAlmostEqual(entry["value"], 0.9)
        expected_mean, expected_std, expected_ci = mean_ci([0.8, 0.9, 1.0])
        self.assertAlmostEqual(entry["std"], expected_std)
        self.assertAlmostEqual(entry["ci95"], expected_ci)
        self.assertEqual(entry["n"], 3)
        self.assertEqual(entry["seeds"], [0, 1, 2])
        self.assertEqual(entry["n_mode"], "seeds")

    def test_duplicate_seeds_are_excluded_not_maximised(self):
        entries = build_entries([observation(seed=0, value=0.8), observation(seed=0, value=0.9)])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["duplicate_seeds_excluded"], [0])
        self.assertIsNone(entries[0]["value"])
        self.assertEqual(entries[0]["status"], "NO_NUMERIC_VALUE")

    def test_unseeded_runs_are_repetitions_not_seeds(self):
        entries = build_entries([observation(seed=None, value=0.9, seeds=[]),
                                 observation(seed=None, value=0.8, seeds=[])])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["n"], 2)
        self.assertEqual(entries[0]["n_mode"], "unseeded-runs")
        self.assertEqual(entries[0]["seeds"], [])

    def test_label_is_appended_to_display(self):
        entries = build_entries([observation(seed=0, value=0.9, label="damp")])
        self.assertEqual(entries[0]["display"], "BASE · damp")


class ComparabilityTests(unittest.TestCase):
    def reference(self, **overrides):
        record = observation(seed=None, value=0.8, n=2, seeds=[0, 1], method=PRIMARY_METHOD,
                             display="FRT-ABL (V10)", split="validation",
                             data_protocol="common-toy", supervision="sum+labels",
                             selection="best-validation-epoch", provenance=PRIMARY)
        record.update(overrides)
        return record

    def test_differing_protocol_is_never_rankable_and_yields_no_delta(self):
        result = compare(build_entries([self.reference(), observation(value=0.9)]))
        group = result["groups"][0]
        self.assertEqual(group["ranking_status"], "NOT_RANKABLE_PROTOCOL_DIFFERS")
        member = next(row for row in group["members"] if row["method"] == "base")
        self.assertEqual(member["comparability"], "PROTOCOL_DIFFERS")
        self.assertFalse(member["rankable"])
        self.assertIsNone(member["delta_vs_reference"])
        self.assertEqual(sorted(member["differences"]),
                         ["data_protocol", "selection", "split", "supervision"])
        self.assertEqual(result["paired_deltas"], [])

    def test_identical_identity_is_rankable_and_pairs_by_seed(self):
        entries = build_entries([
            observation(method=PRIMARY_METHOD, display="FRT-ABL (V10)", seed=0, value=0.80,
                        split="test", data_protocol="native-shared", supervision="sum-only",
                        selection="validation-selected"),
            observation(method=PRIMARY_METHOD, display="FRT-ABL (V10)", seed=1, value=0.82,
                        split="test", data_protocol="native-shared", supervision="sum-only",
                        selection="validation-selected"),
            observation(seed=0, value=0.79, split="test", data_protocol="native-shared",
                        supervision="sum-only", selection="validation-selected"),
            observation(seed=1, value=0.80, split="test", data_protocol="native-shared",
                        supervision="sum-only", selection="validation-selected"),
        ])
        result = compare(entries)
        group = result["groups"][0]
        self.assertEqual(group["ranking_status"], "RANKABLE")
        deltas = result["paired_deltas"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["paired_seeds"], [0, 1])
        self.assertAlmostEqual(deltas[0]["mean_delta"], (0.79 - 0.80 + 0.80 - 0.82) / 2)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_write_comparison_marks_protocol_differences(self):
        write_json(self.root / "summary.json", {"groups": [
            {"task": "toy", "variant": "full", "mean_validation_accuracy": 0.81,
             "std_validation_accuracy": 0.01, "n_with_accuracy": 2, "seeds": [0, 1]}]})
        write_json(self.root / "base.json", {"acc": 0.91})
        cfg = base_config()
        output = self.root / "out"
        summary = write_comparison(output, cfg, project=self.root,
                                   runs_root=self.root / "runs")
        self.assertEqual(summary["ranking_status"], "NOT_ESTABLISHED")
        self.assertEqual(len(summary["groups"]), 1)
        self.assertEqual(summary["groups"][0]["task"], "toy")
        for name in ("summary.json", "comparison.csv", "comparability.csv", "report.md"):
            self.assertTrue((output / name).is_file(), name)
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("NOT_RANKABLE_PROTOCOL_DIFFERS", report)
        self.assertIn("不允许写成名次", report)
        rows = (output / "comparison.csv").read_text(encoding="utf-8")
        self.assertIn("PROTOCOL_DIFFERS", rows)
        # The output directory is append-only by default.
        with self.assertRaises(ComparisonError):
            write_comparison(output, cfg, project=self.root, runs_root=self.root / "runs")

    def test_missing_artifact_is_reported_not_guessed(self):
        write_json(self.root / "summary.json", {"groups": []})
        cfg = base_config()
        summary = write_comparison(self.root / "out", cfg, project=self.root,
                                   runs_root=self.root / "runs")
        self.assertEqual(summary["entries"], [])
        kinds = [issue["kind"] for issue in summary["issues"]]
        self.assertIn("MISSING_ARTIFACT", kinds)

    def test_write_adopted_freezes_hashes_and_refuses_overwrite(self):
        write_json(self.root / "summary.json", {"groups": []})
        write_json(self.root / "base.json", {"acc": 0.91})
        cfg = base_config()
        payload = write_adopted(self.root / "runs", cfg=cfg, project=self.root)
        self.assertEqual(len(payload["observations"]), 1)
        self.assertIn("base.json", payload["sources"])
        self.assertEqual(len(payload["sources"]["base.json"]["sha256"]), 64)
        self.assertTrue((self.root / "runs" / "adopted" / "observations.json").is_file())
        with self.assertRaises(ComparisonError):
            write_adopted(self.root / "runs", cfg=cfg, project=self.root)


class InertnessTests(unittest.TestCase):
    def test_importing_comparison_does_not_load_torch(self):
        code = ("import sys; import methods.frt_abl.comparison; "
                "print('torch' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
