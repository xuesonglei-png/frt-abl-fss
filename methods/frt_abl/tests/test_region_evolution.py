"""Synthetic unit tests for the three-region evolution analyzer.

No experiment data, torch model or training run is used here.  The fixture writes
the same CSV/JSONL schema as ``train.py`` and asserts the transition, waiting,
reliability and weight accounting rules.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from methods.frt_abl.region_evolution import (
    _eligible_from_fields,
    _is_reliable,
    analyze_run,
    check_unique_seeds,
    discover_runs,
    load_run_meta,
    summarize_transitions,
    write_outputs,
)

DECISION_FIELDS = [
    "sample_id", "candidate", "evidence", "region", "q", "lower", "upper",
    "weight", "trained", "age", "complete", "margin", "stable_count",
    "unknown_count", "n_supported", "original_valid", "eligible",
    "curriculum_weight", "effective_weight",
]

# Four observed rounds.  s1: B,B,P,P; s2: B,P,N,B; s3: P,P,P,P;
# s4: N,N,B,P; s5: B,B,B,B; s6: B,P,P,P (first positive is not gate-eligible
# because alpha is None, then becomes reliable at the next round).
STATES = {
    "s1": ["boundary", "boundary", "positive", "positive"],
    "s2": ["boundary", "positive", "negative", "boundary"],
    "s3": ["positive", "positive", "positive", "positive"],
    "s4": ["negative", "negative", "boundary", "positive"],
    "s5": ["boundary", "boundary", "boundary", "boundary"],
    "s6": ["boundary", "positive", "positive", "positive"],
}


def _row(sample_id, region):
    if region == "positive":
        return {
            "sample_id": sample_id,
            "candidate": json.dumps([1, 2]),
            "evidence": "supported",
            "region": region,
            "q": 0.99,
            "lower": 0.96,
            "upper": 0.99,
            "weight": 0.9,
            "trained": True,
            "age": 0,
            "complete": True,
            "margin": 0.1,
            "stable_count": 2,
            "unknown_count": 0,
            "n_supported": 1,
            "original_valid": True,
            "eligible": True,
            "curriculum_weight": 1.0,
            "effective_weight": 0.9,
        }
    if region == "negative":
        return {
            "sample_id": sample_id,
            "candidate": json.dumps([0, 0]),
            "evidence": "refuted",
            "region": region,
            "q": 0.05,
            "lower": 0.02,
            "upper": 0.05,
            "weight": 0.0,
            "trained": False,
            "age": 0,
            "complete": True,
            "margin": 0.0,
            "stable_count": 1,
            "unknown_count": 0,
            "n_supported": 0,
            "original_valid": True,
            "eligible": False,
            "curriculum_weight": 1.0,
            "effective_weight": 0.0,
        }
    if region == "conflict":
        evidence = "inconsistent"
    else:
        evidence = "unresolved"
    return {
        "sample_id": sample_id,
        "candidate": json.dumps([0, 0]),
        "evidence": evidence,
        "region": region,
        "q": 0.2,
        "lower": 0.1,
        "upper": 0.4,
        "weight": 0.0,
        "trained": False,
        "age": 0,
        "complete": True,
        "margin": 0.0,
        "stable_count": 1,
        "unknown_count": 1,
        "n_supported": 0,
        "original_valid": True,
        "eligible": False,
        "curriculum_weight": 1.0,
        "effective_weight": 0.0,
    }


def make_run(root, name="frt-abl-v10-test__mnist-add2__full__seed-0__20261029-000000-000000"):
    run = Path(root) / name
    run.mkdir(parents=True)
    config = {
        "task": "mnist-add2",
        "variant": "full",
        "seed": 0,
        "method_version": "frt-abl-v10-test",
        "rounds": 4,
        "stable_rounds": 2,
        "allow_truncated": False,
        "min_margin": 0.02,
        "alpha_start": 0.95,
        "beta": 0.2,
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "status.json").write_text(json.dumps({
        "status": "TRAINING_COMPLETED_TEST_NOT_RUN",
        "smoke": False,
    }), encoding="utf-8")

    alphas = [None, None, 0.95, 0.95]
    accuracies = [0.50, 0.60, 0.70, 0.80]
    losses = [1.0, 0.9, 0.8, 0.7]
    with (run / "history.jsonl").open("w", encoding="utf-8") as handle:
        for round_index in range(4):
            regions = {
                region: sum(STATES[sample][round_index] == region for sample in STATES)
                for region in ("positive", "negative", "boundary", "conflict")
            }
            handle.write(json.dumps({
                "round": round_index,
                "alpha": alphas[round_index],
                "regions": regions,
                "accepted_candidates": regions["positive"],
                "coverage": regions["positive"] / len(STATES),
                "loss": losses[round_index],
                "validation_task": {
                    "correct": int(accuracies[round_index] * len(STATES)),
                    "total": len(STATES),
                    "accuracy": accuracies[round_index],
                },
            }) + "\n")

    for round_index in range(4):
        path = run / f"decisions_round_{round_index:03}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DECISION_FIELDS)
            writer.writeheader()
            for sample_id, regions in STATES.items():
                writer.writerow(_row(sample_id, regions[round_index]))
    return run


class RegionEvolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = make_run(self.root)
        self.meta = load_run_meta(self.run_dir)
        self.analysis = analyze_run(self.meta, persist_rounds=2)

    def tearDown(self):
        self.temp.cleanup()

    def test_round_region_counts_and_transitions(self):
        rows = self.analysis.round_rows
        self.assertEqual([row["boundary"] for row in rows], [4, 2, 2, 2])
        self.assertEqual([row["positive"] for row in rows], [1, 3, 3, 4])
        self.assertEqual([row["negative"] for row in rows], [1, 1, 1, 0])
        self.assertEqual(rows[1]["converted_boundary_to_positive"], 2)
        self.assertEqual(rows[1]["reliable_conversions"], 0)  # alpha was None
        self.assertEqual(rows[1]["reliable_sustained_conversions"], 1)  # s6 later
        self.assertEqual(rows[2]["converted_boundary_to_positive"], 1)
        self.assertEqual(rows[2]["reliable_conversions"], 1)
        self.assertEqual(rows[2]["reliable_sustained_conversions"], 1)
        self.assertEqual(rows[3]["converted_boundary_to_positive"], 1)
        self.assertEqual(rows[3]["reliable_conversions"], 1)
        self.assertEqual(rows[3]["reliable_sustained_conversions"], 0)  # censored
        self.assertTrue(all(row["history_regions_match"] is True for row in rows))

    def test_waiting_spells_and_reliability_summary(self):
        summary = self.analysis.summary
        self.assertEqual(summary["n_rounds"], 4)
        self.assertEqual(summary["n_samples"], 6)
        self.assertEqual(summary["spell_count"], 6)
        self.assertEqual(summary["converted_spells"], 4)
        self.assertEqual(summary["reliable_conversions"], 2)
        self.assertEqual(summary["eventual_reliable_conversions"], 3)
        self.assertEqual(summary["eventual_reliable_sustained_conversions"], 2)
        self.assertEqual(summary["censored_spells"], 2)
        self.assertAlmostEqual(summary["wait_converted_mean"], 5 / 4)
        self.assertEqual(summary["wait_converted_max"], 2)
        self.assertEqual(summary["ever_boundary_samples"], 5)
        self.assertEqual(summary["end_in_boundary_samples"], 2)
        self.assertAlmostEqual(summary["conversion_rate_per_spell"], 4 / 6)
        self.assertAlmostEqual(summary["reliable_conversion_share"], 2 / 4)
        self.assertAlmostEqual(summary["eventual_reliable_share"], 3 / 4)
        self.assertAlmostEqual(summary["eventual_reliable_sustained_share"], 2 / 4)
        self.assertEqual(summary["spells_started_unresolved"], 6)
        self.assertEqual(summary["eventual_reliable_conversions_from_unresolved"], 3)
        self.assertAlmostEqual(
            summary["eventual_reliable_conversion_rate_from_unresolved"], 3 / 6
        )
        self.assertAlmostEqual(summary["first_reliable_offset_mean"], 1 / 3)
        self.assertAlmostEqual(summary["eventual_reliable_within_1_share"], 2 / 4)
        self.assertAlmostEqual(summary["eventual_reliable_within_2_share"], 3 / 4)
        self.assertAlmostEqual(summary["best_val_accuracy"], 0.80)
        self.assertEqual(summary["best_val_round"], 3)

    def test_actual_training_weights(self):
        summary = self.analysis.summary
        self.assertEqual(summary["trained_count_total"], 11)
        self.assertAlmostEqual(summary["trained_weight_pooled_mean"], 0.9)
        self.assertAlmostEqual(summary["new_conversion_weight_mean"], 0.9)
        self.assertAlmostEqual(summary["post_conversion_weight_sum_mean"], 1.575)
        self.assertAlmostEqual(summary["post_conversion_trained_rounds_mean"], 7 / 4)

    def test_transition_summary_pools_counts(self):
        rows = summarize_transitions([self.analysis])
        boundary_to_positive = [
            row for row in rows
            if row["from_region"] == "boundary" and row["to_region"] == "positive"
        ]
        self.assertEqual(len(boundary_to_positive), 1)
        self.assertEqual(boundary_to_positive[0]["count"], 4)
        positive_to_negative = [
            row for row in rows
            if row["from_region"] == "positive" and row["to_region"] == "negative"
        ]
        self.assertEqual(positive_to_negative[0]["count"], 1)

    def test_reliability_requires_gate_open_and_alpha(self):
        row = {
            "region": "positive", "evidence": "supported", "eligible": True,
            "stable_count": 2, "unknown_count": 0, "complete": True,
            "n_supported": 1, "margin": 0.1, "lower": 0.96,
        }
        self.assertTrue(_is_reliable(row, 0.95, self.meta.config))
        self.assertFalse(_is_reliable(row, None, self.meta.config))
        self.assertFalse(_is_reliable({**row, "lower": 0.90}, 0.95, self.meta.config))
        fallback = {
            "evidence": "supported", "stable_count": 2, "unknown_count": 0,
            "complete": True, "n_supported": 1, "margin": 0.1,
        }
        self.assertTrue(_eligible_from_fields(fallback, self.meta.config))
        self.assertFalse(_eligible_from_fields({**fallback, "unknown_count": 1},
                                               self.meta.config))

    def test_pipeline_writes_expected_outputs(self):
        output = self.root / "analysis"
        manifest = write_outputs([self.analysis], output, persist_rounds=2)
        for name in ("region_rounds.csv", "transitions.csv", "boundary_spells.csv",
                     "run_summary.json", "summary.json", "curve_summary.csv",
                     "transition_summary.csv", "analysis_manifest.json", "REPORT.md"):
            self.assertTrue((output / name).exists(), name)
        self.assertEqual(manifest["n_runs"], 1)
        self.assertFalse(manifest["formal_test_read"])
        report = (output / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("三域演化分析报告", report)
        with (output / "boundary_spells.csv").open(encoding="utf-8", newline="") as handle:
            spell_rows = list(csv.DictReader(handle))
        self.assertEqual(len(spell_rows), 6)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["groups"][0]["n_runs"], 1)
        self.assertEqual(summary["groups"][0]["task"], "mnist-add2")

    def test_duplicate_seed_guard(self):
        duplicate = replace(self.meta, run_id="duplicate-run")
        with self.assertRaises(ValueError):
            check_unique_seeds([self.meta, duplicate])
        self.assertEqual(
            check_unique_seeds([self.meta, duplicate], allow_duplicates=True),
            {("mnist-add2", "full", "frt-abl-v10-test", 0):
             [self.meta.run_id, duplicate.run_id]},
        )

    def test_zero_boundary_run_is_handled(self):
        run = self.root / "zero-boundary"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({
            "task": "mnist-add2", "variant": "full", "seed": 0,
            "method_version": "frt-abl-v10-test", "rounds": 2,
            "stable_rounds": 2, "allow_truncated": False, "min_margin": 0.02,
            "alpha_start": 0.95, "beta": 0.2,
        }), encoding="utf-8")
        (run / "status.json").write_text(json.dumps({
            "status": "TRAINING_COMPLETED_TEST_NOT_RUN", "smoke": False,
        }), encoding="utf-8")
        with (run / "history.jsonl").open("w", encoding="utf-8") as handle:
            for round_index in range(2):
                handle.write(json.dumps({
                    "round": round_index, "alpha": 0.95,
                    "regions": {"positive": 2, "negative": 0, "boundary": 0, "conflict": 0},
                    "validation_task": {"accuracy": 0.5, "correct": 1, "total": 2},
                }) + "\n")
        for round_index in range(2):
            with (run / f"decisions_round_{round_index:03}.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=DECISION_FIELDS)
                writer.writeheader()
                writer.writerow(_row("s1", "positive"))
                writer.writerow(_row("s2", "positive"))
        meta = load_run_meta(run)
        analysis = analyze_run(meta)
        self.assertEqual(analysis.summary["spell_count"], 0)
        self.assertIsNone(analysis.summary["wait_converted_mean"])
        self.assertTrue(any("no boundary spells" in warning
                            for warning in analysis.summary["warnings"]))
        output = self.root / "zero-output"
        write_outputs([analysis], output, persist_rounds=2)
        report = (output / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("没有出现 boundary spell", report)

    def test_discovery_respects_variant_and_status_filters(self):
        runs, skipped = discover_runs([self.root], variant="full")
        self.assertEqual([meta.run_id for meta in runs], [self.run_dir.name])
        runs, skipped = discover_runs([self.root], variant="no_boundary")
        self.assertEqual(runs, [])
        self.assertTrue(any("variant=full" in item["reason"] for item in skipped))


if __name__ == "__main__":
    unittest.main()
