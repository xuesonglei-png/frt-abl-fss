"""Sensitivity protocol/aggregation tests; no torch, no data, no training run."""
import json
import tempfile
import unittest
import uuid
from copy import deepcopy
from pathlib import Path

from methods.frt_abl.policies import METHOD_VERSION
from methods.frt_abl.sensitivity import (BASELINE_ARM, COMPLETE_TRAINING_STATUS,
                                         DEFAULT_SENSITIVITY_CONFIG,
                                         assert_single_factor, build_arm_config,
                                         build_arms, campaign_identity, collect_runs,
                                         declared_factors, default_factor_keys,
                                         history_metrics, load_sensitivity_config,
                                         prepare_campaign, summarize, usable_runs,
                                         validate_value, write_summary)

FACTOR_KEYS = ("knn_k", "graph_sigma", "risk_limit",
               "curriculum_ramp_rounds", "full_boundary_weight")


def baseline_cfg(task="mnist-add2"):
    return {
        "track": "METHOD_LABEL_BUDGET",
        "task": task,
        "variant": "full",
        "method_version": METHOD_VERSION,
        "threshold_policy": "risk_guarded",
        "curriculum_policy": "soft_weight",
        "knn_k": 10,
        "graph_sigma": 0.5,
        "risk_limit": 0.1,
        "curriculum_ramp_rounds": 10,
        "curriculum_min_weight": 0.5,
        "full_boundary_weight": 0.1 if task == "mnist-add2" else 0.2,
        "output_root": "methods/frt_abl/runs-v10",
        "seed": 4,
    }


def make_spec(tmp_path, seeds=(0, 1)):
    spec = deepcopy(load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG))
    spec["output_root"] = str(tmp_path / "runs")
    spec["summary_root"] = str(tmp_path / "summary")
    spec["seeds"] = list(seeds)
    return spec


def make_baselines():
    return {"mnist-add2": baseline_cfg("mnist-add2"), "hwf": baseline_cfg("hwf")}


def history_records(accuracy, rounds=3, coverage=1.0, accepted_share=1.0,
                    n_weak=200, boundary_loss=0.01):
    records = []
    accepted = int(n_weak * accepted_share)
    for index in range(rounds):
        records.append({
            "round": index, "alpha": 0.97,
            # The loosest threshold sees every eligible gate candidate; the chosen
            # threshold accepts `accepted_share` of them.
            "gate": [{"alpha": 0.99, "count": int(accepted * 0.8), "passed": True},
                     {"alpha": 0.97, "count": accepted, "passed": True},
                     {"alpha": 0.95, "count": n_weak, "passed": True}],
            "regions": {"positive": accepted, "boundary": n_weak - accepted},
            "accepted_candidates": accepted,
            "unsupported_weak": 0,
            "coverage": coverage,
            "q_stats": {"count": n_weak},
            "graph_changed_candidates": 3,
            "loss_components_per_update": {"anchor": 0.1, "pseudo": 0.2,
                                           "boundary": boundary_loss},
            "validation_task": {"accuracy": max(0.0, accuracy - 0.001 * (rounds - index - 1))},
        })
    return records


def write_run(campaign_dir, task, arm, seed, accuracy, suffix="", coverage=1.0,
              accepted_share=1.0, campaign=None, status=None, with_history=True):
    campaign_dir = Path(campaign_dir)
    config_path = campaign_dir / "configs" / task / f"{arm}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = seed
    if campaign is not None:
        config["sensitivity"]["campaign"] = campaign
    name = f"{task}__{arm.replace('=', '-')}__seed-{seed}{suffix}__{uuid.uuid4().hex[:6]}"
    run_dir = campaign_dir / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": status or COMPLETE_TRAINING_STATUS,
        "stopped_reason": None,
        "best_validation_accuracy": accuracy,
        "optimizer_steps": 1234,
        "code_revision": METHOD_VERSION,
    }), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({
        "python": "3.11.9", "torch": "2.5.1", "numpy": "2.1.0",
        "device": "cpu", "threads": 2, "code_revision": METHOD_VERSION,
    }), encoding="utf-8")
    (run_dir / "source_manifest.json").write_text(json.dumps({"train.py": "abc"}), encoding="utf-8")
    if with_history:
        with (run_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
            for record in history_records(accuracy, coverage=coverage,
                                          accepted_share=accepted_share):
                handle.write(json.dumps(record) + "\n")
    return run_dir


class ProtocolTests(unittest.TestCase):
    def test_declared_protocol_loads_with_frozen_baseline(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        factors = declared_factors(spec)
        self.assertEqual([factor.key for factor in factors], list(FACTOR_KEYS) + [
            "curriculum_min_weight"])
        self.assertEqual(default_factor_keys(spec), list(FACTOR_KEYS))
        self.assertTrue(Path(spec["output_root"]).is_absolute())
        self.assertEqual(spec["baseline"]["method_version"], METHOD_VERSION)

    def test_every_grid_contains_the_frozen_value(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        for task, cfg in make_baselines().items():
            for factor in declared_factors(spec):
                if factor.key not in cfg:
                    continue
                self.assertTrue(any(factor.is_baseline_level(cfg[factor.key], level)
                                    for level in factor.levels),
                                f"{task}/{factor.key} grid lacks its baseline")

    def test_optional_factor_requires_explicit_selection(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        arms = build_arms(spec, baseline_cfg(), list(FACTOR_KEYS))
        self.assertFalse(any(arm["factor"] == "curriculum_min_weight" for arm in arms))
        arms = build_arms(spec, baseline_cfg(), ["curriculum_min_weight"])
        self.assertTrue(any(arm["factor"] == "curriculum_min_weight" for arm in arms))


class ArmTests(unittest.TestCase):
    def test_one_factor_per_arm_and_unique_ids(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        cfg = baseline_cfg()
        arms = build_arms(spec, cfg, list(FACTOR_KEYS))
        self.assertEqual(arms[0]["arm"], BASELINE_ARM)
        self.assertEqual(arms[0]["overrides"], {})
        ids = [arm["arm"] for arm in arms]
        self.assertEqual(len(ids), len(set(ids)))
        for arm in arms[1:]:
            self.assertEqual(list(arm["overrides"]), [arm["factor"]])
            merged = build_arm_config(spec, cfg, arm, "campaign-x", "/tmp/runs")
            changed = {key for key in set(cfg) | set(merged)
                       if cfg.get(key) != merged.get(key)} - {
                           "output_root", "sensitivity", "seed", "variant", "run_tag"}
            self.assertEqual(changed, {arm["factor"]}, arm["arm"])

    def test_variant_flip_is_rejected_as_a_second_factor(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        cfg = baseline_cfg()
        arm = {"arm": "knn_k=20", "factor": "knn_k", "level": 20, "overrides": {"knn_k": 20}}
        merged = build_arm_config(spec, cfg, arm, "c", "/tmp/runs")
        merged["variant"] = "no_boundary"
        with self.assertRaises(ValueError):
            assert_single_factor(cfg, merged, arm)

    def test_relative_grid_uses_each_task_baseline(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        arms = build_arms(spec, baseline_cfg("mnist-add2"), ["full_boundary_weight"])
        values = {arm["level"]: arm["overrides"]["full_boundary_weight"]
                  for arm in arms if arm["factor"] == "full_boundary_weight"}
        self.assertEqual(values[2.0], 0.2)
        self.assertEqual(values[0.0], 0.0)
        arms = build_arms(spec, baseline_cfg("hwf"), ["full_boundary_weight"])
        values = {arm["level"]: arm["overrides"]["full_boundary_weight"]
                  for arm in arms if arm["factor"] == "full_boundary_weight"}
        self.assertAlmostEqual(values[2.0], 0.4)
        self.assertAlmostEqual(values[3.0], 0.6)

    def test_grid_must_contain_the_baseline_value(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        cfg = baseline_cfg()
        cfg["knn_k"] = 7
        with self.assertRaises(ValueError):
            build_arms(spec, cfg, ["knn_k"])

    def test_invalid_values_are_rejected(self):
        for key, value in (("knn_k", 0), ("graph_sigma", 0.0), ("risk_limit", 1.0),
                           ("curriculum_ramp_rounds", 0), ("curriculum_min_weight", 0.0),
                           ("full_boundary_weight", -0.1)):
            with self.assertRaises(ValueError):
                validate_value(key, value)

    def test_arm_config_refuses_a_second_factor(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        cfg = baseline_cfg()
        arm = {"arm": "knn_k=20", "factor": "knn_k", "level": 20, "overrides": {"knn_k": 20}}
        merged = build_arm_config(spec, cfg, arm, "c", "/tmp/runs")
        merged["graph_sigma"] = 0.9
        with self.assertRaises(ValueError):
            assert_single_factor(cfg, merged, arm)

    def test_campaign_identity_is_pinned_but_seed_agnostic(self):
        spec = load_sensitivity_config(DEFAULT_SENSITIVITY_CONFIG)
        baselines = make_baselines()
        first = campaign_identity(spec, baselines, list(FACTOR_KEYS), "rev-1")
        self.assertEqual(first, campaign_identity(spec, baselines, list(FACTOR_KEYS), "rev-1"))
        self.assertNotEqual(first, campaign_identity(spec, baselines, list(FACTOR_KEYS), "rev-2"))
        self.assertNotEqual(first, campaign_identity(spec, baselines, ["knn_k"], "rev-1"))
        moved = make_baselines()
        moved["hwf"]["risk_limit"] = 0.2
        self.assertNotEqual(first, campaign_identity(spec, moved, list(FACTOR_KEYS), "rev-1"))


class PlanningTests(unittest.TestCase):
    def test_plan_writes_configs_and_unions_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_spec(Path(temporary), seeds=(4,))
            baselines = make_baselines()
            plan = prepare_campaign(spec, baselines, list(FACTOR_KEYS), (4,), "rev-1")
            campaign_dir = Path(plan["campaign_dir"])
            self.assertTrue((campaign_dir / "plan.json").is_file())
            self.assertEqual(plan["n_runs"], sum(plan["n_arms"].values()))
            config_path = campaign_dir / "configs" / "mnist-add2" / "knn_k=20.json"
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["knn_k"], 20)
            self.assertEqual(stored["graph_sigma"], 0.5)
            self.assertEqual(stored["sensitivity"]["campaign"], plan["campaign"])
            self.assertEqual(stored["output_root"], str(campaign_dir / "runs"))
            extended = prepare_campaign(spec, baselines, list(FACTOR_KEYS), (0, 1, 4), "rev-1")
            self.assertEqual(extended["campaign"], plan["campaign"])
            self.assertEqual(extended["seeds"], [0, 1, 4])
            self.assertEqual(extended["n_runs"], sum(extended["n_arms"].values()) * 3)

    def test_config_drift_inside_a_campaign_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_spec(Path(temporary), seeds=(4,))
            plan = prepare_campaign(spec, make_baselines(), ["knn_k"], (4,), "rev-1")
            target = Path(plan["campaign_dir"]) / "configs" / "mnist-add2" / "knn_k=20.json"
            target.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_campaign(spec, make_baselines(), ["knn_k"], (4,), "rev-1")


class MetricsTests(unittest.TestCase):
    def test_history_metrics_reads_coverage_and_curve(self):
        metrics = history_metrics(history_records(0.9, rounds=4, coverage=0.75,
                                                  accepted_share=0.5))
        self.assertEqual(metrics["rounds"], 4)
        self.assertAlmostEqual(metrics["exposure_coverage_mean"], 0.75)
        self.assertAlmostEqual(metrics["acceptance_coverage_final"], 0.5)
        self.assertAlmostEqual(metrics["boundary_share_final"], 0.5)
        self.assertAlmostEqual(metrics["explanation_coverage_final"], 1.0)
        self.assertAlmostEqual(metrics["gate_coverage_final"], 0.5)
        self.assertEqual(metrics["gate_accepted_final"], 100.0)
        self.assertEqual(metrics["gate_candidates_final"], 200.0)
        self.assertAlmostEqual(metrics["validation_auc"], (0.897 + 0.898 + 0.899 + 0.9) / 4)
        self.assertEqual(metrics["rounds_to_best"], 4)
        self.assertEqual(metrics["alpha_none_rounds"], 0)

    def test_gate_coverage_is_zero_when_the_gate_is_closed(self):
        records = history_records(0.9, rounds=2)
        records[-1]["alpha"] = None
        metrics = history_metrics(records)
        self.assertEqual(metrics["gate_accepted_final"], 0.0)
        self.assertEqual(metrics["gate_coverage_final"], 0.0)

    def test_gate_coverage_is_zero_when_no_candidate_passes(self):
        records = history_records(0.9, rounds=2)
        for record in records:
            record["gate"] = [{"alpha": 0.99, "count": 0, "passed": False},
                              {"alpha": 0.95, "count": 0, "passed": False}]
            record["alpha"] = None
        metrics = history_metrics(records)
        self.assertEqual(metrics["gate_accepted_final"], 0.0)
        self.assertEqual(metrics["gate_coverage_final"], 0.0)

    def test_history_metrics_tolerates_missing_fields(self):
        metrics = history_metrics([{"round": 0}])
        self.assertEqual(metrics["rounds"], 1)
        self.assertIsNone(metrics["acceptance_coverage_final"])
        self.assertIsNone(metrics["validation_auc"])

    def test_duplicate_seeds_are_excluded_not_maximised(self):
        run = {"seed": 0, "status": COMPLETE_TRAINING_STATUS, "stopped_reason": None,
               "best_validation_accuracy": 0.9}
        self.assertEqual(len(usable_runs([run])), 1)
        duplicate = dict(run, best_validation_accuracy=0.99)
        self.assertEqual(usable_runs([run, duplicate]), [])
        failed = dict(run, status="FAILED")
        self.assertEqual(usable_runs([failed]), [])
        stopped = dict(run, stopped_reason="GATE_CLOSED")
        self.assertEqual(usable_runs([stopped]), [])
        # A failed/stopped retry must not veto the completed run next to it.
        self.assertEqual(len(usable_runs([run, failed])), 1)
        self.assertEqual(len(usable_runs([run, stopped])), 1)


class SummarizeTests(unittest.TestCase):
    def _campaign(self, temporary, factor_keys=("knn_k", "risk_limit")):
        spec = make_spec(Path(temporary), seeds=(0, 1))
        plan = prepare_campaign(spec, make_baselines(), list(factor_keys), (0, 1), "rev-1")
        campaign_dir = Path(plan["campaign_dir"])
        accuracies = {
            BASELINE_ARM: 0.980,
            "knn_k=1": 0.979, "knn_k=3": 0.9805, "knn_k=20": 0.9795, "knn_k=40": 0.979,
            "risk_limit=0.02": 0.930, "risk_limit=0.05": 0.979,
            "risk_limit=0.2": 0.900, "risk_limit=0.3": 0.900,
        }
        for task in ("mnist-add2", "hwf"):
            arm_ids = {record["arm"] for record in plan["arms"][task]}
            for arm, accuracy in accuracies.items():
                if arm not in arm_ids:
                    continue
                for seed in (0, 1):
                    write_run(campaign_dir, task, arm, seed, accuracy)
        return plan, campaign_dir

    def test_summary_detects_broad_and_narrow_regions(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, campaign_dir = self._campaign(temporary)
            summary = summarize(campaign_dir)
            self.assertEqual(summary["campaign"], plan["campaign"])
            self.assertEqual(summary["n_completed"], 18 * 2)
            self.assertEqual(summary["excluded_runs"], [])
            factors = {item["factor"]: item
                       for item in summary["tasks"]["mnist-add2"]["factors"]}
            self.assertEqual(factors["knn_k"]["verdict"], "broad_confirmed")
            self.assertEqual(factors["knn_k"]["descriptive_plateau"]["n_levels"], 5)
            self.assertEqual(factors["risk_limit"]["verdict"], "narrow")
            self.assertEqual(factors["risk_limit"]["descriptive_plateau"]["n_levels"], 2)
            self.assertTrue(factors["knn_k"]["levels"][0]["paired"])
            self.assertEqual(factors["risk_limit"]["levels"][2]["arm"], BASELINE_ARM)
            self.assertEqual(summary["tasks"]["mnist-add2"]["consistency"]["code_revisions"],
                             [METHOD_VERSION])
            self.assertFalse(summary["tasks"]["mnist-add2"]["consistency"]["mixed_sources"])

    def test_margin_boundary_is_inclusive(self):
        """A delta of exactly -margin must count as within-margin (float-safe)."""
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_spec(Path(temporary), seeds=(0, 1))
            plan = prepare_campaign(spec, make_baselines(), ["full_boundary_weight"],
                                    (0, 1), "rev-1")
            campaign_dir = Path(plan["campaign_dir"])
            for seed in (0, 1):
                write_run(campaign_dir, "hwf", BASELINE_ARM, seed, 0.800)
                write_run(campaign_dir, "hwf", "full_boundary_weight=0.5x", seed, 0.795)
            summary = summarize(campaign_dir)
            item = summary["tasks"]["hwf"]["factors"][0]
            row = {entry["arm"]: entry
                   for entry in item["levels"]}["full_boundary_weight=0.5x"]
            self.assertAlmostEqual(row["delta_mean"], -0.005)
            self.assertTrue(row["within_margin"])
            self.assertTrue(row["valid"])
            self.assertTrue(row["confirmed_valid"])

    def test_missing_coverage_fails_closed(self):
        """No coverage report must make a level invalid, never silently valid."""
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_spec(Path(temporary), seeds=(0, 1))
            plan = prepare_campaign(spec, make_baselines(), ["knn_k"], (0, 1), "rev-1")
            campaign_dir = Path(plan["campaign_dir"])
            accuracies = {BASELINE_ARM: 0.980, "knn_k=1": 0.979, "knn_k=3": 0.9805,
                          "knn_k=20": 0.9795, "knn_k=40": 0.979}
            for arm, accuracy in accuracies.items():
                for seed in (0, 1):
                    write_run(campaign_dir, "mnist-add2", arm, seed, accuracy,
                              with_history=arm != "knn_k=1")
            summary = summarize(campaign_dir)
            item = summary["tasks"]["mnist-add2"]["factors"][0]
            row = {entry["arm"]: entry for entry in item["levels"]}["knn_k=1"]
            self.assertIs(row["coverage_ok"], False)
            self.assertFalse(row["valid"])
            self.assertFalse(row["confirmed_valid"])

    def test_unpaired_levels_block_the_verdict(self):
        """Different seeds on baseline vs arm cannot be compared; grid is incomplete."""
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_spec(Path(temporary), seeds=(0, 1))
            plan = prepare_campaign(spec, make_baselines(), ["knn_k"], (0, 1), "rev-1")
            campaign_dir = Path(plan["campaign_dir"])
            write_run(campaign_dir, "mnist-add2", BASELINE_ARM, 0, 0.980)
            for arm in ("knn_k=1", "knn_k=3", "knn_k=20", "knn_k=40"):
                write_run(campaign_dir, "mnist-add2", arm, 1, 0.979)
            summary = summarize(campaign_dir)
            item = summary["tasks"]["mnist-add2"]["factors"][0]
            self.assertEqual(item["verdict"], "incomplete_grid")
            self.assertEqual(len(item["unpaired_levels"]), 4)
            self.assertTrue(all(not row["paired"] for row in item["levels"]
                                if row["arm"] != BASELINE_ARM))
            self.assertTrue(all(not row["valid"] for row in item["levels"]
                                if row["arm"] != BASELINE_ARM))

    def test_duplicate_completed_seeds_are_surfaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_spec(Path(temporary), seeds=(0,))
            plan = prepare_campaign(spec, make_baselines(), ["knn_k"], (0,), "rev-1")
            campaign_dir = Path(plan["campaign_dir"])
            write_run(campaign_dir, "mnist-add2", BASELINE_ARM, 0, 0.980)
            write_run(campaign_dir, "mnist-add2", BASELINE_ARM, 0, 0.981, suffix="-retry")
            summary = summarize(campaign_dir)
            duplicates = summary["tasks"]["mnist-add2"]["duplicate_seeds"]
            self.assertEqual(duplicates.get(BASELINE_ARM), [0])
            self.assertEqual(summary["tasks"]["mnist-add2"]["baseline"]["n"], 0)

    def test_coverage_collapse_removes_a_level_from_the_region(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_spec(Path(temporary), seeds=(0, 1))
            plan = prepare_campaign(spec, make_baselines(), ["knn_k"], (0, 1), "rev-1")
            campaign_dir = Path(plan["campaign_dir"])
            accuracies = {BASELINE_ARM: 0.980, "knn_k=1": 0.979, "knn_k=3": 0.9805,
                          "knn_k=20": 0.9795, "knn_k=40": 0.980}
            for arm, accuracy in accuracies.items():
                for seed in (0, 1):
                    # Two middle levels lose most of their accepted positives, so a
                    # stable-accuracy arm still fails the coverage side condition.
                    share = 0.2 if arm in {"knn_k=3", "knn_k=20"} else 1.0
                    write_run(campaign_dir, "mnist-add2", arm, seed, accuracy,
                              accepted_share=share)
            summary = summarize(campaign_dir)
            item = summary["tasks"]["mnist-add2"]["factors"][0]
            rows = {entry["arm"]: entry for entry in item["levels"]}
            self.assertFalse(rows["knn_k=3"]["coverage_ok"])
            self.assertFalse(rows["knn_k=3"]["valid"])
            self.assertTrue(rows[BASELINE_ARM]["valid"])
            # Only isolated valid levels survive; no contiguous block contains the
            # baseline level any more.
            self.assertEqual(item["verdict"], "isolated")

    def test_missing_arm_marks_the_grid_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, campaign_dir = self._campaign(temporary, factor_keys=("knn_k",))
            for run_dir in (campaign_dir / "runs").iterdir():
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                if config["sensitivity"]["arm"] != "knn_k=20":
                    continue
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                status["status"] = "FAILED"
                (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            summary = summarize(campaign_dir)
            item = summary["tasks"]["mnist-add2"]["factors"][0]
            self.assertEqual(item["verdict"], "incomplete_grid")
            self.assertIn(20.0, item["missing_levels"])

    def test_foreign_campaign_runs_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, campaign_dir = self._campaign(temporary)
            write_run(campaign_dir, "mnist-add2", BASELINE_ARM, 4, 0.99,
                      suffix="-foreign", campaign="other-campaign")
            runs, excluded = collect_runs(campaign_dir / "runs", plan["campaign"])
            self.assertEqual(len(excluded), 1)
            self.assertIn("not this sensitivity campaign", excluded[0]["reason"])
            self.assertEqual(len(runs), 36)

    def test_write_summary_writes_csv_json_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, campaign_dir = self._campaign(temporary)
            summary, output = write_summary(campaign_dir, Path(temporary) / "summary")
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "levels.csv").is_file())
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("参数敏感性实验报告", report)
            self.assertIn("宽泛有效区间", report)
            stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["campaign"], plan["campaign"])
            with self.assertRaises(FileExistsError):
                write_summary(campaign_dir, Path(temporary) / "summary")
            write_summary(campaign_dir, Path(temporary) / "summary", overwrite=True)


if __name__ == "__main__":
    unittest.main()
