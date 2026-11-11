"""Pure synthetic regression tests. No torch, dataset, model or optimizer.

Written for a later verification stage; the code-only task does not run them.
"""

from copy import deepcopy
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ...core import Evidence, Region
from ..check_source import audit

PACKAGE = "methods.frt_abl.ablation_v11"
SINGLE = ("confidence_only", "fixed_threshold", "no_curriculum", "no_weight", "no_boundary")


def modules(entry):
    return (
        importlib.import_module(f"{PACKAGE}.{entry}.settings"),
        importlib.import_module(f"{PACKAGE}.{entry}.protocol"),
    )


def candidate(identifier="supported", lower=0.4, upper=0.8, q=0.7, labels=(1,)):
    return SimpleNamespace(
        sample={"id": identifier, "images": ["synthetic"], "target": 4, "labels": [1]},
        labels=labels,
        lower=lower,
        upper=upper,
        q=q,
        weight=0.0,
        curriculum_weight=1.0,
        region=Region.BOUNDARY,
        evidence=Evidence.SUPPORTED if labels else Evidence.UNRESOLVED,
        stable_count=0,
        search=SimpleNamespace(n_supported=2, complete=False, unknown_count=1, margin=0.0),
    )


class IndependentProtocolTests(unittest.TestCase):
    def test_static_single_factor_boundaries(self):
        self.assertEqual(audit()["status"], "STATIC_SOURCE_CHECK_PASSED")

    def test_every_soft_arm_admits_supported_candidates_when_gate_is_closed(self):
        for entry in ("full", *SINGLE, "fixed_no_curriculum"):
            settings, protocol = modules(entry)
            for task in ("mnist-add2", "hwf"):
                with self.subTest(entry=entry, task=task):
                    cfg = settings.build_config(task)
                    supported = candidate()
                    unresolved = candidate("unresolved", labels=None)
                    protocol.partition_candidates([supported, unresolved], cfg, None)
                    self.assertEqual(supported.region, Region.POSITIVE)
                    self.assertEqual(unresolved.region, Region.BOUNDARY)
                    self.assertEqual(unresolved.weight, 0.0)
                    scale = cfg["soft_scale_floor"] + (1 - cfg["soft_scale_floor"]) * (0.4 / 0.95)
                    expected = scale if entry == "no_weight" else 0.4 * 0.7 * 0.6 * scale
                    self.assertAlmostEqual(supported.weight, expected)

    def test_no_weight_preserves_soft_scaling_without_base_reliability(self):
        settings, protocol = modules("no_weight")
        cfg = settings.build_config("hwf")
        a, b = candidate(), candidate(q=0.01, upper=0.99)
        protocol.partition_candidates([a, b], cfg, 0.95)
        self.assertEqual(a.weight, b.weight)
        self.assertGreater(a.weight, 0.5)
        self.assertLess(a.weight, 1.0)

    def test_confidence_arm_changes_bounds_but_keeps_soft_admission(self):
        settings, protocol = modules("confidence_only")
        cfg = settings.build_config("hwf")
        weak, gate = candidate(), candidate("gate")
        protocol.build_graph([weak], [gate], [], cfg)
        self.assertEqual((weak.lower, weak.upper), (0.7, 0.7))
        self.assertEqual((gate.lower, gate.upper), (0.7, 0.7))
        protocol.partition_candidates([weak], cfg, None)
        self.assertAlmostEqual(weak.weight, 0.7**2 * (0.5 + 0.5 * 0.7 / 0.95))

    def test_all_gate_arms_keep_full_preference_and_same_risk_family(self):
        for entry in ("full", *SINGLE, "fixed_no_curriculum"):
            settings, protocol = modules(entry)
            cfg = settings.build_config("hwf")
            with self.subTest(entry=entry), patch.object(
                protocol, "choose_alpha", return_value=(0.95, [])
            ) as choose:
                protocol.choose_threshold([candidate()], cfg, object())
                self.assertTrue(choose.call_args.kwargs["prefer_high"])
                self.assertEqual(choose.call_args.kwargs["min_coverage"], 0.9)
                self.assertEqual(choose.call_args.kwargs["family_size"], 20)
                grid = choose.call_args.args[3]
                expected = (
                    [0.95]
                    if entry in {"fixed_threshold", "fixed_no_curriculum"}
                    else [0.95, 0.96, 0.97, 0.98, 0.99]
                )
                self.assertEqual(grid, expected)

    def test_all_gate_arms_preserve_lowest_passing_fallback(self):
        for entry in ("full", *SINGLE, "fixed_no_curriculum"):
            settings, protocol = modules(entry)
            cfg = settings.build_config("hwf")
            with self.subTest(entry=entry), patch.object(
                protocol, "choose_alpha", side_effect=[(None, []), (0.95, [])]
            ) as choose:
                self.assertEqual(protocol.choose_threshold([candidate()], cfg, object())[0], 0.95)
                self.assertFalse(choose.call_args.kwargs["prefer_high"])
                self.assertIsNone(choose.call_args.kwargs["min_coverage"])

    def test_no_curriculum_changes_only_rank_weight_not_accepted_set(self):
        settings, full = modules("full")
        _, removed = modules("no_curriculum")
        cfg = settings.build_config("hwf")
        rows = [candidate("low", lower=0.5), candidate("high", lower=0.9)]
        full.partition_candidates(rows, cfg, 0.95)
        ranked = full.select_curriculum(deepcopy(rows), cfg, 0)
        plain = removed.select_curriculum(deepcopy(rows), cfg, 0)
        self.assertEqual({p.sample["id"] for p in ranked}, {p.sample["id"] for p in plain})
        self.assertTrue(all(p.curriculum_weight == 1.0 for p in plain))
        self.assertNotEqual(ranked[0].curriculum_weight, ranked[1].curriculum_weight)
        self.assertAlmostEqual(
            sum(p.weight for p in rows), sum(p.weight * p.curriculum_weight for p in ranked)
        )

    def test_same_anchor_schedule_is_independent_of_boundary_batches(self):
        _, full = modules("full")
        _, removed = modules("no_boundary")
        for steps in (1, 8, 71, 410):
            batches = [[0]] if steps == 1 else [[0], [1]]
            self.assertEqual(
                full.spread_batches(batches, steps), removed.spread_batches(batches, steps)
            )

    def test_settings_are_fresh_and_reject_another_experiment_identity(self):
        for entry in ("full", *SINGLE, "standard_abl", "anchor_only", "fixed_no_curriculum"):
            settings, _ = modules(entry)
            a = settings.build_config("hwf", 0)
            a["alpha_grid"].clear()
            self.assertEqual(len(settings.build_config("hwf", 0)["alpha_grid"]), 20)
            invalid = settings.build_config("hwf", 0)
            invalid["variant"] = "anchor_only" if entry != "anchor_only" else "full"
            with self.assertRaises(ValueError):
                settings.validate_config(invalid)

    def test_references_are_explicitly_not_single_module_ablations(self):
        for entry, kind in (
            ("standard_abl", "combined_reference"),
            ("anchor_only", "supervised_reference"),
        ):
            settings, protocol = modules(entry)
            cfg = settings.build_config("hwf")
            self.assertEqual(protocol.experiment_record(cfg)["comparison_kind"], kind)


if __name__ == "__main__":
    unittest.main()
