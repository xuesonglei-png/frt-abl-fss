"""v2 regression specifications, added without execution on 2026-10-27."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import unittest

from methods.frt_abl.core import Region
from methods.frt_abl.policies import (METHOD_VERSION, curriculum, resolve_policy,
                                    spread_batches, threshold_options, validate_policy_config)
from methods.frt_abl.aggregate import COMPLETE, comparison_id, paired_deltas, usable


def config(**overrides):
    cfg = {"method_version": METHOD_VERSION, "task": "hwf", "variant": "full",
           "threshold_policy": "risk_guarded", "curriculum_policy": "soft_weight",
           "alpha_grid": [round(.80+i*.01, 2) for i in range(20)], "alpha_family": 20,
           "alpha_start": .95, "alpha_min": .80, "alpha_decay_step": .01,
           "curriculum_min_weight": .5, "curriculum_ramp_rounds": 10,
           "loss_ramp_rounds": 10, "train_fraction": .25}
    cfg.update(overrides)
    return cfg


def proposal(identifier, score, length=1, weight=.8):
    return SimpleNamespace(sample={"id": identifier, "images": ["image"]*length, "target": 4},
                           region=Region.POSITIVE, lower=score, q=score, weight=weight,
                           curriculum_weight=1.)


class PolicyTests(unittest.TestCase):
    def test_single_module_removal_changes_exactly_one_factor(self):
        full = asdict(resolve_policy(config()))
        for variant, factor in (("confidence_only", "graph"), ("no_weight", "reliability_weight"),
                                ("no_boundary", "boundary"), ("fixed_threshold", "threshold"),
                                ("no_curriculum", "curriculum")):
            altered = asdict(resolve_policy(config(variant=variant)))
            self.assertEqual({k for k in full if full[k] != altered[k]}, {factor})

    def test_factorial_covers_exactly_four_combinations(self):
        actual = {(resolve_policy(config(variant=v)).threshold, resolve_policy(config(variant=v)).curriculum)
                  for v in ("full", "fixed_threshold", "no_curriculum", "fixed_no_curriculum")}
        self.assertEqual(actual, {("risk_guarded", "soft_weight"), ("fixed", "soft_weight"),
                                  ("risk_guarded", "all"), ("fixed", "all")})

    def test_risk_threshold_never_falls_with_rounds(self):
        for r in (0, 1, 10, 49):
            grid, floor = threshold_options(config(), r)
            self.assertEqual(floor, .95)
            self.assertTrue(all(a >= .95 for a in grid))
        self.assertEqual(threshold_options(config(variant="fixed_threshold"), 49), ([.95], .95))

    def test_legacy_diagnostic_is_explicit(self):
        cfg = config(threshold_policy="linear_decay", curriculum_policy="hard_fraction")
        self.assertEqual(threshold_options(cfg, 49)[1], .8)
        self.assertEqual(threshold_options(dict(cfg, variant="fixed_threshold"), 49), ([.95], .95))

    def test_fixed_threshold_preserves_soft_curriculum(self):
        data = [proposal(str(i), .8+i*.02) for i in range(8)]
        a = curriculum(deepcopy(data), config(), 0)
        b = curriculum(deepcopy(data), config(variant="fixed_threshold"), 0)
        self.assertEqual([(p.sample["id"], p.curriculum_weight) for p in a],
                         [(p.sample["id"], p.curriculum_weight) for p in b])
        self.assertEqual(len(a), len(data))

    def test_soft_curriculum_preserves_long_samples_and_stratum_mass(self):
        data = [proposal("s1", .98), proposal("s2", .99), proposal("l1", .90, 7), proposal("l2", .91, 7)]
        selected = curriculum(data, config(), 0)
        self.assertEqual(len(selected), 4)
        for n in (1, 7):
            group = [p for p in selected if len(p.sample["images"]) == n]
            self.assertAlmostEqual(sum(p.weight for p in group), sum(p.weight*p.curriculum_weight for p in group))
            self.assertTrue(all(p.curriculum_weight > 0 for p in group))
        self.assertGreater(data[1].curriculum_weight, data[0].curriculum_weight)

    def test_constant_membership_does_not_invent_ranking(self):
        selected = curriculum([proposal("a", .99), proposal("b", .99)], config(), 0)
        self.assertTrue(all(p.curriculum_weight == 1 for p in selected))

    def test_late_curriculum_and_no_curriculum_are_neutral(self):
        for cfg, r in ((config(), 10), (config(variant="no_curriculum"), 0)):
            chosen = curriculum([proposal("a", .8), proposal("b", .99)], cfg, r)
            self.assertEqual([p.curriculum_weight for p in chosen], [1., 1.])

    def test_boundary_is_never_accepted_by_waiting(self):
        deferred = proposal("x", .99)
        deferred.region = Region.BOUNDARY
        self.assertEqual(curriculum([deferred], config(), 1000), [])

    def test_batches_used_once_across_entire_round(self):
        batches = [["a"], ["b"]]
        self.assertEqual(spread_batches(batches, 8), {2: ["a"], 6: ["b"]})
        self.assertEqual(spread_batches([], 8), {})
        with self.assertRaises(ValueError):
            spread_batches(batches, 1)

    def test_full_and_fixed_use_same_hypothesis_family(self):
        for v in ("full", "fixed_threshold"):
            cfg = config(variant=v, alpha_family=1)
            with self.assertRaises(ValueError):
                validate_policy_config(cfg)


class SummaryTests(unittest.TestCase):
    @staticmethod
    def row(seed, value, status=COMPLETE):
        return {"seed": seed, "status": status, "best_validation_accuracy": value, "optimizer_steps": 100}

    def test_failed_and_same_seed_retries_are_not_independent_results(self):
        rows = [self.row(0, .9), self.row(0, .99), self.row(1, .99, "FAILED"), self.row(4, .8)]
        self.assertEqual([r["seed"] for r in usable(rows)], [4])

    def test_pairing_keeps_losses_and_ties(self):
        groups = {("hwf", "full", "id"): [self.row(4, .8)],
                  ("hwf", "no_boundary", "id"): [self.row(4, .81)],
                  ("hwf", "fixed_threshold", "id"): [self.row(4, .8)]}
        rows = paired_deltas(groups)
        self.assertTrue(all(r["strictly_positive_on_all_paired_seeds"] is False for r in rows))
        self.assertTrue(all(len(r["paired_seeds"]) == 1 for r in rows))

    def test_versions_and_budgets_cannot_be_mixed(self):
        cfg = config(seed=4, output_root="out")
        env = {"code_revision": METHOD_VERSION, "prepared_manifest_sha256": "data"}
        original = comparison_id(cfg, env, {"train.py": "sha"})
        self.assertEqual(original, comparison_id(dict(cfg, seed=1, variant="no_boundary"), env, {"train.py": "sha"}))
        self.assertNotEqual(original, comparison_id(dict(cfg, method_version="old"), env, {"train.py": "sha"}))
        self.assertNotEqual(original, comparison_id(dict(cfg, loss_ramp_rounds=1), env, {"train.py": "sha"}))


if __name__ == "__main__":
    unittest.main()
