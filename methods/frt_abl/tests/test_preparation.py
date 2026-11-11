import unittest

from methods.frt_abl.prepare import allocate
from methods.frt_abl.audit import summarize


class PreparationTests(unittest.TestCase):
    def test_roles_image_disjoint_and_purge_reused_images(self):
        rows = [{"id": f"s{i}", "images": [f"i{i//2}"], "target": 1} for i in range(30)]
        cfg = {"task": "hwf", "validation_count": 1, "anchor_count": 1,
               "calibration_fit_count": 1, "calibration_gate_count": 1}
        roles, purged = allocate(cfg, {"train": rows, "test": [{"id": "test", "images": ["test-img"], "target": 2}]})
        sets = [{i for s in samples for i in s["images"]} for samples in roles.values()]
        for i, a in enumerate(sets):
            for b in sets[i+1:]:
                self.assertFalse(a & b)
        self.assertEqual(purged, 4)

    def test_offline_risk_coverage_denominators(self):
        rows = [{"sample_id": str(i), "candidate": [i] if i < 2 else None,
                 "lower": 1-i*.2, "q": .8, "correct": i == 0,
                 "trained": i < 2, "region": "positive" if i < 2 else "boundary", "age": i}
                for i in range(3)]
        report, curve = summarize(rows)
        self.assertEqual(report["trained_correction_error_rate"], .5)
        self.assertEqual(report["wrong_training_fraction_of_all_weak"], 1/3)
        self.assertEqual(report["candidate_found_rate"], 2/3)
        self.assertEqual(report["candidate_accuracy"], .5)
        self.assertEqual(report["negative_count"], 0)
        self.assertEqual(report["eligible_count"], 0)
        self.assertEqual(curve[-1]["coverage"], 2/3)
        self.assertEqual(curve[-1]["correction_risk"], .5)


if __name__ == "__main__":
    unittest.main()
