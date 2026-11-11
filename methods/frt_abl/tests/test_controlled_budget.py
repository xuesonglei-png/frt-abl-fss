import copy
import unittest

from methods.frt_abl.controlled_budget import build_roles, exposure_schedule, nested_prefix


def reservoir():
    roles = {}
    for role in ("anchor", "calibration_fit", "calibration_gate", "weak", "validation", "test"):
        roles[role] = [{"id": f"{role}-{i}", "images": [f"{role}/{i}"], "target": 1}
                       for i in range(12 if role != "weak" else 40)]
        if role in ("anchor", "calibration_fit", "calibration_gate"):
            for row in roles[role]:
                row["labels"] = [0]
    return roles


class ControlledBudgetTests(unittest.TestCase):
    def test_nested_roles_keep_unused_labels_out_of_weak(self):
        source = reservoir()
        snapshot = copy.deepcopy(source)
        low = build_roles(source, dict(anchor=2, calibration_fit=3, calibration_gate=4))
        high = build_roles(source, dict(anchor=7, calibration_fit=8, calibration_gate=9))
        for role in ("anchor", "calibration_fit", "calibration_gate"):
            self.assertEqual(low[role], high[role][:len(low[role])])
        for role in ("weak", "validation", "test"):
            self.assertEqual(low[role], high[role])
        self.assertEqual(snapshot, source)

    def test_rejects_cross_role_image_reuse_and_unbudgeted_truth(self):
        source = reservoir()
        source["weak"][0]["images"] = source["anchor"][0]["images"]
        with self.assertRaisesRegex(ValueError, "Image leakage"):
            build_roles(source, dict(anchor=2, calibration_fit=3, calibration_gate=4))
        source = reservoir()
        source["weak"][0]["labels"] = [0]
        with self.assertRaisesRegex(ValueError, "Unbudgeted"):
            build_roles(source, dict(anchor=2, calibration_fit=3, calibration_gate=4))

    def test_warmup_and_full_round_steps_equal_despite_different_label_counts(self):
        source = reservoir()
        roles = [build_roles(source, dict(anchor=n, calibration_fit=n, calibration_gate=n))
                 for n in (2, 5, 11)]
        warmup, rounds, epochs = exposure_schedule(roles, 4, 5)
        self.assertEqual((warmup, rounds, epochs), (30, 10, [30, 15, 10]))
        roles[0]["weak"].pop()
        with self.assertRaisesRegex(ValueError, "fixed"):
            exposure_schedule(roles, 4, 5)

    def test_budget_uses_unique_images_and_fails_closed_if_reservoir_insufficient(self):
        rows = [{"images": ["a", "b"]}, {"images": ["b", "c"]}]
        self.assertEqual(len(nested_prefix(rows, 3)), 2)
        with self.assertRaisesRegex(ValueError, "cannot reach"):
            nested_prefix(rows, 4)


if __name__ == "__main__":
    unittest.main()
