"""Anchor-only training regression with deterministic synthetic images."""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from methods.frt_abl import train
from methods.frt_abl.prepare import PROJECT, load_config, write_json


class SyntheticStore:
    def __init__(self, cfg):
        pass

    def batch(self, rows, device):
        lengths = [len(row["images"]) for row in rows]
        grid = torch.arange(784, dtype=torch.float32).reshape(1, 28, 28)
        images = [(grid + int(image)).remainder(255) / 255
                  for row in rows for image in row["images"]]
        return torch.stack(images).to(device), lengths


class AnchorTrainingTests(unittest.TestCase):
    def config(self, root, variant="anchor_only"):
        cfg = load_config(PROJECT / "methods/frt_abl/configs/hwf.json")
        cfg.update(variant=variant, seed=4, rounds=2, warmup_epochs=1, batch_size=2,
                   search_workers=1, prepared_root=str(root / "prepared"),
                   output_root=str(root / "runs"), threads=1)
        prepared = Path(cfg["prepared_root"])
        prepared.mkdir()
        write_json(prepared / "manifest.json", {"synthetic": True})
        for index, (role, count) in enumerate((("anchor", 3), ("calibration_fit", 2),
                                               ("calibration_gate", 2), ("weak", 5),
                                               ("validation", 2))):
            rows = [{"id": f"{role}-{i}", "images": [str(index * 10 + i)],
                     "target": i % 9 + 1} for i in range(count)]
            if role in {"anchor", "calibration_fit", "calibration_gate"}:
                for i, row in enumerate(rows):
                    row["labels"] = [i % 9]
            write_json(prepared / f"{role}.json", rows)
        return cfg

    def execute(self, module, cfg, forbid_search=False):
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "verify_prepared", return_value={}))
            stack.enter_context(patch("methods.frt_abl.runtime.ImageStore", SyntheticStore))
            if forbid_search:
                for name in ("propose", "Calibrator"):
                    stack.enter_context(patch(f"methods.frt_abl.runtime.{name}",
                                              side_effect=AssertionError(f"Unexpected {name}")))
                stack.enter_context(patch("concurrent.futures.ProcessPoolExecutor",
                                          side_effect=AssertionError("Unexpected search pool")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            module.run(cfg)
        return next(Path(cfg["output_root"]).iterdir())

    def test_no_search_and_exact_training_equivalence_to_saved_entrypoint(self):
        baseline_path = PROJECT / "methods/frt_abl/anchor-fix-20261104/source_before/train.py"
        spec = importlib.util.spec_from_file_location("methods.frt_abl._anchor_before_fix", baseline_path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            old = self.execute(baseline, cfg)
            cfg["output_root"] = str(root / "fixed")
            cfg["search_workers"] = 2  # Must not even construct the worker pool.
            new = self.execute(train, cfg, forbid_search=True)
            for checkpoint in ("best.pt", "last.pt"):
                a = torch.load(old / checkpoint, weights_only=False)
                b = torch.load(new / checkpoint, weights_only=False)
                if checkpoint == "last.pt":
                    self.assertEqual(a["optimizer_steps"], 8)  # 2 warmup + 2 x 3 updates.
                self.assertEqual(a["optimizer_steps"], b["optimizer_steps"])
                self.assertEqual(a["validation_task"], b["validation_task"])
                for role in ("student", "teacher"):
                    for key in a[role]:
                        self.assertTrue(torch.equal(a[role][key], b[role][key]), (checkpoint, role, key))
                for parameter, state in a["optimizer"]["state"].items():
                    for key, value in state.items():
                        self.assertTrue(torch.equal(value, b["optimizer"]["state"][parameter][key]))
            history = [json.loads(line) for line in (new / "history.jsonl").read_text().splitlines()]
            for row in history:
                self.assertTrue(row["proposal_skipped"])
                self.assertEqual(row["weak_sequences_total"], 5)
                self.assertEqual(row["coverage"], 0)
                self.assertEqual(row["loss_term_steps"], {"anchor": 3})
                self.assertEqual(row["proposal_images"], 0)
                self.assertEqual(row["candidate_checks"], 0)
                self.assertEqual(row["search_calls"], 0)
            status = json.loads((new / "status.json").read_text())
            self.assertEqual(status["status"], "TRAINING_COMPLETED_TEST_NOT_RUN")
            self.assertFalse(status["formal_test_read"])


if __name__ == "__main__":
    unittest.main()
