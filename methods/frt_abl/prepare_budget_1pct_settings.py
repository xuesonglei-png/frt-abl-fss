"""Prepare budget arms using the complete 1% configuration unchanged.

Only the nested labelled-image prefix changes across 1/3/5/10 percent arms.
The 1% training settings, including warmup_epochs=20, rounds and all method
hyperparameters, are copied verbatim.  This is a separate protocol because
its optimizer-step exposure differs from the fixed-pool-v2 protocol.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from .controlled_budget import (BUDGET_ROLES, build_roles, nested_prefix,
                                role_budget_record)
from .budget import budget_plan
from .prepare import PROJECT, digest, load_config, verify_prepared, write_json

PROTOCOL = "annotation-budget-1pct-settings-v1"
RESERVOIR = PROJECT / "methods/frt_abl/prepared/budget-fixed-pool-v2/hwf-10pct"
BASE_CONFIG = PROJECT / "methods/frt_abl/configs/budget-fixed-pool-v2/hwf-1pct.json"
PREPARED_ROOT = PROJECT / "methods/frt_abl/prepared/budget-1pct-settings-v1"
CONFIG_ROOT = PROJECT / "methods/frt_abl/configs/budget-1pct-settings-v1"
OUTPUT_ROOT = PROJECT / "methods/frt_abl/runs-budget-1pct-settings-v1"
LEVELS = (0.01, 0.03, 0.05, 0.10)


def prepare(execute=False, seed=5):
    base = load_config(BASE_CONFIG)
    reservoir_receipt = json.loads((RESERVOIR / "manifest.json").read_text(encoding="utf-8"))
    reservoir = {role: json.loads((RESERVOIR / f"{role}.json").read_text(encoding="utf-8"))
                 for role in (*BUDGET_ROLES, "weak", "validation", "test")}
    universe = {"expressions": 10000, "unique_images": 35411}
    entries = []
    for level in LEVELS:
        plan = budget_plan(base, level, universe)
        roles = build_roles(reservoir, plan["target_role_unique_images"])
        counts = role_budget_record(roles)
        actual = sum(item["unique_labeled_images"] for item in counts.values())
        anchor_batches = math.ceil(len(roles["anchor"]) / base["batch_size"])
        warmup_updates = anchor_batches * base["warmup_epochs"]
        round_steps = math.ceil(len(roles["weak"]) / base["batch_size"])
        record = dict(plan, protocol=PROTOCOL, actual_labeled_images=actual,
                      actual_fraction=actual / universe["unique_images"],
                      reservoir_manifest_sha256=digest(RESERVOIR / "manifest.json"),
                      fixed_weak_sha256=digest(RESERVOIR / "weak.json"),
                      fixed_validation_sha256=digest(RESERVOIR / "validation.json"),
                      warmup_epochs=base["warmup_epochs"], warmup_updates=warmup_updates,
                      round_optimizer_steps=round_steps,
                      expected_total_optimizer_steps=warmup_updates + round_steps * base["rounds"],
                      annotation_roles=counts,
                      unchanged_from="budget-fixed-pool-v2/hwf-1pct",
                      unused_reservoir_policy="withheld_at_all_lower_budgets")
        label = plan["level_label"]
        dest = PREPARED_ROOT / f"hwf-{label}"
        config_path = CONFIG_ROOT / f"hwf-{label}__seed-{seed}.json"
        generated = deepcopy(base)
        generated.update(seed=seed, experiment=PROTOCOL, experiment_version=PROTOCOL,
                         prepared_root=str(dest), output_root=str(OUTPUT_ROOT),
                         run_tag=f"budget-1pct-settings-{label}",
                         controlled_budget=record)
        for role, field in (("anchor", "anchor_count"),
                            ("calibration_fit", "calibration_fit_count"),
                            ("calibration_gate", "calibration_gate_count")):
            generated[field] = len(roles[role])
        entries.append((label, roles, record, generated, config_path, dest))

    if execute:
        if PREPARED_ROOT.exists() or CONFIG_ROOT.exists():
            raise FileExistsError("Refusing to overwrite pure-rate budget artifacts")
        PREPARED_ROOT.mkdir(parents=True)
        CONFIG_ROOT.mkdir(parents=True)
        for label, roles, record, generated, config_path, dest in entries:
            dest.mkdir()
            files = {}
            for role, rows in roles.items():
                path = dest / f"{role}.json"
                write_json(path, rows)
                files[path.name] = digest(path)
            for role in ("weak", "validation", "test"):
                if files[f"{role}.json"] != digest(RESERVOIR / f"{role}.json"):
                    raise ValueError(f"Fixed role hash differs: {role}")
            receipt = deepcopy(reservoir_receipt)
            receipt.update(experiment=PROTOCOL, experiment_version=PROTOCOL,
                           counts={role: len(rows) for role, rows in roles.items()},
                           files=files, annotation_budget=record["annotation_roles"],
                           controlled_budget=record,
                           warning="Pure-rate follow-up: all 1% config settings unchanged.")
            write_json(dest / "manifest.json", receipt)
            write_json(config_path, generated)
            verify_prepared(load_config(config_path))
        campaign = {"protocol": PROTOCOL, "status": "PREPARED_NOT_TRAINED",
                    "seed": seed, "base_config": str(BASE_CONFIG),
                    "entries": [{"level_label": label, "config": str(config_path),
                                 "prepared_root": str(dest),
                                 "expected_total_optimizer_steps": record["expected_total_optimizer_steps"]}
                                for label, _roles, record, _generated, config_path, dest in entries]}
        write_json(PREPARED_ROOT / "campaign.json", campaign)
    return [{"level_label": label, "config": str(config_path),
             "counts": {role: len(rows) for role, rows in roles.items()},
             "expected_total_optimizer_steps": record["expected_total_optimizer_steps"]}
            for label, roles, record, _generated, config_path, _dest in entries]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(prepare(args.execute, args.seed), ensure_ascii=False, indent=2))
