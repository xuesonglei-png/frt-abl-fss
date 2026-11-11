"""HWF nested annotation budgets with fixed weak data and optimizer exposure.

Uses the already prepared maximum-budget reservoir. Unannotated reservoir rows
remain withheld, rather than migrating into weak data. No training code changes
are needed: weak batches dominate the round schedule, and warmup epoch counts
are normalized to the same number of optimizer updates.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from .budget import (BUDGET_ROLES, COUNT_FIELD, budget_plan, level_label,
                     role_budget_record, verify_budget_prepared)
from .prepare import PROJECT, digest, load_config, verify_prepared, write_json

PROTOCOL = "annotation-budget-fixed-pool-v2"
DEFAULT_ROOT = PROJECT / "methods/frt_abl/prepared/budget-fixed-pool-v2"
DEFAULT_CONFIGS = PROJECT / "methods/frt_abl/configs/budget-fixed-pool-v2"


def nested_prefix(rows, target):
    chosen, images = [], set()
    for row in rows:
        chosen.append(deepcopy(row))
        images.update(row["images"])
        if len(images) >= target:
            return chosen
    raise ValueError(f"Reservoir cannot reach {target} unique images")


def build_roles(reservoir, targets):
    roles = {role: nested_prefix(reservoir[role], targets[role]) for role in BUDGET_ROLES}
    roles.update({role: deepcopy(reservoir[role]) for role in ("weak", "validation", "test")})
    images = {role: {image for row in rows for image in row["images"]}
              for role, rows in roles.items()}
    for index, role in enumerate(images):
        for other in list(images)[index + 1:]:
            if images[role] & images[other]:
                raise ValueError(f"Image leakage: {role}/{other}")
    for role in ("weak", "validation", "test"):
        if any("labels" in row for row in roles[role]):
            raise ValueError(f"Unbudgeted symbol labels in {role}")
    return roles


def exposure_schedule(role_sets, batch_size, base_warmup_epochs):
    anchor_batches = [math.ceil(len(roles["anchor"]) / batch_size) for roles in role_sets]
    # The LCM handles general batch counts; for the real HWF 1/3/10% arms
    # these counts are 1/2/4, so warmup is exactly 20 updates in every arm.
    updates = math.lcm(*anchor_batches) * base_warmup_epochs
    weak_counts = {len(roles["weak"]) for roles in role_sets}
    if len(weak_counts) != 1 or not next(iter(weak_counts)):
        raise ValueError("Weak pool must be fixed and nonempty")
    round_steps = math.ceil(next(iter(weak_counts)) / batch_size)
    if any(count > round_steps for count in anchor_batches):
        raise ValueError("Weak batches must dominate the natural round schedule")
    return updates, round_steps, [updates // count for count in anchor_batches]


def prepare(levels=(0.01, 0.03, 0.10), execute=False,
            reservoir_config="methods/frt_abl/configs/budget/hwf-10pct.json"):
    cfg = load_config(reservoir_config)
    if cfg["task"] != "hwf":
        raise ValueError("This controlled protocol is scoped to HWF")
    receipt = verify_prepared(cfg)
    verify_budget_prepared(cfg, receipt)
    if len(set(levels)) != len(levels) or not levels or max(levels) > cfg["budget"]["level"]:
        raise ValueError("Unique levels must fit inside the maximum-budget reservoir")
    root = Path(cfg["prepared_root"])
    reservoir = {role: json.loads((root / f"{role}.json").read_text(encoding="utf-8"))
                 for role in (*BUDGET_ROLES, "weak", "validation", "test")}
    universe = {"expressions": cfg["budget"]["train_expressions"],
                "unique_images": cfg["budget"]["train_image_universe"]}
    plans = [budget_plan(cfg, level, universe) for level in sorted(levels)]
    role_sets = [build_roles(reservoir, plan["target_role_unique_images"]) for plan in plans]
    warmup_updates, round_steps, warmup_epochs = exposure_schedule(
        role_sets, cfg["batch_size"], cfg["warmup_epochs"])
    entries = []
    for plan, roles, epochs in zip(plans, role_sets, warmup_epochs):
        label = plan["level_label"]
        dest = DEFAULT_ROOT / f"hwf-{label}"
        config_path = DEFAULT_CONFIGS / f"hwf-{label}.json"
        counts = role_budget_record(roles)
        actual = sum(row["unique_labeled_images"] for row in counts.values())
        record = dict(plan, protocol=PROTOCOL, actual_labeled_images=actual,
                      actual_fraction=actual / universe["unique_images"],
                      reservoir_manifest_sha256=digest(root / "manifest.json"),
                      fixed_weak_sha256=digest(root / "weak.json"),
                      fixed_validation_sha256=digest(root / "validation.json"),
                      warmup_updates=warmup_updates, round_optimizer_steps=round_steps,
                      expected_total_optimizer_steps=warmup_updates + round_steps * cfg["rounds"],
                      annotation_roles=counts,
                      unused_reservoir_policy="withheld_at_all_lower_budgets")
        generated = deepcopy(cfg)
        for key in ("budget", "base_prepared_root"):
            generated.pop(key, None)
        generated.update(experiment=PROTOCOL, experiment_version=PROTOCOL,
                         controlled_budget=record, warmup_epochs=epochs,
                         prepared_root=str(dest),
                         output_root=str(PROJECT / "methods/frt_abl/runs-budget-fixed-pool-v2"),
                         run_tag=f"budget-fixed-{label}")
        for role in BUDGET_ROLES:
            generated[COUNT_FIELD[role]] = len(roles[role])
        entry = dict(config=str(config_path), level=plan["level"],
                     level_label=label, actual_fraction=record["actual_fraction"],
                     counts={role: len(rows) for role, rows in roles.items()},
                     warmup_epochs=epochs, warmup_updates=warmup_updates,
                     round_steps=round_steps, total_steps=record["expected_total_optimizer_steps"])
        if execute:
            if dest.exists() or config_path.exists():
                raise FileExistsError(f"Refusing to overwrite {dest} or {config_path}")
            dest.mkdir(parents=True)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            files = {}
            for role, rows in roles.items():
                path = dest / f"{role}.json"
                # Identical serialization preserves exact weak/validation/test hashes.
                write_json(path, rows)
                files[path.name] = digest(path)
            for role in ("weak", "validation", "test"):
                if files[f"{role}.json"] != digest(root / f"{role}.json"):
                    raise ValueError(f"Fixed role hash differs: {role}")
            new_receipt = deepcopy(receipt)
            new_receipt.pop("budget", None)
            new_receipt.update(experiment=PROTOCOL, experiment_version=PROTOCOL,
                               controlled_budget=record, files=files,
                               counts=entry["counts"], annotation_budget=counts,
                               warning="Fixed maximum-budget reservoir; unused rows withheld. "
                                       "Not comparable to v1 or COMMON runs.")
            write_json(dest / "manifest.json", new_receipt)
            write_json(config_path, generated)
            verify_controlled(load_config(config_path))
        entries.append(entry)
    report = dict(protocol=PROTOCOL, status="PREPARED" if execute else "PLAN_ONLY",
                  primary_metric="best_validation_accuracy", formal_test_predictions_read=False,
                  inference_test_disabled=True, entries=entries)
    if execute:
        write_json(DEFAULT_ROOT / "campaign.json", report)
    return report


def verify_controlled(cfg):
    receipt = verify_prepared(cfg)
    record = cfg["controlled_budget"]
    if receipt["controlled_budget"] != record or record["protocol"] != PROTOCOL:
        raise ValueError("Controlled budget receipt/config mismatch")
    root = Path(cfg["prepared_root"])
    for role, field in (("weak", "fixed_weak_sha256"), ("validation", "fixed_validation_sha256")):
        if digest(root / f"{role}.json") != record[field]:
            raise ValueError(f"Fixed {role} identity mismatch")
    if math.ceil(cfg["anchor_count"] / cfg["batch_size"]) * cfg["warmup_epochs"] != record["warmup_updates"]:
        raise ValueError("Warmup update schedule mismatch")
    if math.ceil(receipt["counts"]["weak"] / cfg["batch_size"]) != record["round_optimizer_steps"]:
        raise ValueError("Round update schedule mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.01, 0.03, 0.10])
    args = parser.parse_args()
    print(json.dumps(prepare(args.levels, args.execute), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
