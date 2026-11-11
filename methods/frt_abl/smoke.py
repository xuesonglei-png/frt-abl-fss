"""Build a small SMOKE_ONLY prepared root from an existing prepared root.

The smoke evaluation set is taken from the *validation* role (train-derived
expressions/images), never from the formal test manifest.  Every selected role
stays image-disjoint across roles, and the smoke test set excludes images used
by the smoke validation role.
"""
import argparse
import json
from pathlib import Path

from .prepare import PROJECT, digest, load_config, verify_prepared, write_json


def image_set(rows):
    return {image for row in rows for image in row["images"]}


def head(rows, count):
    if count < 0 or count > len(rows):
        raise ValueError(f"Cannot take {count} rows from {len(rows)}")
    return rows[:count]


def disjoint_tail(rows, count, forbidden):
    chosen, used = [], set(forbidden)
    for row in rows:
        images = set(row["images"])
        if images & used:
            continue
        chosen.append(row)
        used |= images
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Only {len(chosen)} image-disjoint rows available, need {count}")


def build(source_cfg_path, dest, config_out, weak, anchor, fit, gate, validation, evaluation):
    cfg = load_config(source_cfg_path)
    source = Path(cfg["prepared_root"])
    dest = Path(dest) if Path(dest).is_absolute() else PROJECT / dest
    config_out = Path(config_out) if Path(config_out).is_absolute() else PROJECT / config_out
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite smoke fixture: {dest}")
    roles = {role: json.loads((source / f"{role}.json").read_text(encoding="utf-8"))
             for role in ("anchor", "calibration_fit", "calibration_gate", "weak", "validation")}
    selected = {
        "anchor": head(roles["anchor"], anchor),
        "calibration_fit": head(roles["calibration_fit"], fit),
        "calibration_gate": head(roles["calibration_gate"], gate),
        "weak": head(roles["weak"], weak),
        "validation": head(roles["validation"], validation),
    }
    selected["smoke_test"] = disjoint_tail(
        roles["validation"][len(selected["validation"]):], evaluation,
        image_set(selected["validation"]))
    sets = {role: image_set(rows) for role, rows in selected.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f"Cross-role image overlap in smoke fixture: {a}/{b}")
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    for role, rows in selected.items():
        path = dest / f"{role}.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        files[path.name] = digest(path)
    budget = {role: {"expressions": len(selected[role]),
                     "unique_labeled_images": len(image_set(selected[role]))}
              for role in ("anchor", "calibration_fit", "calibration_gate")}
    receipt = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    receipt.update({
        "status": "SMOKE_ONLY",
        "counts": {role: len(rows) for role, rows in selected.items()},
        "annotation_budget": budget, "image_overlap_across_roles": 0,
        "purged_train_expressions": None,
        "files": files,
        "smoke_source_manifest_sha256": digest(source / "manifest.json"),
        "warning": "SMOKE_ONLY subset; never cite as an experiment result; formal test was not copied.",
    })
    write_json(dest / "manifest.json", receipt)

    smoke_cfg = dict(cfg)
    smoke_cfg.update({
        "prepared_root": str(dest), "output_root": str(PROJECT / "methods" / "frt_abl" / "runs-smoke-v10"),
        "anchor_count": len(selected["anchor"]),
        "calibration_fit_count": len(selected["calibration_fit"]),
        "calibration_gate_count": len(selected["calibration_gate"]),
        "rounds": 1, "warmup_epochs": 1, "batch_size": min(32, cfg["batch_size"]),
        "retry_per_round": min(8, cfg["retry_per_round"]), "seed": 0,
    })
    if cfg["task"] == "hwf":
        smoke_cfg["validation_count"] = len(selected["validation"])
    config_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_out, smoke_cfg)
    loaded = load_config(config_out)
    verify_prepared(loaded)
    return receipt, smoke_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Source experiment config")
    parser.add_argument("--dest", required=True, help="Prepared smoke directory")
    parser.add_argument("--config-out", required=True, help="Smoke config JSON to write")
    parser.add_argument("--weak", type=int, default=256)
    parser.add_argument("--anchor", type=int, default=32)
    parser.add_argument("--fit", type=int, default=64)
    parser.add_argument("--gate", type=int, default=64)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--eval", dest="evaluation", type=int, default=128)
    args = parser.parse_args()
    receipt, smoke_cfg = build(args.config, args.dest, args.config_out, args.weak,
                               args.anchor, args.fit, args.gate, args.validation,
                               args.evaluation)
    print(json.dumps({"status": "SMOKE_FIXTURE_READY",
                      "source_data_root": receipt.get("data_root"),
                      "smoke_prepared_root": smoke_cfg["prepared_root"],
                      "smoke_config": args.config_out, "counts": receipt["counts"],
                      "formal_test_copied": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
