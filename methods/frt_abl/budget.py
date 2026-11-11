"""Annotation-budget experiment: image-level label budget and role preparation.

The V10 method is held fixed. Only the total number of annotated *images* in the
anchor + calibration roles changes (default 1%, 3%, 5%, 10% of the training
images). Validation and test are evaluation roles and are never counted in the
annotation budget; the formal test manifest is never read here.

This module is inert on import. It writes files only when `budget prepare
--execute` is passed explicitly. Reusing the same deterministic role ordering as
`prepare.py` makes the 3% MNIST budget reproduce the frozen V10 prepared roles
(270/270/270 expressions) exactly, which is useful as a regression check.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from .prepare import (PROJECT, digest, hwf_source, load_config, mnist_source,
                      read_csv, stable_key, verify_prepared, write_json)

EXPERIMENT = "annotation-budget-v1"
BUDGET_VERSION = "frt-abl-budget-v1-20261029"
DEFAULT_LEVELS = (0.01, 0.03, 0.05, 0.10)
BUDGET_ROLES = ("anchor", "calibration_fit", "calibration_gate")
ROLE_ORDER = ("calibration_gate", "calibration_fit", "anchor")
COUNT_FIELD = {"anchor": "anchor_count",
               "calibration_fit": "calibration_fit_count",
               "calibration_gate": "calibration_gate_count"}
ROLE_SHARES = {"mnist-add2": {"anchor": 1, "calibration_fit": 1, "calibration_gate": 1},
               "hwf": {"anchor": 1, "calibration_fit": 3, "calibration_gate": 3}}
DEFAULT_CONFIGS = tuple(str(PROJECT / "methods" / "frt_abl" / "configs" / name)
                        for name in ("mnist.json", "hwf.json"))
DEFAULT_PREPARED_ROOT = PROJECT / "methods" / "frt_abl" / "prepared" / "budget-v1"
DEFAULT_CONFIG_OUT = PROJECT / "methods" / "frt_abl" / "configs" / "budget"
DEFAULT_OUTPUT_ROOT = PROJECT / "methods" / "frt_abl" / "runs-budget-v1"


def _absolute(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def project_relative(path):
    """Store project-internal paths relative to the project root."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT))
    except ValueError:
        return str(resolved)


def level_label(level):
    if not 0 < float(level) < 1:
        raise ValueError(f"Budget level must lie in (0,1): {level}")
    text = f"{float(level) * 100:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") + "pct"


def split_integer_budget(total, weights):
    """Split an integer image budget by weights with largest remainders.

    Every role receives at least one image so that no role becomes empty at the
    smallest budget. The returned values always sum to `total`.
    """
    total = int(total)
    keys = list(weights)
    if len(keys) < 1 or any(weights[k] <= 0 for k in keys):
        raise ValueError("Budget weights must be positive")
    if total < len(keys):
        raise ValueError(f"Budget {total} is too small for {len(keys)} roles")
    denominator = sum(weights.values())
    raw = {k: total * weights[k] / denominator for k in keys}
    result = {k: max(1, math.floor(raw[k])) for k in keys}
    difference = total - sum(result.values())
    if difference > 0:
        order = sorted(keys, key=lambda k: (-(raw[k] - math.floor(raw[k])), k))
        if difference > len(order):
            raise AssertionError("Largest-remainder split exceeded one pass")
        for key in order[:difference]:
            result[key] += 1
    elif difference < 0:
        while difference < 0:
            candidates = [k for k in keys if result[k] > 1]
            if not candidates:
                raise ValueError("Cannot reduce every role to a positive budget")
            key = min(candidates, key=lambda k: (raw[k] - math.floor(raw[k]), k))
            result[key] -= 1
            difference += 1
    if sum(result.values()) != total:
        raise AssertionError("Budget split does not conserve the total")
    return result


def universe_from_samples(samples):
    images = {image for sample in samples["train"] for image in sample["images"]}
    return {"expressions": len(samples["train"]), "unique_images": len(images)}


def load_train_universe(cfg):
    """Read-only count of the training image universe used by the budget.

    The denominator is unique training images, not expressions: HWF reuses the
    same symbol images across expressions, and the annotation receipt is
    image-based.
    """
    root = _absolute(cfg["data_root"])
    if cfg["task"] == "mnist-add2":
        pairs = _absolute(cfg["common_manifest"]) / "pairs_train.csv"
        rows = read_csv(pairs)
        images = {image for row in rows for image in (row["left_id"], row["right_id"])}
        return {"task": cfg["task"], "expressions": len(rows), "unique_images": len(images),
                "source": "pairs_train.csv", "sha256": digest(pairs)}
    path = root / "expr_train.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    images = {image for row in rows for image in row["img_paths"]}
    return {"task": cfg["task"], "expressions": len(rows), "unique_images": len(images),
            "source": "expr_train.json", "sha256": digest(path)}


def budget_plan(cfg, level, universe):
    if cfg.get("task") not in ROLE_SHARES:
        raise ValueError(f"Unknown task: {cfg.get('task')}")
    total_target = int(round(float(level) * universe["unique_images"]))
    if total_target < len(BUDGET_ROLES):
        raise ValueError("The requested budget is smaller than the three roles")
    targets = split_integer_budget(total_target, ROLE_SHARES[cfg["task"]])
    return {"level": float(level), "level_label": level_label(level),
            "train_image_universe": universe["unique_images"],
            "train_expressions": universe["expressions"],
            "target_total_unique_labeled_images": total_target,
            "target_role_unique_images": targets}


def _select_role(ordered, amount, assigned, reserved, mode):
    """Select expressions in the frozen role order.

    `mode="count"` is used only for the fixed HWF validation role; budget roles
    stop once the requested number of *new unique images* is reached. Expressions
    whose images intersect a reserved role are purged before selection.
    """
    chosen = []
    images = set()
    for sample in ordered:
        if sample["id"] in assigned or reserved.intersection(sample["images"]):
            continue
        chosen.append(sample)
        images.update(sample["images"])
        if mode == "count" and len(chosen) >= amount:
            break
        if mode == "images" and len(images) >= amount:
            break
    return chosen, images


def allocate_budget_roles(cfg, samples, targets):
    """Deterministic, image-disjoint roles for one annotation budget.

    HWF reserves the fixed validation role first, then calibration_gate,
    calibration_fit and anchor, exactly following the frozen V10 allocation
    order. A larger budget extends the earlier roles; all roles remain image
    disjoint and no test/validation image is used for annotation.
    """
    if set(targets) != set(BUDGET_ROLES):
        raise ValueError("Budget targets must cover exactly the three labeled roles")
    if any(not isinstance(value, int) or value < 1 for value in targets.values()):
        raise ValueError("Budget targets must be positive integers")
    ordered = sorted(samples["train"], key=lambda s: stable_key(s["id"], "roles-20261026"))
    reserved, assigned, roles = set(), set(), {}
    if cfg["task"] == "hwf":
        chosen, images = _select_role(ordered, int(cfg["validation_count"]), assigned,
                                      reserved, "count")
        if len(chosen) != int(cfg["validation_count"]):
            raise ValueError(f"Cannot create image-disjoint validation: "
                             f"{len(chosen)} < {cfg['validation_count']}")
        roles["validation"] = chosen
        assigned.update(s["id"] for s in chosen)
        reserved.update(images)
    for role in ROLE_ORDER:
        chosen, images = _select_role(ordered, targets[role], assigned, reserved, "images")
        if not chosen:
            raise ValueError(f"Cannot allocate a nonempty {role} role")
        if len(images) < targets[role]:
            raise ValueError(f"Cannot reach {targets[role]} disjoint labeled images for "
                             f"{role}: only {len(images)} available")
        roles[role] = chosen
        assigned.update(s["id"] for s in chosen)
        reserved.update(images)
    roles["weak"] = [s for s in samples["train"]
                     if s["id"] not in assigned and not reserved.intersection(s["images"])]
    if cfg["task"] == "mnist-add2":
        roles["validation"] = samples["validation"]
    roles["test"] = samples["test"]
    if not roles["weak"]:
        raise ValueError("No weak data remains after image-leakage prevention")
    image_sets = {role: {i for s in rows for i in s["images"]} for role, rows in roles.items()}
    names = list(image_sets)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if image_sets[first] & image_sets[second]:
                raise ValueError(f"Cross-role image leakage: {first}/{second}")
    purged = len(samples["train"]) - len(assigned) - len(roles["weak"])
    return roles, purged


def role_budget_record(roles):
    return {role: {"expressions": len(roles[role]),
                   "unique_labeled_images": len({i for s in roles[role] for i in s["images"]})}
            for role in BUDGET_ROLES}


def budget_record(cfg, plan, actual_roles, actual_total):
    fraction = actual_total / plan["train_image_universe"]
    return {
        "experiment": EXPERIMENT,
        "version": BUDGET_VERSION,
        "level": plan["level"],
        "level_label": plan["level_label"],
        "train_image_universe": plan["train_image_universe"],
        "train_expressions": plan["train_expressions"],
        "target_total_unique_labeled_images": plan["target_total_unique_labeled_images"],
        "actual_total_unique_labeled_images": actual_total,
        "actual_fraction": fraction,
        "role_share": dict(ROLE_SHARES[cfg["task"]]),
        "target_role_unique_images": dict(plan["target_role_unique_images"]),
        "actual_role_unique_images": {role: actual_roles[role]["unique_labeled_images"]
                                      for role in BUDGET_ROLES},
        "role_expressions": {role: actual_roles[role]["expressions"]
                             for role in BUDGET_ROLES},
        "validation_role": "held out for evaluation; not counted in the annotation budget",
        "selection_order": list(ROLE_ORDER),
        "selection_salt": "roles-20261026",
    }


def prepare_one(base_cfg, level, prepared_root, config_path, output_root, overwrite=False):
    """Materialize one task × budget prepared directory and its training config."""
    cfg = copy.deepcopy(base_cfg)
    if cfg["task"] == "mnist-add2":
        samples, truth, source_hashes = mnist_source(cfg)
    else:
        samples, truth, source_hashes = hwf_source(cfg)
    universe = universe_from_samples(samples)
    plan = budget_plan(cfg, level, universe)
    roles, purged = allocate_budget_roles(cfg, samples, plan["target_role_unique_images"])
    actual_roles = role_budget_record(roles)
    actual_total = sum(actual_roles[role]["unique_labeled_images"] for role in BUDGET_ROLES)
    record = budget_record(cfg, plan, actual_roles, actual_total)

    prepared_root, config_path = Path(prepared_root), Path(config_path)
    if prepared_root.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite prepared manifests: {prepared_root}")
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite budget config: {config_path}")
    prepared_root.mkdir(parents=True, exist_ok=False)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for role, rows in roles.items():
        rows = [dict(sample, labels=truth[sample["id"]])
                if role in BUDGET_ROLES else sample for sample in rows]
        path = prepared_root / f"{role}.json"
        write_json(path, rows)
        files[path.name] = digest(path)
    receipt = {
        "schema_version": 2,
        "status": "PREPARED_NOT_TRAINED",
        "track": cfg["track"],
        "experiment": EXPERIMENT,
        "experiment_version": BUDGET_VERSION,
        "budget": record,
        "task": cfg["task"],
        "symbols": cfg["symbols"],
        "data_root": cfg["data_root"],
        "source_hashes": source_hashes,
        "counts": {role: len(rows) for role, rows in roles.items()},
        "annotation_budget": actual_roles,
        "image_overlap_across_roles": 0,
        "purged_train_expressions": purged,
        "files": files,
        "warning": ("Image-level annotation budget on a new label-budget track; "
                    "not directly comparable with COMMON result-only baselines."),
        "preprocessing": ("uint8/255; grayscale; HWF resized 28x28; "
                          "no identity-changing augmentation"),
    }
    write_json(prepared_root / "manifest.json", receipt)

    budget_cfg = build_budget_config(base_cfg, record, prepared_root, output_root)
    write_json(config_path, budget_cfg)
    return {"task": cfg["task"], "level": record["level"],
            "level_label": record["level_label"],
            "prepared_root": project_relative(prepared_root),
            "config_path": project_relative(config_path),
            "counts": receipt["counts"], "annotation_budget": actual_roles,
            "target_role_unique_images": plan["target_role_unique_images"],
            "actual_total_unique_labeled_images": actual_total,
            "actual_fraction": record["actual_fraction"],
            "weak_expressions": len(roles["weak"]),
            "purged_train_expressions": purged}


def build_budget_config(base_cfg, record, prepared_root, output_root):
    cfg = copy.deepcopy(base_cfg)
    cfg["prepared_root"] = project_relative(prepared_root)
    cfg["output_root"] = project_relative(output_root)
    for role in BUDGET_ROLES:
        cfg[COUNT_FIELD[role]] = record["role_expressions"][role]
    cfg["experiment"] = EXPERIMENT
    cfg["experiment_version"] = BUDGET_VERSION
    cfg["run_tag"] = f"budget-{record['level_label']}"
    cfg["budget"] = record
    cfg["base_prepared_root"] = str(base_cfg.get("prepared_root"))
    return cfg


def validate_budget_config(cfg):
    """Fail closed when a generated budget config and its metadata disagree."""
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("Not an annotation-budget config")
    if cfg.get("experiment_version") != BUDGET_VERSION:
        raise ValueError("Annotation-budget config version mismatch")
    record = cfg.get("budget")
    if not isinstance(record, dict):
        raise ValueError("Missing budget metadata")
    required = ("experiment", "version", "level", "level_label", "train_image_universe",
                "target_total_unique_labeled_images", "actual_total_unique_labeled_images",
                "actual_fraction", "role_share", "target_role_unique_images",
                "actual_role_unique_images", "role_expressions")
    for key in required:
        if key not in record:
            raise ValueError(f"Missing budget field: {key}")
    if record["experiment"] != EXPERIMENT or record["version"] != BUDGET_VERSION:
        raise ValueError("Budget metadata identity mismatch")
    if not 0 < float(record["level"]) < 1:
        raise ValueError("Budget level must lie in (0,1)")
    if record["level_label"] != level_label(record["level"]):
        raise ValueError("Budget level label mismatch")
    if record["role_share"] != ROLE_SHARES[cfg["task"]]:
        raise ValueError("Budget role shares do not match the frozen role design")
    for field in ("target_role_unique_images", "actual_role_unique_images", "role_expressions"):
        if set(record[field]) != set(BUDGET_ROLES):
            raise ValueError(f"{field} must cover exactly the three labeled roles")
    for role in BUDGET_ROLES:
        expressions = record["role_expressions"][role]
        images = record["actual_role_unique_images"][role]
        target = record["target_role_unique_images"][role]
        if expressions != cfg[COUNT_FIELD[role]]:
            raise ValueError(f"Config count differs from budget metadata: {role}")
        if not isinstance(expressions, int) or expressions < 1:
            raise ValueError(f"Budget role must be nonempty: {role}")
        if not isinstance(images, int) or images < expressions:
            raise ValueError(f"Unique labeled images cannot be below expressions: {role}")
        if not isinstance(target, int) or target < 1:
            raise ValueError(f"Invalid target image budget: {role}")
        if images < target:
            raise ValueError(f"Actual image budget below target: {role}")
    total = record["actual_total_unique_labeled_images"]
    if total != sum(record["actual_role_unique_images"].values()):
        raise ValueError("Actual total budget does not equal the role sum")
    if total < record["target_total_unique_labeled_images"]:
        raise ValueError("Actual budget below the requested total")
    universe = record["train_image_universe"]
    if not isinstance(universe, int) or universe < total:
        raise ValueError("Training image universe must cover the actual budget")
    if not math.isclose(record["actual_fraction"], total / universe, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Actual budget fraction is inconsistent")


def verify_budget_prepared(cfg, receipt=None):
    """Verify that the prepared receipt belongs to this exact budget config."""
    if receipt is None:
        root = _absolute(cfg["prepared_root"])
        receipt = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if receipt.get("experiment") != EXPERIMENT:
        raise ValueError("Prepared manifest is not an annotation-budget manifest")
    if receipt.get("experiment_version") != BUDGET_VERSION:
        raise ValueError("Prepared budget manifest version mismatch")
    record = cfg["budget"]
    saved = receipt.get("budget") or {}
    for key in ("experiment", "version", "level_label", "train_image_universe",
                "train_expressions", "role_share",
                "target_total_unique_labeled_images", "actual_total_unique_labeled_images",
                "target_role_unique_images", "actual_role_unique_images", "role_expressions"):
        if saved.get(key) != record.get(key):
            raise ValueError(f"Prepared budget metadata differs from config: {key}")
    if not math.isclose(float(saved.get("level", -1)), float(record["level"]),
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("Prepared budget level differs from config")
    if not math.isclose(float(saved.get("actual_fraction", -1)), float(record["actual_fraction"]),
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("Prepared actual budget fraction differs from config")
    for role in BUDGET_ROLES:
        if receipt["counts"][role] != cfg[COUNT_FIELD[role]]:
            raise ValueError(f"Prepared role count differs from config: {role}")
    return receipt


def v10_reference(base_cfg, universe):
    """Record the frozen V10 budget as a reference point when it is available."""
    root = _absolute(base_cfg["prepared_root"])
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    receipt = json.loads(manifest.read_text(encoding="utf-8"))
    annotation = receipt.get("annotation_budget") or {}
    if set(annotation) != set(BUDGET_ROLES):
        return None
    total = sum(annotation[role]["unique_labeled_images"] for role in BUDGET_ROLES)
    return {"prepared_root": project_relative(root), "labeled_images": total,
            "fraction": total / universe["unique_images"],
            "annotation_budget": annotation}


def _reset(path, expected_parent):
    """Delete one generated artifact, refusing to touch an unexpected path."""
    path, parent = Path(path).resolve(), Path(expected_parent).resolve()
    if path.parent != parent:
        raise ValueError(f"Refusing to reset a path outside {parent}: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _source(cfg):
    return mnist_source(cfg) if cfg["task"] == "mnist-add2" else hwf_source(cfg)


def _simulate(base_cfg, level):
    samples, _truth, _hashes = _source(base_cfg)
    universe = universe_from_samples(samples)
    plan = budget_plan(base_cfg, level, universe)
    roles, purged = allocate_budget_roles(base_cfg, samples, plan["target_role_unique_images"])
    actual = role_budget_record(roles)
    total = sum(actual[role]["unique_labeled_images"] for role in BUDGET_ROLES)
    return {"level": plan["level"], "level_label": plan["level_label"],
            "train_image_universe": universe["unique_images"],
            "target_total_unique_labeled_images": plan["target_total_unique_labeled_images"],
            "target_role_unique_images": plan["target_role_unique_images"],
            "actual_total_unique_labeled_images": total,
            "actual_fraction": total / universe["unique_images"],
            "annotation_budget": actual,
            "weak_expressions": len(roles["weak"]),
            "purged_train_expressions": purged}


def command_prepare(args):
    configs = [load_config(path) for path in args.configs]
    tasks = [cfg["task"] for cfg in configs]
    if len(set(tasks)) != len(tasks):
        raise SystemExit("Each task must appear once in --configs")
    levels = []
    for level in args.levels:
        if not 0 < level < 1:
            raise SystemExit(f"Budget level must lie in (0,1): {level}")
        if level not in levels:
            levels.append(level)
    destination_root = _absolute(args.dest_root).resolve()
    config_out = _absolute(args.config_out).resolve()
    output_root = _absolute(args.output_root).resolve()
    entries = []
    for cfg in configs:
        for level in levels:
            label = level_label(level)
            entries.append({"task": cfg["task"], "level": level, "level_label": label,
                            "config": cfg,
                            "prepared_root": destination_root / f'{cfg["task"]}-{label}',
                            "config_path": config_out / f'{cfg["task"]}-{label}.json'})
    if args.execute:
        collisions = [str(entry["prepared_root"]) for entry in entries
                      if entry["prepared_root"].exists()]
        collisions += [str(entry["config_path"]) for entry in entries
                       if entry["config_path"].exists()]
        if collisions and not args.overwrite:
            raise SystemExit("Refusing to overwrite existing budget artifacts:\n" +
                             "\n".join(collisions))
        if args.overwrite:
            for entry in entries:
                _reset(entry["prepared_root"], destination_root)
                _reset(entry["config_path"], config_out)
    prepared = []
    for entry in entries:
        cfg = entry["config"]
        if args.execute:
            record = prepare_one(cfg, entry["level"], entry["prepared_root"],
                                 entry["config_path"], output_root, overwrite=False)
            print(f'PREPARED {cfg["task"]} {entry["level_label"]} '
                  f'actual={record["actual_total_unique_labeled_images"]} '
                  f'weak={record["weak_expressions"]}', flush=True)
            prepared.append(record)
            continue
        universe = load_train_universe(cfg)
        plan = budget_plan(cfg, entry["level"], universe)
        item = {"task": cfg["task"], "level": plan["level"],
                "level_label": plan["level_label"],
                "prepared_root": project_relative(entry["prepared_root"]),
                "config_path": project_relative(entry["config_path"]),
                "train_image_universe": universe["unique_images"],
                "target_total_unique_labeled_images": plan["target_total_unique_labeled_images"],
                "target_role_unique_images": plan["target_role_unique_images"]}
        if args.check:
            item["allocation"] = _simulate(cfg, entry["level"])
        prepared.append(item)
    if args.execute:
        reference = {}
        for cfg in configs:
            universe = load_train_universe(cfg)
            item = v10_reference(cfg, universe)
            if item:
                reference[cfg["task"]] = item
        manifest = {"schema_version": 1, "status": "PREPARED_NOT_TRAINED",
                    "experiment": EXPERIMENT, "experiment_version": BUDGET_VERSION,
                    "levels": levels, "output_root": project_relative(output_root),
                    "entries": prepared, "v10_reference": reference,
                    "note": ("Validation/test roles are not part of the annotation budget. "
                             "Formal test is never read by preparation.")}
        write_json(destination_root / "manifest.json", manifest)
    print(json.dumps({"status": "PREPARED_NOT_TRAINED" if args.execute else "PLAN_ONLY",
                      "experiment": EXPERIMENT, "n_entries": len(prepared),
                      "levels": levels,
                      "entries": prepared},
                     ensure_ascii=False, indent=2))


def command_verify(args):
    paths = args.configs
    if not paths:
        paths = [str(path) for path in sorted(DEFAULT_CONFIG_OUT.glob("*.json"))]
    if not paths:
        raise SystemExit("No budget configs found; run `budget prepare --execute` first")
    records = []
    for path in paths:
        cfg = load_config(path)
        validate_budget_config(cfg)
        verify_prepared(cfg)
        receipt = verify_budget_prepared(cfg)
        records.append({"config": project_relative(path), "task": cfg["task"],
                        "level": cfg["budget"]["level"],
                        "level_label": cfg["budget"]["level_label"],
                        "counts": receipt["counts"],
                        "annotation_budget": receipt["annotation_budget"]})
    print(json.dumps({"status": "BUDGET_CONFIGS_VERIFIED", "n": len(records),
                      "records": records}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare",
                                    help="Plan or create per-budget prepared manifests and configs")
    prepare.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    prepare.add_argument("--levels", nargs="+", type=float, default=list(DEFAULT_LEVELS))
    prepare.add_argument("--dest-root", default=str(DEFAULT_PREPARED_ROOT))
    prepare.add_argument("--config-out", default=str(DEFAULT_CONFIG_OUT))
    prepare.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    prepare.add_argument("--execute", action="store_true",
                         help="Actually write prepared manifests and budget configs")
    prepare.add_argument("--overwrite", action="store_true",
                         help="Replace only the generated task-level artifacts")
    prepare.add_argument("--check", action="store_true",
                         help="Plan-only mode: also simulate exact allocation counts")
    prepare.set_defaults(handler=command_prepare)
    verify = subparsers.add_parser("verify", help="Verify generated budget configs and manifests")
    verify.add_argument("--configs", nargs="+", default=None)
    verify.set_defaults(handler=command_verify)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
