"""Build HWF length/operator-combination generalization prepared roots.

This module is data-preparation only: it never imports torch and never starts a
model, an optimizer or an experiment.  The existing V10 ``full`` algorithm and
its label budget are reused unchanged; only the eligible expression pool and
the held-out operator combinations differ.

Two axes are supported:

* **length**: train only on expressions whose token length is in
  ``train_lengths`` (normally 1/3/5) and evaluate on ``eval_lengths`` (normally
  1/3/5/7).  The primary out-of-distribution group is
  ``primary_eval_length`` (normally 7).
* **operator combination** (optional): deterministically withhold a fixed
  number of operator combinations at selected training lengths.  A withheld
  combination never appears in any training role (anchor, calibration, weak or
  validation).  Longer expressions are tagged when one of their contiguous
  operator windows equals a withheld combination, so the effect can also be
  measured inside the length-7 test group.

The produced role manifests keep the existing schema used by ``train.py`` and
``evaluate.py`` and add a ``generalization`` metadata block to every sample.
``train.py`` ignores the extra block; ``evaluate_generalization.py`` uses it for
grouped reporting.
"""
import argparse
import json
from pathlib import Path

from .prepare import (allocate, digest, hwf_source, load_config, stable_key,
                      verify_prepared, write_json)

GENERALIZATION_SCHEMA = 2
DEFAULT_HOLDOUT_SALT = "frt-abl-combo-holdout-v1"
DEFAULT_GROUP_BY = (
    "length",
    "combo",
    "combo_group",
    "contains_holdout_bigram",
    "exact_combo_seen_in_train",
    "length_x_combo_group",
    "length_x_contains_holdout_bigram",
)
ALLOWED_GROUP_BY = set(DEFAULT_GROUP_BY)
OPERATOR_DIRS = {"times": "*", "div": "/"}
OPERATOR_CHARS = ("+", "-", "*", "/")
EXPRESSION_OPERATOR_ALIASES = {"×": "*", "÷": "/", "−": "-", "*": "*", "/": "/",
                               "+": "+", "-": "-"}


def combo_label(combo):
    """Stable, JSON/CSV-safe label for an operator combination."""
    return "".join(combo) if combo else "()"


def operator_combo(images):
    """Extract the ordered operator sequence from HWF image paths.

    HWF stores multiplication/division in ``times``/``div`` directories while
    evaluation maps those directories to ``*``/``/``.  The expression text is
    deliberately not trusted here: grouping must describe the images that the
    model actually sees.
    """
    operators = []
    for image in images:
        head = str(image).replace("\\", "/").split("/")[0]
        if head in OPERATOR_DIRS:
            operators.append(OPERATOR_DIRS[head])
        elif head in OPERATOR_CHARS:
            operators.append(head)
        elif head.isdigit():
            continue
        else:
            raise ValueError(f"Unrecognized HWF image directory {head!r} in {image!r}")
    return tuple(operators)


def expression_operators(expression):
    operators = []
    for character in str(expression):
        if character in EXPRESSION_OPERATOR_ALIASES:
            operators.append(EXPRESSION_OPERATOR_ALIASES[character])
    return tuple(operators)


def validate_generalization_config(cfg):
    """Validate the optional ``generalization`` block before touching data."""
    if cfg.get("task") != "hwf":
        raise ValueError("The length/combo generalization split is defined only for HWF")
    block = cfg.get("generalization")
    if not isinstance(block, dict):
        raise ValueError("Missing generalization configuration block")
    name = str(block.get("name", "")).strip()
    if not name:
        raise ValueError("generalization.name must be a non-empty string")

    def odd_lengths(values, field):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"generalization.{field} must be a non-empty list")
        parsed = [int(value) for value in values]
        if any(value < 1 or value % 2 == 0 for value in parsed):
            raise ValueError(f"generalization.{field} must contain odd positive lengths")
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"generalization.{field} must be unique")
        return sorted(parsed)

    train_lengths = odd_lengths(block.get("train_lengths"), "train_lengths")
    eval_lengths = odd_lengths(block.get("eval_lengths"), "eval_lengths")
    validation_lengths = odd_lengths(block.get("validation_lengths", train_lengths),
                                     "validation_lengths")
    if not set(validation_lengths) <= set(train_lengths):
        raise ValueError("generalization.validation_lengths must be a subset of train_lengths")
    if set(validation_lengths) != set(train_lengths):
        # The shared image-disjoint allocator draws validation from the same
        # eligible pool as training; a partial validation length list would be
        # silently ignored, so reject it until a dedicated OOD validation role
        # is implemented.
        raise ValueError("generalization.validation_lengths must equal train_lengths")
    primary = int(block.get("primary_eval_length", 0))
    if primary not in eval_lengths:
        raise ValueError("generalization.primary_eval_length must be in eval_lengths")
    if not set(eval_lengths) - set(train_lengths):
        raise ValueError("At least one eval length must be unseen during training")
    if not set(eval_lengths) & set(train_lengths):
        raise ValueError("eval_lengths must include at least one in-distribution length")
    if not isinstance(cfg.get("validation_count"), int) or cfg["validation_count"] < 1:
        raise ValueError("validation_count must be a positive integer")

    group_by = block.get("group_by", list(DEFAULT_GROUP_BY))
    if not isinstance(group_by, (list, tuple)) or not group_by:
        raise ValueError("generalization.group_by must be a non-empty list")
    if len(set(group_by)) != len(group_by):
        raise ValueError("generalization.group_by must not contain duplicates")
    unknown = [key for key in group_by if key not in ALLOWED_GROUP_BY]
    if unknown:
        raise ValueError(f"Unknown generalization.group_by keys: {unknown}")

    holdout = block.get("combo_holdout", {"enabled": False})
    if not isinstance(holdout, dict):
        raise ValueError("generalization.combo_holdout must be an object")
    if holdout.get("enabled", False):
        lengths = odd_lengths(holdout.get("lengths"), "combo_holdout.lengths")
        if not set(lengths) <= set(train_lengths):
            raise ValueError("Holdout lengths must be a subset of train_lengths")
        if any(length < 3 for length in lengths):
            raise ValueError("Operator-combination holdout requires lengths >= 3")
        counts = holdout.get("counts")
        if not isinstance(counts, dict):
            raise ValueError("generalization.combo_holdout.counts must be an object")
        for length in lengths:
            value = counts.get(str(length))
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"combo_holdout.counts[{length}] must be a positive integer")
        salt = str(holdout.get("salt", DEFAULT_HOLDOUT_SALT)).strip()
        if not salt:
            raise ValueError("generalization.combo_holdout.salt must be non-empty")
    return {"train_lengths": train_lengths, "eval_lengths": eval_lengths,
            "validation_lengths": validation_lengths, "primary_eval_length": primary,
            "group_by": list(group_by), "name": name}


def _count_for(counts, length):
    if str(length) in counts:
        return int(counts[str(length)])
    if length in counts:
        return int(counts[length])
    raise ValueError(f"Missing combo_holdout.counts entry for length {length}")


def _normalise_holdout(holdout, all_train_rows):
    """Resolve the configured holdout counts into concrete operator combinations."""
    block = dict(holdout or {})
    enabled = bool(block.get("enabled", False))
    lengths = sorted({int(value) for value in block.get("lengths", [5])}) if enabled else []
    counts = {str(key): int(value) for key, value in (block.get("counts") or {"5": 4}).items()}
    salt = str(block.get("salt", DEFAULT_HOLDOUT_SALT))
    heldout_by_length = {}
    heldout_units = []
    if enabled:
        for length in lengths:
            combos = sorted({operator_combo(row["images"]) for row in all_train_rows
                             if len(row["images"]) == length})
            if not combos:
                raise ValueError(f"No training expressions with length {length} for holdout")
            count = _count_for(counts, length)
            if count >= len(combos):
                raise ValueError(f"Cannot hold out {count} of {len(combos)} length-{length} combos")
            ordered = sorted(combos, key=lambda combo: (stable_key("|".join(combo), salt), combo))
            chosen = ordered[:count]
            heldout_by_length[str(length)] = [list(combo) for combo in chosen]
            heldout_units.extend(chosen)
    # Each withheld combination is matched as a contiguous operator window in a
    # test expression.  Length-5 holdouts therefore match length-7 expressions
    # as well; the comparison stays at the operator-transition level.
    units = sorted({(len(unit), tuple(unit)) for unit in heldout_units})
    return {
        "enabled": enabled,
        "salt": salt,
        "lengths": lengths,
        "counts": counts if enabled else {},
        "heldout_by_length": heldout_by_length,
        "heldout_units": [{"width": width, "combo": list(combo), "label": combo_label(combo)}
                          for width, combo in units],
    }


def _contains_withheld(sample, holdout):
    """True when the expression exposes any withheld operator window."""
    if not holdout["enabled"]:
        return False
    return bool(_holdout_hits(operator_combo(sample["images"]), holdout))


def _holdout_hits(combo, holdout):
    if not holdout["enabled"]:
        return []
    hits = []
    for unit in holdout["heldout_units"]:
        width = unit["width"]
        target = tuple(unit["combo"])
        for start in range(max(0, len(combo) - width + 1)):
            if combo[start:start + width] == target:
                hits.append(unit["label"])
    return sorted(set(hits))


def sample_metadata(sample, train_combos, holdout):
    """Per-sample grouping metadata; safe to persist in every prepared role."""
    combo = operator_combo(sample["images"])
    length = len(sample["images"])
    hits = _holdout_hits(combo, holdout)
    exact_seen = combo in train_combos
    combo_group = "seen" if exact_seen and not hits else "unseen"
    return {
        "length": length,
        "combo": combo_label(combo),
        "combo_tuple": list(combo),
        "exact_combo_seen_in_train": exact_seen,
        "contains_holdout_bigram": bool(hits),
        "holdout_bigram_hits": hits,
        "combo_group": combo_group,
    }


def group_key(metadata, dimension):
    """Return the report value for one sample on one grouping dimension."""
    if dimension == "length":
        return str(metadata["length"])
    if dimension == "combo":
        return metadata["combo"]
    if dimension == "combo_group":
        return metadata["combo_group"]
    if dimension == "contains_holdout_bigram":
        return str(metadata["contains_holdout_bigram"])
    if dimension == "exact_combo_seen_in_train":
        return str(metadata["exact_combo_seen_in_train"])
    if dimension == "length_x_combo_group":
        return f'{metadata["length"]}|{metadata["combo_group"]}'
    if dimension == "length_x_contains_holdout_bigram":
        return f'{metadata["length"]}|{metadata["contains_holdout_bigram"]}'
    raise ValueError(f"Unknown group dimension: {dimension}")


def group_counts(rows, dimensions):
    counts = {dimension: {} for dimension in dimensions}
    for row in rows:
        metadata = row["generalization"]
        for dimension in dimensions:
            key = group_key(metadata, dimension)
            counts[dimension][key] = counts[dimension].get(key, 0) + 1
    return {dimension: dict(sorted(values.items())) for dimension, values in counts.items()}


def _verify_raw_rows(raw_rows, samples, split):
    """Check that expression text and image paths describe the same operators."""
    if len(raw_rows) != len(samples):
        raise ValueError(f"Raw {split} rows and parsed samples differ in count")
    by_id = {sample["id"]: sample for sample in samples}
    for row in raw_rows:
        sample = by_id.get(row["id"])
        if sample is None:
            raise ValueError(f"Raw {split} row not present in parsed samples: {row['id']}")
        from_text = expression_operators(row.get("expr", ""))
        from_images = operator_combo(sample["images"])
        if from_text != from_images:
            raise ValueError(
                f"{split} expression/image operator mismatch for {row['id']}: "
                f"expr={from_text!r}, images={from_images!r}")


def prepare_generalization(cfg, dry_run=False):
    """Build (or, with ``dry_run``, only plan) a generalization prepared root."""
    resolved = validate_generalization_config(cfg)
    root = Path(cfg["data_root"])
    samples, truth, source_hashes = hwf_source(cfg)

    raw_train = json.loads((root / "expr_train.json").read_text(encoding="utf-8"))
    raw_test = json.loads((root / "expr_test.json").read_text(encoding="utf-8"))
    _verify_raw_rows(raw_train, samples["train"], "train")
    _verify_raw_rows(raw_test, samples["test"], "test")

    train_lengths = set(resolved["train_lengths"])
    eval_lengths = set(resolved["eval_lengths"])
    holdout = _normalise_holdout(cfg.get("generalization", {}).get("combo_holdout", {}),
                                 samples["train"])

    by_length = [sample for sample in samples["train"]
                 if len(sample["images"]) in train_lengths]
    heldout_filtered = [sample for sample in by_length
                        if not _contains_withheld(sample, holdout)]
    excluded = {
        "not_in_train_lengths": len(samples["train"]) - len(by_length),
        "heldout_operator_combo": len(by_length) - len(heldout_filtered),
    }
    if not heldout_filtered:
        raise ValueError("No training expressions remain after length/combo filtering")

    test_rows = [sample for sample in samples["test"] if len(sample["images"]) in eval_lengths]
    if not test_rows:
        raise ValueError("No test expressions match eval_lengths")
    primary = resolved["primary_eval_length"]
    primary_count = sum(len(sample["images"]) == primary for sample in test_rows)
    if primary_count < 1:
        raise ValueError(f"No test expressions of primary length {primary}")

    train_images = {image for sample in heldout_filtered for image in sample["images"]}
    test_images = {image for sample in test_rows for image in sample["images"]}
    overlap = train_images & test_images
    if overlap:
        raise ValueError(f"Train/test image leakage before allocation: {len(overlap)} images")

    roles, purged = allocate(cfg, {"train": heldout_filtered, "validation": [],
                                   "test": test_rows})
    labelled = {"anchor", "calibration_fit", "calibration_gate"}
    for role, rows in roles.items():
        updated = []
        for sample in rows:
            item = dict(sample)
            if role in labelled:
                item["labels"] = truth[sample["id"]]
            updated.append(item)
        roles[role] = updated

    training_roles = ("weak", "anchor", "calibration_fit", "calibration_gate", "validation")
    train_combos = {operator_combo(sample["images"])
                    for role in training_roles for sample in roles[role]}
    for role, rows in roles.items():
        roles[role] = [dict(sample, generalization=sample_metadata(sample, train_combos, holdout))
                       for sample in rows]

    dimensions = resolved["group_by"]
    counts = {role: len(rows) for role, rows in roles.items()}
    budget = {role: {"expressions": len(roles[role]),
                     "unique_labeled_images": len({image for sample in roles[role]
                                                   for image in sample["images"]})}
              for role in labelled}
    test_counts = group_counts(roles["test"], dimensions)
    train_counts = group_counts([sample for role in training_roles for sample in roles[role]],
                                dimensions)
    receipt = {
        "schema_version": GENERALIZATION_SCHEMA,
        "status": "PREPARED_NOT_TRAINED",
        "track": cfg["track"],
        "task": cfg["task"],
        "symbols": cfg["symbols"],
        "data_root": cfg["data_root"],
        "source_hashes": source_hashes,
        "generalization": {
            "name": resolved["name"],
            "train_lengths": resolved["train_lengths"],
            "eval_lengths": resolved["eval_lengths"],
            "validation_lengths": resolved["validation_lengths"],
            "primary_eval_length": primary,
            "group_by": dimensions,
            "combo_holdout": holdout,
            "train_combos": sorted(combo_label(combo) for combo in train_combos),
            "train_group_counts": train_counts,
            "test_group_counts": test_counts,
            "primary_test_count": primary_count,
            "group_definitions": {
                "length": "number of perceived symbol tokens in the expression",
                "combo": "ordered operator sequence; '()' for a bare digit",
                "combo_group": ("seen = exact combination present in training roles and no "
                                "withheld operator window; unseen otherwise"),
                "contains_holdout_bigram": ("True when any contiguous operator window is a "
                                            "withheld combination"),
                "exact_combo_seen_in_train": "exact operator sequence present in training roles",
                "length_x_combo_group": "length and combo_group cross-tabulation",
                "length_x_contains_holdout_bigram": "length and holdout-window cross-tabulation",
            },
        },
        "counts": counts,
        "annotation_budget": budget,
        "image_overlap_across_roles": 0,
        "purged_train_expressions": purged,
        "excluded_train_expressions": excluded,
        "source_counts": {"expr_train": len(samples["train"]),
                          "expr_test": len(samples["test"])},
        "files": {},
        "warning": ("New HWF length/combo generalization split; not comparable with native HWF "
                    "or with the V10 full-trained-on-all-lengths split."),
        "preprocessing": "uint8/255; grayscale; HWF resized 28x28; no identity-changing augmentation",
    }

    if dry_run:
        receipt["status"] = "DRY_RUN_ONLY"
        return receipt

    dest = Path(cfg["prepared_root"])
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared manifests: {dest}")
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    for role, rows in roles.items():
        path = dest / f"{role}.json"
        write_json(path, rows)
        files[path.name] = digest(path)
    receipt["files"] = files
    write_json(dest / "manifest.json", receipt)
    return receipt


def verify_generalization_prepared(cfg):
    """Verify the prepared split *and* that it matches the config's split definition."""
    receipt = verify_prepared(cfg)
    block = cfg.get("generalization")
    if not block:
        return receipt
    saved = receipt.get("generalization")
    if not isinstance(saved, dict):
        raise ValueError("Generalization config points at a prepared root without generalization metadata")
    expected = validate_generalization_config(cfg)
    for key in ("name", "train_lengths", "eval_lengths", "validation_lengths",
                "primary_eval_length"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"Generalization prepared metadata mismatch: {key}")
    requested = block.get("combo_holdout", {"enabled": False})
    actual = saved.get("combo_holdout", {})
    if bool(requested.get("enabled", False)) != bool(actual.get("enabled", False)):
        raise ValueError("Generalization prepared metadata mismatch: combo_holdout.enabled")
    if actual.get("enabled", False):
        if sorted(int(value) for value in requested.get("lengths", [])) != sorted(
                int(value) for value in actual.get("lengths", [])):
            raise ValueError("Generalization prepared metadata mismatch: combo_holdout.lengths")
        counts = {str(key): int(value) for key, value in (requested.get("counts") or {}).items()}
        if counts != {str(key): int(value) for key, value in actual.get("counts", {}).items()}:
            raise ValueError("Generalization prepared metadata mismatch: combo_holdout.counts")
        salt = str(requested.get("salt", DEFAULT_HOLDOUT_SALT))
        if salt != actual.get("salt"):
            raise ValueError("Generalization prepared metadata mismatch: combo_holdout.salt")
        # Same raw-data hash, salt, lengths and counts give the same
        # deterministic held-out combination choice; no further resolution is
        # needed here.
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and print counts without writing prepared manifests")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = prepare_generalization(cfg, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
