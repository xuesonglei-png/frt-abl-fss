"""Manifest preparation only: no torch imports, model calls, downloads or experiments."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct
from .policies import VARIANTS, validate_policy_config


PROJECT = Path(__file__).resolve().parents[2]


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if cfg["track"] != "METHOD_LABEL_BUDGET":
        raise ValueError("Label-assisted method must not be registered as COMMON")
    if cfg["task"] not in {"mnist-add2", "hwf"}:
        raise ValueError("Unknown task")
    if cfg["task"] == "mnist-add2" and cfg["symbols"] != list("0123456789"):
        raise ValueError("Addition requires ordered digit labels 0..9")
    if cfg["task"] == "hwf" and (len(cfg["symbols"]) != 13 or set(cfg["symbols"]) != set("123456789+-*/")):
        raise ValueError("HWF requires the thirteen nonzero digits/operators")
    if any(len(pair) != 2 or any(not isinstance(v, int) or not 0 <= v < 10 for v in pair)
           for pair in cfg["missing_facts"]):
        raise ValueError("Invalid missing addition fact")
    if cfg["task"] != "mnist-add2" and cfg["missing_facts"]:
        raise ValueError("Missing-fact protocol is currently defined only for addition")
    if cfg["seed"] not in (0, 1, 2, 3, 4, 5):
        raise ValueError("Registered seeds are 0..5")
    if cfg["variant"] not in VARIANTS:
        raise ValueError("Unknown ablation")
    if not 0 <= cfg["beta"] < cfg["alpha_min"] <= cfg["alpha_start"] <= 1:
        raise ValueError("Invalid thresholds")
    if cfg["batch_size"] < 1 or cfg["rounds"] < 1 or cfg["beam_size"] < 1 or cfg["max_checks"] < 1:
        raise ValueError("Invalid iteration/search budget")
    if cfg["task"] == "mnist-add2" and cfg["max_checks"] < 100:
        # The exhaustive addition search has exactly 100 candidates; a smaller
        # cap marks every search incomplete and silently disables acceptance.
        raise ValueError("MNIST max_checks must cover the 100 exhaustive candidates")
    if cfg["warmup_epochs"] < 1 or cfg["stable_rounds"] < 1:
        raise ValueError("Invalid epoch/ramp/stability setting")
    if cfg["retry_factor"] < 1 or cfg["retry_per_round"] < 0:
        raise ValueError("Invalid retry budget")
    if cfg["threads"] < 1:
        raise ValueError("threads must be positive")
    for key in ("search_workers", "search_parallel_min", "search_chunksize"):
        if key not in cfg:
            raise ValueError(f"Missing search-parallel field: {key}")
    if cfg["search_workers"] < 1 or cfg["search_parallel_min"] < 0 or cfg["search_chunksize"] < 1:
        raise ValueError("Invalid search-parallel setting")
    if cfg["graph_chunk_size"] < 1:
        raise ValueError("graph_chunk_size must be positive")
    if cfg["calibration_min_count"] < 1 or not 0 < cfg["risk_limit"] < 1:
        raise ValueError("Invalid calibration gate")
    for key in ("target_center", "target_scale", "target_share", "alpha_family", "gate_patience"):
        if key not in cfg:
            raise ValueError(f"Missing frozen config field: {key}")
    if cfg["alpha_family"] < 1 or cfg["gate_patience"] < 0:
        raise ValueError("Invalid alpha family / gate patience")
    if not 0 <= cfg["ema_decay"] < 1 or not 0 < cfg["train_fraction"] <= 1:
        raise ValueError("Invalid EMA/curriculum setting")
    if cfg["pseudo_weight"] < 0 or cfg["boundary_weight"] < 0 or cfg["min_margin"] < 0:
        raise ValueError("Invalid loss/gate weights")
    for field in ("anchor_count", "calibration_fit_count", "calibration_gate_count"):
        if cfg[field] < 1:
            raise ValueError("The current design requires explicit nonzero labeled roles")
    if cfg["knn_k"] < 1 or cfg["graph_sigma"] <= 0:
        raise ValueError("The graph variant requires knn_k >= 1 and positive sigma")
    if cfg["target_scale"] <= 0 or not 0 < cfg["target_share"] < 1:
        raise ValueError("Invalid frozen target normalization")
    validate_policy_config(cfg)
    for key in ("data_root", "common_manifest", "prepared_root", "output_root"):
        if key in cfg:
            p = Path(cfg[key])
            cfg[key] = str(p if p.is_absolute() else PROJECT / p)
    return cfg


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_key(sample_id, salt):
    return hashlib.sha256(f"frt-abl-v1|{salt}|{sample_id}".encode()).hexdigest()


def idx_bytes(root, name):
    p = Path(root) / name
    if p.is_file():
        return p.read_bytes()
    return gzip.decompress(Path(str(p) + ".gz").read_bytes())


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mnist_source(cfg):
    src, raw = Path(cfg["common_manifest"]), Path(cfg["data_root"])
    if digest(src / "manifest.json") != cfg["common_manifest_sha256"]:
        raise ValueError("COMMON manifest identity changed")
    expected = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    # Verify decoded IDX bytes against the frozen COMMON source identity.
    for name, item in expected["raw_data"].items():
        blob = idx_bytes(raw, name)
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError(f"MNIST raw hash mismatch: {name}")
    hashes = expected.get("files", expected.get("file_hashes", {}))
    for name, entry in hashes.items():
        if name == "labels_eval_only.csv":
            continue
        checksum = entry["sha256"] if isinstance(entry, dict) else entry
        if digest(src / name) != checksum:
            raise ValueError(f"COMMON CSV changed: {name}")
    # All source CSVs are also pinned in our output receipt independently of
    # the upstream manifest's field spelling.
    samples, truth = {}, {}
    for split in ("train", "validation", "test"):
        samples[split] = [{"id": r["pair_id"], "images": [r["left_id"], r["right_id"]],
                           "target": int(r["y_sum"])}
                          for r in read_csv(src / f"pairs_{split}.csv")]
    # Explicitly authorized labeled extension reads original TRAIN IDX labels,
    # never COMMON labels_eval_only.csv. No validation/test symbol labels exported.
    labels = idx_bytes(raw, "train-labels-idx1-ubyte")
    magic, count = struct.unpack(">II", labels[:8])
    if magic != 2049 or count != 60000 or len(labels) != 60008:
        raise ValueError("Invalid MNIST label IDX")
    for s in samples["train"]:
        truth[s["id"]] = [labels[8+int(i.split(":")[1])] for i in s["images"]]
    source_hashes = {p.name: digest(p) for p in src.glob("*.csv")
                     if p.name != "labels_eval_only.csv"}
    source_hashes["manifest.json"] = digest(src / "manifest.json")
    return samples, truth, source_hashes


def hwf_source(cfg):
    root = Path(cfg["data_root"])
    symbols = cfg["symbols"]
    samples, truth = {}, {}
    hashes = {}
    for split, name in (("train", "expr_train.json"), ("test", "expr_test.json")):
        path = root / name
        hashes[name] = digest(path)
        rows = json.loads(path.read_text())
        samples[split] = []
        for r in rows:
            paths = r["img_paths"]
            if any(not (root / "Handwritten_Math_Symbols" / p).is_file() for p in paths):
                raise FileNotFoundError(r["id"])
            samples[split].append({"id": r["id"], "images": paths, "target": r["res"]})
            # Ground truth is exported ONLY for explicitly budgeted train roles.
            if split == "train":
                truth[r["id"]] = [symbols.index({"times": "*", "div": "/"}.get(p.split("/")[0], p.split("/")[0]))
                                  for p in paths]
    samples["validation"] = []
    return samples, truth, hashes


def allocate(cfg, samples):
    """Deterministic, seed-independent image-disjoint annotation/validation roles.

    HWF expressions may reuse images: reserve evaluation/calibration/anchor
    roles first, then purge conflicting train expressions, with a full receipt.
    This is a NEW split, never presented as native HWF or COMMON training.
    """
    source_train = samples["train"]
    ordered = sorted(source_train, key=lambda s: stable_key(s["id"], "roles-20261026"))
    reserved = set()
    roles = {}
    targets = {}
    if cfg["task"] == "hwf":
        targets["validation"] = cfg["validation_count"]
    targets.update({"calibration_gate": cfg["calibration_gate_count"],
                    "calibration_fit": cfg["calibration_fit_count"],
                    "anchor": cfg["anchor_count"]})
    assigned = set()
    for role, count in targets.items():
        chosen = []
        # Images may repeat within a role; never across roles.
        for s in ordered:
            if s["id"] not in assigned and not reserved.intersection(s["images"]):
                chosen.append(s)
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            raise ValueError(f"Cannot create image-disjoint {role}: {len(chosen)} < {count}")
        roles[role] = chosen
        assigned.update(s["id"] for s in chosen)
        reserved.update(i for s in chosen for i in s["images"])
    roles["weak"] = [s for s in source_train if s["id"] not in assigned
                     and not reserved.intersection(s["images"])]
    if cfg["task"] == "mnist-add2":
        roles["validation"] = samples["validation"]
    roles["test"] = samples["test"]
    if not roles["weak"]:
        raise ValueError("No weak data remains after image leakage prevention")
    image_sets = {role: {i for s in rows for i in s["images"]} for role, rows in roles.items()}
    names = list(image_sets)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if image_sets[a] & image_sets[b]:
                raise ValueError(f"Cross-role image leakage: {a}/{b}")
    return roles, len(source_train) - len(assigned) - len(roles["weak"])


def prepare(cfg):
    dest = Path(cfg["prepared_root"])
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared manifests: {dest}")
    samples, truth, source_hashes = mnist_source(cfg) if cfg["task"] == "mnist-add2" else hwf_source(cfg)
    roles, purged = allocate(cfg, samples)
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    for role, rows in roles.items():
        rows = [dict(s, labels=truth[s["id"]]) if role in {"anchor", "calibration_fit", "calibration_gate"}
                else s for s in rows]
        path = dest / f"{role}.json"
        write_json(path, rows)
        files[path.name] = digest(path)
    # An annotation budget is unique images, not number of repeatedly seen tokens.
    budget = {r: {"expressions": len(roles[r]),
                  "unique_labeled_images": len({i for s in roles[r] for i in s["images"]})}
              for r in ("anchor", "calibration_fit", "calibration_gate")}
    receipt = {"schema_version": 1, "status": "PREPARED_NOT_TRAINED", "track": cfg["track"],
               "task": cfg["task"], "symbols": cfg["symbols"], "data_root": cfg["data_root"],
               "source_hashes": source_hashes, "counts": {r: len(v) for r, v in roles.items()},
               "annotation_budget": budget, "image_overlap_across_roles": 0,
               "purged_train_expressions": purged, "files": files,
               "warning": "New label-budget track; not directly comparable with COMMON result-only baselines.",
               "preprocessing": "uint8/255; grayscale; HWF resized 28x28; no identity-changing augmentation"}
    write_json(dest / "manifest.json", receipt)
    return receipt


def verify_prepared(cfg):
    root = Path(cfg["prepared_root"])
    receipt = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if receipt["task"] != cfg["task"] or receipt["symbols"] != cfg["symbols"] or receipt["data_root"] != cfg["data_root"]:
        raise ValueError("Prepared data/config mismatch")
    expected_counts = {"anchor": cfg["anchor_count"], "calibration_fit": cfg["calibration_fit_count"],
                       "calibration_gate": cfg["calibration_gate_count"]}
    if cfg["task"] == "hwf":
        expected_counts["validation"] = cfg["validation_count"]
    if any(receipt["counts"][role] != count for role, count in expected_counts.items()):
        raise ValueError("Prepared annotation/split budget differs from config")
    for name, checksum in receipt["files"].items():
        if digest(root / name) != checksum:
            raise ValueError(f"Prepared manifest changed: {name}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = prepare(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
