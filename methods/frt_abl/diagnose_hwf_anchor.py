"""Read-only audit of archived HWF anchor-only runs, using anchor/validation only.

No training, abduction, formal test evaluation, or checkpoint modification occurs.
Run: python -m methods.frt_abl.diagnose_hwf_anchor --output <new-json-path>
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .prepare import PROJECT, digest, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def label(image, symbols):
    name = image.split("/")[0]
    return symbols.index({"times": "*", "div": "/"}.get(name, name))


def summarize_role(rows, symbols):
    images = [image for row in rows for image in row["images"]]
    weighted = Counter()
    for row in rows:
        for image in row["images"]:
            weighted[symbols[label(image, symbols)]] += 1 / len(row["images"])
    return {
        "expressions": len(rows), "tokens": len(images),
        "unique_images": len(set(images)),
        "token_fraction_by_class": {s: sum(label(i, symbols) == j for i in images) / len(images)
                                    for j, s in enumerate(symbols)},
        "lengths": dict(sorted(Counter(len(row["images"]) for row in rows).items())),
        "unique_images_by_class": dict(Counter(symbols[label(i, symbols)] for i in set(images))),
        "sequence_balanced_class_mass": dict(weighted),
    }


def score(rows, predictions, cfg, kb):
    total_tokens = correct_tokens = correct_sequences = correct_tasks = invalid = 0
    classes = {s: Counter() for s in cfg["symbols"]}
    groups = {}
    confusion = Counter()
    for row in rows:
        truth = [label(i, cfg["symbols"]) for i in row["images"]]
        predicted = [predictions[i] for i in row["images"]]
        matches = sum(a == b for a, b in zip(truth, predicted))
        value = kb.value(predicted)
        hit = value is not None and math.isclose(float(value), float(row["target"]),
                                               rel_tol=0., abs_tol=1e-8)
        total_tokens += len(truth)
        correct_tokens += matches
        correct_sequences += matches == len(truth)
        correct_tasks += hit
        invalid += value is None
        for a, b in zip(truth, predicted):
            classes[cfg["symbols"][a]]["total"] += 1
            classes[cfg["symbols"][a]]["correct"] += a == b
            if a != b:
                confusion[(cfg["symbols"][a], cfg["symbols"][b])] += 1
        for group in (f"length-{len(truth)}",
                      "with-division" if cfg["symbols"].index("/") in truth else "without-division"):
            counts = groups.setdefault(group, Counter())
            counts["total"] += 1
            counts["task_correct"] += hit
            counts["sequence_correct"] += matches == len(truth)
    for counts in groups.values():
        counts["task_accuracy"] = counts["task_correct"] / counts["total"]
        counts["sequence_accuracy"] = counts["sequence_correct"] / counts["total"]
    for counts in classes.values():
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else None
    unique = {i for row in rows for i in row["images"]}
    return {
        "expressions": len(rows), "symbol_accuracy": correct_tokens / total_tokens,
        "unique_image_accuracy": sum(predictions[i] == label(i, cfg["symbols"]) for i in unique) / len(unique),
        "sequence_accuracy": correct_sequences / len(rows),
        "task_accuracy": correct_tasks / len(rows), "invalid_expressions": invalid,
        "per_class": classes, "groups": groups,
        "top_confusions": [{"truth": a, "predicted": b, "count": n}
                           for (a, b), n in confusion.most_common(15)],
    }


def audit(run_root, output):
    import torch
    from .core import ArithmeticKB
    from .policies import resolve_policy
    from .runtime import ImageStore, SymbolCNN

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    run_dirs = sorted(run_root.glob("*__hwf__anchor_only__seed-*"))
    if not run_dirs:
        raise FileNotFoundError("No archived HWF anchor_only runs")
    first_cfg = read_json(run_dirs[0] / "config.json")
    root = Path(first_cfg["prepared_root"])
    roles = {r: read_json(root / f"{r}.json") for r in
             ("anchor", "calibration_fit", "calibration_gate", "weak", "validation")}
    symbols = first_cfg["symbols"]
    receipt = read_json(root / "manifest.json")
    for role in roles:
        if digest(root / f"{role}.json") != receipt["files"][f"{role}.json"]:
            raise ValueError(f"Prepared file changed: {role}")
    source_train_path = Path(first_cfg["data_root"]) / "expr_train.json"
    if digest(source_train_path) != receipt["source_hashes"]["expr_train.json"]:
        raise ValueError("Original training source changed")
    original_train = [{"images": row["img_paths"]} for row in read_json(source_train_path)]
    image_sets = {r: {i for row in rows for i in row["images"]} for r, rows in roles.items()}
    label_mismatches = {
        r: sum(row["labels"] != [label(i, symbols) for i in row["images"]] for row in roles[r])
        for r in ("anchor", "calibration_fit", "calibration_gate")}
    report = {
        "scope": "Archived checkpoints; anchor and validation inference only; no formal test read",
        "roles": {r: summarize_role(rows, symbols) for r, rows in roles.items()},
        "original_train": summarize_role(original_train, symbols),
        "prepared_manifest_sha256": digest(root / "manifest.json"),
        "label_mapping_mismatches": label_mismatches,
        "image_overlap": {f"{a}/{b}": len(image_sets[a] & image_sets[b])
                          for i, a in enumerate(roles) for b in list(roles)[i + 1:]},
        "training_label_keys_in_weak": sum("labels" in row for row in roles["weak"]),
        "runs": [],
    }
    torch.set_num_threads(first_cfg["threads"])
    store = ImageStore(first_cfg)
    ids = sorted(image_sets["anchor"] | image_sets["validation"])
    pixels = torch.stack([store.get(i) for i in ids])
    kb = ArithmeticKB("hwf", symbols)
    report["ground_truth_arithmetic_mismatches"] = {
        role: sum(not math.isclose(float(kb.value([label(i, symbols) for i in row["images"]])),
                                  float(row["target"]), rel_tol=0., abs_tol=1e-8)
                  for row in rows)
        for role, rows in roles.items()}
    source_root = Path(__file__).parent
    for run_dir in run_dirs:
        cfg = read_json(run_dir / "config.json")
        for key in ("prepared_root", "symbols", "data_root"):
            if cfg[key] != first_cfg[key]:
                raise ValueError(f"Mixed data settings: {run_dir.name}: {key}")
        history = [json.loads(l) for l in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        best = max(history, key=lambda h: h["validation_task"]["accuracy"])
        source = read_json(run_dir / "source_manifest.json")
        manifest_hash = digest(root / "manifest.json")
        entry = {
            "run_id": run_dir.name, "seed": cfg["seed"],
            "best_round": best["round"], "recorded_best_validation": best["validation_task"]["accuracy"],
            "source_matches_current": {name: digest(source_root / name) == source.get(name)
                                       for name in ("train.py", "runtime.py", "core.py", "policies.py", "prepare.py")},
            "pseudo_loss_steps_total": sum(h["loss_term_steps"].get("pseudo", 0) for h in history),
            "boundary_loss_steps_total": sum(h["loss_term_steps"].get("boundary", 0) for h in history),
            "max_trained_pseudo_sequences": max(h["trained_pseudo_sequences"] for h in history),
            "optimizer_steps": history[-1]["optimizer_steps"],
            "search_calls": history[-1].get("search_calls_total"),
            "candidate_checks_total_from_rounds": sum(h["candidate_checks"] for h in history),
            "round_steps": sorted({h["round_steps"] for h in history}),
            "final_anchor_loss": history[-1]["loss_components_per_update"]["anchor"],
            "policy_pseudo": resolve_policy(cfg).pseudo,
            "checkpoints": {},
        }
        for name in ("best", "last"):
            checkpoint = torch.load(run_dir / f"{name}.pt", map_location="cpu", weights_only=False)
            if checkpoint["prepared_manifest_sha256"] != manifest_hash:
                raise ValueError("Checkpoint/prepared manifest hash mismatch")
            if name == "best" and checkpoint["round"] != best["round"]:
                raise ValueError("Saved best checkpoint does not match validation selection")
            metrics = {"sha256": digest(run_dir / f"{name}.pt"), "round": checkpoint["round"]}
            for weight in ("teacher", "student"):
                model = SymbolCNN(len(symbols)).eval()
                model.load_state_dict(checkpoint[weight])
                with torch.no_grad():
                    predicted = torch.cat([model(x)[0].argmax(-1) for x in pixels.split(128)]).tolist()
                lookup = dict(zip(ids, predicted))
                metrics[weight] = {r: score(roles[r], lookup, cfg, kb) for r in ("anchor", "validation")}
            if name == "best":
                actual = metrics["teacher"]["validation"]["task_accuracy"]
                if actual != entry["recorded_best_validation"]:
                    raise ValueError(f"Cannot reproduce archived validation accuracy: {actual}")
            entry["checkpoints"][name] = metrics
        report["runs"].append(entry)
        print(json.dumps({"seed": cfg["seed"], "best_round": best["round"],
                          "metrics": {w: {r: {k: v for k, v in m.items()
                                                if k in ("task_accuracy", "symbol_accuracy", "sequence_accuracy")}
                                          for r, m in entry["checkpoints"]["best"][w].items()}
                                      for w in ("teacher", "student")}}), flush=True)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "methods/frt_abl/runs-v10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_root, args.output)


if __name__ == "__main__":
    main()
