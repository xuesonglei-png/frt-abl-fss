"""Independent checkpoint evaluation. Inert unless --execute-evaluation is passed.

The formal test manifest is only read after explicit approval
(`--approve-formal-test`).  Use `--split smoke` with a smoke prepared root for
interface checks; the metrics JSON records which split was read.
"""
import argparse
import json
import time
from pathlib import Path

from .prepare import digest, idx_bytes, load_config, verify_prepared, write_json


def symbol_index(name, symbols):
    key = {"times": "*", "div": "/"}.get(name, name)
    try:
        return symbols.index(key)
    except ValueError as error:
        raise ValueError(f"Unknown HWF symbol directory: {name}") from error


def evaluate(cfg, checkpoint_path, output, split="test", approved_formal_test=False):
    import csv
    import math
    import torch
    from .core import ArithmeticKB, abduce
    from .runtime import ImageStore, SymbolCNN

    if split not in {"test", "smoke"}:
        raise ValueError("split must be 'test' or 'smoke'")
    formal = split == "test"
    if formal and not approved_formal_test:
        raise PermissionError("Formal test evaluation requires --approve-formal-test")
    torch.set_num_threads(cfg["threads"])
    # Only locally created checkpoints from this method are intended inputs.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["prepared_manifest_sha256"] != digest(Path(cfg["prepared_root"]) / "manifest.json"):
        raise ValueError("Checkpoint belongs to another prepared data manifest")
    saved_cfg = checkpoint["config"]
    for key in ("task", "symbols", "prepared_root", "data_root"):
        if saved_cfg[key] != cfg[key]:
            raise ValueError(f"Checkpoint/config mismatch: {key}")
    root = Path(cfg["prepared_root"])
    test_name = "test.json" if formal else "smoke_test.json"
    test_path = root / test_name
    if not test_path.is_file():
        raise FileNotFoundError(
            f"{test_name} not found in {root}; build a smoke manifest or use --split test with approval")
    test = json.loads(test_path.read_text(encoding="utf-8"))

    load_started = time.perf_counter()
    model = SymbolCNN(len(cfg["symbols"]))
    model.load_state_dict(checkpoint["teacher"])
    model.eval()
    model_load_seconds = time.perf_counter() - load_started
    store = ImageStore(cfg)
    # Complete oracle exists only in offline evaluation, never in training's
    # missing-fact knowledge base or in candidate generation.
    kb_started = time.perf_counter()
    oracle = ArithmeticKB(cfg["task"], cfg["symbols"])
    kb_init_seconds = time.perf_counter() - kb_started
    label_cache = {}

    def truth_labels(sample):
        if cfg["task"] != "mnist-add2":
            return [symbol_index(i.split("/")[0], cfg["symbols"]) for i in sample["images"]]
        prefix = sample["images"][0].split(":")[0]
        name = "train-labels-idx1-ubyte" if prefix == "train" else "t10k-labels-idx1-ubyte"
        if name not in label_cache:
            label_cache[name] = idx_bytes(Path(cfg["data_root"]), name)[8:]
        labels = label_cache[name]
        return [labels[int(i.split(":")[1])] for i in sample["images"]]

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    native = {"task": 0, "symbol": 0, "sequence": 0, "symbols_total": 0, "covered": 0}
    argmax = {"task": 0, "symbol": 0, "sequence": 0, "symbols_total": 0}
    total = 0
    eval_started = time.perf_counter()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f, torch.no_grad():
        writer = csv.writer(f)
        writer.writerow(["sample_id", "target", "native_symbols", "native_result",
                         "native_task_correct", "native_symbol_correct", "native_sequence_correct",
                         "native_covered", "argmax_symbols", "argmax_result",
                         "argmax_task_correct", "argmax_symbol_correct", "argmax_sequence_correct"])
        for start in range(0, len(test), cfg["batch_size"]):
            batch = test[start:start + cfg["batch_size"]]
            images, lengths = store.batch(batch, "cpu")
            logits, _ = model(images)
            probabilities = logits.softmax(-1)
            predicted = logits.argmax(-1).tolist()
            offset = 0
            for sample, length in zip(batch, lengths):
                z_argmax = tuple(predicted[offset:offset + length])
                probs = probabilities[offset:offset + length].tolist()
                offset += length
                truth = truth_labels(sample)
                z_native, covered, _ = abduce(
                    probs, sample["target"], oracle, beam_size=cfg["beam_size"],
                    max_checks=cfg["max_checks"], edit_penalty=cfg["edit_penalty"])
                value_native = oracle.value(z_native)
                value_argmax = oracle.value(z_argmax)
                native_hit = value_native is not None and math.isclose(
                    float(value_native), float(sample["target"]), rel_tol=0., abs_tol=1e-8)
                argmax_hit = value_argmax is not None and math.isclose(
                    float(value_argmax), float(sample["target"]), rel_tol=0., abs_tol=1e-8)
                native_matches = sum(a == b for a, b in zip(z_native, truth))
                argmax_matches = sum(a == b for a, b in zip(z_argmax, truth))
                native_sequence = native_matches == length
                argmax_sequence = argmax_matches == length
                native["task"] += native_hit
                native["symbol"] += native_matches
                native["symbols_total"] += length
                native["sequence"] += native_sequence
                native["covered"] += covered
                argmax["task"] += argmax_hit
                argmax["symbol"] += argmax_matches
                argmax["symbols_total"] += length
                argmax["sequence"] += argmax_sequence
                total += 1
                writer.writerow([sample["id"], sample["target"], json.dumps(z_native),
                                 float(value_native) if value_native is not None else None,
                                 native_hit, native_matches, native_sequence, covered,
                                 json.dumps(z_argmax),
                                 float(value_argmax) if value_argmax is not None else None,
                                 argmax_hit, argmax_matches, argmax_sequence])
    evaluation_seconds = time.perf_counter() - eval_started
    metrics = {
        "status": "OFFLINE_EVALUATION_COMPLETED", "split": split, "formal_test": formal,
        "approved_formal_test": bool(approved_formal_test),
        "task_correct": native["task"], "task_total": total,
        "task_accuracy": native["task"] / total,
        "symbol_correct": native["symbol"], "symbol_total": native["symbols_total"],
        "symbol_accuracy": native["symbol"] / native["symbols_total"],
        "sequence_correct": native["sequence"],
        "sequence_accuracy": native["sequence"] / total,
        "pair_symbol_accuracy": native["sequence"] / total,
        "coverage": native["covered"] / total, "fallback_count": total - native["covered"],
        "argmax_task_correct": argmax["task"],
        "argmax_task_accuracy": argmax["task"] / total,
        "argmax_symbol_correct": argmax["symbol"], "argmax_symbol_total": argmax["symbols_total"],
        "argmax_symbol_accuracy": argmax["symbol"] / argmax["symbols_total"],
        "argmax_sequence_correct": argmax["sequence"],
        "argmax_pair_symbol_accuracy": argmax["sequence"] / total,
        "checkpoint_sha256": digest(checkpoint_path),
        "prepared_manifest_sha256": digest(root / "manifest.json"),
        "model_load_seconds": model_load_seconds, "kb_init_seconds": kb_init_seconds,
        "evaluation_seconds": evaluation_seconds, "peak_memory_mb": None,
        "primary_outcome": "argmax_task_accuracy",
        "native_uses_ground_truth_result": True,
        "prediction_mode": ("argmax: answer-free perception followed by arithmetic evaluation; "
                            "native: abduction conditioned on the ground-truth sample result, "
                            "fallback to raw argmax"),
        "note": ("Symbol truth is used only for scoring. Native prediction consumes the true "
                 "expression result and is a conditional symbol-recovery metric; native task "
                 "accuracy largely measures search coverage, not answer-free task generalization."),
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("test", "smoke"), default="test")
    parser.add_argument("--execute-evaluation", action="store_true")
    parser.add_argument("--approve-formal-test", action="store_true",
                        help="Explicitly allow reading the formal test manifest (once, after freezing)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    verify_prepared(cfg)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.execute_evaluation:
        print(json.dumps({"status": "CHECKPOINT_PRESENT_NOT_EVALUATED",
                          "sha256": digest(args.checkpoint), "split": args.split}))
        return
    if args.split == "test" and not args.approve_formal_test:
        print(json.dumps({"status": "FORMAL_TEST_REQUIRES_APPROVAL",
                          "split": "test", "test_read": False,
                          "note": "Re-run with --approve-formal-test only after the model and config are frozen."}))
        raise SystemExit(2)
    print(json.dumps(evaluate(cfg, args.checkpoint, args.output, split=args.split,
                              approved_formal_test=args.approve_formal_test), indent=2))


if __name__ == "__main__":
    main()
