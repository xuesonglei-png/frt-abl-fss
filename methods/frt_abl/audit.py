"""Offline correction-quality audit; not imported by training. No model execution."""
import argparse
import csv
import json
from pathlib import Path

from .prepare import digest, idx_bytes, load_config, verify_prepared, write_json


def _optional_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def _optional_int(row, key):
    value = row.get(key, "")
    return int(float(value)) if value not in ("", None) else None


def _optional_bool(row, key):
    value = row.get(key, "")
    return {"True": True, "False": False}.get(value)


def summarize(rows):
    """Truth must already have been attached by an evaluation-only caller."""
    total = len(rows)
    found = [r for r in rows if r["candidate"] is not None]
    accepted = [r for r in rows if r["region"] == "positive"]
    trained = [r for r in rows if r["trained"]]
    negative = [r for r in rows if r["region"] == "negative"]
    eligible = [r for r in rows if r.get("eligible") is True]
    unique_eligible = [r for r in eligible if r.get("n_supported") == 1]
    curve = []
    ranked = sorted(found, key=lambda r: (-r["lower"], -r["q"], r["sample_id"]))
    errors = 0
    for count, r in enumerate(ranked, 1):
        errors += not r["correct"]
        curve.append({"count": count, "coverage": count/total,
                      "errors": errors, "correction_risk": errors/count})
    return {"total": total, "candidate_found": len(found),
            "candidate_found_rate": len(found)/total if total else None,
            "candidate_accuracy": (sum(r["correct"] for r in found)/len(found)
                                   if found else None),
            "accepted": len(accepted),
            "negative_count": len(negative),
            "eligible_count": len(eligible),
            "unique_eligible_count": len(unique_eligible),
            "trained": len(trained), "training_coverage": len(trained)/total if total else None,
            "accepted_correction_accuracy": sum(r["correct"] for r in accepted)/len(accepted) if accepted else None,
            "trained_correction_error_rate": sum(not r["correct"] for r in trained)/len(trained) if trained else None,
            "wrong_training_fraction_of_all_weak": sum(not r["correct"] for r in trained)/total if total else None,
            "boundary_count": sum(r["region"] == "boundary" for r in rows),
            "max_wait_age": max((r["age"] for r in rows), default=0)}, curve


def audit(cfg, decisions, output):
    verify_prepared(cfg)
    root = Path(cfg["prepared_root"])
    weak = {s["id"]: s for s in json.loads((root/"weak.json").read_text(encoding="utf-8"))}
    labels = idx_bytes(Path(cfg["data_root"]), "train-labels-idx1-ubyte")[8:] if cfg["task"] == "mnist-add2" else None
    with Path(decisions).open(encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f))
    if len(data) != len(weak) or {r["sample_id"] for r in data} != set(weak):
        raise ValueError("Decision records do not match the prepared weak set")
    parsed = []
    for r in data:
        sample = weak[r["sample_id"]]
        if labels is not None:
            truth = [labels[int(i.split(":")[1])] for i in sample["images"]]
        else:
            truth = [cfg["symbols"].index({"times": "*", "div": "/"}.get(i.split("/")[0], i.split("/")[0]))
                     for i in sample["images"]]
        candidate = json.loads(r["candidate"])
        parsed.append({"sample_id": r["sample_id"], "candidate": candidate,
                       "correct": candidate == truth, "q": float(r["q"]), "lower": float(r["lower"]),
                       "trained": r["trained"] == "True", "region": r["region"], "age": int(r["age"]),
                       "margin": _optional_float(r, "margin"),
                       "stable_count": _optional_int(r, "stable_count"),
                       "unknown_count": _optional_int(r, "unknown_count"),
                       "n_supported": _optional_int(r, "n_supported"),
                       "eligible": _optional_bool(r, "eligible")})
    report, curve = summarize(parsed)
    report.update({"status": "OFFLINE_DIAGNOSTIC_ONLY", "decisions_sha256": digest(decisions),
                   "prepared_manifest_sha256": digest(root/"manifest.json"),
                   "note": "Uses hidden weak labels offline; never feed these results to training or model selection."})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"correction_quality.json", report)
    with (output/"risk_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["count", "coverage", "errors", "correction_risk"])
        writer.writeheader()
        writer.writerows(curve)
    return report


def audit_directory(cfg, decisions_dir, output):
    """Audit every decisions_round_*.csv in a directory and write a summary."""
    decisions_dir = Path(decisions_dir)
    files = sorted(decisions_dir.glob("decisions_round_*.csv"))
    if not files:
        raise ValueError(f"No decisions_round_*.csv files in {decisions_dir}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rounds = []
    for path in files:
        report = audit(cfg, path, output / path.stem)
        rounds.append({"round_file": path.name, "report": report})
    summary = {"status": "OFFLINE_DIAGNOSTIC_ONLY", "n_rounds": len(rounds),
               "decisions_dir": str(decisions_dir),
               "note": "Per-round artifacts are in subdirectories; weak labels are used offline only.",
               "rounds": rounds}
    write_json(output / "all_rounds.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--decisions", required=True,
                        help="A decisions CSV, or a run directory with --all-rounds")
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-rounds", action="store_true",
                        help="Audit every decisions_round_*.csv under --decisions")
    parser.add_argument("--execute-audit", action="store_true")
    args = parser.parse_args()
    if not args.execute_audit:
        print("NOT_EXECUTED: offline hidden-label diagnostic requires --execute-audit")
        return
    cfg = load_config(args.config)
    if args.all_rounds:
        print(json.dumps(audit_directory(cfg, args.decisions, args.output),
                         ensure_ascii=False, indent=2))
    else:
        print(json.dumps(audit(cfg, args.decisions, args.output), indent=2))


if __name__ == "__main__":
    main()
