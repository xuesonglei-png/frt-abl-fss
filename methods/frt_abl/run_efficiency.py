"""Plan or run the serial computational-efficiency protocol.

Same prepared data, same frozen config, same machine, one process at a time:
the runner refuses to start unless `--execute --confirm-efficiency-runs` is
given, and it never evaluates the formal test split.  Each training command
enables `train.py --efficiency`, which writes `efficiency.json` and
`efficiency_curve.csv` into the run directory.

Default plan: the v10 `full` benchmark on MNIST and HWF at design seed 4.
Add `--variants` to include ablations for a like-for-like cost comparison.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .efficiency import (DEFAULT_BUDGETS_SECONDS, DEFAULT_TARGETS,
                         hardware_fingerprint, now_iso, parse_seconds)
from .policies import METHOD_VERSION, VARIANTS
from .prepare import PROJECT, digest, idx_bytes, load_config, verify_prepared, write_json

SEEDS = (0, 1, 2, 3, 4)
DEFAULT_CONFIGS = tuple(str(PROJECT / "methods" / "frt_abl" / "configs" / name)
                        for name in ("mnist.json", "hwf.json"))


def effective_config(path, args):
    cfg = load_config(path)
    verify_prepared(cfg)
    if args.threads is not None:
        cfg["threads"] = args.threads
    if args.search_workers is not None:
        cfg["search_workers"] = args.search_workers
    if args.device is not None:
        cfg["device"] = args.device
    return cfg


def warm_cache(cfg):
    """Read the prepared manifests and raw payload bytes once (best effort)."""
    root = Path(cfg["prepared_root"])
    for role in ("anchor", "calibration_fit", "calibration_gate", "weak", "validation"):
        path = root / f"{role}.json"
        if path.is_file():
            path.read_bytes()
    if cfg["task"] == "mnist-add2":
        for name in ("train-images-idx3-ubyte", "t10k-images-idx3-ubyte",
                     "train-labels-idx1-ubyte"):
            try:
                idx_bytes(Path(cfg["data_root"]), name)
            except FileNotFoundError:
                pass
    else:
        for name in ("expr_train.json", "expr_test.json"):
            path = Path(cfg["data_root"]) / name
            if path.is_file():
                rows = json.loads(path.read_text())
                seen = set()
                for row in rows:
                    for image in row["img_paths"]:
                        if image in seen:
                            continue
                        seen.add(image)
                        try:
                            (Path(cfg["data_root"]) / "Handwritten_Math_Symbols" / image).read_bytes()
                        except OSError:
                            pass
    return {"task": cfg["task"], "prepared_root": str(root), "warmed_at": now_iso()}


def build_plan(args):
    installs = []
    for path in args.configs:
        cfg = effective_config(path, args)
        targets = args.targets or list(DEFAULT_TARGETS.get(cfg["task"], ()))
        for variant in args.variants:
            for seed in args.seeds:
                command = [args.python, "-m", "methods.frt_abl.train",
                           "--config", str(Path(path).resolve()),
                           "--variant", variant, "--seed", str(seed),
                           "--execute-training", "--efficiency",
                           "--output-root", str(Path(args.output_root).resolve())]
                for target in targets:
                    command += ["--target-accuracy", repr(float(target))]
                if args.budget_stop is not None:
                    command += ["--time-budget-seconds", str(float(args.budget_stop))]
                if args.threads is not None:
                    command += ["--threads", str(args.threads)]
                if args.search_workers is not None:
                    command += ["--search-workers", str(args.search_workers)]
                if args.device is not None:
                    command += ["--device", args.device]
                installs.append({
                    "task": cfg["task"], "variant": variant, "seed": seed,
                    "config": str(Path(path).resolve()),
                    "config_sha256": digest(path),
                    "prepared_manifest_sha256": digest(Path(cfg["prepared_root"]) / "manifest.json"),
                    "targets": [float(x) for x in targets],
                    "command": command,
                })
    return installs


def protocol_payload(args, plan, hardware):
    targets = {}
    for item in plan:
        targets.setdefault(item["task"], item["targets"])
    return {
        "schema_version": 1,
        "status": "PLAN_ONLY",
        "code_revision": METHOD_VERSION,
        "created_at": now_iso(),
        "targets": targets,
        "budgets_seconds": [float(x) for x in args.budgets],
        "budget_stop_seconds": (None if args.budget_stop is None else float(args.budget_stop)),
        "variants": list(args.variants), "seeds": list(args.seeds),
        "n_runs": len(plan), "serial": True,
        "hardware": hardware,
        "output_root": str(Path(args.output_root).resolve()),
        "runs": plan,
        "notes": [
            "Same prepared manifests, frozen config, machine and thread counts are required.",
            "Default variants contain only the v10 full benchmark; add --variants for ablations.",
            "No formal test split is read by this runner or by train --efficiency.",
            "Targets and budgets are declared before execution and are not tuned afterwards.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["full"],
                        help="Default is the v10 full benchmark; add ablations explicitly")
    parser.add_argument("--seeds", nargs="+", type=int, default=[4])
    parser.add_argument("--targets", nargs="+", type=float, default=None,
                        help="Validation accuracy targets; default per-task pre-declared points")
    parser.add_argument("--budgets", nargs="+", type=parse_seconds,
                        default=list(DEFAULT_BUDGETS_SECONDS),
                        help="Budget points for the summary; accepts 300, 5m, 1h")
    parser.add_argument("--budget-stop", type=parse_seconds, default=None,
                        help="Optional hard stop at one budget (saves the smaller budgets for offline analysis)")
    parser.add_argument("--output-root", default=str(PROJECT / "methods" / "frt_abl" / "runs-efficiency-v10"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--search-workers", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--warm-cache", action="store_true",
                        help="Read raw data once before timed runs (reduce cold-cache bias)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually launch the serial runs (requires --confirm-efficiency-runs)")
    parser.add_argument("--confirm-efficiency-runs", action="store_true")
    parser.add_argument("--allow-hardware-change", action="store_true",
                        help="Permit execution despite a different hardware fingerprint")
    args = parser.parse_args()

    if any(seed not in SEEDS for seed in args.seeds):
        raise SystemExit("Registered seeds are 0..4")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        raise SystemExit("Duplicate seeds/variants are not independent repetitions")
    if args.targets is not None and any(not 0 < target <= 1 for target in args.targets):
        raise SystemExit("--targets must lie in (0, 1]")
    if args.threads is not None and args.threads < 1:
        raise SystemExit("--threads must be positive")
    if args.search_workers is not None and args.search_workers < 1:
        raise SystemExit("--search-workers must be positive")
    if args.execute and not args.confirm_efficiency_runs:
        raise SystemExit("Refusing to launch efficiency runs without --confirm-efficiency-runs")
    if args.execute and args.device == "cuda":
        probe = subprocess.run(
            [args.python, "-c",
             "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
            capture_output=True, text=True)
        if probe.returncode != 0:
            raise SystemExit(f"--python does not provide CUDA: {args.python}")
    if args.budget_stop is not None and args.budget_stop < max(args.budgets):
        print(json.dumps({"warning": "budget_stop is smaller than the largest summary budget; "
                                     "that budget will be censored"}, ensure_ascii=False),
              file=sys.stderr)

    output_root = Path(args.output_root).resolve()
    protocol_path = output_root / "efficiency_protocol.json"
    if protocol_path.exists():
        raise SystemExit(f"Protocol already exists; choose a new --output-root: {protocol_path}")

    plan = build_plan(args)
    first_cfg = effective_config(args.configs[0], args)
    hardware = hardware_fingerprint(first_cfg)
    protocol = protocol_payload(args, plan, hardware)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    if not args.execute:
        print("PLAN_ONLY: no training launched.", file=sys.stderr)
        return

    current = hardware_fingerprint(first_cfg)
    if (current["fingerprint"] != hardware["fingerprint"]
            and not args.allow_hardware_change):
        raise SystemExit("Hardware fingerprint changed between plan and execution; "
                         "rerun the plan or pass --allow-hardware-change")

    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    protocol["status"] = "EXECUTING"
    protocol["execution"] = {"started_at": now_iso(), "completed": [], "failed": []}
    protocol["cache_warmup"] = []
    write_json(protocol_path, protocol)
    try:
        if args.warm_cache:
            for path in args.configs:
                item = warm_cache(effective_config(path, args))
                protocol["cache_warmup"].append(item)
                print(f"WARM  {item['task']}", flush=True)
            write_json(protocol_path, protocol)

        for index, item in enumerate(plan, 1):
            log_path = logs / (f"{index:03d}-{item['task']}-{item['variant']}-"
                               f"seed-{item['seed']}.log")
            print(f"START {index}/{len(plan)} {item['task']} {item['variant']} "
                  f"seed-{item['seed']}", flush=True)
            started = now_iso()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(item["command"], cwd=str(PROJECT),
                                           stdout=log, stderr=subprocess.STDOUT)
            finished = now_iso()
            if completed.returncode != 0:
                protocol["execution"]["failed"].append({
                    "run": item, "log": str(log_path), "returncode": completed.returncode,
                    "started_at": started, "finished_at": finished})
                protocol["status"] = "FAILED"
                write_json(protocol_path, protocol)
                raise SystemExit(f"Run failed ({completed.returncode}); see {log_path}")
            protocol["execution"]["completed"].append({
                "run": item, "log": str(log_path), "returncode": 0,
                "started_at": started, "finished_at": finished})
            write_json(protocol_path, protocol)
            print(f"DONE  {index}/{len(plan)} {item['task']} {item['variant']} "
                  f"seed-{item['seed']}", flush=True)
        protocol["status"] = "EXECUTED"
    finally:
        protocol["execution"]["finished_at"] = now_iso()
        write_json(protocol_path, protocol)


if __name__ == "__main__":
    main()
