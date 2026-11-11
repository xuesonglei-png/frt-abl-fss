"""Plan, freeze provenance for, and summarize the V10 × five-baseline comparison.

Planning is the default: the frozen protocol is validated, every adoption source
is checked for presence and the exact downstream commands are printed.  Nothing
is trained, no model is executed and no formal-test manifest is read.

``--execute-adoption`` re-reads the finished native artifacts and freezes one
provenance record (path + SHA-256 per source).  ``--summarize-only`` writes the
comparison tables and the Markdown report.  Both are separate from any training
gate on purpose: the five baselines already have complete registered results,
and re-running them cannot make them share the COMMON manifest.

Typical stages:

    python -m methods.frt_abl.run_comparison
    python -m methods.frt_abl.run_comparison --check
    python -m methods.frt_abl.run_comparison --execute-adoption
    python -m methods.frt_abl.run_comparison --summarize-only
"""
import argparse
import json
import sys
from pathlib import Path

from .comparison import (DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_RUNS, PROTOCOL_VERSION,
                         ComparisonError, load_comparison_config, relative_path,
                         resolve_path, timestamp, write_adopted, write_comparison)
from .prepare import PROJECT


def adoption_sources(cfg, project=PROJECT, methods=None):
    """Resolve every declared adoption source without reading its content."""
    wanted = set(methods) if methods else None
    sources = []
    for baseline in cfg["baselines"]:
        if wanted is not None and baseline["method"] not in wanted:
            continue
        for spec in baseline["adoptions"]:
            if spec["parser"] == "status_row":
                sources.append({"method": baseline["method"], "parser": "status_row",
                                "task": spec["task"], "path": relative_path(
                                    project / "reproduction_status.csv", project),
                                "exists": (Path(project) / "reproduction_status.csv").is_file()})
                continue
            if "path_template" in spec:
                paths = [(seed, resolve_path(spec["path_template"].format(seed=seed), project))
                         for seed in spec.get("seeds") or []]
            else:
                paths = [(spec.get("seed"), resolve_path(spec["path"], project))]
            for seed, path in paths:
                sources.append({"method": baseline["method"], "parser": spec["parser"],
                                "task": spec["task"], "metric": spec.get("metric")
                                or ",".join(spec.get("fields", {})),
                                "seed": seed, "path": relative_path(path, project),
                                "exists": path.is_file()})
    return sources


def primary_summary_state(cfg, project=PROJECT):
    spec = cfg["primary"]
    path = resolve_path(spec["summary"], project)
    state = {"summary": relative_path(path, project), "exists": path.is_file(),
             "variants": spec["variants"], "tasks": [],
             "test_evaluations": len(spec.get("test_evaluations", []))}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            state["tasks"] = sorted({group.get("task") for group in payload.get("groups", [])
                                     if group.get("variant") in spec["variants"]})
        except (OSError, json.JSONDecodeError):
            state["exists"] = False
    return state


def build_plan(cfg, project=PROJECT, config_path=DEFAULT_CONFIG, methods=None):
    sources = adoption_sources(cfg, project, methods)
    missing = [item for item in sources if not item["exists"]]
    primary = primary_summary_state(cfg, project)
    wanted = set(methods) if methods else None
    return {
        "status": "PLAN_ONLY",
        "protocol_version": PROTOCOL_VERSION,
        "config": relative_path(config_path, project),
        "generated_at": timestamp(),
        "seeds": cfg["seeds"],
        "primary": primary,
        "baselines": [{"method": item["method"], "display": item["display"],
                       "paper": item["paper"],
                       "tasks": sorted({spec["task"] for spec in item["adoptions"]})}
                      for item in cfg["baselines"]
                      if wanted is None or item["method"] in wanted],
        "sources": sources,
        "n_sources": len(sources),
        "n_missing": len(missing),
        "note": ("Adoption is read-only: no baseline is re-run and no model is executed. "
                 "Missing sources are reported, never replaced by a guessed number."),
    }


def main():
    # Baseline display names contain characters such as "³"; a GBK console would
    # raise UnicodeEncodeError on the plan JSON.  Files are always UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runs", default=str(DEFAULT_RUNS),
                        help="Harness run root (adopted/ and runs/ records)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Summary output directory")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Restrict planning/adoption to these baseline ids")
    parser.add_argument("--check", action="store_true",
                        help="Validate the protocol and every adoption source, then exit")
    parser.add_argument("--execute-adoption", action="store_true",
                        help="Freeze the adopted observations with per-source hashes")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Aggregate primary + adopted + harness runs and write the report")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild an existing adoption record or summary directory")
    parser.add_argument("--plan-out", default=None,
                        help="Optional path for the plan JSON (plan mode never writes elsewhere)")
    args = parser.parse_args()

    try:
        cfg = load_comparison_config(args.config)
    except ComparisonError as error:
        raise SystemExit(str(error))

    if args.execute_adoption and args.summarize_only:
        raise SystemExit("--execute-adoption and --summarize-only are separate stages")
    if args.methods and args.summarize_only:
        raise SystemExit("--methods only filters planning/adoption; summarize always uses the full protocol")
    if args.methods:
        known = {baseline["method"] for baseline in cfg["baselines"]}
        unknown = sorted(set(args.methods) - known)
        if unknown:
            raise SystemExit(f"Unknown baseline ids: {unknown}; known: {sorted(known)}")

    if args.summarize_only:
        try:
            summary = write_comparison(args.output, cfg, runs_root=args.runs,
                                       overwrite=args.overwrite)
        except ComparisonError as error:
            raise SystemExit(str(error))
        print(json.dumps({"status": summary["status"],
                          "output": relative_path(args.output, PROJECT),
                          "n_entries": summary["n_entries"],
                          "n_groups": len(summary["groups"]),
                          "ranking_status": summary["ranking_status"],
                          "n_issues": len(summary["issues"]),
                          "note": "Descriptive only; no training, model or formal test was read."},
                         ensure_ascii=False, indent=2))
        return

    if args.execute_adoption:
        try:
            payload = write_adopted(args.runs, cfg=cfg, overwrite=args.overwrite,
                                    methods=args.methods)
        except ComparisonError as error:
            raise SystemExit(str(error))
        print(json.dumps({"status": payload["status"],
                          "record": relative_path(Path(args.runs) / "adopted" / "observations.json",
                                                  PROJECT),
                          "n_observations": len(payload["observations"]),
                          "n_sources": len(payload["sources"]),
                          "n_issues": len(payload["issues"]),
                          "note": "Artifacts were re-read, not executed."},
                         ensure_ascii=False, indent=2))
        return

    plan = build_plan(cfg, config_path=args.config, methods=args.methods)
    if args.plan_out:
        path = Path(args.plan_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.check:
        problems = []
        if not plan["primary"]["exists"]:
            problems.append(f"primary summary missing: {plan['primary']['summary']}")
        problems += [f"{item['method']} source missing: {item['path']}"
                     for item in plan["sources"] if not item["exists"]]
        if problems:
            raise SystemExit("--check failed:\n- " + "\n- ".join(problems))


if __name__ == "__main__":
    main()
