"""Static source audit only: AST parsing, literal inspection and source comparison.

This module does not import any experiment, torch, data loader or trainer.
It neither executes tests nor creates results, models, checkpoints or processes.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_CHANGES = {
    "confidence_only": {"build_graph"},
    "fixed_threshold": {"choose_threshold"},
    "no_curriculum": {"select_curriculum"},
    "no_weight": {"partition_candidates"},
    "no_boundary": set(),
    "fixed_no_curriculum": {"choose_threshold", "select_curriculum"},
}
REFERENCES = ("standard_abl", "anchor_only")


def functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }


def literal_configs(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    build = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_config"
    )
    configs = [
        ast.literal_eval(n.value)
        for n in ast.walk(build)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
    ]
    if len(configs) != 2:
        raise ValueError(f"{path}: expected two complete task literals")
    return {cfg["task"]: cfg for cfg in configs}


def audit():
    errors = []
    files = sorted(ROOT.rglob("*.py"))
    for path in files:
        # compile returns a code object; it is never passed to exec or eval.
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    base = functions(ROOT / "full/protocol.py")
    base_cfg = literal_configs(ROOT / "full/settings.py")
    base_train = functions(ROOT / "full/train.py")
    diffs = {}
    for entry in ("full", *EXPECTED_CHANGES, *REFERENCES):
        for name in ("settings.py", "protocol.py", "train.py"):
            path = ROOT / entry / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 2 or (node.module and "ablation_v11" in node.module):
                        errors.append(f"{entry}/{name}: cross-experiment import")
                    imported = {alias.name for alias in node.names}
                    if imported & {
                        "resolve_policy",
                        "policy_record",
                        "load_config",
                        "partition",
                        "curriculum",
                    }:
                        errors.append(f"{entry}/{name}: legacy dispatch/config import")
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "policy"
                ):
                    errors.append(f"{entry}/{name}: module flag dispatch")
                if name != "settings.py" and isinstance(node, (ast.If, ast.IfExp, ast.Match)):
                    test = node.subject if isinstance(node, ast.Match) else node.test
                    if any(
                        isinstance(n, ast.Constant) and n.value == "variant" for n in ast.walk(test)
                    ):
                        errors.append(f"{entry}/{name}: variant-dependent runtime branch")
                if isinstance(node, ast.Constant) and node.value == "--variant":
                    errors.append(f"{entry}/{name}: shared variant CLI")
            if name == "train.py":
                for node in tree.body:
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                        errors.append(f"{entry}: top-level execution outside main guard")
        cfgs = literal_configs(ROOT / entry / "settings.py")
        for task, cfg in cfgs.items():
            changed = {
                k for k in cfg.keys() | base_cfg[task].keys() if cfg.get(k) != base_cfg[task].get(k)
            }
            if changed != (set() if entry == "full" else {"variant"}):
                errors.append(f"{entry}/{task}: non-target parameter drift: {sorted(changed)}")
            if cfg["variant"] != entry:
                errors.append(f"{entry}/{task}: incorrect experiment identity")
            if cfg["boundary_weight"] != (0.2 if task == "hwf" else 0.1):
                errors.append(f"{entry}/{task}: lost full boundary coefficient")
        if entry in EXPECTED_CHANGES:
            current = functions(ROOT / entry / "protocol.py")
            changed = {
                name
                for name in base.keys() | current.keys()
                if name != "experiment_record" and base.get(name) != current.get(name)
            }
            if changed != EXPECTED_CHANGES[entry]:
                errors.append(f"{entry}: unexpected protocol changes: {sorted(changed)}")
            engine = functions(ROOT / entry / "train.py")
            engine_changes = {
                name
                for name in engine.keys() | base_train.keys()
                if engine.get(name) != base_train.get(name)
            }
            expected_engine = {"run"} if entry == "no_boundary" else set()
            if engine_changes != expected_engine:
                errors.append(
                    f"{entry}: unintended training-loop changes: {sorted(engine_changes)}"
                )
            if entry == "no_boundary":
                if "kl_div" in engine["run"] or "boundary_loss" in engine["run"]:
                    errors.append("no_boundary: boundary optimization was not removed")
            diffs[entry] = {
                "protocol_functions": sorted(changed),
                "training_functions": sorted(engine_changes),
            }
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "status": "STATIC_SOURCE_CHECK_PASSED",
        "python_files": len(files),
        "independent_entries": 9,
        "complete_task_parameter_blocks": 18,
        "single_module_changes": diffs,
        "experiment_modules_imported": False,
        "experiments_executed": False,
        "unit_tests_executed": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
