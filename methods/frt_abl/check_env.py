"""Read-only environment/preflight checker. Never constructs a model or reads test targets."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .prepare import PROJECT, load_config, verify_prepared

MNIST_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
               "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
DEFAULT_CONFIGS = tuple(str(PROJECT / "methods" / "frt_abl" / "configs" / name)
                        for name in ("mnist.json", "hwf.json", "smoke-mnist.json", "smoke-hwf.json"))


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_config(path):
    checks = []
    errors = []

    def record(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        if not ok:
            errors.append(name)

    try:
        cfg = load_config(path)
    except Exception as error:  # schema errors are the point of this checker
        record("config_load", False, repr(error))
        return {"config": str(path), "checks": checks, "errors": errors}

    record("config_load", True, f'{cfg["task"]}/{cfg["variant"]}')
    record("track", cfg["track"] == "METHOD_LABEL_BUDGET", cfg["track"])
    data_root = Path(cfg["data_root"])
    record("data_root_exists", data_root.is_dir(), data_root)
    if cfg["task"] == "mnist-add2":
        for name in MNIST_FILES:
            present = (data_root / name).is_file() or (data_root / (name + ".gz")).is_file()
            record(f"raw_{name}", present, data_root / name)
        common = Path(cfg["common_manifest"])
        record("common_manifest", (common / "manifest.json").is_file(), common / "manifest.json")
    else:
        record("raw_expr_train", (data_root / "expr_train.json").is_file(), data_root / "expr_train.json")
        record("raw_expr_test", (data_root / "expr_test.json").is_file(), data_root / "expr_test.json")
        record("raw_symbols_dir", (data_root / "Handwritten_Math_Symbols").is_dir(),
               data_root / "Handwritten_Math_Symbols")
    prepared = Path(cfg["prepared_root"])
    try:
        receipt = verify_prepared(cfg)
        record("prepared_manifest", True, f'{prepared} ({sum(receipt["counts"].values())} rows)')
        if "smoke" in prepared.name:
            record("smoke_test_file", (prepared / "smoke_test.json").is_file(),
                   prepared / "smoke_test.json")
    except Exception as error:
        record("prepared_manifest", False, repr(error))
    record("no_formal_test_loaded_by_checker", True, "hashes only; no model/test inference")
    return {"config": str(path), "checks": checks, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                        help="Configs to check (defaults: both experiments + both smoke configs)")
    args = parser.parse_args()
    torch_version = package_version("torch")
    numpy_version = package_version("numpy")
    pillow_version = package_version("Pillow")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "numpy": numpy_version,
        "Pillow": pillow_version,
        "executable": sys.executable,
        "project": str(PROJECT),
    }
    issues = []
    if sys.version_info < (3, 11):
        issues.append("python<3.11")
    for name, version in (("torch", torch_version), ("numpy", numpy_version), ("Pillow", pillow_version)):
        if version is None:
            issues.append(f"missing:{name}")
    configs = [check_config(path) for path in args.configs]
    for item in configs:
        if item["errors"]:
            issues.append(f'{item["config"]}:{",".join(item["errors"])}')
    report = {"status": "ENV_READY" if not issues else "ENV_ISSUES",
              "environment": environment, "configs": configs, "issues": issues,
              "training_started": False, "model_constructed": False,
              "formal_test_content_used": False,
              "note": "Integrity hashes only. No training/evaluation is run by this checker."}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
