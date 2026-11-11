"""Complete parameter definitions for confidence_only; no imported defaults or overrides."""

from pathlib import Path
import math

PROJECT = Path(__file__).resolve().parents[4]
CODE_REVISION = "frt-abl-v11-independent-20261104"
EXPERIMENT_VARIANT = "confidence_only"


def build_config(task, seed=4):
    if task == "mnist-add2":
        cfg = {
            "track": "METHOD_LABEL_BUDGET",
            "task": "mnist-add2",
            "variant": "confidence_only",
            "method_version": "frt-abl-v11-independent-20261104",
            "alpha_grid": [
                0.8,
                0.81,
                0.82,
                0.83,
                0.84,
                0.85,
                0.86,
                0.87,
                0.88,
                0.89,
                0.9,
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.96,
                0.97,
                0.98,
                0.99,
            ],
            "curriculum_min_weight": 0.5,
            "curriculum_ramp_rounds": 10,
            "loss_ramp_rounds": 10,
            "symbols": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "data_root": "C:/Users/15841/Desktop/数据集集合/MNIST",
            "common_manifest": "datasets/mnist/manifests/mnist-add2-v1",
            "common_manifest_sha256": "0df11fca001506df6abead6ebe31057a41b350eb99fb5dd9453286a32ab9ad14",
            "prepared_root": "methods/frt_abl/prepared/mnist-label-budget-v1",
            "output_root": "methods/frt_abl/runs-v11-independent",
            "anchor_count": 270,
            "calibration_fit_count": 270,
            "calibration_gate_count": 270,
            "seed": 4,
            "device": "cpu",
            "threads": 2,
            "batch_size": 32,
            "rounds": 30,
            "warmup_epochs": 5,
            "learning_rate": 0.0001,
            "ema_decay": 0.99,
            "beam_size": 256,
            "max_checks": 512,
            "edit_penalty": 0.1,
            "retry_factor": 2,
            "retry_per_round": 256,
            "missing_facts": [],
            "knn_k": 10,
            "graph_sigma": 0.5,
            "graph_chunk_size": 512,
            "search_workers": 2,
            "search_parallel_min": 128,
            "search_chunksize": 16,
            "target_center": 9.0,
            "target_scale": 7.0,
            "target_share": 0.05,
            "alpha_start": 0.95,
            "beta": 0.2,
            "risk_limit": 0.1,
            "calibration_min_count": 50,
            "stable_rounds": 2,
            "min_margin": 0.02,
            "allow_truncated": False,
            "alpha_family": 20,
            "gate_patience": 0,
            "pseudo_weight": 1.0,
            "boundary_weight": 0.1,
            "experiment": "independent-ablation-v11",
            "soft_scale_floor": 0.0,
            "gate_coverage_floor": 0.9,
            "reference_limit": 2048,
        }
    elif task == "hwf":
        cfg = {
            "track": "METHOD_LABEL_BUDGET",
            "task": "hwf",
            "variant": "confidence_only",
            "method_version": "frt-abl-v11-independent-20261104",
            "alpha_grid": [
                0.8,
                0.81,
                0.82,
                0.83,
                0.84,
                0.85,
                0.86,
                0.87,
                0.88,
                0.89,
                0.9,
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.96,
                0.97,
                0.98,
                0.99,
            ],
            "curriculum_min_weight": 0.5,
            "curriculum_ramp_rounds": 10,
            "loss_ramp_rounds": 10,
            "symbols": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"],
            "data_root": "C:/Users/15841/Desktop/数据集集合/hwf/data",
            "prepared_root": "methods/frt_abl/prepared/hwf-label-budget-v1",
            "output_root": "methods/frt_abl/runs-v11-independent",
            "validation_count": 1000,
            "anchor_count": 100,
            "calibration_fit_count": 300,
            "calibration_gate_count": 300,
            "seed": 4,
            "device": "cpu",
            "threads": 2,
            "batch_size": 16,
            "rounds": 50,
            "warmup_epochs": 5,
            "learning_rate": 0.001,
            "ema_decay": 0.99,
            "beam_size": 256,
            "max_checks": 512,
            "edit_penalty": 0.1,
            "retry_factor": 2,
            "retry_per_round": 128,
            "missing_facts": [],
            "knn_k": 10,
            "graph_sigma": 0.5,
            "graph_chunk_size": 512,
            "search_workers": 2,
            "search_parallel_min": 128,
            "search_chunksize": 16,
            "target_center": 6.0,
            "target_scale": 29.5,
            "target_share": 0.05,
            "alpha_start": 0.95,
            "beta": 0.2,
            "risk_limit": 0.1,
            "calibration_min_count": 50,
            "stable_rounds": 2,
            "min_margin": 0.02,
            "allow_truncated": True,
            "alpha_family": 20,
            "gate_patience": 0,
            "pseudo_weight": 1.0,
            "boundary_weight": 0.2,
            "experiment": "independent-ablation-v11",
            "soft_scale_floor": 0.5,
            "gate_coverage_floor": 0.9,
            "reference_limit": 2048,
        }
    else:
        raise ValueError(f"Unknown task: {task}")
    # Seed selects an independent repetition, never a module combination.
    cfg["seed"] = seed
    for key in ("data_root", "common_manifest", "prepared_root", "output_root"):
        if key in cfg:
            path = Path(cfg[key])
            cfg[key] = str(path if path.is_absolute() else PROJECT / path)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg["variant"] != EXPERIMENT_VARIANT or cfg["method_version"] != CODE_REVISION:
        raise ValueError("This entry point only accepts its own independent experiment")
    if cfg["task"] not in {"mnist-add2", "hwf"} or cfg["seed"] not in range(5):
        raise ValueError("Expected a registered task and seed 0..4")
    if cfg["track"] != "METHOD_LABEL_BUDGET" or cfg.get("budget"):
        raise ValueError("This experiment uses the original prepared label-budget split")
    expected = list("0123456789") if cfg["task"] == "mnist-add2" else list("123456789+-*/")
    if cfg["symbols"] != expected:
        raise ValueError("Unexpected symbol order")
    for key in (
        "rounds",
        "warmup_epochs",
        "batch_size",
        "threads",
        "search_workers",
        "beam_size",
        "max_checks",
        "knn_k",
        "graph_chunk_size",
        "reference_limit",
        "search_chunksize",
        "calibration_min_count",
        "alpha_family",
        "stable_rounds",
        "curriculum_ramp_rounds",
        "loss_ramp_rounds",
        "anchor_count",
        "calibration_fit_count",
        "calibration_gate_count",
    ):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "graph_sigma", "target_scale"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    grid = cfg["alpha_grid"]
    if (
        not grid
        or grid != sorted(set(grid))
        or cfg["alpha_start"] not in grid
        or any(not math.isfinite(a) or not 0 < a <= 1 for a in grid)
        or cfg["alpha_family"] < len(grid)
    ):
        raise ValueError("Invalid declared alpha grid or hypothesis family")
    if not 0 <= cfg["beta"] < cfg["alpha_start"] <= 1:
        raise ValueError("Invalid thresholds")
    if not 0 < cfg["risk_limit"] < 1 or not 0 < cfg["gate_coverage_floor"] <= 1:
        raise ValueError("Invalid risk or gate coverage")
    if not 0 <= cfg["soft_scale_floor"] <= 1 or not 0 < cfg["curriculum_min_weight"] <= 1:
        raise ValueError("Invalid soft admission or curriculum parameter")
    if not 0 <= cfg["ema_decay"] < 1 or not 0 < cfg["target_share"] < 1:
        raise ValueError("Invalid EMA or target share")
    for key in ("pseudo_weight", "boundary_weight", "min_margin", "edit_penalty"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if cfg["retry_factor"] < 1 or cfg["retry_per_round"] < 0 or cfg["search_parallel_min"] < 0:
        raise ValueError("Invalid search retry/parallel parameters")
    if cfg["gate_patience"] != 0 or cfg["missing_facts"]:
        raise ValueError(
            "The independent ablation protocol keeps all rounds and complete knowledge"
        )
    if cfg["task"] == "mnist-add2" and cfg["max_checks"] < 100:
        raise ValueError("Addition search must cover all 100 candidates")
    if cfg["task"] == "hwf" and cfg["validation_count"] < 1:
        raise ValueError("Validation count must be positive")
