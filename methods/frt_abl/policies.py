"""Explicit ablation factors and sampling policies; no data/model execution."""
from collections import defaultdict
from dataclasses import asdict, dataclass
import math


METHOD_VERSION = "frt-abl-v10-20261028"
VARIANTS = ("full", "confidence_only", "fixed_threshold", "no_curriculum",
            "fixed_no_curriculum", "no_weight", "no_boundary", "standard_abl", "anchor_only")
MODULE_VARIANTS = tuple(v for v in VARIANTS if v != "fixed_no_curriculum")
FACTORIAL_VARIANTS = ("full", "fixed_threshold", "no_curriculum", "fixed_no_curriculum")


@dataclass(frozen=True)
class AblationPolicy:
    graph: bool = True
    reliability_weight: bool = True
    boundary: bool = True
    gate: bool = True
    pseudo: bool = True
    threshold: str = "risk_guarded"
    curriculum: str = "soft_weight"


def resolve_policy(cfg):
    variant = cfg["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    factors = dict(graph=True, reliability_weight=True, boundary=True, gate=True, pseudo=True,
                   threshold=cfg.get("threshold_policy", "risk_guarded"),
                   curriculum=cfg.get("curriculum_policy", "soft_weight"))
    if variant == "confidence_only":
        factors["graph"] = False
    if variant in {"fixed_threshold", "fixed_no_curriculum"}:
        factors["threshold"] = "fixed"
    if variant in {"no_curriculum", "fixed_no_curriculum"}:
        factors["curriculum"] = "all"
    if variant == "no_weight":
        factors["reliability_weight"] = False
    if variant == "no_boundary":
        factors["boundary"] = False
    if variant in {"standard_abl", "anchor_only"}:
        factors.update(graph=False, reliability_weight=False, boundary=False, gate=False,
                       threshold="none", curriculum="all", pseudo=variant == "standard_abl")
    return AblationPolicy(**factors)


def policy_record(cfg):
    return {"method_version": METHOD_VERSION, **asdict(resolve_policy(cfg))}


def validate_policy_config(cfg):
    if cfg.get("method_version") != METHOD_VERSION:
        raise ValueError("Explicit v2 method_version required; archived configs are not silently migrated")
    if cfg.get("threshold_policy") not in {"risk_guarded", "linear_decay"}:
        raise ValueError("threshold_policy must be risk_guarded or diagnostic linear_decay")
    if cfg.get("curriculum_policy") not in {"soft_weight", "hard_fraction"}:
        raise ValueError("curriculum_policy must be soft_weight or diagnostic hard_fraction")
    grid = cfg.get("alpha_grid", [])
    if not grid or any(not math.isfinite(x) or not 0 <= x <= 1 for x in grid):
        raise ValueError("Invalid predeclared alpha_grid")
    if list(grid) != sorted(set(grid)) or cfg["alpha_start"] not in grid:
        raise ValueError("alpha_grid must be sorted/unique and include alpha_start")
    if cfg["alpha_family"] < len(grid):
        raise ValueError("alpha_family must cover the full grid, including fixed-threshold ablations")
    if not 0 < cfg.get("curriculum_min_weight", 0) <= 1:
        raise ValueError("curriculum_min_weight must lie in (0,1]")
    if not 0 < cfg.get("full_gate_coverage_floor", 0.90) <= 1:
        raise ValueError("full_gate_coverage_floor must lie in (0,1]")
    if cfg.get("curriculum_ramp_rounds", 0) < 1 or cfg.get("loss_ramp_rounds", 0) < 1:
        raise ValueError("Separate curriculum/loss ramps must be positive")
    if not math.isfinite(cfg.get("alpha_decay_step", -1)) or cfg["alpha_decay_step"] < 0:
        raise ValueError("Invalid diagnostic decay step")
    resolve_policy(cfg)


def threshold_options(cfg, round_index):
    policy = resolve_policy(cfg)
    if policy.threshold == "none":
        return [], cfg["alpha_start"]
    if policy.threshold == "fixed":
        return [cfg["alpha_start"]], cfg["alpha_start"]
    floor = cfg["alpha_start"]
    if policy.threshold == "linear_decay":
        floor = max(cfg["alpha_min"], floor-cfg["alpha_decay_step"]*round_index)
    # High initial threshold stays the lower limit in v2. Gate evidence may
    # require a higher threshold; elapsed rounds never relax evidence quality.
    return [x for x in cfg["alpha_grid"] if x >= floor], floor


def stratum(sample, cfg):
    if cfg["task"] == "hwf":
        return f'length-{len(sample["images"])}'
    return f'sum-bin-{int(sample["target"]) // 4}'


def curriculum(proposals, cfg, round_index):
    """Return accepted candidates without dropping any in the soft curriculum.

    Within each structure stratum, relative rank weights ease hard candidates
    in. Weighted mass in that stratum is preserved. Ties receive equal weight,
    so constant q/L cannot invent information from sample IDs.
    """
    policy = resolve_policy(cfg)
    positive = [p for p in proposals if p.region.value == "positive"]
    if not positive:
        return []
    for p in positive:
        p.curriculum_weight = 1.0
    if policy.curriculum == "all":
        return sorted(positive, key=lambda p: p.sample["id"])
    if policy.curriculum == "hard_fraction":
        fraction = min(1., cfg["train_fraction"] + (1-cfg["train_fraction"])*
                       round_index/cfg["curriculum_ramp_rounds"])
        ranked = sorted(positive, key=lambda p: (-p.lower, -p.q, p.sample["id"]))
        return ranked[:math.ceil(len(ranked)*fraction)]
    progress = min(1., max(0, round_index)/cfg["curriculum_ramp_rounds"])
    minimum = cfg["curriculum_min_weight"] + (1-cfg["curriculum_min_weight"])*progress
    groups = defaultdict(list)
    for p in positive:
        groups[stratum(p.sample, cfg)].append(p)
    for group in groups.values():
        group.sort(key=lambda p: (p.lower, p.q, p.sample["id"]))
        if (group[0].lower, group[0].q) == (group[-1].lower, group[-1].q):
            # Exactly neutral on a constant signal, including singleton strata.
            continue
        start = 0
        while start < len(group):
            end = start + 1
            while end < len(group) and (group[end].lower, group[end].q) == (group[start].lower, group[start].q):
                end += 1
            rank = ((start+end-1)/2)/(len(group)-1) if len(group) > 1 else .5
            for p in group[start:end]:
                p.curriculum_weight = minimum + (1-minimum)*rank
            start = end
        base_mass = sum(p.weight for p in group)
        weighted_mass = sum(p.weight*p.curriculum_weight for p in group)
        scale = base_mass/weighted_mass if weighted_mass > 0 else 1.
        for p in group:
            p.curriculum_weight *= scale
    return sorted(positive, key=lambda p: p.sample["id"])


def spread_batches(batches, steps):
    """Use each batch once, distributed across the whole optimizer round."""
    if steps < len(batches) or steps < 1:
        raise ValueError("Step budget cannot accommodate all batches")
    if not batches:
        return {}
    # Bin centres give an injective mapping when steps >= n; the unused prefix
    # and suffix are both bounded by half a bin rather than a whole epoch tail.
    return {((2*i+1)*steps // (2*len(batches))): batch for i, batch in enumerate(batches)}
