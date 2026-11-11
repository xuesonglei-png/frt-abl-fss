"""Independent experiment logic for fixed_threshold; no calls to full or module switches."""

from collections import defaultdict
from ...core import Evidence, Region, choose_alpha


def experiment_record(cfg):
    """Provenance only; these fields never dispatch training behavior."""
    return {
        "method_version": cfg["method_version"],
        "implementation": "fixed_threshold",
        "comparison_kind": "single_module",
        "changed_components": ["adaptive_threshold"],
        "soft_admission": "supported_candidates",
    }


def acceptance_eligible(p, cfg):
    n_supported = getattr(p.search, "n_supported", None)
    if n_supported == 1 and p.search.complete and (p.search.unknown_count == 0):
        margin_ok = True
    else:
        margin_ok = p.search.margin >= cfg["min_margin"]
    return (
        p.evidence == Evidence.SUPPORTED
        and p.stable_count >= cfg["stable_rounds"]
        and margin_ok
        and (p.search.unknown_count == 0)
        and (p.search.complete or cfg["allow_truncated"])
    )


def build_graph(weak, gate, bank, cfg):
    from ...runtime import graph_bounds

    graph_bounds(weak, bank, cfg)
    graph_bounds(gate, bank, cfg)


def choose_threshold(gate, cfg, calibrator):
    """Preserve full's high-threshold preference, coverage rule and fallback."""
    if calibrator is None:
        return (None, [])
    grid = [cfg["alpha_start"]]
    args = (
        [p.lower for p in gate],
        [p.labels == tuple(p.sample["labels"]) for p in gate],
        [acceptance_eligible(p, cfg) for p in gate],
        grid,
        cfg["risk_limit"],
        cfg["calibration_min_count"],
        cfg["alpha_start"],
    )
    alpha, report = choose_alpha(
        *args,
        family_size=cfg["alpha_family"],
        prefer_high=True,
        min_coverage=cfg["gate_coverage_floor"],
    )
    if alpha is None:
        alpha, report = choose_alpha(
            *args, family_size=cfg["alpha_family"], prefer_high=False, min_coverage=None
        )
    return (alpha, report)


def partition_candidates(proposals, cfg, alpha):
    """Full's supported-candidate soft admission, including closed-gate fallback."""
    reference = alpha if alpha is not None else cfg["alpha_start"]
    for p in proposals:
        if p.labels is None:
            p.region = Region.BOUNDARY
            p.weight = 0.0
        else:
            p.region = Region.POSITIVE
            soft = min(1.0, max(0.0, p.lower / reference))
            floor = cfg["soft_scale_floor"]
            scale = floor + (1.0 - floor) * soft
            p.weight = p.lower * p.q * (1.0 - (p.upper - p.lower)) * scale


def stratum(sample, cfg):
    if cfg["task"] == "hwf":
        return f"length-{len(sample['images'])}"
    return f"sum-bin-{int(sample['target']) // 4}"


def select_curriculum(proposals, cfg, round_index):
    """Return accepted candidates without dropping any in the soft curriculum.

    Within each structure stratum, relative rank weights ease hard candidates
    in. Weighted mass in that stratum is preserved. Ties receive equal weight,
    so constant q/L cannot invent information from sample IDs.
    """
    positive = [p for p in proposals if p.region.value == "positive"]
    if not positive:
        return []
    for p in positive:
        p.curriculum_weight = 1.0
    progress = min(1.0, max(0, round_index) / cfg["curriculum_ramp_rounds"])
    minimum = cfg["curriculum_min_weight"] + (1 - cfg["curriculum_min_weight"]) * progress
    groups = defaultdict(list)
    for p in positive:
        groups[stratum(p.sample, cfg)].append(p)
    for group in groups.values():
        group.sort(key=lambda p: (p.lower, p.q, p.sample["id"]))
        if (group[0].lower, group[0].q) == (group[-1].lower, group[-1].q):
            continue
        start = 0
        while start < len(group):
            end = start + 1
            while end < len(group) and (group[end].lower, group[end].q) == (
                group[start].lower,
                group[start].q,
            ):
                end += 1
            rank = (start + end - 1) / 2 / (len(group) - 1) if len(group) > 1 else 0.5
            for p in group[start:end]:
                p.curriculum_weight = minimum + (1 - minimum) * rank
            start = end
        base_mass = sum((p.weight for p in group))
        weighted_mass = sum((p.weight * p.curriculum_weight for p in group))
        scale = base_mass / weighted_mass if weighted_mass > 0 else 1.0
        for p in group:
            p.curriculum_weight *= scale
    return sorted(positive, key=lambda p: p.sample["id"])


def spread_batches(batches, steps):
    """Use each batch once, distributed across the whole optimizer round."""
    if steps < len(batches) or steps < 1:
        raise ValueError("Step budget cannot accommodate all batches")
    if not batches:
        return {}
    return {(2 * i + 1) * steps // (2 * len(batches)): batch for i, batch in enumerate(batches)}
