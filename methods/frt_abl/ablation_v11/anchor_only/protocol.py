"""Independent experiment logic for anchor_only; no calls to full or module switches."""

from ...core import Evidence, Region


def experiment_record(cfg):
    """Provenance only; these fields never dispatch training behavior."""
    return {
        "method_version": cfg["method_version"],
        "implementation": "anchor_only",
        "comparison_kind": "supervised_reference",
        "changed_components": ["weak_supervision"],
        "soft_admission": "not_applicable",
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
    """Replace graph approximations with calibrated confidence only."""
    for p in weak + gate:
        p.lower = p.upper = p.q


def choose_threshold(gate, cfg, calibrator):
    return (None, [])


def partition_candidates(proposals, cfg, alpha):
    """Supervised reference has no weak-training candidates."""
    if proposals:
        raise ValueError("Anchor-only must not generate weak proposals")


def stratum(sample, cfg):
    if cfg["task"] == "hwf":
        return f"length-{len(sample['images'])}"
    return f"sum-bin-{int(sample['target']) // 4}"


def select_curriculum(proposals, cfg, round_index):
    """Keep the same supported positives; remove only curriculum rank weights."""
    positive = [p for p in proposals if p.region == Region.POSITIVE]
    for p in positive:
        p.curriculum_weight = 1.0
    return sorted(positive, key=lambda p: p.sample["id"])


def spread_batches(batches, steps):
    """Use each batch once, distributed across the whole optimizer round."""
    if steps < len(batches) or steps < 1:
        raise ValueError("Step budget cannot accommodate all batches")
    if not batches:
        return {}
    return {(2 * i + 1) * steps // (2 * len(batches)): batch for i, batch in enumerate(batches)}
