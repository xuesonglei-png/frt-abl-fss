"""Independent fixed_threshold training entry point; default is plan-only."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
from ...prepare import digest, verify_prepared, write_json
from .settings import CODE_REVISION, build_config, validate_config
from .protocol import (
    experiment_record,
    spread_batches,
    stratum,
    acceptance_eligible,
    build_graph,
    choose_threshold,
    partition_candidates,
    select_curriculum,
)


def preflight(cfg, smoke=False, max_updates=None, efficiency=None):
    validate_config(cfg)
    receipt = verify_prepared(cfg)
    return {
        "status": "READY_NOT_RUN",
        "task": cfg["task"],
        "variant": cfg["variant"],
        "formal_experiment_config_frozen": False,
        "split_review": "PREPARED_IMAGE_DISJOINT_LABEL_BUDGET",
        "track": cfg["track"],
        "counts": receipt["counts"],
        "annotation_budget": receipt["annotation_budget"],
        "budget": cfg.get("budget"),
        "target_normalization": {
            "center": cfg["target_center"],
            "scale": cfg["target_scale"],
            "share": cfg["target_share"],
        },
        "alpha_family": cfg["alpha_family"],
        "gate_patience": cfg["gate_patience"],
        "device": cfg["device"],
        "threads": cfg["threads"],
        "search_workers": cfg["search_workers"],
        "smoke_mode": smoke,
        "max_updates": max_updates,
        "code_revision": CODE_REVISION,
        "resolved_policy": experiment_record(cfg),
        "efficiency": efficiency or {"enabled": False},
        "model_constructed": False,
        "training_started": False,
        "note": "Preflight is file-only; this invocation does not execute a model. Independent V11 implementation; no model was executed by preflight.",
    }


def run(cfg, max_updates=None, smoke=False, efficiency=None):
    """Called only after explicit --execute-training. Never called by preparation.

    `max_updates` is a hard optimizer-step cap for protocol smoke runs; `smoke`
    only changes the run/status labels, it does not relax any statistical gate.
    `efficiency` enables the optional efficiency protocol:
    {"targets": [...], "time_budget_seconds": float|None,
     "sample_interval_seconds": float}.  It changes no numerical operation.
    """
    import csv
    from contextlib import contextmanager, nullcontext
    import math
    import platform
    import time
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    import torch
    from torch.nn import functional as F
    from ...core import ArithmeticKB, Region
    from ...runtime import (
        Calibrator,
        ImageStore,
        SymbolCNN,
        propose,
        search_worker_init,
        update_teacher,
        weighted_sequence_ce,
    )

    clock_started = time.perf_counter()
    validate_config(cfg)
    verify_prepared(cfg)
    torch.set_num_threads(cfg["threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if cfg["device"].startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable")
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    if smoke:
        tag = "__smoke"
    elif efficiency:
        tag = "__efficiency"
    else:
        tag = ""
    run_tag = f"__{cfg['run_tag']}" if cfg.get("run_tag") else ""
    run_id = f"{CODE_REVISION}__{cfg['task']}__{cfg['variant']}__seed-{cfg['seed']}{run_tag}{tag}__{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')}"
    out = Path(cfg["output_root"]) / run_id
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    write_json(out / "resolved_policy.json", experiment_record(cfg))
    source_root = Path(__file__).resolve().parents[1]
    write_json(
        out / "source_manifest.json",
        {
            **{
                str(p.relative_to(source_root)): digest(p)
                for p in source_root.rglob("*.py")
                if "__pycache__" not in p.parts
            },
            **{
                "shared/" + name: digest(source_root.parent / name)
                for name in ("core.py", "runtime.py", "prepare.py", "policies.py", "efficiency.py")
            },
        },
    )
    write_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": cfg["device"],
            "threads": cfg["threads"],
            "code_revision": CODE_REVISION,
            "experiment": cfg.get("experiment"),
            "prepared_manifest_sha256": digest(Path(cfg["prepared_root"]) / "manifest.json"),
            "config_sha256": digest(out / "config.json"),
        },
    )
    recorder = None
    if efficiency:
        from ...efficiency import EfficiencyRecorder

        recorder = EfficiencyRecorder(
            out,
            cfg,
            run_id,
            targets=efficiency.get("targets"),
            time_budget_seconds=efficiency.get("time_budget_seconds"),
            sample_interval_seconds=efficiency.get("sample_interval_seconds", 0.25),
            started_perf=clock_started,
            torch_module=torch,
        )
        recorder.start()
        recorder.add_phase("setup", time.perf_counter() - clock_started)

    def phase(name):
        return recorder.phase(name) if recorder is not None else nullcontext()

    write_json(
        out / "status.json",
        {
            "status": "RUNNING",
            "test_evaluated": False,
            "formal_test_read": False,
            "smoke": smoke,
            "max_updates": max_updates,
        },
    )
    init_started = time.perf_counter()
    root = Path(cfg["prepared_root"])
    roles = {
        role: json.loads((root / f"{role}.json").read_text(encoding="utf-8"))
        for role in ("anchor", "calibration_fit", "calibration_gate", "weak", "validation")
    }
    store = ImageStore(cfg)
    kb = ArithmeticKB(cfg["task"], cfg["symbols"], cfg["missing_facts"])
    student = SymbolCNN(len(cfg["symbols"])).to(cfg["device"])
    trainable = list(student.parameters())
    optimizer = torch.optim.Adam(trainable, lr=cfg["learning_rate"])
    if recorder is not None:
        recorder.add_phase("model_init", time.perf_counter() - init_started)

    def batches(rows, role, round_index):
        rng = random.Random(f"{cfg['seed']}|{role}|{round_index}")
        ids = list(range(len(rows)))
        rng.shuffle(ids)
        return [
            [rows[i] for i in ids[start : start + cfg["batch_size"]]]
            for start in range(0, len(ids), cfg["batch_size"])
        ]

    @contextmanager
    def role_rng(role, round_index, step_index):
        import hashlib

        key = f"{cfg['seed']}|{role}|{round_index}|{step_index}"
        role_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63 - 1)
        devices = list(range(torch.cuda.device_count())) if cfg["device"].startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(role_seed)
            if devices:
                torch.cuda.manual_seed_all(role_seed)
            yield

    def image_count(batch):
        return sum((len(s["images"]) for s in batch))

    def role_stats(rows):
        return (
            sum((len(s["images"]) for s in rows)),
            (len(rows) + cfg["batch_size"] - 1) // cfg["batch_size"],
        )

    def supervised_batch(batch):
        x, lengths = store.batch(batch, cfg["device"])
        logits, features = student(x)
        loss = weighted_sequence_ce(
            logits, lengths, [s["labels"] for s in batch], [1.0] * len(batch)
        )
        return loss

    def step(loss, teacher=None):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            (
                p.grad is not None and (not torch.isfinite(p.grad).all())
                for p in student.parameters()
            )
        ):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        if teacher is not None:
            update_teacher(student, teacher, cfg["ema_decay"])

    @torch.no_grad()
    def validation():
        teacher.eval()
        correct = total = seen = forwards = 0
        for start in range(0, len(roles["validation"]), cfg["batch_size"]):
            batch = roles["validation"][start : start + cfg["batch_size"]]
            x, lengths = store.batch(batch, cfg["device"])
            logits, _ = teacher(x)
            forwards += 1
            seen += image_count(batch)
            predicted, offset = (logits.argmax(-1).tolist(), 0)
            for sample, n in zip(batch, lengths):
                value = kb.value(predicted[offset : offset + n])
                correct += value is not None and math.isclose(
                    float(value), float(sample["target"]), rel_tol=0.0, abs_tol=1e-08
                )
                total += 1
                offset += n
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "images": seen,
            "forwards": forwards,
        }

    total_steps = 0
    started = time.perf_counter()
    total_proposal_images = total_proposal_forwards = 0
    total_training_images = total_training_forwards = 0
    total_validation_images = total_validation_forwards = 0
    total_candidate_checks = total_search_calls = 0
    search_pool = None
    try:
        student.train()
        warmup_stopped = False
        warmup_budget_hit = False
        with phase("warmup"):
            for epoch in range(cfg["warmup_epochs"]):
                for batch_index, batch in enumerate(batches(roles["anchor"], "warmup", epoch)):
                    with role_rng("warmup", epoch, batch_index):
                        step(supervised_batch(batch))
                    total_steps += 1
                    total_training_images += image_count(batch)
                    total_training_forwards += 1
                    if max_updates is not None and total_steps >= max_updates:
                        warmup_stopped = True
                        break
                    if recorder is not None:
                        recorder.set_counters(
                            optimizer_steps=total_steps,
                            training_images=total_training_images,
                            training_forwards=total_training_forwards,
                        )
                        if recorder.budget_exceeded():
                            warmup_stopped = True
                            warmup_budget_hit = True
                            break
                if warmup_stopped:
                    break
        if warmup_stopped:
            if warmup_budget_hit:
                status = "STOPPED_TIME_BUDGET_IN_WARMUP"
                stopped_reason = "TIME_BUDGET"
            else:
                status = (
                    "SMOKE_ONLY_STOPPED_IN_WARMUP" if smoke else "STOPPED_MAX_UPDATES_IN_WARMUP"
                )
                stopped_reason = "MAX_UPDATES"
            efficiency_summary = None
            if recorder is not None:
                recorder.set_counters(
                    optimizer_steps=total_steps,
                    training_images=total_training_images,
                    training_forwards=total_training_forwards,
                )
                efficiency_summary = recorder.finish(
                    status,
                    extra={
                        "optimizer_steps": total_steps,
                        "training_images": total_training_images,
                        "training_forwards": total_training_forwards,
                    },
                )
            write_json(
                out / "status.json",
                {
                    "status": status,
                    "test_evaluated": False,
                    "formal_test_read": False,
                    "smoke": smoke,
                    "max_updates": max_updates,
                    "optimizer_steps": total_steps,
                    "stopped_reason": stopped_reason,
                    "time_budget_seconds": (
                        efficiency.get("time_budget_seconds") if efficiency else None
                    ),
                    "training_images": total_training_images,
                    "training_forwards": total_training_forwards,
                    "efficiency": efficiency_summary,
                    "note": "Warmup ended before a teacher existed; no checkpoint was saved.",
                },
            )
            return
        teacher = deepcopy(student).eval()
        teacher.requires_grad_(False)
        if cfg["search_workers"] > 1:
            search_pool = ProcessPoolExecutor(
                max_workers=cfg["search_workers"], initializer=search_worker_init, initargs=(kb,)
            )
        state, best_val = ({}, -1.0)
        stopped_reason = None
        gate_closed_streak = 0
        weak_batches = (len(roles["weak"]) + cfg["batch_size"] - 1) // cfg["batch_size"]
        for round_index in range(cfg["rounds"]):
            if max_updates is not None and total_steps >= max_updates:
                stopped_reason = "MAX_UPDATES"
                break
            if recorder is not None:
                if recorder.budget_exceeded():
                    stopped_reason = "TIME_BUDGET"
                    break
                recorder.begin_round(round_index)
            round_started = time.perf_counter()
            round_steps = 0
            loss_sum = 0.0
            proposal_images = proposal_forwards = 0
            training_images = training_forwards = 0
            with phase("proposal"):
                retries = sorted(
                    (s for s in roles["weak"] if state.get(s["id"], {}).get("age", 0) > 0),
                    key=lambda s: (-state[s["id"]]["age"], s["id"]),
                )[: cfg["retry_per_round"]]
                retry_ids = {s["id"] for s in retries}
                args = (teacher, store, kb, cfg, state)
                weak = propose(roles["weak"], *args, retry_ids=retry_ids, pool=search_pool)
                anchors = propose(roles["anchor"], *args, pool=search_pool)
                fitting = propose(roles["calibration_fit"], *args, pool=search_pool)
                gate = propose(roles["calibration_gate"], *args, pool=search_pool)
                for rows in (
                    roles["weak"],
                    roles["anchor"],
                    roles["calibration_fit"],
                    roles["calibration_gate"],
                ):
                    images, forwards = role_stats(rows)
                    proposal_images += images
                    proposal_forwards += forwards
            total_proposal_images += proposal_images
            total_proposal_forwards += proposal_forwards
            round_search_calls = (
                len(roles["weak"])
                + len(roles["anchor"])
                + len(roles["calibration_fit"])
                + len(roles["calibration_gate"])
            )
            total_search_calls += round_search_calls
            with phase("calibration"):
                fit_supported = [p for p in fitting if p.labels is not None]
                fit_labels = [tuple(p.sample["labels"]) == p.labels for p in fit_supported]
                calibrator, alpha, gate_report = (None, None, [])
                if fit_supported:
                    calibrator = Calibrator()
                    calibrator.fit([p.reliability_features for p in fit_supported], fit_labels)
                    scored = weak + gate
                    scores = calibrator.predict([p.reliability_features for p in scored])
                    for p, q in zip(scored, scores):
                        p.q = q if p.labels is not None else 0.0
                for p in anchors:
                    p.q = float(p.labels is not None and p.labels == tuple(p.sample["labels"]))
            bank = [p for p in weak + anchors if p.labels is not None]
            unsupported_weak = sum((p.labels is None for p in weak))
            with phase("graph"):
                build_graph(weak, gate, bank, cfg)
            with phase("decision"):
                alpha, gate_report = choose_threshold(gate, cfg, calibrator)
                gate_closed_streak = gate_closed_streak + 1 if alpha is None else 0
                partition_candidates(weak, cfg, alpha)
                selected = select_curriculum(weak, cfg, round_index)
                boundary = [p for p in weak if p.region == Region.BOUNDARY]
            anchor_batches = batches(roles["anchor"], "anchor", round_index)
            pos_batches = batches(selected, "positive", round_index)
            boundary_batches = batches(boundary, "boundary", round_index)
            count = max(len(anchor_batches), len(pos_batches), len(boundary_batches), weak_batches)
            positive_slots = spread_batches(pos_batches, count)
            boundary_slots = spread_batches(boundary_batches, count)
            ramp = min(1.0, (round_index + 1) / cfg["loss_ramp_rounds"])
            trained_ids = set()
            loss_components = {"anchor": 0.0, "pseudo": 0.0, "boundary": 0.0}
            term_steps = Counter()
            positive_exposure = Counter()
            student.train()
            partial_round = False
            update_started = time.perf_counter()
            for i in range(count):
                if max_updates is not None and total_steps >= max_updates:
                    stopped_reason = "MAX_UPDATES"
                    break
                if recorder is not None and recorder.budget_exceeded():
                    stopped_reason = "TIME_BUDGET"
                    partial_round = True
                    break
                anchor_batch = anchor_batches[i % len(anchor_batches)]
                with role_rng("anchor", round_index, i):
                    loss = supervised_batch(anchor_batch)
                loss_components["anchor"] += float(loss.detach())
                term_steps["anchor"] += 1
                training_images += image_count(anchor_batch)
                training_forwards += 1
                current_positive_ids = []
                if i in positive_slots:
                    batch = positive_slots[i]
                    x, lengths = store.batch([p.sample for p in batch], cfg["device"])
                    with role_rng("positive", round_index, i):
                        logits, features = student(x)
                    training_images += image_count([p.sample for p in batch])
                    training_forwards += 1
                    pseudo_loss = (
                        ramp
                        * cfg["pseudo_weight"]
                        * weighted_sequence_ce(
                            logits,
                            lengths,
                            [p.labels for p in batch],
                            [p.weight * p.curriculum_weight for p in batch],
                        )
                    )
                    loss = loss + pseudo_loss
                    loss_components["pseudo"] += float(pseudo_loss.detach())
                    term_steps["pseudo"] += 1
                    current_positive_ids = [p.sample["id"] for p in batch]
                    positive_exposure.update((stratum(p.sample, cfg) for p in batch))
                if i in boundary_slots:
                    batch = boundary_slots[i]
                    x, lengths = store.batch([p.sample for p in batch], cfg["device"])
                    with torch.no_grad():
                        targets = teacher(x)[0].softmax(-1)
                    training_images += 2 * image_count([p.sample for p in batch])
                    training_forwards += 2
                    with role_rng("boundary", round_index, i):
                        logits, features = student(x)
                    token_loss = F.kl_div(logits.log_softmax(-1), targets, reduction="none").sum(-1)
                    pieces, offset = ([], 0)
                    for n in lengths:
                        pieces.append(token_loss[offset : offset + n].mean())
                        offset += n
                    boundary_weight = cfg["boundary_weight"]
                    boundary_loss = ramp * boundary_weight * torch.stack(pieces).mean()
                    loss = loss + boundary_loss
                    loss_components["boundary"] += float(boundary_loss.detach())
                    term_steps["boundary"] += 1
                step(loss, teacher)
                trained_ids.update(current_positive_ids)
                total_steps += 1
                round_steps += 1
                loss_sum += float(loss.detach())
            if recorder is not None:
                recorder.add_phase("update", time.perf_counter() - update_started)
            total_training_images += training_images
            total_training_forwards += training_forwards
            chosen_ids = trained_ids
            weak_ids = {p.sample["id"] for p in weak}
            for p in weak + fitting + gate + anchors:
                p.age = (
                    p.age + 1
                    if p.sample["id"] in weak_ids and p.sample["id"] not in chosen_ids
                    else 0
                )
                state[p.sample["id"]] = {
                    "labels": p.labels,
                    "stable_count": p.stable_count,
                    "age": p.age,
                }
            with phase("validation"):
                val = validation()
            total_validation_images += val["images"]
            total_validation_forwards += val["forwards"]
            round_candidate_checks = sum(
                (p.search.checked for p in weak + fitting + gate + anchors)
            )
            total_candidate_checks += round_candidate_checks
            supported = [p for p in weak if p.labels is not None]

            def stats(values):
                if not values:
                    return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
                mean = sum(values) / len(values)
                return {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": mean,
                    "std": math.sqrt(sum(((v - mean) ** 2 for v in values)) / len(values)),
                }

            weights = [p.weight * p.curriculum_weight for p in selected]
            report = {
                "round": round_index,
                "alpha": alpha,
                "gate": gate_report,
                "proposal_skipped": False,
                "weak_sequences_total": len(roles["weak"]),
                "code_revision": CODE_REVISION,
                "resolved_policy": experiment_record(cfg),
                "alpha_family": cfg["alpha_family"],
                "calibration_fit_correct": sum(fit_labels),
                "calibration_fit_total": len(fit_labels),
                "q_stats": stats([p.q for p in supported]),
                "lower_stats": stats([p.lower for p in supported]),
                "rough_width_stats": stats([p.upper - p.lower for p in supported]),
                "graph_changed_candidates": sum(
                    (abs(p.lower - p.q) > 1e-08 or abs(p.upper - p.q) > 1e-08 for p in supported)
                ),
                "effective_weight_stats": stats(weights),
                "curriculum_weight_stats": stats([p.curriculum_weight for p in selected]),
                "positive_effective_sample_size": (
                    sum(weights) ** 2 / sum((w * w for w in weights))
                    if weights and any(weights)
                    else 0.0
                ),
                "loss_components_per_update": {
                    k: v / round_steps if round_steps else 0.0 for k, v in loss_components.items()
                },
                "loss_term_steps": dict(term_steps),
                "positive_exposure_by_stratum": dict(positive_exposure),
                "calibrator_degenerate": calibrator is None or calibrator.degenerate,
                "regions": dict(Counter((p.region.value for p in weak))),
                "accepted_candidates": sum((p.region == Region.POSITIVE for p in weak)),
                "unique_accepted": sum(
                    (
                        p.region == Region.POSITIVE and getattr(p.search, "n_supported", None) == 1
                        for p in weak
                    )
                ),
                "eligible_unique": sum(
                    (
                        acceptance_eligible(p, cfg) and getattr(p.search, "n_supported", None) == 1
                        for p in weak
                    )
                ),
                "eligible_unique_truncated": sum(
                    (
                        acceptance_eligible(p, cfg)
                        and getattr(p.search, "n_supported", None) == 1
                        and (not p.search.complete)
                        for p in weak
                    )
                ),
                "unsupported_weak": unsupported_weak,
                "reference_candidates": len(bank),
                "scheduled_pseudo_sequences": len(selected),
                "trained_pseudo_sequences": len(chosen_ids),
                "coverage": len(chosen_ids) / len(roles["weak"]),
                "candidate_checks": round_candidate_checks,
                "candidate_checks_total": total_candidate_checks,
                "search_calls": round_search_calls,
                "search_calls_total": total_search_calls,
                "truncated_searches": sum((not p.search.complete for p in weak)),
                "retry_samples": len(retry_ids),
                "loss": loss_sum / round_steps if round_steps else 0.0,
                "round_steps": round_steps,
                "partial_round": partial_round,
                "validation_task": val,
                "optimizer_steps": total_steps,
                "gate_closed_streak": gate_closed_streak,
                "proposal_images": proposal_images,
                "proposal_forwards": proposal_forwards,
                "training_images": training_images,
                "training_forwards": training_forwards,
                "validation_images": val["images"],
                "validation_forwards": val["forwards"],
                "inference_images_total": total_proposal_images
                + total_training_images
                + total_validation_images,
                "inference_forwards_total": total_proposal_forwards
                + total_training_forwards
                + total_validation_forwards,
                "round_seconds": time.perf_counter() - round_started,
                "elapsed_seconds": time.perf_counter() - started,
                "max_wait_age": max((p.age for p in weak), default=0),
            }
            if recorder is not None:
                recorder.set_counters(
                    optimizer_steps=total_steps,
                    candidate_checks=total_candidate_checks,
                    search_calls=total_search_calls,
                    proposal_images=total_proposal_images,
                    proposal_forwards=total_proposal_forwards,
                    training_images=total_training_images,
                    training_forwards=total_training_forwards,
                    validation_images=total_validation_images,
                    validation_forwards=total_validation_forwards,
                )
                point = recorder.record_round(
                    {
                        "round": round_index,
                        "accuracy": val["accuracy"],
                        "correct": val["correct"],
                        "total": val["total"],
                        "elapsed_seconds": recorder.elapsed(),
                        "round_seconds": time.perf_counter() - round_started,
                        "partial_round": partial_round,
                    }
                )
                report["efficiency_elapsed_seconds"] = point["elapsed_seconds"]
                report["efficiency_phase_seconds"] = point["round_phase_seconds"]
                report["peak_process_tree_rss_mb"] = recorder.peak_now()["process_tree_rss_peak_mb"]
                report["time_budget_remaining_seconds"] = recorder.budget_remaining()
            report_started = time.perf_counter()
            with (out / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(report) + "\n")
            with (out / f"decisions_round_{round_index:03}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sample_id",
                        "candidate",
                        "evidence",
                        "region",
                        "q",
                        "lower",
                        "upper",
                        "weight",
                        "trained",
                        "age",
                        "complete",
                        "margin",
                        "stable_count",
                        "unknown_count",
                        "n_supported",
                        "original_valid",
                        "eligible",
                        "curriculum_weight",
                        "effective_weight",
                    ]
                )
                for p in weak:
                    writer.writerow(
                        [
                            p.sample["id"],
                            json.dumps(p.labels),
                            p.evidence.value,
                            p.region.value,
                            p.q,
                            p.lower,
                            p.upper,
                            p.weight,
                            p.sample["id"] in chosen_ids,
                            p.age,
                            p.search.complete,
                            p.search.margin,
                            p.stable_count,
                            p.search.unknown_count,
                            getattr(p.search, "n_supported", ""),
                            getattr(p.search, "original_valid", ""),
                            acceptance_eligible(p, cfg),
                            p.curriculum_weight,
                            p.weight * p.curriculum_weight,
                        ]
                    )
            with (out / f"gate_round_{round_index:03}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sample_id",
                        "candidate",
                        "q",
                        "lower",
                        "upper",
                        "margin",
                        "stable_count",
                        "unknown_count",
                        "n_supported",
                        "complete",
                        "correct",
                        "eligible",
                    ]
                )
                for p in gate:
                    writer.writerow(
                        [
                            p.sample["id"],
                            json.dumps(p.labels),
                            p.q,
                            p.lower,
                            p.upper,
                            p.search.margin,
                            p.stable_count,
                            p.search.unknown_count,
                            getattr(p.search, "n_supported", ""),
                            p.search.complete,
                            p.labels == tuple(p.sample["labels"]),
                            acceptance_eligible(p, cfg),
                        ]
                    )
            if recorder is not None:
                recorder.add_phase("report", time.perf_counter() - report_started)
            checkpoint_started = time.perf_counter()
            checkpoint = {
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "aux_head": None,
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "round": round_index,
                "prepared_manifest_sha256": digest(root / "manifest.json"),
                "optimizer_steps": total_steps,
                "state": state,
                "calibrator": calibrator.state_dict() if calibrator else None,
                "alpha": alpha,
                "validation_task": val,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "batch_rng_scheme": "seed-role-round-v2",
                "resolved_policy": experiment_record(cfg),
                "code_revision": CODE_REVISION,
                "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(checkpoint, out / "last.pt")
            if val["accuracy"] > best_val:
                best_val = val["accuracy"]
                torch.save(checkpoint, out / "best.pt")
            if recorder is not None:
                recorder.add_phase("checkpoint", time.perf_counter() - checkpoint_started)
            print(json.dumps(report), flush=True)
            if (
                stopped_reason is None
                and cfg["gate_patience"]
                and (gate_closed_streak >= cfg["gate_patience"])
            ):
                stopped_reason = "GATE_CLOSED"
            if stopped_reason is not None:
                break
        best_path = out / "best.pt"
        if smoke:
            status = (
                "SMOKE_ONLY_COMPLETED_NOT_TESTED"
                if stopped_reason is None
                else f"SMOKE_ONLY_STOPPED_{stopped_reason}"
            )
        elif stopped_reason == "GATE_CLOSED":
            status = "STOPPED_GATE_CLOSED"
        elif stopped_reason == "MAX_UPDATES":
            status = "STOPPED_MAX_UPDATES"
        elif stopped_reason == "TIME_BUDGET":
            status = "STOPPED_TIME_BUDGET"
        else:
            status = "TRAINING_COMPLETED_TEST_NOT_RUN"
        efficiency_summary = None
        if recorder is not None:
            recorder.set_counters(
                optimizer_steps=total_steps,
                candidate_checks=total_candidate_checks,
                search_calls=total_search_calls,
                proposal_images=total_proposal_images,
                proposal_forwards=total_proposal_forwards,
                training_images=total_training_images,
                training_forwards=total_training_forwards,
                validation_images=total_validation_images,
                validation_forwards=total_validation_forwards,
            )
            efficiency_summary = recorder.finish(
                status,
                extra={
                    "optimizer_steps": total_steps,
                    "candidate_checks": total_candidate_checks,
                    "search_calls": total_search_calls,
                    "proposal_images": total_proposal_images,
                    "proposal_forwards": total_proposal_forwards,
                    "training_images": total_training_images,
                    "training_forwards": total_training_forwards,
                    "validation_images": total_validation_images,
                    "validation_forwards": total_validation_forwards,
                },
            )
        write_json(
            out / "status.json",
            {
                "status": status,
                "best_validation_accuracy": best_val,
                "code_revision": CODE_REVISION,
                "resolved_policy": experiment_record(cfg),
                "experiment": cfg.get("experiment"),
                "budget": cfg.get("budget"),
                "checkpoint_sha256": digest(best_path) if best_path.exists() else None,
                "test_evaluated": False,
                "formal_test_read": False,
                "smoke": smoke,
                "max_updates": max_updates,
                "time_budget_seconds": (efficiency or {}).get("time_budget_seconds"),
                "stopped_reason": stopped_reason,
                "optimizer_steps": total_steps,
                "proposal_images": total_proposal_images,
                "proposal_forwards": total_proposal_forwards,
                "training_images": total_training_images,
                "training_forwards": total_training_forwards,
                "validation_images": total_validation_images,
                "validation_forwards": total_validation_forwards,
                "candidate_checks": total_candidate_checks,
                "search_calls": total_search_calls,
                "efficiency": efficiency_summary,
                "note": "No formal test was read. This entry point does not evaluate formal test data.",
            },
        )
    except BaseException as error:
        efficiency_summary = None
        if recorder is not None:
            try:
                efficiency_summary = recorder.finish("FAILED", error=repr(error))
            except Exception:
                efficiency_summary = None
        write_json(
            out / "status.json",
            {
                "status": "FAILED",
                "error": repr(error),
                "test_evaluated": False,
                "formal_test_read": False,
                "smoke": smoke,
                "max_updates": max_updates,
                "efficiency": efficiency_summary,
            },
        )
        raise
    finally:
        if search_pool is not None:
            search_pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("mnist-add2", "hwf"))
    parser.add_argument("--seed", type=int, choices=range(5), default=4)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    cfg = build_config(args.task, args.seed)
    limit = args.max_updates
    if args.smoke:
        limit = 20 if limit is None else limit
        if not 1 <= limit <= 20:
            parser.error("Smoke mode requires 1..20 updates")
    elif limit is not None and limit < 1:
        parser.error("--max-updates must be positive")
    if not args.execute_training:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "config": cfg,
                    "implementation": experiment_record(cfg),
                    "training_started": False,
                    "model_constructed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            preflight(cfg, smoke=args.smoke, max_updates=limit), ensure_ascii=False, indent=2
        )
    )
    run(cfg, smoke=args.smoke, max_updates=limit)


if __name__ == "__main__":
    main()
