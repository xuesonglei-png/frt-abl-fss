"""Post-hoc robustness evaluation for the frozen full v10 model.

This module never trains, never changes the v10 pipeline and never reads the
formal test split without explicit approval.  It re-evaluates the *frozen*
`full` checkpoint under three pre-registered perturbations:

1. perception noise on the 28x28 image (Gaussian additive noise or salt-and-
   pepper corruption; the noise field is a deterministic function of the image
   id, so a condition is reproducible and independent of batch order);
2. knowledge incompleteness: a nested subset of KB facts (addition) or
   operator-application cells (handwritten arithmetic) is removed.  The abducer
   still has to say "unresolved" instead of falling back to the full arithmetic
   oracle; the complete oracle is used only for offline scoring;
3. candidate-search budget: the beam width and the candidate-check cap are
   scaled down relative to the v10 setting.

The module is import-light on purpose (no torch at import time) so the matrix
can be planned without building a model or touching a dataset.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .core import ArithmeticKB, Evidence
from .evaluate import symbol_index
from .prepare import digest, idx_bytes, stable_key, write_json

PROTOCOL_VERSION = "frt-abl-robustness-v1"
DEFAULT_BASELINE_METHOD = "frt-abl-v10-20261028"
COMPLETE_TRAINING_STATUS = "TRAINING_COMPLETED_TEST_NOT_RUN"
SPLITS = ("validation", "test")
STAGES = ("single", "combined", "all")

PREDICTION_FIELDS = (
    "sample_id", "length", "target", "truth_symbols",
    "raw_symbols", "raw_value", "raw_task_correct", "raw_symbol_matches",
    "raw_sequence_correct", "raw_syntax_valid", "raw_unresolved",
    "native_symbols", "native_value", "native_task_correct", "native_symbol_matches",
    "native_sequence_correct", "covered", "search_complete", "candidates_checked",
    "unknown_candidates", "n_supported", "margin",
)


# ---------------------------------------------------------------------------
# Perturbation specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """One cell of the robustness matrix (clean / single-factor / combined)."""

    condition_id: str
    stage: str
    noise_kind: str
    noise_sigma: float
    knowledge_ratio: float
    budget_scale: float
    beam_size: int
    max_checks: int

    def __post_init__(self):
        if not self.condition_id:
            raise ValueError("A condition needs a non-empty id")
        if self.stage not in {"clean", "single", "combined"}:
            raise ValueError(f"Unknown condition stage: {self.stage}")
        if self.noise_kind not in {"none", "gaussian", "salt_pepper"}:
            raise ValueError(f"Unknown noise kind: {self.noise_kind}")
        if not 0.0 <= self.noise_sigma <= 0.5:
            raise ValueError("Noise strength must lie in [0, 0.5]")
        if self.noise_kind == "none" and self.noise_sigma != 0.0:
            raise ValueError("A noise-free condition must have zero strength")
        if self.noise_kind != "none" and self.noise_sigma <= 0.0:
            raise ValueError("A noisy condition needs a positive strength")
        if not 0.0 <= self.knowledge_ratio < 1.0:
            raise ValueError("Knowledge-missing ratio must lie in [0, 1)")
        if not 0.0 < self.budget_scale <= 1.0:
            raise ValueError("Search-budget scale must lie in (0, 1]")
        if self.beam_size < 1 or self.max_checks < 1:
            raise ValueError("Search budgets must be positive")

    def as_dict(self):
        return {"condition_id": self.condition_id, "stage": self.stage,
                "noise_kind": self.noise_kind, "noise_sigma": self.noise_sigma,
                "knowledge_ratio": self.knowledge_ratio,
                "budget_scale": self.budget_scale,
                "beam_size": self.beam_size, "max_checks": self.max_checks}


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same_number(a, b):
    return _is_number(a) and _is_number(b) and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)


def _contains(values, target):
    return any(_same_number(value, target) for value in values)


def _check_levels(values, name, upper, allow_zero=True, descending=False):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    if any(not _is_number(v) or not 0 <= float(v) <= upper for v in values):
        raise ValueError(f"{name} must contain numbers in [0, {upper}]")
    levels = [float(v) for v in values]
    if len(set(levels)) != len(levels):
        raise ValueError(f"{name} must be unique")
    ordered = sorted(levels, reverse=descending)
    if levels != ordered:
        raise ValueError(f"{name} must be sorted {'descending' if descending else 'ascending'}")
    if allow_zero and 0.0 not in levels:
        raise ValueError(f"{name} must include the clean 0.0 level")
    return levels


def validate_robustness_config(spec):
    """Validate the declared grids; no filesystem access, no model."""
    if spec.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Robustness protocol_version must be {PROTOCOL_VERSION}")
    baseline = spec.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("Missing baseline block")
    if baseline.get("variant") != "full":
        raise ValueError("The robustness baseline is the full method only")
    if not isinstance(baseline.get("method_version"), str) or not baseline["method_version"]:
        raise ValueError("Missing baseline method_version")
    if not isinstance(baseline.get("runs_root"), str) or not baseline["runs_root"]:
        raise ValueError("Missing baseline runs_root")
    seeds = spec.get("seeds")
    if (not isinstance(seeds, (list, tuple)) or not seeds
            or any(not isinstance(s, int) or s not in (0, 1, 2, 3, 4) for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a non-empty unique subset of 0..4")
    if spec.get("split") not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    for key in ("output_root", "summary_root"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError(f"Missing {key}")

    noise = spec.get("noise")
    if not isinstance(noise, dict) or noise.get("kind") not in {"gaussian", "salt_pepper"}:
        raise ValueError("noise.kind must be gaussian or salt_pepper")
    if not isinstance(noise.get("seed_salt"), str) or not noise["seed_salt"]:
        raise ValueError("Missing noise.seed_salt")
    levels_by_kind = noise.get("levels")
    if not isinstance(levels_by_kind, dict) or set(levels_by_kind) != {"gaussian", "salt_pepper"}:
        raise ValueError("noise.levels must declare both gaussian and salt_pepper")
    for kind, levels in levels_by_kind.items():
        _check_levels(levels, f"noise.levels.{kind}", upper=0.5)

    knowledge = spec.get("knowledge")
    if not isinstance(knowledge, dict):
        raise ValueError("Missing knowledge block")
    _check_levels(knowledge.get("levels"), "knowledge.levels", upper=0.999)
    if any(float(v) >= 1.0 for v in knowledge["levels"]):
        raise ValueError("knowledge.levels must stay below 1.0")
    if not isinstance(knowledge.get("units"), dict):
        raise ValueError("Missing knowledge.units documentation")

    budget = spec.get("search_budget")
    if not isinstance(budget, dict):
        raise ValueError("Missing search_budget block")
    scales = _check_levels(budget.get("levels"), "search_budget.levels", upper=1.0,
                           allow_zero=False, descending=True)
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("search_budget.levels must be strictly positive")
    if not _same_number(scales[0], 1.0):
        raise ValueError("The first search-budget level must be the full 1.0 budget")
    for task in ("mnist-add2", "hwf"):
        ladder = budget.get(task)
        if not isinstance(ladder, dict):
            raise ValueError(f"Missing search_budget.{task} ladder")
        beams, checks = ladder.get("beam_size"), ladder.get("max_checks")
        if not isinstance(beams, list) or not isinstance(checks, list):
            raise ValueError(f"search_budget.{task} needs beam_size and max_checks lists")
        if len(beams) != len(scales) or len(checks) != len(scales):
            raise ValueError(f"search_budget.{task} ladders must match the declared levels")
        if any(not isinstance(v, int) or v < 1 for v in beams + checks):
            raise ValueError(f"search_budget.{task} values must be positive integers")
    combined = spec.get("combined")
    if not isinstance(combined, dict):
        raise ValueError("Missing combined block")
    chosen_noise = levels_by_kind[noise["kind"]]
    for key, upper in (("noise_sigma", 0.5), ("knowledge_ratio", 0.999), ("budget_scale", 1.0)):
        values = combined.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"combined.{key} must be a non-empty list")
        if any(not _is_number(v) or float(v) <= 0 or float(v) > upper for v in values):
            raise ValueError(f"combined.{key} values are out of range")
    for sigma in combined["noise_sigma"]:
        if not _contains(chosen_noise, sigma) or float(sigma) == 0.0:
            raise ValueError("combined.noise_sigma must reuse positive single-factor levels")
    for ratio in combined["knowledge_ratio"]:
        if not _contains(knowledge["levels"], ratio) or float(ratio) == 0.0:
            raise ValueError("combined.knowledge_ratio must reuse positive single-factor levels")
    for scale in combined["budget_scale"]:
        if not _contains(scales, scale):
            raise ValueError("combined.budget_scale must reuse declared budget levels")
    return spec


def load_robustness_config(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return validate_robustness_config(spec)


def _budget_pair(spec, task, scale):
    levels = spec["search_budget"]["levels"]
    ladder = spec["search_budget"][task]
    for level, beam, checks in zip(levels, ladder["beam_size"], ladder["max_checks"]):
        if _same_number(level, scale):
            return int(beam), int(checks)
    raise ValueError(f"Undeclared search-budget scale {scale} for {task}")


def _format_level(value):
    return f"{float(value):g}"


def build_conditions(spec, task, stage="single"):
    """Build the pre-registered condition list for one task and stage."""
    validate_robustness_config(spec)
    if task not in ("mnist-add2", "hwf"):
        raise ValueError(f"Unknown task: {task}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    noise_kind = spec["noise"]["kind"]
    noise_levels = spec["noise"]["levels"][noise_kind]
    knowledge_levels = spec["knowledge"]["levels"]
    budget_levels = spec["search_budget"]["levels"]

    def make(condition_id, condition_stage, kind, sigma, ratio, scale):
        beam, checks = _budget_pair(spec, task, scale)
        return Condition(condition_id, condition_stage, kind, float(sigma), float(ratio),
                         float(scale), beam, checks)

    clean_beam, clean_checks = _budget_pair(spec, task, 1.0)
    conditions = [Condition("clean", "clean", "none", 0.0, 0.0, 1.0,
                            clean_beam, clean_checks)]
    if stage in {"single", "all"}:
        for sigma in noise_levels:
            if float(sigma) == 0.0:
                continue
            conditions.append(make(f"noise-{noise_kind}-{_format_level(sigma)}", "single",
                                   noise_kind, sigma, 0.0, 1.0))
        for ratio in knowledge_levels:
            if float(ratio) == 0.0:
                continue
            conditions.append(make(f"knowledge-{_format_level(ratio)}", "single",
                                   "none", 0.0, ratio, 1.0))
        for scale in budget_levels:
            if _same_number(scale, 1.0):
                continue
            conditions.append(make(f"budget-{_format_level(scale)}", "single",
                                   "none", 0.0, 0.0, scale))
    if stage in {"combined", "all"}:
        grid = spec["combined"]
        for sigma in grid["noise_sigma"]:
            for ratio in grid["knowledge_ratio"]:
                for scale in grid["budget_scale"]:
                    condition_id = (f"combo-n{_format_level(sigma)}-k{_format_level(ratio)}"
                                    f"-b{_format_level(scale)}")
                    conditions.append(make(condition_id, "combined", noise_kind,
                                           sigma, ratio, scale))
    seen = set()
    unique = []
    for condition in conditions:
        if condition.condition_id in seen:
            raise ValueError(f"Duplicate condition id: {condition.condition_id}")
        seen.add(condition.condition_id)
        unique.append(condition)
    return unique


# ---------------------------------------------------------------------------
# Knowledge incompleteness
# ---------------------------------------------------------------------------

def knowledge_units(task):
    """The declared unit of knowledge that can be removed.

    * addition: one fact ``(a, b)`` per ordered digit pair -> 100 units;
    * handwritten arithmetic: one operator-application cell
      ``(operator, right-operand digit 1..9)`` -> 36 units.  This is the
      eval-only generalisation of the addition missing-fact protocol; v10
      training never saw a masked operator table.
    """
    if task == "mnist-add2":
        return tuple((a, b) for a in range(10) for b in range(10))
    if task == "hwf":
        return tuple((op, digit) for op in "+-*/" for digit in range(1, 10))
    raise ValueError(f"Unknown task: {task}")


def masked_units(task, ratio):
    """Deterministic, nested removal set for a missing-knowledge ratio.

    Units are ordered by a stable hash and the first ``round(ratio * N)`` are
    removed.  Ratios are therefore nested (0.1 is a subset of 0.2), so the
    degradation curve is not confounded by a fresh random mask at every level.
    """
    if not 0.0 <= float(ratio) < 1.0:
        raise ValueError("Knowledge-missing ratio must lie in [0, 1)")
    units = knowledge_units(task)
    if float(ratio) == 0.0:
        return frozenset()
    ordered = sorted(units, key=lambda unit: stable_key(f"{task}:{unit[0]}:{unit[1]}",
                                                        "knowledge-ablation"))
    count = min(len(units), max(1, int(round(float(ratio) * len(units)))))
    return frozenset(ordered[:count])


class MaskedKnowledge:
    """KB view with a removed subset; unknown stays unknown.

    The interface is the subset used by ``core.search``: ``task``, ``symbols``,
    ``syntax_valid``, ``domains`` and ``check``.  ``value`` never consults the
    complete arithmetic oracle for a removed unit and never guesses.
    """

    def __init__(self, task, symbols, knowledge_ratio=0.0):
        self.base = ArithmeticKB(task, symbols)
        self.task = task
        self.symbols = tuple(symbols)
        self.tolerance = self.base.tolerance
        self.knowledge_ratio = float(knowledge_ratio)
        self.masked = masked_units(task, knowledge_ratio)

    def syntax_valid(self, labels):
        return self.base.syntax_valid(labels)

    def domains(self, length):
        return self.base.domains(length)

    def value(self, labels):
        if self.task == "mnist-add2":
            if tuple(labels) in self.masked:
                return None
            return self.base.value(labels)
        return self._hwf_value(labels)

    def _hwf_value(self, labels):
        if not labels or len(labels) % 2 == 0:
            return None
        digits = tuple("123456789")
        operators = ("+", "-", "*", "/")
        syms = []
        for position, index in enumerate(labels):
            if not isinstance(index, int) or not 0 <= index < len(self.symbols):
                return None
            symbol = self.symbols[index]
            if symbol not in (digits if position % 2 == 0 else operators):
                return None
            syms.append(symbol)
        # Standard precedence, left associative, rational intermediate values.
        total, term, sign = Fraction(0), Fraction(int(syms[0])), 1
        for position in range(1, len(syms), 2):
            operator, number = syms[position], int(syms[position + 1])
            if (operator, number) in self.masked:
                return None
            if operator == "*":
                term *= number
            elif operator == "/":
                term /= number
            else:
                total += sign * term
                term, sign = number, (1 if operator == "+" else -1)
        return total + sign * term

    def check(self, labels, target):
        value = self.value(labels)
        if value is None:
            return Evidence.UNRESOLVED
        return (Evidence.SUPPORTED if math.isclose(float(value), float(target),
                rel_tol=0.0, abs_tol=self.tolerance) else Evidence.REFUTED)


# ---------------------------------------------------------------------------
# Perception noise
# ---------------------------------------------------------------------------

class ImageNoise:
    """Deterministic per-image corruption, applied after the standard loader.

    The Gaussian field seed does not depend on sigma, so increasing sigma adds
    amplitude to the *same* noise realisation; degradation curves are nested.
    """

    def __init__(self, kind, sigma, salt="frt-abl-robustness-v1"):
        if kind not in {"none", "gaussian", "salt_pepper"}:
            raise ValueError(f"Unknown noise kind: {kind}")
        if not 0.0 <= float(sigma) <= 0.5:
            raise ValueError("Noise strength must lie in [0, 0.5]")
        self.kind, self.sigma, self.salt = kind, float(sigma), salt
        self._seeds = {}

    def _seed(self, image_id):
        cached = self._seeds.get(image_id)
        if cached is None:
            key = f"{self.salt}|{self.kind}|{image_id}"
            cached = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**63 - 1)
            self._seeds[image_id] = cached
        return cached

    def apply(self, image, image_id):
        import torch

        if self.kind == "none" or self.sigma <= 0.0:
            return image
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed(image_id))
        if self.kind == "gaussian":
            noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
            return torch.clamp(image + self.sigma * noise, 0.0, 1.0)
        probability = self.sigma
        draw = torch.rand(image.shape, generator=generator)
        corrupted = image.clone()
        corrupted[draw < probability / 2] = 0.0
        corrupted[draw >= 1.0 - probability / 2] = 1.0
        return corrupted


class NoisyImageStore:
    """Compose the standard ImageStore with a noise model (lazy torch import)."""

    def __init__(self, cfg, noise, store=None):
        if store is None:
            from .runtime import ImageStore
            store = ImageStore(cfg)
        self.store, self.noise = store, noise

    def get(self, image_id):
        return self.noise.apply(self.store.get(image_id), image_id)

    def batch(self, samples, device):
        import torch

        lengths = [len(sample["images"]) for sample in samples]
        images = torch.stack([self.get(image_id) for sample in samples
                              for image_id in sample["images"]]).to(device)
        return images, lengths


# ---------------------------------------------------------------------------
# Frozen-checkpoint evaluation
# ---------------------------------------------------------------------------

def load_frozen_model(cfg, checkpoint_path, require_variant="full"):
    """Load a v10 checkpoint with the same identity checks as evaluate.py."""
    import torch

    from .runtime import SymbolCNN

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint["prepared_manifest_sha256"] != digest(Path(cfg["prepared_root"]) / "manifest.json"):
        raise ValueError("Checkpoint belongs to another prepared data manifest")
    saved = checkpoint["config"]
    for key in ("task", "symbols", "prepared_root", "data_root"):
        if saved[key] != cfg[key]:
            raise ValueError(f"Checkpoint/config mismatch: {key}")
    if require_variant is not None and saved.get("variant") != require_variant:
        raise ValueError(f"Robustness baseline must be the '{require_variant}' checkpoint")
    model = SymbolCNN(len(cfg["symbols"]))
    model.load_state_dict(checkpoint["teacher"])
    model.eval()
    return model, checkpoint


def discover_checkpoint(runs_root, task, seed, method_version=DEFAULT_BASELINE_METHOD,
                        variant="full"):
    """Locate exactly one completed v10 checkpoint for a task/seed."""
    runs_root = Path(runs_root)
    pattern = f"{method_version}__{task}__{variant}__seed-{seed}__*"
    candidates = []
    for run_dir in sorted(p for p in runs_root.glob(pattern) if p.is_dir()):
        try:
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            candidates.append({"run_dir": str(run_dir), "reason": repr(error)})
            continue
        if not (run_dir / "best.pt").is_file():
            candidates.append({"run_dir": str(run_dir), "reason": "best.pt missing"})
            continue
        if (cfg.get("method_version") != method_version or cfg.get("task") != task
                or cfg.get("variant") != variant or cfg.get("seed") != seed):
            candidates.append({"run_dir": str(run_dir), "reason": "run metadata mismatch"})
            continue
        if (status.get("status") != COMPLETE_TRAINING_STATUS
                or status.get("stopped_reason") is not None):
            candidates.append({"run_dir": str(run_dir),
                               "reason": f"status={status.get('status')} "
                                         f"stopped_reason={status.get('stopped_reason')}"})
            continue
        candidates.append({
            "run_dir": str(run_dir), "checkpoint": str(run_dir / "best.pt"),
            "run_id": run_dir.name, "status": status.get("status"),
            "best_validation_accuracy": status.get("best_validation_accuracy"),
            "optimizer_steps": status.get("optimizer_steps"),
            "valid": True})
    usable = [item for item in candidates if item.get("valid")]
    if len(usable) == 1:
        return usable[0]
    if not candidates:
        raise FileNotFoundError(
            f"No v10 full run found for {task} seed {seed} under {runs_root}")
    detail = json.dumps(candidates, ensure_ascii=False, indent=2)
    if not usable:
        raise FileNotFoundError(
            f"No completed v10 full checkpoint for {task} seed {seed}. Candidates:\n{detail}")
    raise ValueError(
        f"Multiple completed v10 full checkpoints for {task} seed {seed}; "
        f"pass --checkpoints explicitly. Candidates:\n{detail}")


def describe_checkpoint(checkpoint_path):
    """Metadata for an explicitly supplied checkpoint directory."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    info = {"checkpoint": str(path), "run_dir": str(path.parent), "run_id": path.parent.name,
            "checkpoint_sha256": digest(path)}
    for name in ("status.json", "config.json"):
        candidate = path.parent / name
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if name == "status.json":
                info["status"] = payload.get("status")
                info["best_validation_accuracy"] = payload.get("best_validation_accuracy")
                info["optimizer_steps"] = payload.get("optimizer_steps")
            else:
                info["method_version"] = payload.get("method_version")
    info.setdefault("status", None)
    info.setdefault("best_validation_accuracy", None)
    info.setdefault("optimizer_steps", None)
    return info


def load_split(prepared_root, split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = Path(prepared_root) / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def truth_labels(sample, cfg, cache):
    """Golden labels for offline scoring only; never used by the abducer."""
    if cfg["task"] != "mnist-add2":
        return [symbol_index(path.split("/")[0], cfg["symbols"]) for path in sample["images"]]
    prefix = sample["images"][0].split(":")[0]
    name = "train-labels-idx1-ubyte" if prefix == "train" else "t10k-labels-idx1-ubyte"
    if name not in cache:
        cache[name] = idx_bytes(Path(cfg["data_root"]), name)[8:]
    labels = cache[name]
    return [labels[int(image_id.split(":")[1])] for image_id in sample["images"]]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


class MetricAccumulator:
    """Counts computed from per-sample records; pure and unit-testable."""

    def __init__(self):
        self.n = 0
        self.task_correct = 0
        self.symbol_correct = 0
        self.symbol_total = 0
        self.sequence_correct = 0
        self.raw_task_correct = 0
        self.raw_symbol_correct = 0
        self.raw_symbol_total = 0
        self.raw_sequence_correct = 0
        self.covered = 0
        self.covered_correct = 0
        self.raw_error = 0
        self.corrected = 0
        self.raw_right = 0
        self.corrupted = 0
        self.raw_unresolved = 0
        self.raw_syntax_invalid = 0
        self.candidates_checked = 0
        self.search_complete = 0
        self.unknown_candidates = 0
        self.by_length = defaultdict(lambda: {"n": 0, "task_correct": 0,
                                              "symbol_correct": 0, "symbol_total": 0})

    def add(self, record):
        length = int(record["length"])
        self.n += 1
        self.task_correct += int(record["native_task_correct"])
        self.symbol_correct += int(record["native_symbol_matches"])
        self.symbol_total += length
        self.sequence_correct += int(record["native_sequence_correct"])
        self.raw_task_correct += int(record["raw_task_correct"])
        self.raw_symbol_correct += int(record["raw_symbol_matches"])
        self.raw_symbol_total += length
        self.raw_sequence_correct += int(record["raw_sequence_correct"])
        self.covered += int(record["covered"])
        self.covered_correct += int(record["covered"]) * int(record["native_task_correct"])
        raw_ok = bool(record["raw_task_correct"])
        native_ok = bool(record["native_task_correct"])
        if not raw_ok:
            self.raw_error += 1
            self.corrected += int(native_ok)
        else:
            self.raw_right += 1
            self.corrupted += int(not native_ok)
        self.raw_unresolved += int(record["raw_unresolved"])
        self.raw_syntax_invalid += int(not record["raw_syntax_valid"])
        self.candidates_checked += int(record["candidates_checked"])
        self.search_complete += int(record["search_complete"])
        self.unknown_candidates += int(record["unknown_candidates"])
        bucket = self.by_length[length]
        bucket["n"] += 1
        bucket["task_correct"] += int(record["native_task_correct"])
        bucket["symbol_correct"] += int(record["native_symbol_matches"])
        bucket["symbol_total"] += length

    def finalize(self):
        by_length = {}
        for length in sorted(self.by_length):
            bucket = self.by_length[length]
            by_length[str(length)] = {
                "n": bucket["n"],
                "task_accuracy": _ratio(bucket["task_correct"], bucket["n"]),
                "symbol_accuracy": _ratio(bucket["symbol_correct"], bucket["symbol_total"]),
            }
        return {
            "n": self.n,
            "task_correct": self.task_correct,
            "task_accuracy": _ratio(self.task_correct, self.n),
            "symbol_correct": self.symbol_correct,
            "symbol_total": self.symbol_total,
            "symbol_accuracy": _ratio(self.symbol_correct, self.symbol_total),
            "sequence_correct": self.sequence_correct,
            "sequence_accuracy": _ratio(self.sequence_correct, self.n),
            "raw_argmax_task_correct": self.raw_task_correct,
            "raw_argmax_task_accuracy": _ratio(self.raw_task_correct, self.n),
            "raw_argmax_symbol_correct": self.raw_symbol_correct,
            "raw_argmax_symbol_total": self.raw_symbol_total,
            "raw_argmax_symbol_accuracy": _ratio(self.raw_symbol_correct,
                                                 self.raw_symbol_total),
            "raw_argmax_sequence_correct": self.raw_sequence_correct,
            "raw_argmax_sequence_accuracy": _ratio(self.raw_sequence_correct, self.n),
            "covered": self.covered,
            "coverage": _ratio(self.covered, self.n),
            "fallback_count": self.n - self.covered,
            "undecided_rate": _ratio(self.n - self.covered, self.n),
            "covered_correct": self.covered_correct,
            "covered_task_accuracy": _ratio(self.covered_correct, self.covered),
            "raw_error_count": self.raw_error,
            "error_correction_count": self.corrected,
            "error_correction_rate": _ratio(self.corrected, self.raw_error),
            "raw_correct_count": self.raw_right,
            "corrupted_count": self.corrupted,
            "corruption_rate": _ratio(self.corrupted, self.raw_right),
            "correction_precision": _ratio(self.corrected,
                                           self.corrected + self.corrupted),
            "raw_argmax_unresolved_count": self.raw_unresolved,
            "raw_argmax_unresolved_rate": _ratio(self.raw_unresolved, self.n),
            "raw_argmax_syntax_invalid_count": self.raw_syntax_invalid,
            "raw_argmax_syntax_invalid_rate": _ratio(self.raw_syntax_invalid, self.n),
            "mean_candidates_checked": _ratio(self.candidates_checked, self.n),
            "search_complete_rate": _ratio(self.search_complete, self.n),
            "mean_unknown_candidates": _ratio(self.unknown_candidates, self.n),
            "by_length": by_length,
        }


def evaluate_condition(cfg, model, condition, samples, output_dir, seed, split,
                       noise_salt="frt-abl-robustness-v1", provenance=None,
                       batch_size=None, device="cpu"):
    """Evaluate one perturbation condition; writes condition/metrics/predictions."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if condition.stage == "clean":
        if condition.noise_sigma or condition.knowledge_ratio or condition.budget_scale != 1.0:
            raise ValueError("The clean condition must not perturb anything")
    import csv
    import torch

    from .core import ArithmeticKB, abduce

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "condition.json", {
        "protocol_version": PROTOCOL_VERSION, "task": cfg["task"], "seed": seed,
        "split": split, "condition": condition.as_dict(), "n_samples": len(samples),
        "provenance": provenance or {},
        "note": "Frozen v10 full checkpoint; perturbations are evaluation-only."})
    write_json(output_dir / "status.json", {"status": "RUNNING", "split": split,
                                            "condition_id": condition.condition_id})
    kb = MaskedKnowledge(cfg["task"], cfg["symbols"], condition.knowledge_ratio)
    oracle = ArithmeticKB(cfg["task"], cfg["symbols"])
    noise = ImageNoise(condition.noise_kind, condition.noise_sigma, noise_salt)
    store = NoisyImageStore(cfg, noise)
    label_cache = {}
    accumulator = MetricAccumulator()
    batch_size = batch_size or cfg["batch_size"]
    started = time.perf_counter()
    try:
        predictions_path = output_dir / "predictions.csv"
        with predictions_path.open("w", newline="", encoding="utf-8") as handle, torch.no_grad():
            writer = csv.writer(handle)
            writer.writerow(PREDICTION_FIELDS)
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                images, lengths = store.batch(batch, device)
                logits, _ = model(images)
                probabilities = logits.softmax(-1).tolist()
                predicted = logits.argmax(-1).tolist()
                offset = 0
                for sample, length in zip(batch, lengths):
                    raw = tuple(predicted[offset:offset + length])
                    rows = probabilities[offset:offset + length]
                    offset += length
                    truth = truth_labels(sample, cfg, label_cache)
                    native, covered, result = abduce(
                        rows, sample["target"], kb,
                        beam_size=condition.beam_size, max_checks=condition.max_checks,
                        edit_penalty=cfg["edit_penalty"])
                    raw_value = oracle.value(raw)
                    native_value = oracle.value(native)
                    raw_task_correct = (raw_value is not None and math.isclose(
                        float(raw_value), float(sample["target"]), rel_tol=0.0, abs_tol=1e-8))
                    native_task_correct = (native_value is not None and math.isclose(
                        float(native_value), float(sample["target"]), rel_tol=0.0, abs_tol=1e-8))
                    raw_matches = sum(a == b for a, b in zip(raw, truth))
                    native_matches = sum(a == b for a, b in zip(native, truth))
                    # A grammar-invalid raw argmax is not a knowledge-unresolved
                    # candidate (same distinction as core/CODE_REVIEW P0-1).
                    raw_syntax_valid = kb.syntax_valid(raw)
                    raw_unresolved = raw_syntax_valid and kb.value(raw) is None
                    record = {
                        "sample_id": sample["id"], "length": length,
                        "target": sample["target"], "truth_symbols": json.dumps(list(truth)),
                        "raw_symbols": json.dumps(list(raw)),
                        "raw_value": None if raw_value is None else float(raw_value),
                        "raw_task_correct": bool(raw_task_correct),
                        "raw_symbol_matches": raw_matches,
                        "raw_sequence_correct": raw_matches == length,
                        "raw_syntax_valid": bool(raw_syntax_valid),
                        "raw_unresolved": bool(raw_unresolved),
                        "native_symbols": json.dumps(list(native)),
                        "native_value": None if native_value is None else float(native_value),
                        "native_task_correct": bool(native_task_correct),
                        "native_symbol_matches": native_matches,
                        "native_sequence_correct": native_matches == length,
                        "covered": bool(covered),
                        "search_complete": bool(result.complete),
                        "candidates_checked": int(result.checked),
                        "unknown_candidates": int(result.unknown_count),
                        "n_supported": result.n_supported,
                        "margin": float(result.margin),
                    }
                    accumulator.add(record)
                    writer.writerow([record[field] for field in PREDICTION_FIELDS])
        elapsed = time.perf_counter() - started
        metrics = accumulator.finalize()
        metrics.update({
            "status": "ROBUSTNESS_EVAL_COMPLETED",
            "protocol_version": PROTOCOL_VERSION,
            "task": cfg["task"], "seed": seed, "split": split,
            "condition": condition.as_dict(),
            "n_samples": len(samples),
            "seconds": elapsed,
            "samples_per_second": _ratio(len(samples), elapsed),
            "checkpoint": (provenance or {}).get("checkpoint"),
            "checkpoint_sha256": (provenance or {}).get("checkpoint_sha256"),
            "knowledge_units_total": len(knowledge_units(cfg["task"])),
            "knowledge_units_masked": len(kb.masked),
            "note": ("Frozen v10 full checkpoint evaluated under an evaluation-only "
                     "perturbation; native predictions use the perturbed KB/search, "
                     "scoring uses the complete offline oracle."),
        })
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "status.json", {
            "status": "ROBUSTNESS_EVAL_COMPLETED", "split": split,
            "condition_id": condition.condition_id,
            "n_samples": len(samples), "task_accuracy": metrics["task_accuracy"]})
        return metrics
    except BaseException as error:
        write_json(output_dir / "status.json", {
            "status": "FAILED", "split": split,
            "condition_id": condition.condition_id, "error": repr(error)})
        raise


# ---------------------------------------------------------------------------
# Summary and degradation relative to the clean condition
# ---------------------------------------------------------------------------

_AGGREGATED_METRICS = (
    "task_accuracy", "symbol_accuracy", "sequence_accuracy",
    "raw_argmax_task_accuracy", "raw_argmax_symbol_accuracy",
    "error_correction_rate", "correction_precision", "corruption_rate",
    "coverage", "undecided_rate", "raw_argmax_unresolved_rate",
    "mean_candidates_checked", "search_complete_rate",
)

_DELTA_METRICS = (
    "task_accuracy", "symbol_accuracy", "error_correction_rate",
    "correction_precision", "coverage", "undecided_rate", "raw_argmax_task_accuracy",
)


def _aggregate(values):
    clean = [float(value) for value in values
             if _is_number(value) and math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(clean),
            "std": statistics.stdev(clean) if len(clean) > 1 else None,
            "n": len(clean)}


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _condition_sort_key(row):
    stage_order = {"clean": 0, "single": 1, "combined": 2}
    condition = row["condition"]
    factor_prefix = row["condition_id"].split("-")[0]
    factor_order = {"clean": 0, "noise": 1, "knowledge": 2, "budget": 3, "combo": 4}
    return (row["task"], stage_order.get(condition.get("stage"), 3),
            factor_order.get(factor_prefix, 5),
            condition.get("noise_sigma") or 0.0,
            condition.get("knowledge_ratio") or 0.0,
            condition.get("budget_scale") or 0.0,
            row["condition_id"])


def summarize(run_root):
    """Aggregate completed conditions and compute deltas vs the clean run."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        raise FileNotFoundError(str(run_root))
    records = []
    for task_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in task_dir.glob("seed-*") if p.is_dir()):
            try:
                seed = int(seed_dir.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            for condition_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                metrics = _load_json(condition_dir / "metrics.json")
                condition = _load_json(condition_dir / "condition.json")
                if not metrics or not condition:
                    continue
                if metrics.get("status") != "ROBUSTNESS_EVAL_COMPLETED":
                    continue
                records.append({"task": metrics.get("task", task_dir.name),
                                "seed": seed,
                                "condition": condition["condition"],
                                "condition_id": condition["condition"]["condition_id"],
                                "stage": condition["condition"]["stage"],
                                "metrics": metrics})
    if not records:
        raise ValueError(f"No completed robustness evaluations under {run_root}")
    seen = set()
    for record in records:
        key = (record["task"], record["seed"], record["condition_id"])
        if key in seen:
            raise ValueError(f"Duplicate completed condition: {key}")
        seen.add(key)
    clean = {(record["task"], record["seed"]): record for record in records
             if record["condition_id"] == "clean"}
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["condition_id"])].append(record)
    rows = []
    for (task, condition_id), group in grouped.items():
        group.sort(key=lambda item: item["seed"])
        per_seed, metric_values, delta_values = [], defaultdict(list), defaultdict(list)
        for record in group:
            clean_record = clean.get((task, record["seed"]))
            per_seed_metrics = {key: record["metrics"].get(key) for key in _AGGREGATED_METRICS}
            per_seed_deltas = {}
            for key in _DELTA_METRICS:
                base_value = clean_record["metrics"].get(key) if clean_record else None
                current = record["metrics"].get(key)
                per_seed_deltas[key] = (current - base_value
                                        if _is_number(current) and _is_number(base_value) else None)
            per_seed.append({"seed": record["seed"],
                             "n_samples": record["metrics"].get("n"),
                             "metrics": per_seed_metrics,
                             "deltas": per_seed_deltas})
            for key in _AGGREGATED_METRICS:
                metric_values[key].append(per_seed_metrics[key])
            for key in _DELTA_METRICS:
                delta_values[key].append(per_seed_deltas[key])
        first = group[0]
        rows.append({
            "task": task, "condition_id": condition_id,
            "condition": first["condition"],
            "seeds": [item["seed"] for item in group],
            "n_seeds": len(group),
            "per_seed": per_seed,
            "metrics": {key: _aggregate(values) for key, values in metric_values.items()},
            "deltas": {key: _aggregate(values) for key, values in delta_values.items()},
        })
    rows.sort(key=_condition_sort_key)
    split = sorted({record["metrics"].get("split") for record in records
                    if record["metrics"].get("split")})
    baseline_document = _load_json(run_root / "baseline.json")
    baseline_consistency = []
    if isinstance(baseline_document, dict):
        for evaluation in baseline_document.get("evaluations", []):
            if not isinstance(evaluation, dict):
                continue
            task, seed = evaluation.get("task"), evaluation.get("seed")
            recorded = evaluation.get("best_validation_accuracy")
            clean_record = clean.get((task, seed))
            observed = (clean_record["metrics"].get("raw_argmax_task_accuracy")
                        if clean_record else None)
            clean_split = clean_record["metrics"].get("split") if clean_record else None
            comparable = clean_split == "validation"
            baseline_consistency.append({
                "task": task, "seed": seed,
                "recorded_best_validation_accuracy": recorded,
                "observed_clean_raw_argmax_task_accuracy": observed,
                "delta": (observed - recorded
                          if comparable and _is_number(observed)
                          and _is_number(recorded) else None),
                "split": clean_split, "comparable": comparable,
            })
    return {
        "status": "ROBUSTNESS_SUMMARY",
        "protocol_version": PROTOCOL_VERSION,
        "run_root": str(run_root),
        "split": split[0] if len(split) == 1 else split,
        "baseline": baseline_document,
        "baseline_consistency": baseline_consistency,
        "n_conditions": len(rows),
        "n_seeds": sorted({record["seed"] for record in records}),
        "conditions": rows,
        "note": ("Deltas are condition minus clean on the same seed; negative accuracy "
                 "deltas are degradation. Validation is the default split; a formal-test "
                 "read requires --approve-formal-test."),
    }


def write_summary(run_root, summary_root, overwrite=False):
    summary = summarize(run_root)
    output = Path(summary_root)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite robustness summary: {output}")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "summary.csv"):
            stale = output / name
            if stale.is_file():
                stale.unlink()
    else:
        output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    fields = ["task", "condition_id", "stage", "noise_kind", "noise_sigma",
              "knowledge_ratio", "budget_scale", "beam_size", "max_checks", "n_seeds",
              "mean_task_accuracy", "delta_task_accuracy",
              "mean_symbol_accuracy", "delta_symbol_accuracy",
              "mean_error_correction_rate", "delta_error_correction_rate",
              "mean_undecided_rate", "delta_undecided_rate",
              "mean_correction_precision", "mean_corruption_rate",
              "mean_coverage", "mean_raw_argmax_task_accuracy"]
    import csv

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["conditions"]:
            condition = row["condition"]
            record = {
                "task": row["task"], "condition_id": row["condition_id"],
                "stage": condition.get("stage"),
                "noise_kind": condition.get("noise_kind"),
                "noise_sigma": condition.get("noise_sigma"),
                "knowledge_ratio": condition.get("knowledge_ratio"),
                "budget_scale": condition.get("budget_scale"),
                "beam_size": condition.get("beam_size"),
                "max_checks": condition.get("max_checks"),
                "n_seeds": row["n_seeds"],
            }
            for key in ("task_accuracy", "symbol_accuracy", "error_correction_rate",
                        "undecided_rate", "correction_precision", "corruption_rate",
                        "coverage", "raw_argmax_task_accuracy"):
                record[f"mean_{key}"] = row["metrics"][key]["mean"]
            for key in ("task_accuracy", "symbol_accuracy", "error_correction_rate",
                        "undecided_rate"):
                record[f"delta_{key}"] = row["deltas"][key]["mean"]
            writer.writerow(record)
    return summary
