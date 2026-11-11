"""Torch components. Import is inert: no model, data, device or process is created."""
from dataclasses import dataclass
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .core import (Candidate, Evidence, Region, SearchResult, decide, fuzzy_bounds,
                   search)
from .prepare import idx_bytes, stable_key
from .policies import curriculum, resolve_policy


class SymbolCNN(nn.Module):
    """New shared backbone for method/ablations; not claimed identical to baselines."""
    def __init__(self, classes):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(6, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 16, 3), nn.ReLU())
        self.feature = nn.Sequential(nn.Linear(16*4*4, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(84, classes))

    def forward(self, x):
        features = self.feature(self.conv(x).flatten(1))
        return self.head(features), features


class ImageStore:
    def __init__(self, cfg):
        self.root, self.task = Path(cfg["data_root"]), cfg["task"]
        self.arrays = {}

    def get(self, image_id):
        if self.task == "mnist-add2":
            split, idx = image_id.split(":")
            if split not in self.arrays:
                name = "train-images-idx3-ubyte" if split == "train" else "t10k-images-idx3-ubyte"
                b = idx_bytes(self.root, name)
                magic, count, rows, cols = struct.unpack(">IIII", b[:16])
                if magic != 2051 or (rows, cols) != (28, 28) or len(b) != 16+count*rows*cols:
                    raise ValueError("Invalid image IDX")
                self.arrays[split] = np.frombuffer(b, np.uint8, offset=16).reshape(count, 28, 28)
            arr = self.arrays[split][int(idx)].copy()
        else:
            # No label inference from directory names in a training data loader.
            with Image.open(self.root / "Handwritten_Math_Symbols" / image_id) as image:
                arr = np.asarray(image.convert("L").resize((28, 28), Image.Resampling.BILINEAR)).copy()
        return torch.from_numpy(arr).float().unsqueeze(0) / 255.0

    def batch(self, samples, device):
        lengths = [len(s["images"]) for s in samples]
        images = torch.stack([self.get(i) for s in samples for i in s["images"]]).to(device)
        return images, lengths


class Calibrator:
    """Low-capacity logistic calibration; labels are whole-sequence correctness."""
    def fit(self, x, y):
        x, y = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Calibration fit requires candidates")
        self.mean, self.scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-3)
        a = torch.cat([torch.ones(len(x), 1, dtype=x.dtype), (x-self.mean)/self.scale], 1)
        self.coef = torch.zeros(a.shape[1], dtype=x.dtype)
        self.coef[0] = math.log((float(y.sum())+1)/(len(y)-float(y.sum())+1))
        self.degenerate = len(torch.unique(y)) < 2
        if self.degenerate:
            return
        # Deterministic regularized Newton steps; independent of training RNG.
        reg = torch.eye(a.shape[1], dtype=x.dtype) * 0.01
        reg[0, 0] = 1e-6
        for _ in range(30):
            p = torch.sigmoid(a @ self.coef)
            grad = a.T @ (p-y)/len(y) + reg @ self.coef
            hess = a.T @ ((p*(1-p)).unsqueeze(1)*a)/len(y) + reg
            step = torch.linalg.solve(hess, grad)
            self.coef -= step.clamp(-2, 2)
            if step.abs().max() < 1e-7:
                break

    def predict(self, features):
        x = torch.as_tensor(features, dtype=torch.float64)
        a = torch.cat([torch.ones(len(x), 1, dtype=x.dtype), (x-self.mean)/self.scale], 1)
        return torch.sigmoid(a @ self.coef).tolist()

    def state_dict(self):
        return {"mean": self.mean, "scale": self.scale, "coef": self.coef, "degenerate": self.degenerate}


@dataclass
class Proposal:
    sample: dict
    labels: tuple | None
    evidence: Evidence
    reliability_features: list
    representation: torch.Tensor
    search: object
    stable_count: int
    age: int
    q: float = 0.0
    lower: float = 0.0
    upper: float = 1.0
    region: Region = Region.BOUNDARY
    weight: float = 0.0
    curriculum_weight: float = 1.0


def target_feature(target, length, cfg):
    """Bounded observable-target block with a length-independent norm share.

    The old `target/100` made large HWF results occupy up to 98.5% of the
    relation vector while typical results occupied <0.3%.  Here the target is
    mapped through a frozen robust tanh and then scaled so a saturated value
    occupies `target_share` of the squared norm of the whole vector for every
    expression length.  The centre/scale are frozen in the config from the
    prepared weak targets; `target_share` is a pre-registered sensitivity knob,
    not something to tune after seeing results.
    """
    share = float(cfg["target_share"])
    if not 0 < share < 1:
        raise ValueError("target_share must lie in (0,1)")
    if cfg["target_scale"] <= 0:
        raise ValueError("target_scale must be positive")
    if length < 1:
        raise ValueError("Expression length must be positive")
    z = math.tanh((float(target) - float(cfg["target_center"])) / float(cfg["target_scale"]))
    return math.sqrt(2 * length * share / (1 - share)) * z


# Process-pool search. The worker only needs the KB and the per-sample
# probabilities; it returns a compact summary without the candidate list.
_WORKER_KB = None


def search_worker_init(kb):
    global _WORKER_KB
    _WORKER_KB = kb
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def search_summary(result):
    best = result.best
    return (best.labels if best is not None else None,
            best.score if best is not None else 0.0,
            result.checked, result.complete, result.unknown_count, result.margin,
            result.generated, result.n_supported, result.original_valid)


def result_from_summary(summary):
    labels, score, checked, complete, unknown, margin, generated, n_supported, original_valid = summary
    best = Candidate(labels, score, Evidence.SUPPORTED) if labels is not None else None
    # Candidate lists are not needed after proposal construction; dropping them
    # keeps full-bank rounds near 0.5 GB instead of several GB.
    return SearchResult([], best, checked, complete, unknown, margin, generated,
                        n_supported, original_valid)


def search_worker(task):
    probabilities, target, beam_size, max_checks, edit_penalty = task
    if _WORKER_KB is None:
        raise RuntimeError("search worker used before initialization")
    result = search(probabilities, target, _WORKER_KB, beam_size=beam_size,
                    max_checks=max_checks, edit_penalty=edit_penalty)
    return search_summary(result)


@torch.no_grad()
def propose(samples, teacher, store, kb, cfg, state, retry_ids=(), pool=None):
    teacher.eval()
    device = next(teacher.parameters()).device
    out = []
    parallel = pool is not None and len(samples) >= cfg.get("search_parallel_min", 512)
    for start in range(0, len(samples), cfg["batch_size"]):
        batch = samples[start:start+cfg["batch_size"]]
        images, lengths = store.batch(batch, device)
        logits, embeddings = teacher(images)
        probs = torch.softmax(logits, -1).cpu()
        embeddings = F.normalize(embeddings.cpu(), dim=-1)
        offset = 0
        payloads = []
        tasks = []
        for sample, length in zip(batch, lengths):
            p, h = probs[offset:offset+length], embeddings[offset:offset+length]
            offset += length
            budget_factor = cfg["retry_factor"] if sample["id"] in retry_ids else 1
            probabilities = p.tolist()
            payloads.append((sample, length, p, h))
            tasks.append((probabilities, sample["target"],
                          cfg["beam_size"]*budget_factor,
                          cfg["max_checks"]*budget_factor, cfg["edit_penalty"]))
        if parallel:
            summaries = list(pool.map(search_worker, tasks,
                                       chunksize=cfg.get("search_chunksize", 16)))
        else:
            summaries = [search_summary(search(probabilities, target, kb,
                                               beam_size=beam_size,
                                               max_checks=max_checks,
                                               edit_penalty=edit_penalty))
                         for probabilities, target, beam_size, max_checks, edit_penalty in tasks]
        for (sample, length, p, h), summary in zip(payloads, summaries):
            result = result_from_summary(summary)
            labels = result.best.labels if result.best else None
            previous = state.get(sample["id"], {})
            stable = previous.get("stable_count", 0)+1 if labels is not None and previous.get("labels") == labels else 1
            age = previous.get("age", 0)
            evidence = Evidence.SUPPORTED if labels is not None else Evidence.UNRESOLVED
            # Failure to find any supported candidate is NOT a negative sample.
            seq = labels or tuple(p.argmax(-1).tolist())
            likelihood = math.exp(sum(math.log(max(float(p[i, v]), 1e-30)) for i, v in enumerate(seq))/length)
            edit = sum(v != int(p[i].argmax()) for i, v in enumerate(seq))/length
            entropy = float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()/math.log(p.shape[1]))
            features = [likelihood, result.margin, edit, min(stable/3, 1), entropy,
                        float(result.complete), float(result.unknown_count > 0)]
            # Ordered perception + ordered candidate code + bounded observed result.
            code = F.one_hot(torch.tensor(seq), len(kb.symbols)).float()
            target = torch.tensor([target_feature(sample["target"], length, cfg)])
            representation = torch.cat([h.flatten(), code.flatten(), target])
            representation = F.normalize(representation, dim=0)
            out.append(Proposal(sample, labels, evidence, features, representation, result, stable, age))
    return out


def graph_bounds(proposals, references, cfg, symmetric=False):
    """Structure-restricted sparse fuzzy graph, capped deterministic reference bank.

    Reflexivity is supplied by fuzzy_bounds. Only training candidates/anchors
    are references; fit/gate labels never become weak-training neighbors.
    """
    if not proposals:
        return
    # A sample with no found explanation has no candidate membership.  Keeping
    # its residual argmax in the bank with q=0 would inject a false negative
    # into every neighbour's lower approximation (DESIGN: unresolved != false).
    references = [p for p in references if p.labels is not None]
    all_items = proposals if symmetric else proposals + references
    q = [p.q for p in all_items]
    edges = [[] for _ in all_items]
    limit = cfg.get("reference_limit", 2048)
    for length in sorted({len(p.sample["images"]) for p in proposals}):
        query_ids = [i for i, p in enumerate(proposals) if len(p.sample["images"]) == length]
        ref_ids = [i for i in (range(len(all_items)) if symmetric else range(len(proposals), len(all_items)))
                   if len(all_items[i].sample["images"]) == length]
        ref_ids.sort(key=lambda i: stable_key(all_items[i].sample["id"], "graph-bank"))
        # Always include labeled anchors in the reference bank; cap weak refs.
        anchors = [i for i in ref_ids if "labels" in all_items[i].sample]
        unlabelled = [i for i in ref_ids if "labels" not in all_items[i].sample]
        ref_ids = anchors + unlabelled[:limit]
        if not ref_ids or cfg["knn_k"] == 0:
            continue
        bank = torch.stack([all_items[i].representation for i in ref_ids])
        # Map sample id -> column(s) so self-edge removal is O(queries) instead
        # of O(queries x bank). This only removes the exact self match; the
        # resulting graph and decisions are unchanged.
        id_to_cols = {}
        for col, j in enumerate(ref_ids):
            id_to_cols.setdefault(all_items[j].sample["id"], []).append(col)
        for start in range(0, len(query_ids), cfg["graph_chunk_size"]):
            ids = query_ids[start:start+cfg["graph_chunk_size"]]
            matrix = torch.stack([proposals[i].representation for i in ids])
            distance = torch.cdist(matrix, bank).square()
            for row, i in enumerate(ids):
                # Explicit self edge is exact; remove it from the k-neighbor budget.
                columns = id_to_cols.get(proposals[i].sample["id"])
                if columns:
                    distance[row, columns] = float("inf")
            k = min(cfg["knn_k"], len(ref_ids))
            values, indices = distance.topk(k, largest=False, sorted=True)
            for row, i in enumerate(ids):
                for d, col in zip(values[row].tolist(), indices[row].tolist()):
                    if not math.isfinite(d):
                        continue
                    j = ref_ids[col]
                    r = math.exp(-d/(2*cfg["graph_sigma"]**2))
                    edges[i].append((j, r))
                    if symmetric:
                        edges[j].append((i, r))
    lo, hi = fuzzy_bounds(q, edges)
    for i, p in enumerate(proposals):
        p.lower, p.upper = lo[i], hi[i]


def acceptance_eligible(p, cfg):
    # A single explanation is globally unique only for a complete search with
    # no unknown competitors. A beam finding one candidate is not that proof.
    n_supported = getattr(p.search, "n_supported", None)
    if n_supported == 1 and p.search.complete and p.search.unknown_count == 0:
        margin_ok = True
    else:
        margin_ok = p.search.margin >= cfg["min_margin"]
    return (p.evidence == Evidence.SUPPORTED and p.stable_count >= cfg["stable_rounds"]
            and margin_ok and p.search.unknown_count == 0
            and (p.search.complete or cfg["allow_truncated"]))


def partition(proposals, cfg, alpha):
    policy = resolve_policy(cfg)
    for p in proposals:
        eligible = acceptance_eligible(p, cfg)
        p.region = decide(p.evidence, p.lower, p.upper,
                          alpha if alpha is not None else 1., cfg["beta"],
                          stable=eligible and alpha is not None)
        if not policy.gate and policy.pseudo and p.evidence == Evidence.SUPPORTED:
            p.region = Region.POSITIVE
        if not policy.pseudo:
            p.region = Region.BOUNDARY
        p.weight = p.lower*p.q*(1-(p.upper-p.lower)) if p.region == Region.POSITIVE else 0.
        if not policy.reliability_weight and p.region == Region.POSITIVE:
            p.weight = 1.


def select_curriculum(proposals, cfg, round_index):
    return curriculum(proposals, cfg, round_index)


def weighted_sequence_ce(logits, lengths, labels, weights):
    """Sequence-balanced, not token-length-weighted; safe differentiable empty case."""
    if not lengths:
        return logits.sum()*0
    if len(lengths) != len(labels) or len(lengths) != len(weights) or sum(lengths) != len(logits):
        raise ValueError("Mismatched structured supervision")
    losses = []
    offset = 0
    for length, y in zip(lengths, labels):
        losses.append(F.cross_entropy(logits[offset:offset+length], torch.tensor(y, device=logits.device), reduction="mean"))
        offset += length
    weights = torch.tensor(weights, device=logits.device, dtype=logits.dtype)
    return (torch.stack(losses)*weights).sum()/weights.sum().clamp_min(1e-8)


@torch.no_grad()
def update_teacher(student, teacher, decay):
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.mul_(decay).add_(source, alpha=1-decay)
    for target, source in zip(teacher.buffers(), student.buffers()):
        target.copy_(source)
