"""Logic and decision primitives, independent of the perception framework."""
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import product
import math


class Evidence(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"
    INCONSISTENT = "inconsistent"


class Region(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"
    CONFLICT = "conflict"


class ArithmeticKB:
    """Finite addition facts or closed arithmetic grammar; no Prolog failure-as-false.

    Missing addition facts remain unknown. The full arithmetic oracle is never
    consulted to recover a removed fact. HWF uses explicitly declared complete
    arithmetic rules (no missing-rule simulation for that task).
    """

    def __init__(self, task, symbols, missing_facts=(), tolerance=1e-8):
        if task not in {"mnist-add2", "hwf"}:
            raise ValueError(task)
        if task != "mnist-add2" and missing_facts:
            raise ValueError("Missing-fact experiment is defined only for addition")
        self.task, self.symbols, self.tolerance = task, tuple(symbols), tolerance
        missing = {tuple(x) for x in missing_facts}
        self.facts = {(a, b): a + b for a in range(10) for b in range(10)
                      if (a, b) not in missing}

    def syntax_valid(self, labels):
        """Grammar validity, independent of whether the fact is known.

        `value`/`check` answer "what does the KB say"; this answers "is this a
        well-formed candidate at all".  A grammar-invalid raw argmax must not be
        counted as an unresolved competitor, and a missing fact must not make a
        well-formed candidate disappear from the candidate list.
        """
        if self.task == "mnist-add2":
            return len(labels) == 2 and all(0 <= v < 10 for v in labels)
        if not labels or len(labels) % 2 == 0:
            return False
        if any(not 0 <= v < len(self.symbols) for v in labels):
            return False
        digits = tuple("123456789")
        operators = ("+", "-", "*", "/")
        return all(self.symbols[v] in (digits if p % 2 == 0 else operators)
                   for p, v in enumerate(labels))

    def domains(self, length):
        if self.task == "mnist-add2":
            if length != 2:
                raise ValueError("Addition requires exactly two symbols")
            return [tuple(range(10))] * 2
        if length % 2 != 1 or length < 1:
            raise ValueError("HWF requires an odd positive length")
        return [tuple(i for i, s in enumerate(self.symbols)
                      if s in (tuple("123456789") if p % 2 == 0 else ("+", "-", "*", "/")))
                for p in range(length)]

    def value(self, labels):
        if self.task == "mnist-add2":
            return self.facts.get(tuple(labels))
        if not labels or len(labels) % 2 == 0:
            return None
        syms = [self.symbols[i] for i in labels]
        if any(s not in (tuple("123456789") if p % 2 == 0 else ("+", "-", "*", "/"))
               for p, s in enumerate(syms)):
            return None
        # Standard precedence, left associative, rational intermediate values.
        total, term, sign = Fraction(0), Fraction(int(syms[0])), 1
        for p in range(1, len(syms), 2):
            op, number = syms[p], Fraction(int(syms[p + 1]))
            if op == "*":
                term *= number
            elif op == "/":
                term /= number
            else:
                total += sign * term
                term, sign = number, (1 if op == "+" else -1)
        return total + sign * term

    def check(self, labels, target):
        value = self.value(labels)
        if value is None:
            return Evidence.UNRESOLVED
        return (Evidence.SUPPORTED if math.isclose(float(value), float(target),
                rel_tol=0.0, abs_tol=self.tolerance) else Evidence.REFUTED)


@dataclass(frozen=True)
class Candidate:
    labels: tuple
    score: float
    evidence: Evidence


@dataclass
class SearchResult:
    candidates: list
    best: Candidate | None
    checked: int
    complete: bool
    unknown_count: int
    margin: float
    generated: int
    # How many distinct explanations were found. 1 means "unique explanation",
    # which is not the same thing as "ambiguous/tied" (margin == 0).
    n_supported: int | None = None
    # False when the raw argmax is not a well-formed expression (HWF only).
    original_valid: bool | None = None


def search(probabilities, target, kb, beam_size=256, edit_penalty=0.1,
           max_checks=512):
    """Exact addition; bounded grammar beam for HWF. Missing search != refutation.

    The ranking objective is additive; all prefixes have the same length. The raw
    argmax is kept as a candidate only when it is a well-formed expression: a
    grammar-invalid argmax is neither a competing explanation nor an unresolved
    fact.  `unknown_count` therefore counts only well-formed candidates that the
    KB cannot decide because a fact is missing.
    """
    if beam_size < 1 or max_checks < 1:
        raise ValueError("Search budgets must be positive")
    n = len(probabilities)
    if not n or any(len(row) != len(kb.symbols) for row in probabilities):
        raise ValueError("Invalid probability matrix")
    if any(not math.isfinite(p) or p < 0 for row in probabilities for p in row):
        raise ValueError("Probabilities must be finite and nonnegative")
    if any(not math.isclose(sum(row), 1.0, abs_tol=1e-5) for row in probabilities):
        raise ValueError("Rows must sum to one")
    original = tuple(max(range(len(row)), key=row.__getitem__) for row in probabilities)
    original_valid = kb.syntax_valid(original)
    domains = kb.domains(n)

    def score(seq):
        return sum(math.log(max(probabilities[p][v], 1e-30))
                   - edit_penalty * (v != original[p]) for p, v in enumerate(seq)) / n

    if kb.task == "mnist-add2":
        sequences = list(product(*domains))
    else:
        beam = [()]
        for domain in domains:
            beam = sorted((seq + (v,) for seq in beam for v in domain),
                          key=lambda seq: (-score(seq), seq))[:beam_size]
        sequences = beam
    injected = [original] if original_valid else []
    sequences = sorted(set(sequences + injected), key=lambda z: (-score(z), z))
    total_space = math.prod(len(x) for x in domains)
    complete = total_space <= beam_size if kb.task == "hwf" else True
    complete = complete and len(sequences) <= max_checks
    selected = sequences[:max_checks]
    candidates = [Candidate(z, score(z), kb.check(z, target)) for z in selected]
    supported = [c for c in candidates if c.evidence == Evidence.SUPPORTED]
    # A single found explanation has no observed runner-up: margin stays zero.
    # n_supported counts only checked candidates. Completeness and unknown
    # competitors must also be checked before claiming a unique explanation.
    margin = supported[0].score - supported[1].score if len(supported) > 1 else 0.0
    return SearchResult(candidates, supported[0] if supported else None,
                        len(candidates), complete,
                        sum(c.evidence == Evidence.UNRESOLVED for c in candidates),
                        margin, len(sequences), len(supported), original_valid)


def abduce(probabilities, target, kb, beam_size=256, edit_penalty=0.1,
           max_checks=512):
    """Native prediction: best supported candidate, else the raw argmax.

    Returns (sequence, covered, result). `covered` is True only when the search
    found a complete explanation; the fallback is explicit and is reported as a
    coverage loss, not silently mixed into abduced predictions.
    """
    result = search(probabilities, target, kb, beam_size=beam_size,
                    edit_penalty=edit_penalty, max_checks=max_checks)
    if result.best is not None:
        return result.best.labels, True, result
    original = tuple(max(range(len(row)), key=row.__getitem__) for row in probabilities)
    return original, False, result


def fuzzy_bounds(memberships, edges):
    """Sparse reflexive Lukasiewicz approximations; absent edges have R=0."""
    if len(memberships) != len(edges):
        raise ValueError("Graph and universe differ")
    if any(not math.isfinite(q) or not 0 <= q <= 1 for q in memberships):
        raise ValueError("Membership outside [0,1]")
    lower, upper = [], []
    for i, neighbors in enumerate(edges):
        lo = hi = memberships[i]  # mandatory R_ii=1
        for j, r in neighbors:
            if not 0 <= j < len(memberships) or not math.isfinite(r) or not 0 <= r <= 1:
                raise ValueError("Invalid fuzzy edge")
            lo = min(lo, min(1., 1. - r + memberships[j]))
            hi = max(hi, max(0., r + memberships[j] - 1.))
        lower.append(lo)
        upper.append(hi)
    return lower, upper


def decide(evidence, lower, upper, alpha, beta, stable=True,
           margin_ok=True, has_unknown=False):
    if not 0 <= beta < alpha <= 1 or not 0 <= lower <= upper <= 1:
        raise ValueError("Invalid three-way thresholds/bounds")
    if evidence == Evidence.INCONSISTENT:
        return Region.CONFLICT
    if evidence == Evidence.REFUTED:
        return Region.NEGATIVE
    if evidence == Evidence.UNRESOLVED:
        return Region.BOUNDARY
    if lower >= alpha and stable and margin_ok and not has_unknown:
        return Region.POSITIVE
    if upper <= beta:
        return Region.NEGATIVE
    return Region.BOUNDARY


def wilson_upper(errors, count, z=1.96):
    if count == 0:
        return 1.0
    p = errors / count
    return (p + z*z/(2*count) + z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1+z*z/count)


def choose_alpha(lower, correct, eligible, grid, risk_limit, min_count, floor,
                 family_size=None, prefer_high=False, min_coverage=None):
    """Fixed-grid held-out gate; Bonferroni simultaneous Wilson approximation.

    `family_size` is the pre-registered number of hypotheses and must not depend
    on which ablation is running: otherwise a one-threshold ablation and the
    full grid would use different critical values (1.96 vs ~3.02) and the
    comparison would confound the gate strength with the threshold policy.
    `prefer_high` is used only by full: among thresholds that pass the same
    risk test, choose the most precise one that still covers at least
    `min_coverage` of eligible gate candidates. This keeps full's threshold
    policy distinct from the fixed-0.95 ablation without weakening the gate.
    This is a diagnostic risk estimate, not a distribution-free guarantee.
    No gate data ever trains the calibrator or perception network.
    """
    from statistics import NormalDist
    tests = int(family_size) if family_size is not None else len(grid)
    if not grid or tests < len(set(grid)):
        raise ValueError("family_size must cover the declared threshold grid")
    if not len(lower) == len(correct) == len(eligible):
        raise ValueError("Gate arrays must have equal lengths")
    if min_coverage is not None and not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must lie in [0, 1]")
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * tests))
    eligible_count = sum(1 for x in eligible if x)
    required = int(min_count)
    if min_coverage is not None:
        required = max(required, math.ceil(float(min_coverage) * eligible_count))
    reports = []
    selected = None
    for threshold in sorted(grid, reverse=prefer_high):
        mask = [i for i, q in enumerate(lower) if q >= threshold and eligible[i]]
        errors = sum(not correct[i] for i in mask)
        bound = wilson_upper(errors, len(mask), z)
        passed = threshold >= floor and len(mask) >= required and bound <= risk_limit
        reports.append({"alpha": threshold, "count": len(mask), "errors": errors,
                        "wilson_upper": bound, "passed": passed})
        if selected is None and passed:
            selected = threshold
    # >1 would violate the algebraic threshold range. Return None to disable
    # pseudo-label acceptance completely when calibration evidence is insufficient.
    return selected, reports
