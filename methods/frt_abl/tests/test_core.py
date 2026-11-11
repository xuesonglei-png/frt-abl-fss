import unittest

from methods.frt_abl.core import (ArithmeticKB, Evidence, Region, abduce, choose_alpha,
                                  decide, fuzzy_bounds, search)


class LogicTests(unittest.TestCase):
    def setUp(self):
        self.kb = ArithmeticKB("mnist-add2", list("0123456789"))

    def test_missing_fact_is_not_refuted(self):
        kb = ArithmeticKB("mnist-add2", list("0123456789"), [(2, 2)])
        self.assertEqual(kb.check((2, 2), 4), Evidence.UNRESOLVED)
        self.assertEqual(kb.check((1, 3), 4), Evidence.SUPPORTED)
        self.assertEqual(kb.check((1, 3), 5), Evidence.REFUTED)

    def test_syntax_validity_is_separate_from_missing_facts(self):
        symbols = list("123456789") + ["+", "-", "*", "/"]
        hwf = ArithmeticKB("hwf", symbols)
        self.assertTrue(hwf.syntax_valid((1, 9, 2)))
        self.assertFalse(hwf.syntax_valid((1, 2, 2)))
        self.assertFalse(hwf.syntax_valid((1, 9)))
        mn = ArithmeticKB("mnist-add2", list("0123456789"))
        self.assertTrue(mn.syntax_valid((0, 0)))
        self.assertFalse(mn.syntax_valid((0,)))

    def test_hwf_invalid_argmax_is_not_an_unresolved_candidate(self):
        symbols = list("123456789") + ["+", "-", "*", "/"]
        kb = ArithmeticKB("hwf", symbols)
        truth = [symbols.index("2"), symbols.index("+"), symbols.index("3")]
        target = 5

        def probabilities(predicted):
            rows = []
            for value in predicted:
                row = [(1 - 0.9) / 12] * 13
                row[value] = 0.9
                rows.append(row)
            return rows

        good = search(probabilities(truth), target, kb)
        self.assertTrue(good.original_valid)
        self.assertEqual(good.unknown_count, 0)
        self.assertGreaterEqual(good.n_supported, 1)
        bad_argmax = (truth[0], symbols.index("7"), truth[2])
        result = search(probabilities(bad_argmax), target, kb)
        self.assertFalse(result.original_valid)
        self.assertIsNotNone(result.best)
        self.assertEqual(result.unknown_count, 0)
        self.assertNotIn(bad_argmax, [c.labels for c in result.candidates])

    def test_missing_fact_stays_a_well_formed_unresolved_candidate(self):
        kb = ArithmeticKB("mnist-add2", list("0123456789"), [(2, 2)])
        probabilities = [[0.0] * 10, [0.0] * 10]
        probabilities[0][2] = probabilities[1][2] = 1.0
        result = search(probabilities, 4, kb)
        self.assertTrue(result.original_valid)
        self.assertIn((2, 2), [c.labels for c in result.candidates])
        self.assertGreaterEqual(result.unknown_count, 1)

    def test_unique_explanation_has_zero_margin_but_n_supported_one(self):
        kb = ArithmeticKB("mnist-add2", list("0123456789"))
        probabilities = [[1.0] + [0.0] * 9, [1.0] + [0.0] * 9]
        result = search(probabilities, 0, kb)
        self.assertEqual(result.best.labels, (0, 0))
        self.assertEqual(result.margin, 0.0)
        self.assertEqual(result.n_supported, 1)

    def test_abduce_reports_supported_and_fallback(self):
        kb = ArithmeticKB("mnist-add2", list("0123456789"))
        supported = [[0.0] * 10, [0.0] * 10]
        supported[0][2] = supported[1][3] = 1.0
        z, covered, result = abduce(supported, 5, kb)
        self.assertEqual(z, (2, 3))
        self.assertTrue(covered)
        self.assertIsNotNone(result.best)
        unsupported = [[1.0] + [0.0] * 9, [1.0] + [0.0] * 9]
        z, covered, result = abduce(unsupported, 19, kb)
        self.assertEqual(z, (0, 0))
        self.assertFalse(covered)
        self.assertIsNone(result.best)

    def test_search_never_claims_unique_from_truncation(self):
        p = [[0.1]*10, [0.1]*10]
        result = search(p, 4, self.kb, max_checks=1)
        self.assertFalse(result.complete)
        self.assertIsNone(result.best)
        self.assertEqual(result.margin, 0)
        self.assertEqual(result.checked, 1)

    def test_addition_exhaustive_enumeration(self):
        result = search([[0.1]*10]*2, 4, self.kb)
        self.assertTrue(result.complete)
        self.assertEqual(result.checked, 100)
        self.assertEqual(sum(c.evidence == Evidence.SUPPORTED for c in result.candidates), 5)

    def test_hwf_precedence_and_left_associativity(self):
        symbols = list("123456789")+["+", "-", "*", "/"]
        kb = ArithmeticKB("hwf", symbols)
        encode = lambda s: tuple(symbols.index(x) for x in s)
        self.assertEqual(kb.value(encode("2+3*4")), 14)
        self.assertEqual(kb.value(encode("8/4/2")), 1)
        self.assertEqual(kb.value(encode("8-3-2")), 3)
        self.assertEqual(kb.check(encode("2+3*4"), 14), Evidence.SUPPORTED)

    def test_no_support_is_boundary(self):
        self.assertEqual(decide(Evidence.UNRESOLVED, 0, 0, .9, .2), Region.BOUNDARY)

    def test_conflict_not_accepted(self):
        self.assertEqual(decide(Evidence.INCONSISTENT, 1, 1, .9, .2), Region.CONFLICT)


class RoughTests(unittest.TestCase):
    def test_reflexivity_and_neighbor_contradiction(self):
        q = [.9, .1, .5]
        lo, hi = fuzzy_bounds(q, [[(1, .95)], [(0, .95)], []])
        self.assertAlmostEqual(lo[0], .15)
        self.assertAlmostEqual(hi[1], .85)
        for l, m, u in zip(lo, q, hi):
            self.assertLessEqual(l, m)
            self.assertLessEqual(m, u)
        self.assertEqual((lo[2], hi[2]), (.5, .5))

    def test_zero_relation_has_no_effect(self):
        self.assertEqual(fuzzy_bounds([.8, .2], [[(1, 0)], []]), ([.8, .2], [.8, .2]))

    def test_unknown_competitor_blocks_acceptance(self):
        self.assertEqual(decide(Evidence.SUPPORTED, .99, 1, .9, .2, has_unknown=True), Region.BOUNDARY)

    def test_gate_fails_closed_with_insufficient_labels(self):
        alpha, _ = choose_alpha([.99]*5, [True]*5, [True]*5, [.8, .9], .1, 50, .8)
        self.assertIsNone(alpha)

    def test_gate_rejects_wrong_corrections(self):
        alpha, _ = choose_alpha([.99]*100, [False]*100, [True]*100, [.8, .9], .1, 50, .8)
        self.assertIsNone(alpha)

    def test_gate_accepts_sufficient_correct_evidence(self):
        alpha, _ = choose_alpha([.99]*300, [True]*300, [True]*300, [.8, .9], .1, 50, .8)
        self.assertEqual(alpha, .8)

    def test_gate_family_size_is_not_ablation_dependent(self):
        lower, correct, eligible = [.96]*60, [True]*60, [True]*60
        self.assertIsNone(choose_alpha(lower, correct, eligible, [.95], .1, 50, .8,
                                       family_size=20)[0])
        self.assertEqual(choose_alpha(lower, correct, eligible, [.95], .1, 50, .8,
                                      family_size=1)[0], .95)

    def test_full_prefers_higher_threshold_while_keeping_coverage(self):
        lower = [.96]*95 + [.95]*5
        correct = [True]*100
        eligible = [True]*100
        low, _ = choose_alpha(lower, correct, eligible, [.95, .96], .1, 50, .95,
                              family_size=2)
        high, _ = choose_alpha(lower, correct, eligible, [.95, .96], .1, 50, .95,
                               family_size=2, prefer_high=True, min_coverage=0.95)
        self.assertEqual(low, .95)
        self.assertEqual(high, .96)


if __name__ == "__main__":
    unittest.main()
