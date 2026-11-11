"""Tiny synthetic tensor checks; no dataset, network forward, optimizer or experiment."""
import unittest
import torch

from methods.frt_abl.runtime import (weighted_sequence_ce, Calibrator, Proposal,
                                   acceptance_eligible, graph_bounds, partition,
                                   result_from_summary, search_worker, search_worker_init,
                                   select_curriculum, target_feature)
from methods.frt_abl.core import Candidate, SearchResult, Evidence, Region


class TensorTests(unittest.TestCase):
    def test_sequence_balancing_and_stop_gradient_labels(self):
        logits = torch.tensor([[2., 0.], [0., 2.], [0., 2.]], requires_grad=True)
        loss = weighted_sequence_ce(logits, [1, 2], [(0,), (0, 0)], [1., 0.])
        loss.backward()
        self.assertGreater(float(logits.grad[0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[1:].abs().sum()), 0)

    def test_empty_positive_loss_finite(self):
        logits = torch.empty((0, 2), requires_grad=True)
        loss = weighted_sequence_ce(logits, [], [], [])
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)

    def test_single_class_calibrator_flag(self):
        calibrator = Calibrator()
        calibrator.fit([[.2, 0.], [.3, 1.]], [False, False])
        self.assertTrue(calibrator.degenerate)
        self.assertTrue(all(0 < q < 1 for q in calibrator.predict([[.2, 0.]])))

    def test_degenerate_calibrator_keeps_an_informative_prior(self):
        features = [[.2, 0.], [.3, 1.]] * 10
        all_correct = Calibrator()
        all_correct.fit(features, [True] * 20)
        all_wrong = Calibrator()
        all_wrong.fit(features, [False] * 20)
        self.assertGreater(all_correct.predict([[.2, 0.]])[0], 0.9)
        self.assertLess(all_wrong.predict([[.2, 0.]])[0], 0.1)

    def test_calibrator_predicts_bounded_scores_with_both_classes(self):
        calibrator = Calibrator()
        calibrator.fit([[0.], [.1], [.9], [1.]], [False, False, True, True])
        low, high = calibrator.predict([[0.], [1.]])
        self.assertFalse(calibrator.degenerate)
        self.assertTrue(0 < low < high < 1)

    def test_train_only_reference_changes_gate_bounds(self):
        c = Candidate((0,), 0., Evidence.SUPPORTED)
        result = SearchResult([c], c, 1, True, 0, .1, 1)
        query = Proposal({"id": "query", "images": ["a"]}, (0,), Evidence.SUPPORTED,
                         [], torch.tensor([1., 0.]), result, 3, 0, q=.9)
        reference = Proposal({"id": "reference", "images": ["b"], "labels": [1]}, (0,), Evidence.SUPPORTED,
                             [], torch.tensor([1., 0.]), result, 3, 0, q=0.)
        cfg = {"knn_k": 1, "graph_sigma": .5, "graph_chunk_size": 8,
               "stable_rounds": 2, "min_margin": .02, "allow_truncated": False,
               "beta": .2, "variant": "full", "train_fraction": .25,
               "curriculum_ramp_rounds": 10, "curriculum_min_weight": .5}
        graph_bounds([query], [reference], cfg)
        self.assertEqual(query.lower, 0.)
        self.assertEqual(query.upper, .9)
        partition([query], cfg, .8)
        self.assertEqual(query.region, Region.BOUNDARY)
        query.lower = query.upper = .99
        partition([query], cfg, None)
        self.assertEqual(query.region, Region.BOUNDARY)
        self.assertEqual(select_curriculum([query], cfg, 0), [])

    def test_target_feature_share_is_independent_of_length(self):
        cfg = {"target_center": 0.0, "target_scale": 1.0, "target_share": 0.05}
        for length in (1, 2, 7):
            value = target_feature(1e9, length, cfg)
            share = value * value / (2 * length + value * value)
            self.assertAlmostEqual(share, 0.05, places=12)
        self.assertEqual(target_feature(0.0, 7, cfg), 0.0)
        huge = target_feature(3024, 7, {"target_center": 6.0, "target_scale": 29.5,
                                        "target_share": 0.05})
        self.assertLess(abs(huge), 1.0)

    def test_unique_explanation_is_eligible_even_with_zero_margin(self):
        c = Candidate((0,), 0., Evidence.SUPPORTED)
        unique = SearchResult([c], c, 1, True, 0, 0., 1, n_supported=1)
        tied = SearchResult([c], c, 1, True, 0, 0., 1, n_supported=2)
        legacy = SearchResult([c], c, 1, True, 0, .1, 1)
        cfg = {"stable_rounds": 2, "min_margin": .02, "allow_truncated": False}
        for result, expected in ((unique, True), (tied, False), (legacy, True)):
            proposal = Proposal({"id": "x", "images": ["a"]}, (0,), Evidence.SUPPORTED,
                                [], torch.tensor([1., 0.]), result, 2, 0)
            self.assertEqual(acceptance_eligible(proposal, cfg), expected)

    def test_truncated_single_candidate_does_not_prove_uniqueness(self):
        c = Candidate((0,), 0., Evidence.SUPPORTED)
        result = SearchResult([c], c, 1, False, 0, 0., 1, n_supported=1)
        p = Proposal({"id": "x", "images": ["a"]}, (0,), Evidence.SUPPORTED,
                     [], torch.tensor([1., 0.]), result, 2, 0)
        cfg = {"stable_rounds": 2, "min_margin": .02, "allow_truncated": True}
        self.assertFalse(acceptance_eligible(p, cfg))

    def test_unsupported_reference_does_not_enter_the_bank(self):
        c = Candidate((0,), 0., Evidence.SUPPORTED)
        result = SearchResult([c], c, 1, True, 0, 0., 1, n_supported=1)
        query = Proposal({"id": "q", "images": ["a"]}, (0,), Evidence.SUPPORTED,
                         [], torch.tensor([1., 0.]), result, 1, 0, q=.9)
        unsupported = Proposal({"id": "u", "images": ["a"]}, None, Evidence.UNRESOLVED,
                               [], torch.tensor([1., 0.]), result, 1, 0, q=0.)
        cfg = {"knn_k": 1, "graph_sigma": .5, "graph_chunk_size": 8}
        graph_bounds([query], [unsupported], cfg)
        self.assertAlmostEqual(query.lower, .9)
        self.assertAlmostEqual(query.upper, .9)

    def test_self_duplicate_is_removed_from_the_neighbor_budget(self):
        c = Candidate((0,), 0., Evidence.SUPPORTED)
        result = SearchResult([c], c, 1, True, 0, 0., 1, n_supported=1)
        query = Proposal({"id": "same", "images": ["a"]}, (0,), Evidence.SUPPORTED,
                         [], torch.tensor([1., 0.]), result, 1, 0, q=.9)
        same = Proposal({"id": "same", "images": ["a"], "labels": [1]}, (0,),
                        Evidence.SUPPORTED, [], torch.tensor([1., 0.]), result, 1, 0, q=0.)
        other = Proposal({"id": "other", "images": ["b"], "labels": [1]}, (0,),
                         Evidence.SUPPORTED, [], torch.tensor([1., 0.]), result, 1, 0, q=0.)
        cfg = {"knn_k": 1, "graph_sigma": .5, "graph_chunk_size": 8}
        graph_bounds([query], [query, same, other], cfg)
        # Both "same" columns are inf; the only retained neighbor is q=0.
        self.assertAlmostEqual(query.lower, 0.)
        self.assertAlmostEqual(query.upper, .9)

    def test_search_worker_summary_matches_the_search_api(self):
        from methods.frt_abl.core import ArithmeticKB, search
        kb = ArithmeticKB("mnist-add2", list("0123456789"))
        probabilities = [[0.0] * 10, [0.0] * 10]
        probabilities[0][2] = probabilities[1][3] = 1.0
        search_worker_init(kb)
        summary = search_worker((probabilities, 5, 256, 512, 0.1))
        reference = search(probabilities, 5, kb, beam_size=256, max_checks=512)
        result = result_from_summary(summary)
        self.assertEqual(result.best.labels, reference.best.labels)
        self.assertEqual(result.checked, reference.checked)
        self.assertEqual(result.complete, reference.complete)
        self.assertEqual(result.unknown_count, reference.unknown_count)
        self.assertEqual(result.n_supported, reference.n_supported)


if __name__ == "__main__":
    unittest.main()
