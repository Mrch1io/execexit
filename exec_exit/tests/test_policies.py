"""Synthetic checks of stopping semantics and charged fixed-record paths."""
from dataclasses import replace
import unittest

from exec_exit.replay.policies import NativeCost, PublicProbe, choose_stop, choose_oracle


def probe(subject, position, tokens=200, *, confidence=.99, valid=True, public="PASS"):
    return PublicProbe(subject, position, tokens, 6, valid, True, valid, public, confidence)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.native = NativeCost("N:a", 1000, 6)

    def test_acceptance_charges_prefix_and_draft_only_once(self):
        result = choose_stop("FIRST", self.native, [probe("a1", 100, 200)])
        self.assertEqual(result["total_generated_tokens"], 300)
        self.assertEqual(result["total_control_tokens"], 6)

    def test_first_invalid_falls_back_without_visiting_second(self):
        result = choose_stop("FIRST", self.native, [probe("a1", 100, 80, valid=False), probe("a2", 300)])
        self.assertFalse(result["accepted_probe"])
        self.assertEqual(result["total_generated_tokens"], 1080)
        self.assertEqual(len(result["visited"]), 1)
        self.assertEqual(result["total_control_tokens"], 12)

    def test_confidence_strict_threshold_unknown_and_early_eos(self):
        result = choose_stop("CONF-50", self.native, [probe("a1", 100, 400, confidence=.95),
            probe("a2", 200, 17, confidence=None), probe("a3", 300, 120, confidence=.951)])
        self.assertEqual(result["total_generated_tokens"], 300+50+17+120)
        self.assertEqual(result["selected_subject_id"], "a3")

    def test_high_confidence_invalid_full_draft_is_paid_then_continue(self):
        result = choose_stop("CONF-50", self.native, [probe("a1", 100, 500, valid=False), probe("a2", 300, 120)])
        self.assertEqual(result["total_generated_tokens"], 300+500+120)
        self.assertEqual(result["selected_subject_id"], "a2")

    def test_all_rejected_adds_probe_cost_to_complete_native(self):
        result = choose_stop("CONF-50", self.native, [probe("a1", 100, 500, confidence=.1), probe("a2", 300, 120, confidence=None)])
        self.assertEqual(result["total_generated_tokens"], 1100)
        self.assertEqual(result["total_control_tokens"], 18)

    def test_eg_unknown_is_not_a_public_pass(self):
        result = choose_stop("EG", self.native, [probe("a1", 100, 80, public="UNKNOWN"), probe("a2", 300, 120)])
        self.assertEqual(result["total_generated_tokens"], 500)
        self.assertEqual(result["public_check_demands"], 2)
        self.assertEqual(result["private_check_demands"], 0)

    def test_oracle_pays_every_preceding_full_draft(self):
        candidates = [probe("a1", 100, 800, confidence=.1), probe("a2", 300, 120)]
        result = choose_oracle(self.native, candidates, {"a1": "FAIL", "a2": "PASS"})
        self.assertEqual(result["total_generated_tokens"], 300+800+120)
        self.assertEqual(result["private_check_demands"], 2)
        self.assertEqual(result["total_control_tokens"], 12)

    def test_parseable_but_length_invalid_is_never_accepted(self):
        candidate = replace(probe("a1", 100), generation_valid=False, parse_ok=True, deliverable=False)
        for policy in ("FIRST", "CONF-50", "EG"):
            self.assertFalse(choose_stop(policy, self.native, [candidate])["accepted_probe"])

    def test_no_candidates_keeps_native_denominator_and_cost(self):
        for policy in ("N", "FIRST", "CONF-50", "EG"):
            result = choose_stop(policy, self.native, [])
            self.assertEqual(result["total_generated_tokens"], 1000)
            self.assertEqual(result["selected_subject_id"], "N:a")

    def test_ordinary_api_rejects_private_dict_and_oracle_name(self):
        with self.assertRaises(TypeError):
            choose_stop("EG", self.native, [{"subject_id": "a1", "private_result": "PASS"}])
        with self.assertRaises(ValueError):
            choose_stop("ORACLE", self.native, [])
        with self.assertRaises(TypeError):
            choose_stop("EG", self.native, [], dev_labels={})

    def test_duplicate_or_unsorted_positions_fail(self):
        for candidates in ([probe("a1", 100), probe("a2", 100)], [probe("a1", 200), probe("a2", 100)]):
            with self.assertRaises(ValueError):
                choose_stop("FIRST", self.native, candidates)


if __name__ == "__main__":
    unittest.main()
