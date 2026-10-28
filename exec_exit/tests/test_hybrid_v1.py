import unittest

from exec_exit.replay.hybrid_v1 import choose_hybrid
from exec_exit.replay.policies import NativeCost, PublicProbe
from exec_exit.replay.policies_v1 import ProbeV1


def probe(position, *, pub="FAIL", valid=True, tokens=100, abort=None):
    return ProbeV1(PublicProbe(f"p{position}", position, tokens, 5, valid,
        valid, valid, pub, .99), abort, "AST" if valid else None)


class HybridBoundaries(unittest.TestCase):
    def setUp(self):
        self.native = NativeCost("N:q", 10000, 5)

    def test_boundary_survives_stride_and_is_charged_once(self):
        ps = [probe(p) for p in (100, 500, 1000, 2048, 3000, 4096, 6000)]
        choice = choose_hybrid("HYB-A", self.native, ps)
        self.assertEqual([v["position"] for v in choice["visited"]], [100, 3000, 4096])
        self.assertEqual(choice["total_generated_tokens"], 4096 + 300)
        self.assertEqual(choice["public_check_demands"], 2)

    def test_boundary_selected_by_stride_is_not_duplicated(self):
        ps = [probe(p) for p in (100, 500, 1000, 2048, 4096, 6000)]
        choice = choose_hybrid("HYB-A", self.native, ps)
        self.assertEqual([v["position"] for v in choice["visited"]], [100, 4096])

    def test_early_public_accept_never_visits_boundary(self):
        choice = choose_hybrid("HYB-B", self.native, [probe(100, pub="PASS"), probe(4096)])
        self.assertEqual(choice["selected_subject_id"], "p100")
        self.assertEqual(choice["total_generated_tokens"], 200)
        self.assertEqual(len(choice["visited"]), 1)

    def test_a_vs_b_differs_only_in_boundary_public_requirement(self):
        for status in ("FAIL", "UNKNOWN"):
            with self.subTest(status=status):
                ps = [probe(100), probe(4096, pub=status)]
                a = choose_hybrid("HYB-A", self.native, ps)
                b = choose_hybrid("HYB-B", self.native, ps)
                self.assertEqual(a["selected_subject_id"], "p4096")
                self.assertEqual(b["selected_subject_id"], "N:q")
                self.assertEqual(b["total_generated_tokens"], 10200)
                self.assertEqual(b["public_check_demands"], 2)

    def test_b_boundary_public_pass_accepts(self):
        choice = choose_hybrid("HYB-B", self.native, [probe(4096, pub="PASS")])
        self.assertEqual(choice["selected_subject_id"], "p4096")
        self.assertEqual(choice["total_generated_tokens"], 4196)

    def test_abort_overrides_parseability_and_charges_visible_prefix(self):
        for policy in ("HYB-A", "HYB-B"):
            choice = choose_hybrid(policy, self.native,
                [probe(100, tokens=200, abort=7), probe(4096, pub="PASS", tokens=200, abort=13)])
            self.assertEqual(choice["selected_subject_id"], "N:q")
            self.assertEqual(choice["total_generated_tokens"], 10020)
            self.assertEqual(choice["public_check_demands"], 0)

    def test_invalid_boundary_and_later_good_probe_keep_native(self):
        for policy in ("HYB-A", "HYB-B"):
            ps = [probe(100), probe(4096, valid=False), probe(6000, pub="PASS")]
            choice = choose_hybrid(policy, self.native, ps)
            self.assertEqual(choice["selected_subject_id"], "N:q")
            self.assertEqual([v["position"] for v in choice["visited"]], [100, 4096])

    def test_native_ended_before_boundary_no_invented_probe(self):
        for policy in ("HYB-A", "HYB-B"):
            choice = choose_hybrid(policy, NativeCost("N:short", 1000), [probe(100)])
            self.assertEqual(choice["selected_subject_id"], "N:short")
            self.assertEqual(choice["total_generated_tokens"], 1100)

    def test_no_candidates_and_reject_private_inputs(self):
        self.assertEqual(choose_hybrid("HYB-A", self.native, [])["total_generated_tokens"], 10000)
        with self.assertRaises(TypeError):
            choose_hybrid("HYB-A", self.native, [{"final_outcome": "PASS"}])


if __name__ == "__main__":
    unittest.main()
