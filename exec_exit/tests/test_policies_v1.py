import unittest

from exec_exit.replay.policies import NativeCost, PublicProbe
from exec_exit.replay.policies_v1 import ProbeV1, choose_v1, choose_oracle_v1, free_perfect_bound, select_candidates


def probe(name, pos, *, abort=None, pub="PASS", ast="same", conf=.99, valid=True, tokens=200):
    return ProbeV1(PublicProbe(name,pos,tokens,5,valid,valid,valid,pub,conf),abort,ast)


class V1PolicyTests(unittest.TestCase):
    def setUp(self):
        self.native = NativeCost("N:q",10000)

    def test_abort_is_rejected_and_charged(self):
        result=choose_v1("EG",self.native,[probe("a",500,abort=40),probe("b",1000)])
        self.assertEqual(result["total_generated_tokens"],1240)
        self.assertEqual(result["selected_subject_id"],"b")

    def test_conf_gate_precedes_later_abort(self):
        result=choose_v1("CONF-50",self.native,[probe("a",500,abort=100,conf=.8)])
        self.assertEqual(result["total_generated_tokens"],10050)

    def test_abort_precedes_conf_gate(self):
        result=choose_v1("CONF-50",self.native,[probe("a",500,abort=20,conf=.8)])
        self.assertEqual(result["total_generated_tokens"],10020)

    def test_eg2_does_not_bridge_reject(self):
        result=choose_v1("EG-2",self.native,[probe("a",100),probe("b",200,abort=10),probe("c",300)])
        self.assertEqual(result["selected_subject_id"],"N:q")

    def test_eg2_two_consecutive_passes(self):
        result=choose_v1("EG-2",self.native,[probe("a",100),probe("b",200)])
        self.assertEqual(result["selected_subject_id"],"b")

    def test_text_no_public_label_used(self):
        result=choose_v1("CONV-TEXT",self.native,[probe("a",100,pub="FAIL"),probe("b",200,pub="FAIL")])
        self.assertEqual(result["selected_subject_id"],"b")
        self.assertEqual(result["public_check_demands"],0)

    def test_text_reset_after_invalid(self):
        result=choose_v1("CONV-TEXT",self.native,[probe("a",100),probe("b",200,valid=False),probe("c",300)])
        self.assertEqual(result["selected_subject_id"],"N:q")

    def test_grid_stride_before_minimum(self):
        chosen=select_candidates([probe("a",100),probe("b",2048),probe("c",3000),probe("d",4096)],k=2,minimum=2048)
        self.assertEqual([p.public.subject_id for p in chosen],["c"])

    def test_fixed_ignores_other_points_and_invalid_fallback(self):
        result=choose_v1("BF@2048",self.native,[probe("a",100),probe("b",2048,abort=7)])
        self.assertEqual(result["total_generated_tokens"],10007)

    def test_oracle_charges_failed_prefixes(self):
        result=choose_oracle_v1(self.native,[probe("a",100,abort=7),probe("b",200)],{"b":"PASS"})
        self.assertEqual(result["total_generated_tokens"],407)

    def test_free_bounds_distinguish_delivery_cost(self):
        ps=[probe("a",100)]
        labels={"a":"PASS","N:q":"FAIL"}
        self.assertEqual(free_perfect_bound(self.native,ps,labels)["total_generated_tokens"],300)
        self.assertEqual(free_perfect_bound(self.native,ps,labels,include_delivery=False)["total_generated_tokens"],100)

    def test_first_does_not_scan_after_invalid(self):
        result=choose_v1("FIRST",self.native,[probe("a",100,abort=7),probe("b",200)])
        self.assertEqual(result["selected_subject_id"],"N:q")


if __name__ == "__main__":
    unittest.main()
