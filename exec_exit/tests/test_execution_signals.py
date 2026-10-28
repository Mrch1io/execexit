import unittest

from exec_exit.replay.policies import PublicProbe, NativeCost
from exec_exit.replay.policies_v1 import ProbeV1
from exec_exit.replay.execution_signals import ExecutionSignals, choose_execution, generator_failed, tool_token_charge


def probe(i, public="PASS", abort=None):
    return ProbeV1(PublicProbe(str(i), i*100, 20, 5, True, True, True, public, None), abort, "same")


class ExecutionSignalTests(unittest.TestCase):
    def test_unknown_never_accepts_and_keeps_full_fallback_cost(self):
        result=choose_execution("EG-S",NativeCost("N",1000),[probe(1)],ExecutionSignals({},{}))
        self.assertEqual((result["selected_subject_id"],result["total_generated_tokens"]),("N",1020))

    def test_agree_uses_previous_pub_even_across_public_failure(self):
        ps=[probe(1),probe(2,"FAIL"),probe(3)]
        result=choose_execution("EG-A",NativeCost("N",1000),ps,ExecutionSignals({},{("1","3"):"PASS"}))
        self.assertEqual(result["selected_subject_id"],"3")
        self.assertEqual(result["visited"][-1]["previous_pub_subject_id"],"1")

    def test_aborted_probe_not_eligible_even_with_passing_signal(self):
        result=choose_execution("EG-S",NativeCost("N",1000),[probe(1,abort=4)],ExecutionSignals({"1":"PASS"},{}))
        self.assertEqual(result["selected_subject_id"],"N")
        self.assertEqual(result["probe_generated_tokens"],4)
        self.assertFalse(result["visited"][0]["stress_check_demand"])

    def test_stress_failure_precedes_agree_and_pub_reference_is_retained(self):
        result=choose_execution("EG-SA",NativeCost("N",1000),[probe(1),probe(2)],ExecutionSignals({"1":"FAIL","2":"PASS"},{("1","2"):"PASS"}))
        self.assertEqual(result["selected_subject_id"],"2")
        self.assertEqual(result["visited"][0]["agreement_status"],"NOT_REQUESTED")

    def test_failure_rule_explicit_and_no_private_labels(self):
        audit={"generator_all_planned_calls_returned":True,"input_validation_counts":{"MODEL_VALIDATOR_REJECTED":1}}
        self.assertFalse(generator_failed(audit,"execution_only"))
        self.assertTrue(generator_failed(audit,"include_validator_rejection"))
        with self.assertRaises(ValueError):
            generator_failed(audit,"pending")
        with self.assertRaises(TypeError):
            choose_execution("EG-A",NativeCost("N",1000),[],{"final_outcome":"PASS"})

    def test_cost_policy_cannot_be_selected_implicitly(self):
        self.assertEqual(tool_token_charge(501,2,"per_sample_full"),501)
        self.assertEqual(tool_token_charge(501,2,"amortized_across_samples"),250.5)
        with self.assertRaises(ValueError):
            tool_token_charge(501,2,"pending")


if __name__=="__main__":
    unittest.main()
