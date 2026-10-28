import unittest
from exec_exit.serve.heldout_service_template import template,issues,paired_poisson_arrivals


class ServiceTemplate(unittest.TestCase):
    def test_draft_keeps_choices_empty_and_cannot_run(self):
        spec=template()
        self.assertIsNone(spec["selected_case_ids"])
        self.assertIn("FREEZE_REQUIRED",issues(spec))
        self.assertFalse(any("WAITING_EXPLICIT_PAUSE" in s for s in issues(spec)))
        self.assertIn("H",spec["serial"]["policies"])
        self.assertIsNone(spec["serial"]["CONF-50"]["k"])
        self.assertEqual(spec["poisson"]["rates_requests_per_second"],[None,None])
    def test_arrivals_shared_and_no_denominator_drops(self):
        ids=["synthetic:a","synthetic:b","synthetic:a"]
        first=paired_poisson_arrivals(ids,2,7)
        second=paired_poisson_arrivals(ids,2,7)
        self.assertEqual(first,second)
        self.assertEqual([r["case_id"] for r in first],ids)
        self.assertTrue(all(a["intended_arrival_seconds"]<b["intended_arrival_seconds"] for a,b in zip(first,first[1:])))
    def test_missing_control_not_silently_accepted(self):
        spec=template()
        spec["serial"]["policies"]=["N","HYB-B-ASYNC"]
        self.assertIn("All six serial controls must remain present",issues(spec))


if __name__=="__main__":
    unittest.main()
