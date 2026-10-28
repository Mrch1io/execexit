"""Meaningful statistical boundary tests; synthetic inputs never count as data."""
import math
import unittest
import numpy as np
from exec_exit.analysis.stats import (summarize, stratified_indices,
    clopper_pearson_upper, minimum_zero_harm_questions)

POLICIES=['N','BF@4096','HYB-B']


def fixture(n=4,samples=4):
    questions=[dict(collection='A' if i%2 else 'B',case_id=f'q{i}',difficulty='easy' if i<2 else 'hard') for i in range(n)]
    rows=[dict(**q,sample_index=s,name=p,outcome='PASS',tokens=100.) for q in questions for s in range(samples) for p in POLICIES]
    return questions,rows


def analyze(q,r,samples=4):
    return summarize(r,q,POLICIES,sample_count=samples,replicates=200,scope='synthetic')


class QuestionStatisticsTests(unittest.TestCase):
    def test_secondary_n_success_denominator_uses_any_of_four_samples(self):
        q,r=fixture(3,4)
        for row in r:
            if row['case_id']=='q0' and row['name']=='N':row['outcome']='FAIL'
            if row['case_id']=='q1' and row['name']=='HYB-B':row['outcome']='FAIL'
            if row['case_id']=='q2' and row['name']=='N' and row['sample_index']!=3:row['outcome']='FAIL'
        summary,_,_=analyze(q,r)
        comparison=summary['populations'][0]['comparisons']['N']
        secondary=comparison['secondary_n_success_subset']
        self.assertEqual(comparison['strict']['questions'],3)
        self.assertEqual(secondary['eligible_questions_known'],2)
        self.assertEqual(secondary['excluded_all_n_failed_questions'],1)
        self.assertEqual(secondary['eligible_harmed_questions_known'],1)
        self.assertEqual(secondary['harmed_proportion'],.5)
        self.assertAlmostEqual(secondary['upper_probability_97_5'],clopper_pearson_upper(1,2))
        self.assertFalse(secondary['used_for_primary_decision'])

    def test_secondary_unknown_n_is_not_all_failed_and_eligible_harm_unknown(self):
        q,r=fixture(3,4)
        for row in r:
            if row['case_id']=='q0' and row['name']=='N':row['outcome']='FAIL' if row['sample_index'] else 'UNKNOWN'
            if row['case_id']=='q1' and row['name']=='HYB-B' and row['sample_index']==0:row['outcome']='UNKNOWN'
            if row['case_id']=='q2' and row['name']=='N':row['outcome']='FAIL'
        summary,_,_=analyze(q,r)
        s=summary['populations'][0]['comparisons']['N']['secondary_n_success_subset']
        self.assertEqual(s['eligible_question_ids'],['q1'])
        self.assertEqual(s['eligibility_unknown_question_ids'],['q0'])
        self.assertEqual(s['excluded_all_n_failed_question_ids'],['q2'])
        self.assertEqual(s['eligible_harm_unknown_question_ids'],['q1'])
        self.assertEqual(s['eligible_questions_bounds'],[1,2])
        self.assertEqual(s['status'],'NO_FORMAL_DECISION_UNKNOWN')
        self.assertIsNone(s['harmed_proportion']);self.assertIsNone(s['upper_probability_97_5'])

    def test_secondary_known_harm_survives_other_unknown_but_no_formal_bound(self):
        q,r=fixture(1,4)
        for row in r:
            if row['name']=='HYB-B' and row['sample_index']==0:row['outcome']='FAIL'
            if row['name']=='N' and row['sample_index']==1:row['outcome']='UNKNOWN'
        summary,_,_=analyze(q,r)
        s=summary['populations'][0]['comparisons']['N']['secondary_n_success_subset']
        self.assertEqual(s['eligible_questions_known'],1)
        self.assertEqual(s['eligible_harmed_questions_known'],1)
        self.assertEqual(s['eligible_harm_unknown_questions'],0)
        self.assertTrue(s['harm_indicator_identified'])
        self.assertIsNone(s['upper_probability_97_5'])

    def test_secondary_empty_subset_not_zero_risk(self):
        q,r=fixture(2,4)
        for row in r:
            if row['name']=='N':row['outcome']='FAIL'
        summary,_,_=analyze(q,r)
        s=summary['populations'][0]['comparisons']['N']['secondary_n_success_subset']
        self.assertEqual(s['status'],'NOT_APPLICABLE_EMPTY_N_SUCCESS_SUBSET')
        self.assertEqual(s['eligible_questions_known'],0)
        self.assertIsNone(s['harmed_proportion']);self.assertIsNone(s['upper_probability_97_5'])

    def test_secondary_mixed_two_four_no_structural_unknown(self):
        q,r=fixture(2,4)
        r=[row for row in r if row['collection']=='B' or row['sample_index']<2]
        for row in r:
            if row['collection']=='A' and row['name']=='N':row['outcome']='FAIL'
        summary,_,_=summarize(r,q,POLICIES,samples_by_collection={'A':2,'B':4},scope='synthetic',replicates=20)
        s=summary['populations'][0]['comparisons']['N']['secondary_n_success_subset']
        self.assertEqual(s['eligible_questions_known'],1)
        self.assertEqual(s['excluded_all_n_failed_questions'],1)
        self.assertEqual(s['eligibility_unknown_questions'],0)
        self.assertEqual(s['unknown_n_sample_labels'],0)
        self.assertEqual(s['harmed_proportion'],0)
        self.assertAlmostEqual(s['upper_probability_97_5'],.975)

    def test_secondary_missing_expected_sample_and_cost_follow_existing_rule(self):
        q,r=fixture(2,2)
        r=[row for row in r if not(row['case_id']=='q0' and row['name']=='N' and row['sample_index']==0)]
        for row in r:
            if row['case_id']=='q0' and row['name']=='N':row['outcome']='FAIL'
        summary,_,_=analyze(q,r,samples=2)
        s=summary['populations'][0]['comparisons']['N']['secondary_n_success_subset']
        self.assertEqual(s['eligibility_unknown_question_ids'],['q0'])
        self.assertIsNone(s['harmed_proportion'])
        q,r=fixture(2,2);r[-1]['tokens']=None
        summary,_,_=analyze(q,r,samples=2)
        s=summary['populations'][0]['comparisons']['N']['secondary_n_success_subset']
        self.assertTrue(s['harm_indicator_identified'])
        self.assertFalse(s['parent_comparison_complete'])
        self.assertIsNone(s['upper_probability_97_5'])

    def test_mixed_sample_plan_weights_questions_and_masks_absent_slots(self):
        q,r=fixture(2)
        r=[x for x in r if x['collection']=='B' or x['sample_index']<2]
        for row in r:
            row['tokens']=1000 if row['collection']=='B' else 100
            if row['name']=='HYB-B' and row['collection']=='A': row['outcome']='FAIL'
        result,means,indices=summarize(r,q,POLICIES,samples_by_collection={'A':2,'B':4},scope='synthetic',replicates=20)
        pop=result['populations'][0];c=pop['comparisons']['N']
        self.assertEqual(result['records_expected'],18)
        self.assertEqual(result['missing_records'],0)
        self.assertEqual(c['pass_difference_pp'],-50)
        self.assertEqual(c['harmed_question_equivalents'],1)
        self.assertEqual(c['strict']['harmed_questions'],1)
        self.assertEqual(c['strict']['possible_additional_harmed_questions'],0)
        self.assertEqual(pop['estimates'][0]['mean_tokens'],550)
        self.assertEqual(pop['estimates'][0]['fixed_samples'],6)
        self.assertEqual(pop['estimates'][0]['unknown_outcomes'],0)
        self.assertEqual({x['case_id']:len(x['outcomes']) for x in means},{'q0':4,'q1':2})
        self.assertEqual(indices.shape,(20,2))

    def test_mixed_missing_expected_slot_is_unknown_not_omitted(self):
        q,r=fixture(2)
        r=[x for x in r if x['collection']=='B' or x['sample_index']<2]
        r=[x for x in r if not(x['collection']=='A' and x['sample_index']==1 and x['name']=='HYB-B')]
        result,_,_=summarize(r,q,POLICIES,samples_by_collection={'A':2,'B':4},scope='synthetic',replicates=20)
        c=result['populations'][0]['comparisons']['N']
        self.assertEqual(result['missing_records'],1)
        self.assertEqual(c['paired_outcome_unknown'],1)
        self.assertEqual(c['pass_difference_pp_bounds'],[-25,0])
        self.assertIsNone(c['nominal_noninferiority'])
        with self.assertRaises(ValueError):
            summarize(r,q,POLICIES,samples_by_collection={'A':2},scope='synthetic')

    def test_explicit_uniform_plan_preserves_all_statistics_and_indices(self):
        q,r=fixture(4,2)
        original,means,indices=summarize(r,q,POLICIES,sample_count=2,scope='development',replicates=20)
        explicit,means2,indices2=summarize(r,q,POLICIES,samples_by_collection={'A':2,'B':2},scope='development',replicates=20)
        self.assertEqual(original,explicit);self.assertEqual(means,means2)
        np.testing.assert_array_equal(indices,indices2)

    def test_cp_closed_form_boundary_and_minimum_n(self):
        self.assertAlmostEqual(clopper_pearson_upper(0,63),1-.025**(1/63),places=14)
        self.assertGreater(clopper_pearson_upper(0,63),.03)
        n=minimum_zero_harm_questions()
        self.assertLess(clopper_pearson_upper(0,n),.03)
        self.assertGreaterEqual(clopper_pearson_upper(0,n-1),.03)
        self.assertEqual(clopper_pearson_upper(12,12),1)
        self.assertAlmostEqual(clopper_pearson_upper(1,2),math.sqrt(.975),places=14)

    def test_cp_nonzero_solves_independent_binomial_polynomial(self):
        for k,n in [(1,63),(5,121),(40,63)]:
            upper=clopper_pearson_upper(k,n)
            cdf=math.fsum(math.comb(n,i)*upper**i*(1-upper)**(n-i) for i in range(k+1))
            self.assertAlmostEqual(cdf,.025,places=12)
            self.assertGreater(upper,clopper_pearson_upper(k-1,n))

    def test_strata_preserved_indices_deterministic_and_shared(self):
        q,_=fixture(12)
        first=stratified_indices(q,50);second=stratified_indices(q,50)
        np.testing.assert_array_equal(first,second)
        for col,question in enumerate(q):
            for picked in first[:,col]:
                self.assertEqual((q[picked]['collection'],q[picked]['difficulty']),(question['collection'],question['difficulty']))

    def test_question_four_sample_mean_ratio_and_any_harm(self):
        q,r=fixture(2)
        for row in r:
            row['tokens']=1000. if row['case_id']=='q0' else 100.
            if row['name']=='HYB-B':
                row['tokens']=500. if row['case_id']=='q0' else 100.
                if row['sample_index']==0: row['outcome']='FAIL'
        result,means,indices=analyze(q,r)
        c=result['populations'][0]['comparisons']['N']
        self.assertEqual(result['questions'],2)
        self.assertAlmostEqual(c['pass_difference_pp'],-25)
        self.assertAlmostEqual(c['token_saving_percent'],100*(1-300/550))
        self.assertNotAlmostEqual(c['token_saving_percent'],25)
        self.assertEqual(c['strict']['harmed_questions'],2)
        self.assertEqual(c['harmed_question_equivalents'],.5)
        self.assertFalse(c['formal_eligible'])
        self.assertEqual(len(means),6)
        self.assertEqual(indices.shape,(200,2))

    def test_missing_record_and_cost_keep_fixed_denominator_no_formal_call(self):
        q,r=fixture(2)
        r=[row for row in r if not (row['case_id']=='q0' and row['name']=='HYB-B' and row['sample_index']==0)]
        result,_,_=analyze(q,r)
        self.assertEqual(result['missing_records'],1)
        self.assertEqual(result['records_expected'],24)
        c=result['populations'][0]['comparisons']['N']
        self.assertIsNone(c['nominal_noninferiority']);self.assertIsNone(c['strict_noninferiority'])
        self.assertEqual(c['pass_difference_pp_bounds'],[-12.5,0])
        self.assertEqual(c['strict']['possible_additional_harmed_questions'],1)
        q,r=fixture(2);r[-1]['tokens']=None
        result,_,_=analyze(q,r)
        c=result['populations'][0]['comparisons']['N']
        self.assertEqual(c['pass_difference_pp'],0)
        self.assertIsNone(c['strict_noninferiority'])

    def test_degenerate_nominal_ci_does_not_establish_strict_ni(self):
        q,r=fixture(63)
        result,_,_=analyze(q,r)
        c=result['populations'][0]['comparisons']['N']
        self.assertEqual(c['pass_difference_pp_ci95'],[0,0])
        self.assertTrue(c['nominal_noninferiority'])
        self.assertFalse(c['strict_noninferiority'])
        self.assertGreater(c['strict']['upper_probability_97_5'],.05)

    def test_pool_is_question_weighted_not_collection_weighted(self):
        q,r=fixture(4)
        for question in q: question['collection']='A' if question['case_id']=='q0' else 'B'
        for row in r:
            row['collection']='A' if row['case_id']=='q0' else 'B'
            if row['name']=='HYB-B' and row['case_id']=='q0': row['outcome']='FAIL'
        result,_,_=analyze(q,r)
        self.assertEqual(result['populations'][0]['comparisons']['N']['pass_difference_pp'],-25)
        self.assertEqual(result['sequence']['primary_populations'],'ALL_FROZEN_COLLECTIONS_POOLED')

    def test_duplicate_unknown_label_and_real_two_sample_boundary(self):
        q,r=fixture(2)
        with self.assertRaises(ValueError): analyze(q,r+[r[0]])
        r[0]['outcome']='UNSTABLE'
        with self.assertRaises(ValueError): analyze(q,r)
        q,r=fixture(2,2)
        with self.assertRaises(ValueError): summarize(r,q,POLICIES,sample_count=2,scope='heldout')
        result,_,_=summarize(r,q,POLICIES,sample_count=2,scope='development',replicates=20)
        self.assertEqual(result['sample_count'],2)
        self.assertEqual(result['sequence']['status'],'FORMULA_ON_DEVELOPMENT_OR_SYNTHETIC_ONLY')

    def test_fixed_sequence_blocks_bf_claim_when_nominal_ni_fails(self):
        q,r=fixture(2)
        for row in r:
            if row['name']=='HYB-B' and row['sample_index']==0: row['outcome']='FAIL'
            if row['name']=='BF@4096': row['outcome']='FAIL'
        result,_,_=analyze(q,r)
        self.assertTrue(result['populations'][0]['comparisons']['BF@4096']['nominal_superiority'])
        self.assertFalse(result['sequence']['nominal_bf_superiority_after_ni'])


if __name__=='__main__': unittest.main()
