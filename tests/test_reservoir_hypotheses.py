"""Numerical safeguards for coherent 4D seismic screening, not fluid truth."""
import unittest
import numpy as np
from reservoir_time_machine.analysis.reservoir_hypotheses import screen_response_regions, candidate_briefing

class ReservoirHypothesesTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(41)
        self.b=rng.normal(size=(64,64,45)).astype("float32")
        self.m=self.b+rng.normal(0,0.15,size=self.b.shape).astype("float32")
        self.m[24:35,24:35,:]=3*self.b[24:35,24:35,:]
        self.m[37:49,35:49,:]=self.b[37:49,35:49,:]
        self.mask=np.zeros((64,64),dtype=bool)
        self.il=np.arange(1000,1064)
        self.xl=np.arange(2000,2064)

    def screen(self):
        return screen_response_regions(self.b,self.m,self.mask,[31,31,22],self.il,self.xl)

    def test_coherent_changed_and_quiet_groups_are_separate(self):
        r=self.screen()
        self.assertEqual(r["status"],"measured")
        self.assertEqual(r["relative_response_status"],"coherent_changed_and_quiet_regions")
        high=r["response_region_groups"]["changed"][0]
        low=r["response_region_groups"]["relatively_quiet"][0]
        self.assertGreater(high["trace_count"],30)
        self.assertGreater(low["trace_count"],30)
        self.assertGreater(high["median_nrms_percent"],r["local_median_nrms_percent"])
        self.assertLess(low["median_nrms_percent"],r["local_median_nrms_percent"])
        self.assertEqual(high["center_segy"]["inline_id"],1000+high["center_array"][0])
        self.assertEqual(low["center_segy"]["crossline_id"],2000+low["center_array"][1])

    def test_conditional_briefing_is_not_drilling_claim(self):
        r=self.screen()
        finding={"id":"field-example","array_location":[31,31,22],"segy_location":{"inline_id":1031,"crossline_id":2031,"sample_index":22},"edge_artifact_risk":0.1}
        b=candidate_briefing(finding,r)
        self.assertIsNotNone(b["quieter_region"])
        self.assertIsNone(b["drilling_location"])
        self.assertIn("hypothesis_only",b["bypassed_hydrocarbon"])
        self.assertEqual(b["recovery_candidate_status"],"reservoir_eligibility_unverified")

    def test_missing_traces_are_not_supposed_quiet_oil(self):
        self.mask[37:49,35:49]=True
        r=self.screen()
        self.assertFalse(any(37<=z["center_array"][0]<49 and 35<=z["center_array"][1]<49 for z in r["response_region_groups"]["relatively_quiet"]))
        self.assertTrue(np.isnan(r["response_map"][37-9,35-9]))

    def test_zero_energy_cannot_become_quiet_target(self):
        self.b[37:49,35:49]=0
        self.m[37:49,35:49]=0
        r=self.screen()
        self.assertFalse(any(37<=z["center_array"][0]<49 and 35<=z["center_array"][1]<49 for z in r["response_region_groups"]["relatively_quiet"]))

    def test_invalid_geometry_or_location_rejected(self):
        with self.assertRaises(ValueError):
            screen_response_regions(self.b,self.m,self.mask[:10],[31,31,22],self.il,self.xl)
        with self.assertRaises(ValueError):
            screen_response_regions(self.b,self.m,self.mask,[31,31,55],self.il,self.xl)

    def test_unchanged_cube_does_not_create_fake_region(self):
        r=screen_response_regions(self.b,self.b,self.mask,[31,31,22],self.il,self.xl)
        self.assertEqual(r["relative_response_status"],"not_resolved")
        self.assertEqual(r["response_region_groups"]["changed"],[])
        self.assertEqual(r["response_region_groups"]["relatively_quiet"],[])

if __name__=="__main__":
    unittest.main()
