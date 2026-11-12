import unittest

import numpy as np

from reservoir_time_machine.data_ingest.spatial_geometry import (
    conditionally_map_well_coordinates,
    fit_inline_crossline_affine,
)


class SpatialGeometryTests(unittest.TestCase):
    def test_affine_fit_reports_spatial_holdout_error(self):
        il, xl = np.meshgrid(np.arange(10, 20), np.arange(100, 112), indexing="ij")
        x = 500_000 + il * 25.0 + xl * 2.0
        y = 6_000_000 - il * 1.5 + xl * 12.5
        result = fit_inline_crossline_affine(il.ravel(), xl.ravel(), x.ravel(), y.ravel())
        self.assertEqual(result["design_rank"], 3)
        self.assertGreater(result["spatial_holdout_count"], 0)
        self.assertLess(result["holdout_error_coordinate_units_unknown"]["p95"], 1e-6)
        self.assertIn("does not identify CRS", result["interpretation_limit"])

    def test_rank_deficient_geometry_is_rejected(self):
        il = np.repeat(np.arange(12), 2)
        xl = np.tile([200, 201], 12)
        x = il * 2 + xl
        y = il * -1 + xl * 3
        # The held-out inline bands are still predictable; rank deficiency is
        # exercised by identical inlines/crosslines and a zero-area support.
        with self.assertRaisesRegex(ValueError, "span enough"):
            fit_inline_crossline_affine(np.ones(24), xl, x, y)

    def test_workbook_coordinate_projection_is_explicitly_conditional(self):
        geometry = {"affine_geometry": {
            "coefficient_rows_inline_crossline_intercept_columns_x_y": [[25, -1.5], [2, 12.5], [500000, 6000000]],
            "holdout_r2_x_y": [0.999, 0.998],
        }}
        wells = [{"well_id": "Well-2", "workbook_xy": [500000 + 25 * 1000 + 2 * 2200,
                                                            6000000 - 1.5 * 1000 + 12.5 * 2200]}]
        result = conditionally_map_well_coordinates(wells, geometry,
            inline_ids=np.arange(900, 1100), crossline_ids=np.arange(2000, 2300))
        location = result["wells"][0]
        self.assertEqual(result["status"], "conditional_only")
        self.assertAlmostEqual(location["conditional_inline_id"], 1000)
        self.assertAlmostEqual(location["conditional_crossline_id"], 2200)
        self.assertTrue(location["inside_header_axis_extent"])
        self.assertIn("not verified", location["interpretation_limit"])

    def test_conditional_projection_rejects_singular_map(self):
        geometry = {"affine_geometry": {
            "coefficient_rows_inline_crossline_intercept_columns_x_y": [[1, 2], [2, 4], [0, 0]]}}
        result = conditionally_map_well_coordinates([], geometry, inline_ids=[1, 2], crossline_ids=[3, 4])
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
