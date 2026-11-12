"""Bounded pilot config tests; no human approval fixture is implied."""
import copy
import json
from pathlib import Path
import unittest

from scripts.hpc.validate_bounded_ml_pilot_config import (
    resolve_attribute_directory, validate_pilot_config)


CONFIG = Path(__file__).resolve().parents[1] / "config/hpc/well2_bounded_pilot.json"


class BoundedPilotConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_experiment_geometry_and_scientific_limits_are_valid(self):
        result = validate_pilot_config(self.config)
        self.assertEqual(result["status"], "pilot_configuration_valid")
        self.assertEqual(self.config["reference_bounds"], [64, 192, 1088, 1216])
        self.assertEqual(self.config["calibration_bounds"], [869, 997, 64, 192])
        self.assertEqual(self.config["sample_bounds"], [850, 901])
        self.assertEqual(self.config["scoring_bounds"], [600, 728, 131, 259, 850, 901])
        self.assertEqual(self.config["full_field_screening_status"], "REVIEW_REQUIRED_BEFORE_EXECUTION")
        self.assertNotIn("authorization", self.config)

    def test_mismatched_geometry_or_disabled_qc_fails(self):
        config = copy.deepcopy(self.config)
        config["calibration_bounds"] = [896, 1024, 64, 192]
        with self.assertRaisesRegex(ValueError, "calibration bounds"):
            validate_pilot_config(config)
        config = copy.deepcopy(self.config)
        config["mandatory_edge_qc"] = False
        with self.assertRaisesRegex(ValueError, "edge QC"):
            validate_pilot_config(config)

    def test_pilot_config_cannot_approve_full_field_screening(self):
        config = copy.deepcopy(self.config)
        config["full_field_screening_status"] = "APPROVED_FOR_SCREENING"
        with self.assertRaisesRegex(ValueError, "cannot approve full-field"):
            validate_pilot_config(config)

    def test_stage1_attribute_path_resolves_below_private_output_root(self):
        root = Path("/private/derived")
        resolved = resolve_attribute_directory(root, self.config)
        self.assertEqual(resolved, root / "full-field-attributes/attributes-0a361aa415beca729b53")
        self.assertNotEqual(resolved, root)
        config = copy.deepcopy(self.config)
        config["attribute_run_directory"] = ""
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            resolve_attribute_directory(root, config)


if __name__ == "__main__":
    unittest.main()
