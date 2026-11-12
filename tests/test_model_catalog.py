"""Keep the capability ledger aligned with the full-field implementation."""

import unittest

from reservoir_time_machine.analysis.model_catalog import model_catalog


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_reports_full_field_status_without_stale_tile_limits(self):
        items = {item["family"]: item for item in model_catalog()}
        expected = {
            "Mahalanobis distance", "PCA / linear autoencoder", "Isolation Forest",
            "Gaussian Mixture Model", "Statistical MAD screen",
            "3D connected components and edge QC", "Bayesian change detector / posterior fusion",
            "Siamese 3D CNN", "Nonlinear 3D convolutional autoencoder",
        }
        self.assertTrue(expected.issubset(items))
        for name in ("Mahalanobis distance", "PCA / linear autoencoder",
                     "Isolation Forest", "Gaussian Mixture Model"):
            self.assertIn("ANGSI RUN PENDING", items[name]["status"])
            self.assertIn("unmeasured", items[name]["resources"].lower())
        self.assertIn("50,000", items["Isolation Forest"]["resources"])
        self.assertNotIn("16384 tile cells", " ".join(i["resources"] for i in items.values()))
        self.assertIn("NO VERIFIED ANGSI CHECKPOINT", items["Siamese 3D CNN"]["status"])
        self.assertIn("NO VERIFIED ANGSI CHECKPOINT", items["Nonlinear 3D convolutional autoencoder"]["status"])
        self.assertIn("NOT CALIBRATED OR INTEGRATED", items["Bayesian change detector / posterior fusion"]["status"])


if __name__ == "__main__":
    unittest.main()
