import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hpc/run_full_field_ml.py"
SPEC = importlib.util.spec_from_file_location("run_full_field_ml", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FindingsPublicationTests(unittest.TestCase):
    source_identity = hashlib.sha256(b"source").hexdigest()

    @staticmethod
    def _write_package(staging, source_identity, *, corrupt=False):
        records = {}
        for key, filename in (("additional_candidate_catalogue", "candidate_catalogue.json"),
                              ("quarantined_qc_catalogue", "qc_catalogue.json")):
            schema = ("rtm-full-field-candidate-catalogue-1"
                      if key == "additional_candidate_catalogue"
                      else "rtm-full-field-qc-catalogue-1")
            body = {"schema": schema, "source_identity": source_identity,
                    "status": "complete_for_supplied_full_volume"}
            from reservoir_time_machine.analysis.full_field_findings import _write_streamed_envelope
            list_key = "candidates" if key == "additional_candidate_catalogue" else "items"
            digest = _write_streamed_envelope(staging / filename, body | {list_key: None}, list_key, [])
            records[key] = {"file": filename, "sha256": "0" * 64 if corrupt else digest}
        (staging / "findings.json").write_text(json.dumps({
            "source_identity": source_identity,
            "status": "complete_for_supplied_full_volume",
            **records,
        }))

    def test_complete_package_is_published_by_directory_rename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root / ".staging"; destination = root / "full-field-findings"
            stage.mkdir(); self._write_package(stage, self.source_identity)
            package = MODULE._publish_findings_package(stage, destination, self.source_identity)
            self.assertEqual(package["status"], "complete_for_supplied_full_volume")
            self.assertTrue(destination.is_dir())
            self.assertFalse(stage.exists())

    def test_invalid_or_existing_destination_is_never_published_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root / ".staging"; destination = root / "full-field-findings"
            stage.mkdir(); self._write_package(stage, self.source_identity, corrupt=True)
            with self.assertRaisesRegex(ValueError, "checksum"):
                MODULE._publish_findings_package(stage, destination, self.source_identity)
            self.assertFalse(destination.exists())
            (stage / "findings.json").write_text(json.dumps({
                "status": "complete_for_supplied_full_volume", "source_identity": self.source_identity,
                "additional_candidate_catalogue": {"file": "candidate_catalogue.json", "sha256": "0" * 64},
                "quarantined_qc_catalogue": {"file": "qc_catalogue.json", "sha256": "0" * 64}}))
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE._publish_findings_package(stage, destination, self.source_identity)
            self.assertTrue(destination.is_dir())


if __name__ == "__main__":
    unittest.main()
