import importlib.util
import ipaddress
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/check_public_docs.py"
SPEC = importlib.util.spec_from_file_location("check_public_docs", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class PublicAddressGuardTests(unittest.TestCase):
    def test_private_and_loopback_ipv4_are_detected_generically(self):
        for packed in (bytes((10, 73, 4, 9)), bytes((192, 168, 7, 11)), bytes((127, 0, 0, 1))):
            value = str(ipaddress.IPv4Address(packed))
            self.assertTrue(CHECKER.private_network_address(value))
            self.assertEqual(list(CHECKER.private_address_lines("first\n" + value)), [2])

    def test_private_ipv6_is_detected_and_global_address_is_allowed(self):
        private_v6 = str(ipaddress.IPv6Address(bytes.fromhex("fd120000000000000000000000000009")))
        global_v4 = str(ipaddress.IPv4Address(bytes((8, 8, 4, 4))))
        self.assertTrue(CHECKER.private_network_address(private_v6))
        self.assertFalse(CHECKER.private_network_address(global_v4))
        self.assertEqual(list(CHECKER.private_address_lines("Year 2026\n" + global_v4)), [])

    def test_checker_contains_no_literal_private_service_endpoint(self):
        source = SCRIPT.read_text()
        private_endpoint = ".".join(("172", "28", "34", "1"))
        self.assertNotIn(private_endpoint, source)

    def test_private_path_guard_uses_generic_roots(self):
        patterns = CHECKER.FORBIDDEN
        path_pattern = patterns["private absolute paths"]
        self.assertIsNotNone(path_pattern.search("/home/person/private/report.txt"))
        self.assertIsNotNone(path_pattern.search("/data/project/input.bin"))
        self.assertIsNone(path_pattern.search("docs/methodology.md"))

    def test_dashboard_source_has_no_hpc_command_or_registry_setup_prompt(self):
        app_source = (SCRIPT.parents[1] / "app.py").read_text()
        shared_root = "/" + "cm/shared/apps/python"
        self.assertNotIn(shared_root, app_source)
        self.assertNotIn("Configure RTM_FINDINGS_REGISTRY", app_source)


if __name__ == "__main__":
    unittest.main()
