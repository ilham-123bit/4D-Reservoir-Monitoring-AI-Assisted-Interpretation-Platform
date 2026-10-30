"""CLI for inspecting and quality-checking authorized Challenge 3 input data."""

import argparse
import json

from .data_audit.audit import audit_challenge3
from .data_ingest.manifest import challenge3_availability
from .settings import ApplicationSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Reservoir Time Machine actual-data utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "audit"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--root", default=None, help="Authorized dataset root (or RTM_CHALLENGE3_ROOT).")
        if command == "audit":
            sub.add_argument("--output", help="Optional local JSON output path; no sample values or coordinates included.")
    args = parser.parse_args()
    root = args.root or str(ApplicationSettings.from_environment().challenge3_root)
    report = challenge3_availability(root) if args.command == "status" else audit_challenge3(root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if getattr(args, "output", None):
        from pathlib import Path
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
