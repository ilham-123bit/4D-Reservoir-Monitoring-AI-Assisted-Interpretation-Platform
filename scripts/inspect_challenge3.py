"""Read-only Challenge 3 inventory. Run on the authorized HPC, not on raw-data copies."""

import argparse
import json
import os
from pathlib import Path

from reservoir_time_machine.data_audit.audit import audit_challenge3


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Challenge 3 file and metadata inventory.")
    parser.add_argument("--root", default=os.environ.get("RTM_CHALLENGE3_ROOT", "."),
                        help="Authorized dataset directory or RTM_CHALLENGE3_ROOT.")
    parser.add_argument("--output", type=Path, help="Optional local JSON report path; report omits sample data and coordinates.")
    args = parser.parse_args()
    report = audit_challenge3(args.root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
