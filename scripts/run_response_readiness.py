#!/usr/bin/env python3
"""Issue or check the immutable response-level RL readiness receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.response_identity import clean_repository_revision  # noqa: E402
from rdan_grpo.response_readiness import (  # noqa: E402
    ResponseReadinessError,
    issue_response_readiness,
    validate_response_readiness,
)


def main(argv: list[str] | None = None) -> int:
    """Run one caller-boundary readiness operation."""

    args = _parse_args(argv)
    revision = clean_repository_revision(args.program, args.rdan_revision)
    operation = validate_response_readiness if args.check else issue_response_readiness
    if args.check:
        receipt = operation(
            args.output,
            args.program,
            args.bootstrap,
            args.evidence,
            rdan_revision=revision,
        )
    else:
        receipt = operation(
            args.program,
            args.bootstrap,
            args.evidence,
            args.output,
            rdan_revision=revision,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        required=True,
        help="repeat in judge calibration, runtime parity, no-update order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rdan-revision", help="exact clean checkout revision, defaults to program repository HEAD")
    parser.add_argument("--check", action="store_true", help="validate the existing output without modifying it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ResponseReadinessError, ValueError) as error:
        print(f"response readiness failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
