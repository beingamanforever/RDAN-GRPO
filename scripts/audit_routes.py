#!/usr/bin/env python3
"""Derive the pinned HIR hard-route inventory and reject unresolved routes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.routes import RouteAuditError, audit_route_inventory, require_resolved_routes  # noqa: E402


def main() -> None:
    """Print static-resolution evidence, optionally seal it, then reject gaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/HIR_trainv1.jsonl")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "configs/data/hir_taxonomy.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/data/hir.json")
    parser.add_argument("--rtt-root", type=Path, default=ROOT.parent / "Rubrics-To-Tokens")
    parser.add_argument("--resolution-output", type=Path)
    args = parser.parse_args()
    try:
        audit = audit_route_inventory(args.source, args.taxonomy, args.manifest, args.rtt_root)
        print(json.dumps(audit.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if args.resolution_output:
            output = json.dumps(audit.resolution_artifact(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            args.resolution_output.write_text(output, encoding="utf-8")
        require_resolved_routes(audit)
    except RouteAuditError as error:
        raise SystemExit(f"route audit blocked: {error}") from error


if __name__ == "__main__":
    main()
