import json
import os
import subprocess
from pathlib import Path

import pytest

from rdan_grpo.routes import (
    EXPECTED_ROUTE_DIGEST,
    PINNED_RTT_REVISION,
    RouteAuditError,
    audit_route_inventory,
    load_rtt_route_maps,
    require_resolved_routes,
    route_digest,
    verify_rtt_checkout,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/HIR_trainv1.jsonl"
TAXONOMY = ROOT / "configs/data/hir_taxonomy.json"
MANIFEST = ROOT / "configs/data/hir.json"
RESOLUTION = ROOT / "configs/artifacts/hir_route_resolution.json"
RTT = ROOT.parent / "Rubrics-To-Tokens"
RUN_ROUTE_E2E = os.environ.get("RDAN_RUN_ROUTE_E2E") == "1"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rtt_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "rtt"
    reward_root = root / "roll/pipeline/rlvr/rewards"
    (reward_root / "muldimif_checkers").mkdir(parents=True)
    (reward_root / "rubrics_llm_judge_reward_worker.py").write_text(
        'INSTRUCTION_ID_TO_IFEVAL = {"known": ("checker", {})}\n',
        encoding="utf-8",
    )
    (reward_root / "ifeval_rule_reward_worker.py").write_text(
        'IF_FUNCTIONS_MAP = {"checker": object()}\n',
        encoding="utf-8",
    )
    (reward_root / "type2_checkers.py").write_text(
        'TYPE2_CHECKERS = {"known2": object()}\n',
        encoding="utf-8",
    )
    (reward_root / "muldimif_checkers/__init__.py").write_text(
        'CONSTRAINT_CHECKER_MAP = {"Known_Route": object()}\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    _git(root, "config", "user.email", "route-tests@example.invalid")
    _git(root, "config", "user.name", "Route Tests")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def test_route_map_parser_and_digest_are_hermetic(tmp_path: Path) -> None:
    checkout, revision = _rtt_fixture(tmp_path)

    assert verify_rtt_checkout(checkout, revision)["revision"] == revision
    assert load_rtt_route_maps(checkout) == {
        "type1": {"known": "checker"},
        "ifeval": {"checker"},
        "type2": {"known2"},
        "type3": {"Known_Route"},
    }
    rows = [
        {
            "source": "type1",
            "route": "known",
            "route_resolvable": True,
            "reason": "static_route_resolved",
            "criteria": 1,
            "rows": 1,
        }
    ]
    assert route_digest(rows) == route_digest(list(rows))
    assert route_digest(rows) != route_digest([{**rows[0], "criteria": 2}])


def test_checkout_rejects_wrong_revision_and_dirty_tree(tmp_path: Path) -> None:
    checkout, revision = _rtt_fixture(tmp_path)

    with pytest.raises(RouteAuditError, match="wrong RTT revision"):
        verify_rtt_checkout(checkout, "0" * 40)

    (checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RouteAuditError, match="RTT checkout is dirty"):
        verify_rtt_checkout(checkout, revision)


@pytest.mark.skipif(not RUN_ROUTE_E2E, reason="set RDAN_RUN_ROUTE_E2E=1 for pinned external-data route audit")
def test_pinned_external_route_audit_matches_sealed_artifact() -> None:
    missing = [path for path in (SOURCE, TAXONOMY, MANIFEST, RTT) if not path.exists()]
    if missing:
        pytest.fail(f"pinned external prerequisites are absent: {missing}")

    first = audit_route_inventory(SOURCE, TAXONOMY, MANIFEST, RTT)
    second = audit_route_inventory(SOURCE, TAXONOMY, MANIFEST, RTT)
    assert first == second
    assert first.sources["rtt"]["revision"] == PINNED_RTT_REVISION
    assert first.counts["route_resolvable"] == 75_657
    assert first.counts["unresolved"] == 799
    assert first.counts["unique_rows_with_unsupported"] == 650
    assert first.digest["sha256"] == EXPECTED_ROUTE_DIGEST
    assert first.resolution_artifact() == json.loads(RESOLUTION.read_text(encoding="utf-8"))
    with pytest.raises(RouteAuditError, match="unresolved hard routes remain"):
        require_resolved_routes(first)
