"""Deterministic resolution audit for hard HIR routes in a pinned RTT tree."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .hir import SOURCES, audit_hir, classify_hir_row

PINNED_HIR_REVISION = "2a95f69eb56cc47edc16a45f939cde479673a4cb"
PINNED_HIR_SHA256 = "465a01c19dc29e2c8d1cf183ccf3135872f7ec94ef10b20b7eb35603164c183b"
PINNED_RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
PINNED_RTT_REPOSITORY = "https://github.com/TURLEing/Rubrics-To-Tokens"
EXPECTED_ROUTE_DIGEST = "a367d5e688fa2996543b123ce7491c8e878515597751bcf38f5b33f9e57d3e22"

_REWARD_ROOT = Path("roll/pipeline/rlvr/rewards")
_ROUTE_FILES = {
    "worker": _REWARD_ROOT / "rubrics_llm_judge_reward_worker.py",
    "ifeval": _REWARD_ROOT / "ifeval_rule_reward_worker.py",
    "type2": _REWARD_ROOT / "type2_checkers.py",
    "type3": _REWARD_ROOT / "muldimif_checkers/__init__.py",
}


class RouteAuditError(ValueError):
    """Raised when route-resolution evidence is unresolved or inconsistent."""


@dataclass(frozen=True)
class RouteAudit:
    """Compact deterministic route-resolution inventory, not an evaluator certificate."""

    schema_version: int
    status: str
    sources: dict[str, Any]
    counts: dict[str, Any]
    routes: tuple[dict[str, Any], ...]
    unsupported_identities: tuple[dict[str, Any], ...]
    digest: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary."""
        value = asdict(self)
        value["routes"] = list(self.routes)
        value["unsupported_identities"] = list(self.unsupported_identities)
        return value

    def resolution_artifact(self) -> dict[str, Any]:
        """Return the compact artifact consumed by the experiment program."""
        return {
            "schema_version": 1,
            "id": "hir_route_resolution_v1",
            "status": self.status,
            "taxonomy_sha256": self.sources["hir"]["taxonomy_sha256"],
            "rtt_revision": self.sources["rtt"]["revision"],
            "rtt_tree": self.sources["rtt"]["tree"],
            "route_digest": self.digest["sha256"],
            "counts": {
                "hard": self.counts["hard"],
                "route_resolvable": self.counts["route_resolvable"],
                "unresolved": self.counts["unresolved"],
            },
            "unsupported_identities": list(self.unsupported_identities),
        }


def audit_route_inventory(
    source_path: str | Path,
    taxonomy_path: str | Path,
    manifest_path: str | Path,
    rtt_root: str | Path,
) -> RouteAudit:
    """Verify source identities and derive the pinned RTT route-resolution inventory."""
    source_path = Path(source_path).resolve()
    taxonomy = _load_json(Path(taxonomy_path).resolve())
    manifest = _load_json(Path(manifest_path).resolve())
    route_config = _object(taxonomy.get("static_rtt_route_audit"), "taxonomy.static_rtt_route_audit")
    expected_revision = _string(route_config.get("revision"), "route audit revision")
    checkout = verify_rtt_checkout(rtt_root, expected_revision)
    _validate_sources(source_path, taxonomy, manifest, route_config)

    with source_path.open(encoding="utf-8") as source:
        taxonomy_audit = audit_hir(source)
    expected_taxonomy = _object(taxonomy.get("expected"), "taxonomy.expected")
    actual_taxonomy = asdict(taxonomy_audit)
    taxonomy_digest = actual_taxonomy.pop("digest")
    _expect(actual_taxonomy == expected_taxonomy, "HIR taxonomy count mismatch")
    _expect(
        taxonomy_digest == _object(taxonomy.get("taxonomy_digest"), "taxonomy.taxonomy_digest").get("sha256"),
        "HIR taxonomy digest mismatch",
    )

    maps = load_rtt_route_maps(Path(rtt_root).resolve())
    routes, identities, counts = _derive_routes(source_path, maps, actual_taxonomy)
    digest = route_digest(routes)
    _expect(digest == EXPECTED_ROUTE_DIGEST, "route digest mismatch")
    _validate_expected_counts(counts, routes, route_config)
    status = "route_resolvable" if counts["unresolved"] == 0 else "route_resolvable_with_gaps"
    return RouteAudit(
        schema_version=1,
        status=status,
        sources={
            "hir": {
                "dataset": manifest["dataset"],
                "revision": manifest["revision"],
                "sha256": PINNED_HIR_SHA256,
                "taxonomy_sha256": taxonomy_digest,
            },
            "rtt": {
                "repository": PINNED_RTT_REPOSITORY,
                "revision": checkout["revision"],
                "tree": checkout["tree"],
            },
        },
        counts=counts,
        routes=routes,
        unsupported_identities=identities,
        digest={
            "algorithm": "sha256",
            "encoding": "ordered UTF-8 JSON lines of aggregate route rows with sorted keys and compact separators",
            "sha256": digest,
        },
    )


def require_resolved_routes(audit: RouteAudit) -> None:
    """Fail closed unless every hard rubric has a statically resolvable route."""
    _expect(audit.status == "route_resolvable" and audit.counts["unresolved"] == 0, "unresolved hard routes remain")


def verify_rtt_checkout(rtt_root: str | Path, expected_revision: str) -> dict[str, str]:
    """Require an exact clean Git checkout before inspecting route code."""
    root = Path(rtt_root).resolve()
    top_level = _git(root, "rev-parse", "--show-toplevel")
    _expect(Path(top_level).resolve() == root, "RTT path is not the checkout root")
    revision = _git(root, "rev-parse", "HEAD")
    _expect(revision == expected_revision, f"wrong RTT revision: {revision}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    _expect(not status, "RTT checkout is dirty")
    return {"revision": revision, "tree": _git(root, "rev-parse", "HEAD^{tree}")}


def load_rtt_route_maps(root: str | Path) -> dict[str, Any]:
    """Parse RTT route maps without importing or executing the checkout."""
    root = Path(root).resolve()
    worker = _assignment_dict(root / _ROUTE_FILES["worker"], "INSTRUCTION_ID_TO_IFEVAL")
    ifeval = _string_dict_keys(_assignment_dict(root / _ROUTE_FILES["ifeval"], "IF_FUNCTIONS_MAP"))
    type1: dict[str, str] = {}
    for key_node, value_node in zip(worker.keys, worker.values, strict=True):
        key = _literal_string(key_node, "type1 route key")
        _expect(isinstance(value_node, ast.Tuple) and value_node.elts, f"invalid type1 route for {key}")
        type1[key] = _literal_string(value_node.elts[0], f"type1 target for {key}")
    _expect(len(type1) == len(worker.keys), "duplicate type1 route key")
    return {
        "type1": type1,
        "ifeval": ifeval,
        "type2": _string_dict_keys(_assignment_dict(root / _ROUTE_FILES["type2"], "TYPE2_CHECKERS")),
        "type3": _string_dict_keys(_assignment_dict(root / _ROUTE_FILES["type3"], "CONSTRAINT_CHECKER_MAP")),
    }


def route_digest(routes: Iterable[dict[str, Any]]) -> str:
    """Hash ordered aggregate route rows."""
    digest = hashlib.sha256()
    for row in routes:
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(f"{encoded}\n".encode())
    return digest.hexdigest()


def _validate_sources(
    source_path: Path,
    taxonomy: dict[str, Any],
    manifest: dict[str, Any],
    route_config: dict[str, Any],
) -> None:
    source = _object(taxonomy.get("source"), "taxonomy.source")
    _expect(source.get("path") == "data/HIR_trainv1.jsonl", "unresolved taxonomy source path")
    _expect(source.get("sha256") == PINNED_HIR_SHA256, "unresolved taxonomy source hash")
    _expect(_sha256(source_path) == PINNED_HIR_SHA256, "HIR source hash mismatch")
    _expect(
        manifest.get("dataset") == "sastpg/HIR-16K" and manifest.get("revision") == PINNED_HIR_REVISION,
        "unresolved HIR dataset identity",
    )
    raw_source = _object(manifest.get("source"), "manifest.source")
    _expect(
        raw_source.get("path") == "HIR_trainv1.jsonl" and raw_source.get("sha256") == PINNED_HIR_SHA256,
        "unresolved HIR source manifest",
    )
    processed = _object(manifest.get("rtt_processed"), "manifest.rtt_processed")
    _expect(
        processed.get("repository") == PINNED_RTT_REPOSITORY and processed.get("revision") == PINNED_RTT_REVISION,
        "unresolved processed HIR identity",
    )
    _expect(
        route_config.get("repository") == PINNED_RTT_REPOSITORY
        and route_config.get("revision") == PINNED_RTT_REVISION
        and route_config.get("status") == "route_resolvable_inventory",
        "unresolved RTT route source identity",
    )


def _derive_routes(
    source_path: Path,
    maps: dict[str, Any],
    taxonomy: dict[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    grouped: dict[tuple[str, str, bool, str], dict[str, Any]] = defaultdict(lambda: {"criteria": 0, "rows": set()})
    identities: list[dict[str, Any]] = []
    source_counts = {source: {"route_resolvable": 0, "unresolved": 0} for source in SOURCES}
    unsupported_rows: set[int | str] = set()
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                hard_mask = classify_hir_row(row)
            except (json.JSONDecodeError, ValueError) as error:
                raise RouteAuditError(f"line {line_number}: {error}") from error
            for index, hard in enumerate(hard_mask):
                if not hard:
                    continue
                route, resolvable, reason = _route_for(row, index, maps)
                grouped[(row["source"], route, resolvable, reason)]["criteria"] += 1
                grouped[(row["source"], route, resolvable, reason)]["rows"].add(row["id"])
                state = "route_resolvable" if resolvable else "unresolved"
                source_counts[row["source"]][state] += 1
                if not resolvable:
                    unsupported_rows.add(row["id"])
                    identities.append(
                        {"source": row["source"], "row_id": row["id"], "rubric_index": index, "route": route}
                    )
    routes = tuple(
        {
            "source": source,
            "route": route,
            "route_resolvable": resolvable,
            "reason": reason,
            "criteria": values["criteria"],
            "rows": len(values["rows"]),
        }
        for (source, route, resolvable, reason), values in sorted(grouped.items())
    )
    stable_identities = tuple(sorted(identities, key=_identity_key))
    resolved = sum(value["route_resolvable"] for value in source_counts.values())
    unresolved = sum(value["unresolved"] for value in source_counts.values())
    return (
        routes,
        stable_identities,
        {
            "rows": taxonomy["rows"],
            "criteria": taxonomy["criteria"],
            "hard": taxonomy["hard"],
            "soft": taxonomy["soft"],
            "route_resolvable": resolved,
            "unresolved": unresolved,
            "unique_rows_with_unsupported": len(unsupported_rows),
            "sources": source_counts,
        },
    )


def _route_for(row: dict[str, Any], index: int, maps: dict[str, Any]) -> tuple[str, bool, str]:
    source = row["source"]
    ground_truth = row["ground_truth"]
    if source == "type1":
        route = ground_truth["instruction_id_list"][index]
        target = maps["type1"].get(route)
        resolvable = target is not None and target in maps["ifeval"]
    elif source == "type2":
        route = ground_truth["instruction_id_list"][index]
        resolvable = route in maps["type2"]
    elif source == "type3":
        constraint = ground_truth["constraints"][index]
        route = f"{constraint[0]}_{constraint[1]}"
        resolvable = route in maps["type3"]
    else:
        route = "embedded_check_following"
        resolvable = _defines_check_following(ground_truth["functions"][index])
    return route, resolvable, "static_route_resolved" if resolvable else "static_route_missing"


def _validate_expected_counts(
    counts: dict[str, Any],
    routes: tuple[dict[str, Any], ...],
    expected: dict[str, Any],
) -> None:
    actual = {
        "supported": counts["route_resolvable"],
        "unsupported": counts["unresolved"],
        "unique_rows_with_unsupported": counts["unique_rows_with_unsupported"],
        "sources": {
            source: {
                "supported": values["route_resolvable"],
                "unsupported": values["unresolved"],
            }
            for source, values in counts["sources"].items()
        },
        "unsupported_keys": {
            source: {
                row["route"]: row["criteria"]
                for row in routes
                if row["source"] == source and not row["route_resolvable"]
            }
            for source in SOURCES
        },
    }
    pinned = {key: expected.get(key) for key in actual}
    _expect(actual == pinned, "route count mismatch")
    _expect(counts["route_resolvable"] + counts["unresolved"] == counts["hard"], "hard route total mismatch")


def _identity_key(identity: dict[str, Any]) -> str:
    return json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assignment_dict(path: Path, name: str) -> ast.Dict:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise RouteAuditError(f"cannot parse {path}: {error}") from error
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Dict)
        and any(isinstance(target, ast.Name) and target.id == name for target in _targets(node))
    ]
    _expect(len(matches) == 1, f"cannot resolve {name} in {path}")
    return matches[0]


def _targets(node: ast.Assign | ast.AnnAssign) -> Iterable[ast.expr]:
    return node.targets if isinstance(node, ast.Assign) else (node.target,)


def _string_dict_keys(node: ast.Dict) -> set[str]:
    keys = [_literal_string(key, "route key") for key in node.keys]
    _expect(len(keys) == len(set(keys)), "duplicate route key")
    return set(keys)


def _literal_string(node: ast.expr | None, name: str) -> str:
    _expect(isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value, f"invalid {name}")
    return node.value


def _defines_check_following(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check_following"
    ]
    if len(matches) != 1:
        return False
    arguments = matches[0].args
    positional = len(arguments.posonlyargs) + len(arguments.args)
    required = positional - len(arguments.defaults)
    return required <= 2 and (positional >= 2 or arguments.vararg is not None)


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RouteAuditError(f"cannot inspect RTT checkout: {error}") from error
    return result.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file, object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise RouteAuditError(f"cannot load {path}: {error}") from error
    return _object(value, str(path))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _expect(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RouteAuditError(f"cannot read {path}: {error}") from error
    return digest.hexdigest()


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RouteAuditError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RouteAuditError(f"{name} must be a non-empty string")
    return value


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RouteAuditError(message)
