#!/usr/bin/env python3
"""Certify Qwen prompt lengths for the English two-route RubricHub subset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load data module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_data = _load_module("rdan_grpo_rubrichub_data", ROOT / "src/rdan_grpo/rubrichub_data.py")
_rules = _load_module("rdan_grpo_rubrichub_rules", ROOT / "src/rdan_grpo/rubrichub_rules.py")
source_row_hash = _data.source_row_hash
RubricHubRuleError = _rules.RubricHubRuleError
verify_rule_certificate = _rules.verify_rule_certificate

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TRANSFORMERS_VERSION = "4.57.0"
TOKENIZER_FILES = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
}
MIN_TOKENS = 5
MAX_TOKENS = 2_048
EXPECTED_CANDIDATES = 1_142
EXPECTED_ACCEPTED = 1_134


def main() -> None:
    """Build or verify the tokenizer evidence and compact certificate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/rubrichub_instruction_following.json")
    parser.add_argument(
        "--checker-certificate",
        type=Path,
        default=ROOT / "configs/artifacts/rubrichub_rule_certificate.json",
    )
    parser.add_argument(
        "--language-certificate",
        type=Path,
        default=ROOT / "data/rubrichub-source/language-id/rubrichub_instruction_following_english_certificate.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots" / MODEL_REVISION,
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "data/rubrichub-source/certificates/rubrichub_tokenizer_evidence.jsonl",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=ROOT / "configs/artifacts/rubrichub_tokenizer_certificate.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = _read_json(args.config)
    if args.check:
        certificate = verify_certificate(
            args.certificate,
            args.evidence,
            args.checker_certificate,
            args.language_certificate,
            args.model_path,
            config["source"],
        )
    else:
        certificate = build_certificate(
            config,
            args.checker_certificate,
            args.language_certificate,
            args.model_path,
            args.evidence,
            args.certificate,
        )
    print(json.dumps(_summary(certificate), sort_keys=True, separators=(",", ":")))


def build_certificate(
    config: dict[str, Any],
    checker_path: Path,
    language_path: Path,
    model_path: Path,
    evidence_path: Path,
    certificate_path: Path,
) -> dict[str, Any]:
    """Generate exact chat-template token counts for all certified candidates."""

    source = config["source"]
    parquet_path = _resolve(ROOT, source["path"])
    _verify_file(parquet_path, source["bytes"], source["sha256"], "RubricHub parquet")
    checker = verify_rule_certificate(checker_path, source)
    selection = checker.get("candidate_selection", {})
    indices = selection.get("source_indices")
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise RubricHubRuleError("checker candidate indices are missing")
    if indices != sorted(set(indices)) or len(indices) != EXPECTED_CANDIDATES:
        raise RubricHubRuleError("checker candidate indices differ from the frozen subset")
    language_ref = selection.get("language_certificate")
    if language_ref != _file_ref(language_path):
        raise RubricHubRuleError("checker candidate selection uses another English certificate")

    tokenizer, runtime = _load_tokenizer(model_path)
    rows = _selected_rows(parquet_path, indices, source["records"])
    evidence = []
    for index in indices:
        row = rows[index]
        prompt = _prompt(row)
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(token_ids, list) or not all(isinstance(token, int) for token in token_ids):
            raise RubricHubRuleError(f"tokenizer returned malformed input ids for row {index}")
        tokens = len(token_ids)
        accepted = MIN_TOKENS < tokens <= MAX_TOKENS
        evidence.append(
            {
                "source_index": index,
                "source_row_sha256": source_row_hash(row),
                "messages_sha256": _digest(messages),
                "input_ids_sha256": _digest(token_ids),
                "input_tokens": tokens,
                "accepted": accepted,
                "reason": "accepted" if accepted else _rejection_reason(tokens),
            }
        )
    accepted = [item for item in evidence if item["accepted"]]
    rejected = [item for item in evidence if not item["accepted"]]
    if len(accepted) != EXPECTED_ACCEPTED:
        raise RubricHubRuleError(
            f"accepted count differs from frozen tokenizer gate: {len(accepted)} != {EXPECTED_ACCEPTED}"
        )
    evidence_ref = _write_jsonl(evidence_path, evidence)
    certificate = {
        "schema_version": 1,
        "id": "rubrichub_qwen3_4b_instruct_2507_tokenizer_v1",
        "status": "frozen",
        "source": _certificate_source(source),
        "selection": {
            "checker_certificate": _file_ref(checker_path),
            "language_certificate": _file_ref(language_path),
            "candidate_rows": len(evidence),
            "candidate_indices_sha256": _digest(indices),
            "candidate_source_hashes_sha256": _digest([item["source_row_sha256"] for item in evidence]),
        },
        "tokenizer": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "files_sha256": TOKENIZER_FILES,
            "transformers_version": TRANSFORMERS_VERSION,
            "chat_template": {
                "messages": "one_user_message",
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
            "acceptance": "5 < input_tokens <= 2048",
        },
        "runtime": runtime,
        "generator": {
            "path": "scripts/certify_rubrichub_tokenizer.py",
            "sha256": _sha256(Path(__file__)),
        },
        "evidence": evidence_ref,
        "results": {
            "candidates": len(evidence),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "frozen_expected_candidates": EXPECTED_CANDIDATES,
            "frozen_expected_accepted": EXPECTED_ACCEPTED,
            "accepted_source_indices": [item["source_index"] for item in accepted],
            "accepted_source_indices_sha256": _digest([item["source_index"] for item in accepted]),
            "rejected_rows": rejected,
            "largest_accepted_input_tokens": max(item["input_tokens"] for item in accepted),
        },
    }
    _write_json(certificate_path, certificate)
    return verify_certificate(
        certificate_path,
        evidence_path,
        checker_path,
        language_path,
        model_path,
        source,
    )


def verify_certificate(
    certificate_path: Path,
    evidence_path: Path,
    checker_path: Path,
    language_path: Path,
    model_path: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Verify immutable identities and every compact-to-full evidence digest."""

    certificate = _read_json(certificate_path)
    if certificate.get("schema_version") != 1 or certificate.get("status") != "frozen":
        raise RubricHubRuleError("tokenizer certificate status is invalid")
    if certificate.get("source") != _certificate_source(source):
        raise RubricHubRuleError("tokenizer certificate source is stale")
    if certificate.get("generator", {}).get("sha256") != _sha256(Path(__file__)):
        raise RubricHubRuleError("tokenizer certificate generator is stale")
    selection = certificate.get("selection", {})
    verify_rule_certificate(checker_path, source)
    if selection.get("checker_certificate") != _file_ref(checker_path):
        raise RubricHubRuleError("tokenizer certificate checker identity is stale")
    if selection.get("language_certificate") != _file_ref(language_path):
        raise RubricHubRuleError("tokenizer certificate language identity is stale")
    _verify_tokenizer_files(model_path)
    expected_tokenizer = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "files_sha256": TOKENIZER_FILES,
        "transformers_version": TRANSFORMERS_VERSION,
        "chat_template": {
            "messages": "one_user_message",
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "acceptance": "5 < input_tokens <= 2048",
    }
    if certificate.get("tokenizer") != expected_tokenizer:
        raise RubricHubRuleError("tokenizer certificate contract is stale")
    evidence_ref = certificate.get("evidence")
    if not isinstance(evidence_ref, dict) or evidence_ref.get("sha256") != _sha256(evidence_path):
        raise RubricHubRuleError("tokenizer evidence was tampered")
    if evidence_ref.get("bytes") != evidence_path.stat().st_size:
        raise RubricHubRuleError("tokenizer evidence size changed")
    rows = []
    try:
        for line in evidence_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RubricHubRuleError("tokenizer evidence row is malformed")
            rows.append(row)
    except json.JSONDecodeError as error:
        raise RubricHubRuleError("tokenizer evidence JSON is malformed") from error
    results = certificate.get("results", {})
    if evidence_ref.get("records") != len(rows) or results.get("candidates") != len(rows):
        raise RubricHubRuleError("tokenizer evidence count changed")
    accepted = [row["source_index"] for row in rows if row.get("accepted") is True]
    indices = [row.get("source_index") for row in rows]
    if indices != sorted(set(indices)) or any(not _valid_evidence_acceptance(row) for row in rows):
        raise RubricHubRuleError("tokenizer evidence acceptance contract changed")
    if selection.get("candidate_rows") != len(rows) or selection.get("candidate_indices_sha256") != _digest(indices):
        raise RubricHubRuleError("tokenizer candidate inventory changed")
    if selection.get("candidate_source_hashes_sha256") != _digest([row.get("source_row_sha256") for row in rows]):
        raise RubricHubRuleError("tokenizer candidate source hashes changed")
    if results.get("accepted_source_indices") != accepted or results.get("accepted") != len(accepted):
        raise RubricHubRuleError("tokenizer accepted row inventory changed")
    if results.get("accepted_source_indices_sha256") != _digest(accepted):
        raise RubricHubRuleError("tokenizer accepted row digest changed")
    if results.get("rejected") != len(rows) - len(accepted):
        raise RubricHubRuleError("tokenizer rejected row count changed")
    return certificate


def _load_tokenizer(model_path: Path) -> tuple[Any, dict[str, str]]:
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RubricHubRuleError("Transformers is required to build the tokenizer certificate") from error
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RubricHubRuleError(f"Transformers version differs: {transformers.__version__} != {TRANSFORMERS_VERSION}")
    _verify_tokenizer_files(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return tokenizer, {
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "implementation": platform.python_implementation(),
    }


def _verify_tokenizer_files(model_path: Path) -> None:
    for name, digest in TOKENIZER_FILES.items():
        path = model_path / name
        if not path.is_file() or _sha256(path) != digest:
            raise RubricHubRuleError(f"tokenizer file differs from the frozen revision: {name}")


def _selected_rows(path: Path, indices: list[int], expected_rows: int) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RubricHubRuleError("pyarrow is required to scan the pinned RubricHub parquet") from error
    selected = set(indices)
    rows = {}
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches():
        for row in batch.to_pylist():
            if seen in selected:
                rows[seen] = row
            seen += 1
    if seen != expected_rows or set(rows) != selected:
        raise RubricHubRuleError("tokenizer source selection differs from the pinned dataset")
    return rows


def _prompt(row: dict[str, Any]) -> str:
    messages = row.get("prompt")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise RubricHubRuleError("RubricHub candidate prompt is malformed")
    return messages[0]["content"]


def _rejection_reason(tokens: int) -> str:
    return "input_tokens_too_short" if tokens <= MIN_TOKENS else "input_tokens_exceed_prompt_length"


def _valid_evidence_acceptance(row: dict[str, Any]) -> bool:
    tokens = row.get("input_tokens")
    accepted = row.get("accepted")
    return (
        isinstance(tokens, int)
        and not isinstance(tokens, bool)
        and isinstance(accepted, bool)
        and accepted == (MIN_TOKENS < tokens <= MAX_TOKENS)
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    digest = hashlib.sha256()
    count = 0
    size = 0
    try:
        with os.fdopen(handle, "wb") as stream:
            for record in records:
                line = _canonical(record) + b"\n"
                stream.write(line)
                digest.update(line)
                count += 1
                size += len(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return {
        "path": "data/rubrichub-source/certificates/rubrichub_tokenizer_evidence.jsonl",
        "records": count,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RubricHubRuleError(f"cannot read JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RubricHubRuleError(f"JSON root must be an object: {path}")
    return payload


def _certificate_source(source: dict[str, Any]) -> dict[str, Any]:
    return {key: source[key] for key in ("dataset", "revision", "file", "sha256", "records")}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_file(path: Path, size: int, digest: str, label: str) -> None:
    if not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
        raise RubricHubRuleError(f"{label} differs from its frozen pin")


def _file_ref(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _summary(certificate: dict[str, Any]) -> dict[str, Any]:
    results = certificate["results"]
    return {
        "id": certificate["id"],
        "status": certificate["status"],
        "candidates": results["candidates"],
        "accepted": results["accepted"],
        "rejected": results["rejected"],
        "evidence_sha256": certificate["evidence"]["sha256"],
    }


if __name__ == "__main__":
    try:
        main()
    except RubricHubRuleError as error:
        raise SystemExit(f"RubricHub tokenizer certification blocked: {error}") from error
