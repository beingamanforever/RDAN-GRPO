#!/usr/bin/env python3
"""Certify Qwen prompt lengths for every processed HIR row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TRANSFORMERS_VERSION = "4.57.0"
TOKENIZER_FILES = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
}
MIN_TOKENS = 5
MAX_TOKENS = 2_048
EXPECTED_ROWS = 16_968
EXPECTED_ACCEPTED = 16_962


class CertificationError(ValueError):
    """Raised when the frozen HIR tokenizer evidence is inconsistent."""


def main() -> None:
    """Build or verify the full HIR tokenizer certificate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=ROOT / "data/HIR_trainv1_rubrics_processed.jsonl")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots" / MODEL_REVISION,
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "data/hir-certificates/hir_qwen_tokenizer_evidence.jsonl",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=ROOT / "configs/artifacts/hir_qwen_tokenizer_certificate.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        certificate = verify_certificate(args.processed, args.model_path, args.evidence, args.certificate)
    else:
        certificate = build_certificate(args.processed, args.model_path, args.evidence, args.certificate)
    print(json.dumps(_summary(certificate), sort_keys=True, separators=(",", ":")))


def build_certificate(
    processed_path: Path,
    model_path: Path,
    evidence_path: Path,
    certificate_path: Path,
) -> dict[str, Any]:
    """Measure the exact non-thinking Qwen prompt length for every HIR row."""

    tokenizer, runtime = _load_tokenizer(model_path)
    evidence = []
    row_ids = []
    for row in _rows(processed_path):
        row_id = _row_id(row)
        messages = _messages(row, row_id)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = tokenizer(text)["input_ids"]
        if not isinstance(token_ids, list) or not all(isinstance(token, int) for token in token_ids):
            raise CertificationError(f"row {row_id!r}: tokenizer returned malformed input ids")
        tokens = len(token_ids)
        accepted = MIN_TOKENS < tokens <= MAX_TOKENS
        row_ids.append(row_id)
        evidence.append(
            {
                "row_id": row_id,
                "source": row.get("source"),
                "messages_sha256": _digest(messages),
                "input_ids_sha256": _digest(token_ids),
                "input_tokens": tokens,
                "accepted": accepted,
                "reason": "accepted" if accepted else _rejection_reason(tokens),
            }
        )
    accepted = [row for row in evidence if row["accepted"]]
    rejected = [row for row in evidence if not row["accepted"]]
    if len(evidence) != EXPECTED_ROWS or len(accepted) != EXPECTED_ACCEPTED:
        raise CertificationError(f"HIR tokenizer counts differ: rows={len(evidence)}, accepted={len(accepted)}")
    evidence_ref = _write_jsonl(evidence_path, evidence)
    certificate = {
        "schema_version": 1,
        "id": "hir_qwen3_4b_instruct_2507_tokenizer_v1",
        "status": "frozen",
        "source": _file_ref(processed_path, records=len(evidence)),
        "tokenizer": _tokenizer_contract(),
        "runtime": runtime,
        "generator": {"path": "scripts/certify_hir_tokenizer.py", "sha256": _sha256(Path(__file__))},
        "evidence": evidence_ref,
        "results": {
            "input_rows": len(evidence),
            "input_row_ids_sha256": _digest(row_ids),
            "accepted": len(accepted),
            "accepted_row_ids_sha256": _digest([row["row_id"] for row in accepted]),
            "rejected": len(rejected),
            "rejected_rows": rejected,
            "largest_accepted_input_tokens": max(row["input_tokens"] for row in accepted),
        },
    }
    _write_json(certificate_path, certificate)
    return verify_certificate(processed_path, model_path, evidence_path, certificate_path)


def verify_certificate(
    processed_path: Path,
    model_path: Path,
    evidence_path: Path,
    certificate_path: Path,
) -> dict[str, Any]:
    """Verify the source, tokenizer, evidence, and compact result inventory."""

    certificate = _read_json(certificate_path)
    if (
        certificate.get("schema_version") != 1
        or certificate.get("id") != "hir_qwen3_4b_instruct_2507_tokenizer_v1"
        or certificate.get("status") != "frozen"
    ):
        raise CertificationError("HIR tokenizer certificate status is invalid")
    rows = _rows(processed_path)
    if certificate.get("source") != _file_ref(processed_path, records=len(rows)):
        raise CertificationError("HIR tokenizer source identity changed")
    if certificate.get("tokenizer") != _tokenizer_contract():
        raise CertificationError("HIR tokenizer contract changed")
    _verify_tokenizer_files(model_path)
    if certificate.get("generator") != {
        "path": "scripts/certify_hir_tokenizer.py",
        "sha256": _sha256(Path(__file__)),
    }:
        raise CertificationError("HIR tokenizer certificate generator changed")
    evidence_ref = certificate.get("evidence")
    if not isinstance(evidence_ref, dict) or evidence_ref != _file_ref(evidence_path, records=len(rows)):
        raise CertificationError("HIR tokenizer evidence was tampered")
    evidence = _rows(evidence_path)
    if len(evidence) != len(rows):
        raise CertificationError("HIR tokenizer evidence count changed")
    accepted_ids = []
    for source, item in zip(rows, evidence, strict=True):
        row_id = _row_id(source)
        if item.get("row_id") != row_id or item.get("source") != source.get("source"):
            raise CertificationError("HIR tokenizer evidence order or source changed")
        if item.get("messages_sha256") != _digest(_messages(source, row_id)):
            raise CertificationError(f"row {row_id!r}: tokenizer message identity changed")
        if not _valid_acceptance(item):
            raise CertificationError(f"row {row_id!r}: tokenizer acceptance changed")
        if item["accepted"]:
            accepted_ids.append(row_id)
    results = certificate.get("results", {})
    rejected = [item for item in evidence if not item["accepted"]]
    expected = {
        "input_rows": len(rows),
        "input_row_ids_sha256": _digest([_row_id(row) for row in rows]),
        "accepted": len(accepted_ids),
        "accepted_row_ids_sha256": _digest(accepted_ids),
        "rejected": len(rejected),
        "rejected_rows": rejected,
        "largest_accepted_input_tokens": max(item["input_tokens"] for item in evidence if item["accepted"]),
    }
    if results != expected or len(rows) != EXPECTED_ROWS or len(accepted_ids) != EXPECTED_ACCEPTED:
        raise CertificationError("HIR tokenizer result inventory changed")
    return certificate


def _load_tokenizer(model_path: Path) -> tuple[Any, dict[str, str]]:
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise CertificationError("Transformers is required for HIR tokenizer certification") from error
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise CertificationError(f"Transformers version differs: {transformers.__version__}")
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
            raise CertificationError(f"tokenizer file differs from the frozen revision: {name}")


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CertificationError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise CertificationError(f"cannot read JSONL: {path}") from error
    return rows


def _row_id(row: dict[str, Any]) -> int | str:
    row_id = row.get("id")
    if isinstance(row_id, bool) or not isinstance(row_id, (int, str)):
        raise CertificationError("HIR row ID is invalid")
    return row_id


def _messages(row: dict[str, Any], row_id: int | str) -> list[dict[str, str]]:
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise CertificationError(f"row {row_id!r}: messages are malformed")
    return messages


def _valid_acceptance(row: dict[str, Any]) -> bool:
    tokens = row.get("input_tokens")
    accepted = row.get("accepted")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not isinstance(accepted, bool):
        return False
    reason = "accepted" if accepted else _rejection_reason(tokens)
    return accepted == (MIN_TOKENS < tokens <= MAX_TOKENS) and row.get("reason") == reason


def _rejection_reason(tokens: int) -> str:
    return "input_tokens_too_short" if tokens <= MIN_TOKENS else "input_tokens_exceed_prompt_length"


def _tokenizer_contract() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "files_sha256": TOKENIZER_FILES,
        "transformers_version": TRANSFORMERS_VERSION,
        "chat_template": {
            "messages": "one_user_message",
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "tokenizer_add_special_tokens": True,
        },
        "acceptance": "5 < input_tokens <= 2048",
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b"".join(_canonical(record) + b"\n" for record in records)
    _atomic_write(path, body)
    return {"path": _display(path), "records": body.count(b"\n"), "bytes": len(body), "sha256": _digest_bytes(body)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CertificationError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise CertificationError(f"JSON root is not an object: {path}")
    return value


def _file_ref(path: Path, records: int | None = None) -> dict[str, Any]:
    reference = {"path": _display(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if records is not None:
        reference["records"] = records
    return reference


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _digest(value: Any) -> str:
    return _digest_bytes(_canonical(value))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(certificate: dict[str, Any]) -> dict[str, Any]:
    results = certificate["results"]
    return {
        "id": certificate["id"],
        "input_rows": results["input_rows"],
        "accepted": results["accepted"],
        "rejected": results["rejected"],
        "evidence_sha256": certificate["evidence"]["sha256"],
    }


if __name__ == "__main__":
    try:
        main()
    except CertificationError as error:
        raise SystemExit(f"HIR tokenizer certification blocked: {error}") from error
