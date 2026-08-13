"""Fail-closed byte receipts for the pinned ROLL FSDP2 to HF update."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY, write_artifact

RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_IDENTITY_KEYS = {
    "model",
    "revision",
    "snapshot_sha256",
    "tokenizer_files_sha256",
    "chat_template_sha256",
}
EXPECTED_PAIRS = ((0, 0), (1, 1))
RTT_BOUNDARY_SHA256 = {
    "roll/configs/worker_config.py": "e9aa9e95a0575dfde86d2c0055eac69653a6b5ae1dd550fd8952f006c20a285d",
    "roll/distributed/executor/model_update_group.py": (
        "7ee9051ac6d778f619ad80345855eef1c98885e45a4c67e35096c38c84fed91f"
    ),
    "roll/distributed/executor/worker.py": "2a808864f9581d120d877afd01e93c3649414a3859d2c40d013d33dcdec5fc26",
    "roll/distributed/strategy/factory.py": "aeb00dabc2b9f6c35c797ef6b92b980b857ca330267b13f5ae5eb71154515fd5",
    "roll/distributed/strategy/fsdp2_strategy.py": (
        "179a570cd383ef846d1e20caa249af07052a0a6bfe4eb89f5beb680b03c07899"
    ),
    "roll/distributed/strategy/hf_strategy.py": "7a0c65db35a1a9afcbb6bdddc22211af29b50e2b1ab3d1354cdbf3c660e7d7b6",
    "roll/distributed/strategy/strategy.py": "3f3fdc874ee8648885bb3e2d2e4e94d4f10ee02832b76b3cf5b3c0d62a8b3d28",
    "roll/pipeline/base_worker.py": "161c8aabae67a70283219cc961178fb57488cc8acad1176f804624f122cfedcb",
    "roll/third_party/fsdp2/model_update.py": "7f1acb4bd549681f19bc8faaa4fe62e70d097c3af7a1096d3636f537db5bf0c1",
    "roll/utils/cuda_ipc_utils.py": "31a667073d8d6985797f1ef67d9738c976ee48385e782d1efb3ff8665ace00b2",
    "roll/utils/send_recv_utils.py": "0918010bdcb713648c226271fb6ef632f512b32917addce5e82880dce8e65dc7",
}


class FSDPHFReceiptError(ValueError):
    """Raised when the diagnostic cannot prove an exact FSDP2 to HF receipt."""


@dataclass(frozen=True)
class FSDPHFTransaction:
    """Identify one actor and HF inference rank pair."""

    transaction_id: str
    actor_rank: int
    infer_rank: int

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise FSDPHFReceiptError("transaction id must be non-empty")
        if (self.actor_rank, self.infer_rank) not in EXPECTED_PAIRS:
            raise FSDPHFReceiptError(
                "FSDP2 to HF receipt requires identity pairing, "
                f"got actor {self.actor_rank} to infer {self.infer_rank}"
            )


class FSDPHFStreamReceipt:
    """Record one actor gather stream or one final HF parameter sequence."""

    def __init__(self, transaction: FSDPHFTransaction, side: str, accelerator_name: str | None) -> None:
        if side not in {"actor", "infer"}:
            raise FSDPHFReceiptError(f"invalid receipt side: {side}")
        self.transaction = transaction
        self.side = side
        self.accelerator_name = accelerator_name
        self.items: list[dict[str, Any]] = []
        self.stream_started = False
        self.stream_complete = False
        self.calls = 0

    def open_actor_stream(self) -> None:
        if self.side != "actor":
            raise FSDPHFReceiptError("only an actor receipt can wrap gather_fsdp2_weights")
        self.calls += 1
        if self.calls != 1 or self.stream_started or self.stream_complete:
            raise FSDPHFReceiptError("gather_fsdp2_weights must be called exactly once per transaction")
        self.stream_started = True

    def wrap_actor_batches(self, batches: Iterable[Any]) -> Iterable[Any]:
        """Hash raw batches as the real FSDP2 update consumes them."""

        for batch in batches:
            for name, tensor in _named_tensors(batch):
                self.items.append(tensor_manifest_entry(name, tensor, len(self.items)))
            yield batch
        self.stream_complete = True

    def finish_infer(self, named_parameters: Iterable[tuple[str, torch.Tensor]]) -> None:
        if self.side != "infer":
            raise FSDPHFReceiptError("only an infer receipt can record final HF parameters")
        self.calls += 1
        if self.calls != 1 or self.stream_started or self.stream_complete:
            raise FSDPHFReceiptError("HF final parameters must be recorded exactly once per transaction")
        self.stream_started = True
        for name, tensor in named_parameters:
            self.items.append(tensor_manifest_entry(name, tensor, len(self.items)))
        self.stream_complete = True

    def snapshot(self) -> dict[str, Any]:
        summary = manifest_summary(self.items)
        rank = self.transaction.actor_rank if self.side == "actor" else self.transaction.infer_rank
        paired_rank = self.transaction.infer_rank if self.side == "actor" else self.transaction.actor_rank
        return {
            "transaction_id": self.transaction.transaction_id,
            "side": self.side,
            "rank": rank,
            "paired_rank": paired_rank,
            "accelerator_name": self.accelerator_name,
            "stream_started": self.stream_started,
            "stream_complete": self.stream_complete,
            "items": list(self.items),
            **summary,
            "transaction": {"calls": self.calls, "complete": self.stream_complete and self.calls == 1},
        }


def tensor_manifest_entry(name: str, tensor: torch.Tensor, index: int) -> dict[str, Any]:
    """Hash the exact tensor bytes without dtype conversion."""

    if not isinstance(name, str) or not name:
        raise FSDPHFReceiptError("weight name must be a non-empty string")
    if not isinstance(tensor, torch.Tensor):
        raise FSDPHFReceiptError(f"weight {name} is not a tensor")
    raw = tensor.detach().contiguous().flatten().view(torch.uint8).cpu().numpy().tobytes()
    return {
        "index": index,
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nbytes": tensor.numel() * tensor.element_size(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def manifest_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "tensor_count": len(items),
        "total_bytes": sum(int(item["nbytes"]) for item in items),
        "manifest_sha256": canonical_sha256(list(items)),
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_fsdp_hf_boundary(root: str | Path) -> dict[str, str]:
    """Verify all inspected RTT FSDP2 to HF boundary bytes."""

    root = Path(root).resolve()
    observed: dict[str, str] = {}
    for relative, expected in RTT_BOUNDARY_SHA256.items():
        path = root / relative
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise FSDPHFReceiptError(f"cannot read RTT boundary file: {relative}") from error
        if digest != expected:
            raise FSDPHFReceiptError(f"unexpected RTT boundary digest for {relative}: {digest}")
        observed[relative] = digest
    return observed


def verify_fsdp_hf_checkout(root: str | Path) -> dict[str, str]:
    """Require the exact clean pinned RTT checkout and inspected file bytes."""

    root = Path(root).resolve()
    git_root = _git(root, "rev-parse", "--show-toplevel")
    if Path(git_root).resolve() != root:
        raise FSDPHFReceiptError(f"RTT root is not the checkout root: {root}")
    revision = _git(root, "rev-parse", "HEAD")
    if revision != RTT_REVISION:
        raise FSDPHFReceiptError(f"unexpected RTT revision: {revision}")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise FSDPHFReceiptError("RTT checkout is dirty")
    return verify_fsdp_hf_boundary(root)


def build_fsdp_hf_receipt_artifact(
    actor_receipts: Sequence[Mapping[str, Any]],
    infer_receipts: Sequence[Mapping[str, Any]],
    *,
    model_identity: Any,
    resolved_config_sha256: str,
    rtt_revision: str,
    rtt_boundary_sha256: Mapping[str, str],
    generation_source_identity: Mapping[str, str],
    transaction_id: str,
    optimizer_updates: int = 0,
    pipeline_steps: int = 0,
    generation_started_before_seal: bool = False,
    update_error: str | None = None,
) -> dict[str, Any]:
    """Build immutable pass or failure evidence for one FSDP2 to HF update."""

    actors = [_sanitize_receipt(receipt) for receipt in actor_receipts]
    infers = [_sanitize_receipt(receipt) for receipt in infer_receipts]
    failures = _receipt_failures(actors, infers, transaction_id)
    if rtt_revision != RTT_REVISION:
        failures.insert(0, {"check": "rtt_revision"})
    if dict(rtt_boundary_sha256) != RTT_BOUNDARY_SHA256:
        failures.insert(0, {"check": "rtt_boundary_sha256"})
    if not _is_sha256(resolved_config_sha256):
        failures.insert(0, {"check": "resolved_config_sha256"})
    generation_identity, generation_identity_valid = _generation_source_identity(generation_source_identity)
    if not generation_identity_valid:
        failures.insert(0, {"check": "generation_source_identity"})
    if optimizer_updates != 0:
        failures.insert(0, {"check": "optimizer_updates"})
    if pipeline_steps != 0:
        failures.insert(0, {"check": "pipeline_steps"})
    if generation_started_before_seal is not False:
        failures.insert(0, {"check": "generation_started_before_seal"})
    if update_error is not None:
        failures.insert(0, {"check": "model_update", "reason": update_error})
    identity, identity_valid = _model_identity(model_identity)
    if not identity_valid:
        failures.insert(0, {"check": "model_identity"})
    artifact = {
        "schema_version": 1,
        "id": "qwen_a100_fsdp2_hf_weight_receipt_v1",
        "status": "receipt_passed" if not failures else "receipt_failed",
        "claim": (
            "FSDP2 gathered full tensors matched the final named parameters of each paired HF model byte for byte"
            if not failures
            else None
        ),
        "diagnostic_target": "FSDP2 gathered full tensors through paired HF model loader",
        "transaction_id": transaction_id,
        "optimizer_updates": optimizer_updates,
        "pipeline_steps": pipeline_steps,
        "generation_started_before_seal": generation_started_before_seal,
        "topology": {
            "accelerator": "A100",
            "actor_dp": 2,
            "actor_tp": 1,
            "actor_pp": 1,
            "infer_dp": 2,
            "infer_tp": 1,
            "infer_pp": 1,
            "pairs": [{"actor_rank": actor, "infer_rank": infer} for actor, infer in EXPECTED_PAIRS],
        },
        "runtime": {
            "rtt_revision": rtt_revision,
            "resolved_config_sha256": resolved_config_sha256,
            "rtt_boundary_sha256": dict(rtt_boundary_sha256),
            **generation_identity,
        },
        "model": identity,
        "actor_receipts": actors,
        "infer_receipts": infers,
        "failures": failures,
    }
    artifact["receipt_manifest_sha256"] = canonical_sha256({"actor_receipts": actors, "infer_receipts": infers})
    return artifact


def seal_fsdp_hf_receipt(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Write a new immutable receipt and raise after sealing any failure."""

    write_artifact(path, artifact)
    if artifact.get("status") != "receipt_passed":
        first = artifact.get("failures", [{}])[0]
        raise FSDPHFReceiptError(f"FSDP2 to HF receipt failed: {first.get('check', 'unknown')}")


def _receipt_failures(
    actors: Sequence[Mapping[str, Any]], infers: Sequence[Mapping[str, Any]], transaction_id: str
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    actor_by_rank = _rank_receipts(actors, "actor", transaction_id, failures)
    infer_by_rank = _rank_receipts(infers, "infer", transaction_id, failures)
    for actor_rank, infer_rank in EXPECTED_PAIRS:
        actor = actor_by_rank.get(actor_rank)
        infer = infer_by_rank.get(infer_rank)
        if actor is None or infer is None:
            continue
        if actor.get("paired_rank") != infer_rank or infer.get("paired_rank") != actor_rank:
            failures.append({"check": "pair", "actor_rank": actor_rank, "infer_rank": infer_rank})
        _validate_receipt(actor, failures)
        _validate_receipt(infer, failures)
        _compare_items(actor.get("items"), infer.get("items"), "pair", actor_rank, infer_rank, failures)
    if len(actor_by_rank) == len(EXPECTED_PAIRS):
        _compare_items(
            actor_by_rank[0].get("items"),
            actor_by_rank[1].get("items"),
            "actor_cross_replica",
            0,
            1,
            failures,
        )
    if len(infer_by_rank) == len(EXPECTED_PAIRS):
        _compare_items(
            infer_by_rank[0].get("items"),
            infer_by_rank[1].get("items"),
            "infer_cross_replica",
            0,
            1,
            failures,
        )
    return failures


def _rank_receipts(
    receipts: Sequence[Mapping[str, Any]], side: str, transaction_id: str, failures: list[dict[str, Any]]
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    expected = {rank for pair in EXPECTED_PAIRS for rank in [pair[0 if side == "actor" else 1]]}
    for receipt in receipts:
        rank = receipt.get("rank")
        if receipt.get("side") != side or receipt.get("transaction_id") != transaction_id:
            failures.append({"check": "identity", "side": side, "rank": rank})
        if not isinstance(rank, int) or isinstance(rank, bool) or rank in result:
            failures.append({"check": "duplicate_rank", "side": side, "rank": rank})
            continue
        result[rank] = receipt
    for rank in expected - result.keys():
        failures.append({"check": "missing_rank", "side": side, "rank": rank})
    for rank in result.keys() - expected:
        failures.append({"check": "unexpected_rank", "side": side, "rank": rank})
    return result


def _validate_receipt(receipt: Mapping[str, Any], failures: list[dict[str, Any]]) -> None:
    side = receipt.get("side")
    rank = receipt.get("rank")
    items = receipt.get("items")
    if not isinstance(items, list) or not items:
        failures.append({"check": "missing", "side": side, "rank": rank})
        return
    if not _well_formed_items(items):
        failures.append({"check": "malformed", "side": side, "rank": rank})
        return
    if receipt.get("stream_started") is not True or receipt.get("stream_complete") is not True:
        failures.append({"check": "incomplete", "side": side, "rank": rank})
    transaction = receipt.get("transaction")
    if (
        not isinstance(transaction, Mapping)
        or transaction.get("calls") != 1
        or transaction.get("complete") is not True
    ):
        failures.append({"check": "transaction", "side": side, "rank": rank})
    if "A100" not in str(receipt.get("accelerator_name")):
        failures.append({"check": "accelerator", "side": side, "rank": rank})
    names = [item["name"] for item in items]
    if len(names) != len(set(names)):
        failures.append({"check": "duplicate", "side": side, "rank": rank})
    for field, expected in manifest_summary(items).items():
        if receipt.get(field) != expected:
            failures.append({"check": field, "side": side, "rank": rank})


def _compare_items(
    left: Any,
    right: Any,
    check: str,
    left_rank: int,
    right_rank: int,
    failures: list[dict[str, Any]],
) -> None:
    if not isinstance(left, list) or not isinstance(right, list):
        failures.append({"check": check, "field": "missing", "left_rank": left_rank, "right_rank": right_rank})
        return
    if not _well_formed_items(left) or not _well_formed_items(right):
        failures.append({"check": check, "field": "malformed", "left_rank": left_rank, "right_rank": right_rank})
        return
    if len(left) != len(right):
        failures.append({"check": check, "field": "count", "left_rank": left_rank, "right_rank": right_rank})
        return
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
        for field in ("index", "name", "shape", "dtype", "nbytes", "sha256"):
            if left_item.get(field) != right_item.get(field):
                failures.append(
                    {
                        "check": check,
                        "field": "order" if field == "index" else field,
                        "index": index,
                        "left_rank": left_rank,
                        "right_rank": right_rank,
                    }
                )
                return


def _well_formed_items(items: Sequence[Any]) -> bool:
    for item in items:
        if not isinstance(item, Mapping):
            return False
        if not isinstance(item.get("index"), int) or isinstance(item.get("index"), bool):
            return False
        if not isinstance(item.get("name"), str) or not item["name"]:
            return False
        shape = item.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape
        ):
            return False
        if not isinstance(item.get("dtype"), str) or not item["dtype"].startswith("torch."):
            return False
        if not isinstance(item.get("nbytes"), int) or isinstance(item.get("nbytes"), bool) or item["nbytes"] < 0:
            return False
        if not _is_sha256(item.get("sha256")):
            return False
    return True


def _sanitize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    transaction = receipt.get("transaction")
    return {
        "transaction_id": receipt.get("transaction_id"),
        "side": receipt.get("side"),
        "rank": receipt.get("rank"),
        "paired_rank": receipt.get("paired_rank"),
        "accelerator_name": receipt.get("accelerator_name"),
        "stream_started": receipt.get("stream_started"),
        "stream_complete": receipt.get("stream_complete"),
        "items": _sanitize_items(receipt.get("items")),
        "tensor_count": receipt.get("tensor_count"),
        "total_bytes": receipt.get("total_bytes"),
        "manifest_sha256": receipt.get("manifest_sha256"),
        "transaction": (
            {"calls": transaction.get("calls"), "complete": transaction.get("complete")}
            if isinstance(transaction, Mapping)
            else None
        ),
    }


def _sanitize_items(value: Any) -> Any:
    if not isinstance(value, list):
        return None
    return [
        (
            {
                "index": item.get("index"),
                "name": item.get("name"),
                "shape": item.get("shape"),
                "dtype": item.get("dtype"),
                "nbytes": item.get("nbytes"),
                "sha256": item.get("sha256"),
            }
            if isinstance(item, Mapping)
            else None
        )
        for item in value
    ]


def _named_tensors(batch: Any) -> Iterable[tuple[str, torch.Tensor]]:
    return batch.items() if isinstance(batch, Mapping) else batch


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _model_identity(value: Any) -> tuple[dict[str, Any], bool]:
    try:
        raw = asdict(value) if is_dataclass(value) and not isinstance(value, type) else dict(value)
    except (TypeError, ValueError):
        return {}, False
    identity = {key: raw.get(key) for key in MODEL_IDENTITY_KEYS}
    valid = (
        set(raw) == MODEL_IDENTITY_KEYS
        and identity["model"] == MODEL
        and identity["revision"] == MODEL_REVISION
        and all(
            _is_sha256(identity[key]) for key in ("snapshot_sha256", "tokenizer_files_sha256", "chat_template_sha256")
        )
    )
    return identity, valid


def _generation_source_identity(value: Any) -> tuple[dict[str, Any], bool]:
    try:
        raw = dict(value)
    except (TypeError, ValueError):
        return {}, False
    identity = {key: raw.get(key) for key in GENERATION_SOURCE_IDENTITY}
    return (identity, True) if raw == GENERATION_SOURCE_IDENTITY else ({}, False)


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise FSDPHFReceiptError(f"cannot verify RTT checkout: {type(error).__name__}") from error
    return result.stdout.strip()
