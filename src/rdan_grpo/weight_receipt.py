"""Fail-closed byte receipts for the diagnostic ROLL weight transfer."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from rdan_grpo.runtime_parity import write_artifact

RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
RECEIPT_WORKER_EXTENSION = "rdan_grpo.roll_weight_receipt.ReceiptWorkerV1"
EXPECTED_PAIRS = ((0, 0), (1, 1))
RECEIPT_CLAIM = (
    "Each paired vLLM loader consumed the same ordered HF-converted tensor bytes sent by its actor "
    "and returned successfully"
)
RECEIPT_NON_CLAIM = (
    "This receipt does not prove that a loader applied the tensors or that they match vLLM's packed internal layout"
)
RTT_BOUNDARY_SHA256 = {
    "roll/distributed/executor/model_update_group.py": (
        "7ee9051ac6d778f619ad80345855eef1c98885e45a4c67e35096c38c84fed91f"
    ),
    "roll/distributed/strategy/megatron_strategy.py": (
        "99d9a0791674e4d3191b27fca5d13677c45cb4a5e05ccecb0fe478bcf4f1c5e6"
    ),
    "roll/distributed/strategy/vllm_strategy.py": "fac85c427e5335574b5046480930ab8dd822154d8d2c97d66eee476344b0c3a6",
    "roll/pipeline/base_worker.py": "161c8aabae67a70283219cc961178fb57488cc8acad1176f804624f122cfedcb",
    "roll/third_party/megatron/model_update.py": "26c15d747d2bf677e1668002f2d485a5ca64331078aedfb428e321e6c5bee2ed",
    "roll/third_party/vllm/async_llm.py": "c9d04e5ec9151edc999bc1afdbb2c589683c0b98610d25473032c29f2d2ef690",
    "roll/third_party/vllm/worker.py": "65d7d23426553eed6bda385ec83b64c435994cb5de4607de600a6ca4d0bbac20",
}


class WeightReceiptError(ValueError):
    """Raised when the diagnostic cannot prove the complete weight receipt."""


class TensorStreamReceipt:
    """Record tensor bytes as a lazy iterable is consumed exactly once."""

    def __init__(
        self,
        transaction_id: str,
        side: str,
        rank: int,
        paired_rank: int,
        accelerator_name: str | None = None,
    ) -> None:
        if side not in {"actor", "infer"}:
            raise ValueError(f"invalid receipt side: {side}")
        self.transaction_id = transaction_id
        self.side = side
        self.rank = rank
        self.paired_rank = paired_rank
        self.accelerator_name = accelerator_name
        self.items: list[dict[str, Any]] = []
        self.stream_started = False
        self.stream_complete = False
        self.loader_calls = 0
        self.loader_successes = 0
        self.loader_failed = False
        self.loader_segments_started = 0
        self.loader_segments_completed = 0
        self.internal_before: dict[str, Any] | None = None
        self.internal_after: dict[str, Any] | None = None

    def wrap(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
        """Hash each tensor immediately before yielding it to the consumer."""

        if self.stream_started:
            raise WeightReceiptError("weight stream was consumed more than once")
        self.stream_started = True
        for name, tensor in weights:
            self.items.append(tensor_manifest_entry(name, tensor, len(self.items)))
            yield name, tensor
        self.stream_complete = True

    def record_batch(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Hash one already-materialized RTT conversion batch without changing it."""

        if self.stream_complete:
            raise WeightReceiptError("weight stream was consumed more than once")
        self.stream_started = True
        for name, tensor in weights:
            self.items.append(tensor_manifest_entry(name, tensor, len(self.items)))

    def wrap_loader_segment(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
        """Hash one RTT bucket lazily as the real vLLM loader consumes it."""

        if self.stream_complete:
            raise WeightReceiptError("weight stream was already sealed")
        self.stream_started = True
        self.loader_segments_started += 1
        for name, tensor in weights:
            self.items.append(tensor_manifest_entry(name, tensor, len(self.items)))
            yield name, tensor
        self.loader_segments_completed += 1

    def finish_stream(self) -> None:
        self.stream_complete = True

    def mark_loader_success(self) -> None:
        self.loader_calls += 1
        self.loader_successes += 1

    def mark_loader_failure(self) -> None:
        self.loader_calls += 1
        self.loader_failed = True

    def set_internal_before(self, named_parameters: Iterable[tuple[str, torch.Tensor]]) -> None:
        if self.internal_before is not None:
            return
        self.internal_before = manifest(named_parameters)

    def set_internal_after(self, named_parameters: Iterable[tuple[str, torch.Tensor]]) -> None:
        if self.internal_after is None:
            self.internal_after = manifest(named_parameters)

    def snapshot(self) -> dict[str, Any]:
        summary = manifest_summary(self.items)
        return {
            "transaction_id": self.transaction_id,
            "side": self.side,
            "rank": self.rank,
            "paired_rank": self.paired_rank,
            "accelerator_name": self.accelerator_name,
            "stream_started": self.stream_started,
            "stream_complete": self.stream_complete,
            "items": list(self.items),
            **summary,
            "loader": {
                "calls": self.loader_calls,
                "successes": self.loader_successes,
                "failed": self.loader_failed,
                "segments_started": self.loader_segments_started,
                "segments_completed": self.loader_segments_completed,
                "loaded": (
                    self.side == "infer"
                    and self.stream_complete
                    and self.loader_calls > 0
                    and self.loader_calls == self.loader_successes
                    and self.loader_segments_started == self.loader_segments_completed
                    and not self.loader_failed
                ),
            },
            "internal_parameters": {
                "before": self.internal_before,
                "after": self.internal_after,
            },
        }


def tensor_manifest_entry(name: str, tensor: torch.Tensor, index: int) -> dict[str, Any]:
    """Describe and hash a tensor without any numeric dtype conversion."""

    if not isinstance(name, str) or not name:
        raise WeightReceiptError("weight name must be a non-empty string")
    if not isinstance(tensor, torch.Tensor):
        raise WeightReceiptError(f"weight {name} is not a tensor")
    contiguous = tensor.detach().contiguous().flatten().view(torch.uint8).cpu()
    return {
        "index": index,
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nbytes": tensor.numel() * tensor.element_size(),
        "sha256": hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
    }


def manifest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    items = [tensor_manifest_entry(name, tensor, index) for index, (name, tensor) in enumerate(named_tensors)]
    return {"items": items, **manifest_summary(items)}


def manifest_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "tensor_count": len(items),
        "total_bytes": sum(int(item["nbytes"]) for item in items),
        "manifest_sha256": canonical_sha256(list(items)),
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_rtt_boundary(root: str | Path) -> dict[str, str]:
    """Verify every inspected RTT boundary against the pinned revision bytes."""

    root = Path(root).resolve()
    observed: dict[str, str] = {}
    for relative, expected in RTT_BOUNDARY_SHA256.items():
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise WeightReceiptError(f"unexpected RTT boundary digest for {relative}: {digest}")
        observed[relative] = digest
    return observed


def verify_rtt_checkout(root: str | Path) -> dict[str, str]:
    """Require the exact clean RTT checkout before verifying inspected boundaries."""

    root = Path(root).resolve()
    git_root = _git(root, "rev-parse", "--show-toplevel")
    if Path(git_root).resolve() != root:
        raise WeightReceiptError(f"RTT_ROOT is not the checkout root: {root}")
    revision = _git(root, "rev-parse", "HEAD")
    if revision != RTT_REVISION:
        raise WeightReceiptError(f"unexpected RTT revision: {revision}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise WeightReceiptError("RTT checkout is dirty")
    return verify_rtt_boundary(root)


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise WeightReceiptError(f"cannot verify RTT checkout: {type(error).__name__}") from error
    return result.stdout.strip()


def build_weight_receipt_artifact(
    actor_receipts: Sequence[Mapping[str, Any]],
    infer_receipts: Sequence[Mapping[str, Any]],
    *,
    model_identity: Any,
    resolved_config_sha256: str,
    rtt_revision: str,
    rtt_boundary_sha256: Mapping[str, str],
    transaction_id: str,
    update_error: str | None = None,
) -> dict[str, Any]:
    """Build pass or failure evidence without including prompts, responses, or environment state."""

    actors = [_sanitize_receipt(receipt) for receipt in actor_receipts]
    infers = [_sanitize_receipt(receipt) for receipt in infer_receipts]
    failures = _receipt_failures(actors, infers, transaction_id)
    if rtt_revision != RTT_REVISION:
        failures.insert(0, {"check": "rtt_revision"})
    if dict(rtt_boundary_sha256) != RTT_BOUNDARY_SHA256:
        failures.insert(0, {"check": "rtt_boundary_sha256"})
    if len(resolved_config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in resolved_config_sha256
    ):
        failures.insert(0, {"check": "resolved_config_sha256"})
    if update_error is not None:
        failures.insert(0, {"check": "model_update", "reason": update_error})
    identity = asdict(model_identity) if is_dataclass(model_identity) else dict(model_identity)
    artifact = {
        "schema_version": 1,
        "id": "qwen_a100_weight_receipt_v1",
        "status": "receipt_passed" if not failures else "receipt_failed",
        "claim": RECEIPT_CLAIM if not failures else None,
        "diagnostic_target": "Actor tensor transport and successful consumption by each paired vLLM loader",
        "non_claim": RECEIPT_NON_CLAIM,
        "transaction_id": transaction_id,
        "optimizer_updates": 0,
        "generation_started_before_seal": False,
        "topology": {
            "accelerator": "A100",
            "actor_dp": 2,
            "actor_tp": 1,
            "actor_pp": 1,
            "pairs": [{"actor_rank": actor, "infer_rank": infer} for actor, infer in EXPECTED_PAIRS],
        },
        "runtime": {
            "rtt_revision": rtt_revision,
            "vllm_version": "0.10.2",
            "resolved_config_sha256": resolved_config_sha256,
            "worker_extension_cls": RECEIPT_WORKER_EXTENSION,
            "rtt_boundary_sha256": dict(rtt_boundary_sha256),
        },
        "model": identity,
        "actor_receipts": actors,
        "infer_receipts": infers,
        "failures": failures,
    }
    artifact["receipt_manifest_sha256"] = canonical_sha256({"actor_receipts": actors, "infer_receipts": infers})
    return artifact


def build_receipt_link(path: str | Path, resolved_config_sha256: str) -> dict[str, str]:
    """Validate one sealed receipt and return its immutable parity linkage."""

    target = Path(path)
    try:
        payload = target.read_bytes()
        artifact = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise WeightReceiptError(f"cannot load weight receipt: {type(error).__name__}") from error
    validate_weight_receipt_artifact(artifact, resolved_config_sha256)
    return {
        "transaction_id": artifact["transaction_id"],
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "resolved_config_sha256": resolved_config_sha256,
    }


def validate_parity_receipt_pair(parity: Mapping[str, Any], receipt_path: str | Path) -> None:
    """Reject parity and receipt artifacts unless their immutable run linkage matches."""

    backend = parity.get("runtime_backend")
    linkage = parity.get("weight_receipt")
    if not isinstance(backend, Mapping) or not isinstance(linkage, Mapping):
        raise WeightReceiptError("parity artifact is missing weight receipt linkage")
    resolved_config_sha256 = backend.get("resolved_config_sha256")
    if not isinstance(resolved_config_sha256, str):
        raise WeightReceiptError("parity artifact is missing the resolved config digest")
    expected = build_receipt_link(receipt_path, resolved_config_sha256)
    if dict(linkage) != expected:
        raise WeightReceiptError("parity and weight receipt artifacts do not share an immutable run linkage")


def validate_weight_receipt_artifact(artifact: Any, resolved_config_sha256: str) -> None:
    """Revalidate the bounded transport claim from one serialized receipt."""

    if not isinstance(artifact, Mapping):
        raise WeightReceiptError("weight receipt artifact must be an object")
    transaction_id = artifact.get("transaction_id")
    runtime = artifact.get("runtime")
    actors = artifact.get("actor_receipts")
    infers = artifact.get("infer_receipts")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("id") != "qwen_a100_weight_receipt_v1"
        or artifact.get("status") != "receipt_passed"
        or artifact.get("claim") != RECEIPT_CLAIM
        or artifact.get("non_claim") != RECEIPT_NON_CLAIM
        or artifact.get("optimizer_updates") != 0
        or artifact.get("generation_started_before_seal") is not False
        or artifact.get("failures") != []
        or not isinstance(transaction_id, str)
        or not transaction_id
        or not isinstance(runtime, Mapping)
        or runtime.get("resolved_config_sha256") != resolved_config_sha256
        or runtime.get("rtt_revision") != RTT_REVISION
        or runtime.get("rtt_boundary_sha256") != RTT_BOUNDARY_SHA256
        or not isinstance(actors, list)
        or not isinstance(infers, list)
    ):
        raise WeightReceiptError("weight receipt artifact identity is invalid")
    failures = _receipt_failures(actors, infers, transaction_id)
    manifest_sha256 = canonical_sha256({"actor_receipts": actors, "infer_receipts": infers})
    if failures or artifact.get("receipt_manifest_sha256") != manifest_sha256:
        raise WeightReceiptError("weight receipt artifact evidence is invalid")


def seal_weight_receipt(path: str | Path, artifact: Mapping[str, Any]) -> None:
    write_artifact(path, artifact)
    if artifact.get("status") != "receipt_passed":
        first = artifact.get("failures", [{}])[0]
        raise WeightReceiptError(f"weight receipt failed: {first.get('check', 'unknown')}")


def _receipt_failures(
    actors: Sequence[Mapping[str, Any]], infers: Sequence[Mapping[str, Any]], transaction_id: str
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    expected_actor_ranks = [pair[0] for pair in EXPECTED_PAIRS]
    expected_infer_ranks = [pair[1] for pair in EXPECTED_PAIRS]
    actor_by_rank = _rank_receipts(actors, "actor", expected_actor_ranks, transaction_id, failures)
    infer_by_rank = _rank_receipts(infers, "infer", expected_infer_ranks, transaction_id, failures)

    for actor_rank, infer_rank in EXPECTED_PAIRS:
        actor = actor_by_rank.get(actor_rank)
        infer = infer_by_rank.get(infer_rank)
        if actor is None or infer is None:
            continue
        if actor.get("paired_rank") != infer_rank or infer.get("paired_rank") != actor_rank:
            failures.append({"check": "pair", "actor_rank": actor_rank, "infer_rank": infer_rank})
        _validate_stream(actor, "actor", failures)
        _validate_stream(infer, "infer", failures)
        _compare_manifests(actor.get("items"), infer.get("items"), "transport", actor_rank, infer_rank, failures)
        _validate_internal(infer, infer_rank, failures)

    if len(actor_by_rank) == len(EXPECTED_PAIRS):
        _compare_manifests(
            actor_by_rank[0].get("items"), actor_by_rank[1].get("items"), "actor_cross_replica", 0, 1, failures
        )
    if len(infer_by_rank) == len(EXPECTED_PAIRS):
        _compare_manifests(
            infer_by_rank[0].get("items"), infer_by_rank[1].get("items"), "infer_cross_replica", 0, 1, failures
        )
        before0 = _internal_items(infer_by_rank[0], "before")
        before1 = _internal_items(infer_by_rank[1], "before")
        after0 = _internal_items(infer_by_rank[0], "after")
        after1 = _internal_items(infer_by_rank[1], "after")
        _compare_manifests(before0, before1, "internal_before_cross_replica", 0, 1, failures)
        _compare_manifests(after0, after1, "internal_after_cross_replica", 0, 1, failures)
    return failures


def _rank_receipts(
    receipts: Sequence[Mapping[str, Any]],
    side: str,
    expected_ranks: Sequence[int],
    transaction_id: str,
    failures: list[dict[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for receipt in receipts:
        rank = receipt.get("rank")
        if receipt.get("side") != side or receipt.get("transaction_id") != transaction_id:
            failures.append({"check": "identity", "side": side, "rank": rank})
        if not isinstance(rank, int) or rank in result:
            failures.append({"check": "duplicate_rank", "side": side, "rank": rank})
            continue
        result[rank] = receipt
    for rank in expected_ranks:
        if rank not in result:
            failures.append({"check": "missing_rank", "side": side, "rank": rank})
    for rank in result.keys() - set(expected_ranks):
        failures.append({"check": "unexpected_rank", "side": side, "rank": rank})
    return result


def _validate_stream(receipt: Mapping[str, Any], side: str, failures: list[dict[str, Any]]) -> None:
    rank = receipt.get("rank")
    items = receipt.get("items")
    if not isinstance(items, list) or not items:
        failures.append({"check": "missing", "side": side, "rank": rank})
        return
    if not _well_formed_items(items):
        failures.append({"check": "malformed", "side": side, "rank": rank})
        return
    if receipt.get("stream_complete") is not True:
        failures.append({"check": "incomplete", "side": side, "rank": rank})
    if "A100" not in str(receipt.get("accelerator_name")):
        failures.append({"check": "accelerator", "side": side, "rank": rank})
    names = [item.get("name") for item in items if isinstance(item, Mapping)]
    if len(names) != len(set(names)):
        failures.append({"check": "duplicate", "side": side, "rank": rank})
    expected = manifest_summary(items)
    for field, value in expected.items():
        if receipt.get(field) != value:
            failures.append({"check": field, "side": side, "rank": rank})
    if side == "infer":
        loader = receipt.get("loader")
        if not isinstance(loader, Mapping) or loader.get("loaded") is not True:
            failures.append({"check": "loader", "side": side, "rank": rank})


def _validate_internal(receipt: Mapping[str, Any], rank: int, failures: list[dict[str, Any]]) -> None:
    before = _internal_items(receipt, "before")
    after = _internal_items(receipt, "after")
    if before is None or after is None:
        failures.append({"check": "internal_missing", "infer_rank": rank})
        return
    internal = receipt.get("internal_parameters")
    assert isinstance(internal, Mapping)
    for state in ("before", "after"):
        value = internal[state]
        if not _well_formed_items(value["items"]):
            failures.append({"check": f"internal_{state}_malformed", "infer_rank": rank})
            return
        expected = manifest_summary(value["items"])
        if any(value.get(field) != expected_value for field, expected_value in expected.items()):
            failures.append({"check": f"internal_{state}_summary", "infer_rank": rank})
    _compare_manifests(before, after, "internal_pre_post", rank, rank, failures)


def _internal_items(receipt: Mapping[str, Any], state: str) -> list[Mapping[str, Any]] | None:
    internal = receipt.get("internal_parameters")
    if not isinstance(internal, Mapping):
        return None
    value = internal.get(state)
    if not isinstance(value, Mapping) or not isinstance(value.get("items"), list):
        return None
    return value["items"]


def _compare_manifests(
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
    fields = ("index", "name", "shape", "dtype", "nbytes", "sha256")
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
        for field in fields:
            if left_item.get(field) != right_item.get(field):
                failures.append(
                    {
                        "check": check,
                        "field": field if field != "index" else "order",
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
        if not isinstance(shape, list) or any(not isinstance(size, int) or size < 0 for size in shape):
            return False
        if not isinstance(item.get("dtype"), str) or not item["dtype"].startswith("torch."):
            return False
        if not isinstance(item.get("nbytes"), int) or item["nbytes"] < 0:
            return False
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
    return True


def _sanitize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    loader = receipt.get("loader")
    internal = receipt.get("internal_parameters")
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
        "loader": (
            {
                "calls": loader.get("calls"),
                "successes": loader.get("successes"),
                "failed": loader.get("failed"),
                "segments_started": loader.get("segments_started"),
                "segments_completed": loader.get("segments_completed"),
                "loaded": loader.get("loaded"),
            }
            if isinstance(loader, Mapping)
            else None
        ),
        "internal_parameters": (
            {
                "before": _sanitize_manifest(internal.get("before")),
                "after": _sanitize_manifest(internal.get("after")),
            }
            if isinstance(internal, Mapping)
            else None
        ),
    }


def _sanitize_manifest(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "items": _sanitize_items(value.get("items")),
        "tensor_count": value.get("tensor_count"),
        "total_bytes": value.get("total_bytes"),
        "manifest_sha256": value.get("manifest_sha256"),
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
