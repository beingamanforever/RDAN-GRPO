"""Exact model contracts for base evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "configs").is_dir():
    ROOT = Path(__file__).resolve().parents[3]
TARGETS = ROOT / "configs/models/targets.json"


class ModelContractError(ValueError):
    """Raised when a baseline model does not match an approved target."""


@dataclass(frozen=True)
class BaselineModel:
    """Immutable identity and request-template contract for one baseline model."""

    target_id: str
    name: str
    revision: str
    served_name: str
    chat_template_kwargs: dict[str, Any]


MODELS = (
    BaselineModel(
        target_id="qwen3_4b",
        name="Qwen/Qwen3-4B-Instruct-2507",
        revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        served_name="qwen3-4b-instruct-2507",
        chat_template_kwargs={"enable_thinking": False},
    ),
    BaselineModel(
        target_id="llama3_2_3b",
        name="meta-llama/Llama-3.2-3B-Instruct",
        revision="0cb88a4f764b7a12671c53f0838cd831a0843b95",
        served_name="llama-3.2-3b-instruct",
        chat_template_kwargs={},
    ),
    BaselineModel(
        target_id="ministral3_3b",
        name="mistralai/Ministral-3-3B-Instruct-2512",
        revision="b35d4dfe56c142746f54dbd64f579faab2744308",
        served_name="ministral-3-3b-instruct-2512",
        chat_template_kwargs={},
    ),
)


def load_model_contract(config: dict[str, Any], targets_path: Path = TARGETS) -> BaselineModel:
    """Return the exact approved model contract selected by a baseline config."""
    try:
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
        records = targets["models"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelContractError(f"invalid target model registry: {error}") from error
    if targets.get("schema_version") != 1 or not isinstance(records, list):
        raise ModelContractError("unsupported target model registry")

    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        target_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(target_id, str) or target_id in by_id:
            raise ModelContractError("target model registry has invalid or duplicate ids")
        by_id[target_id] = record

    model = config.get("model")
    generation = config.get("generation")
    if not isinstance(model, dict) or not isinstance(generation, dict):
        raise ModelContractError("baseline model or generation contract is invalid")
    for contract in MODELS:
        target = by_id.get(contract.target_id)
        if target is None or (target.get("model"), target.get("revision")) != (contract.name, contract.revision):
            raise ModelContractError(f"target registry pin drifted for {contract.target_id}")
        if model == {
            "name": contract.name,
            "revision": contract.revision,
            "served_name": contract.served_name,
        }:
            if generation.get("chat_template_kwargs") != contract.chat_template_kwargs:
                raise ModelContractError(f"chat template kwargs drifted for {contract.target_id}")
            return contract
    raise ModelContractError("baseline model is not an approved exact target pin")
