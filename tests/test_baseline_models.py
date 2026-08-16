from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rdan_grpo import baseline, baseline_models

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/baselines/qwen_base.json",
    ROOT / "configs/baselines/llama3_2_3b_base.json",
    ROOT / "configs/baselines/ministral3_3b_base.json",
)


@pytest.mark.parametrize(("config_path", "contract"), tuple(zip(CONFIGS, baseline_models.MODELS)))
def test_exact_model_configs(config_path: Path, contract: baseline_models.BaselineModel) -> None:
    config = baseline._load_config(config_path)

    assert baseline_models.load_model_contract(config) == contract


@pytest.mark.parametrize(
    ("field", "value"),
    (("name", "ibm-granite/granite-3.1-2b-instruct"), ("revision", "0" * 40), ("served_name", "drifted")),
)
def test_rejects_model_identity_drift(field: str, value: str) -> None:
    config = json.loads(CONFIGS[0].read_text(encoding="utf-8"))
    config["model"][field] = value

    with pytest.raises(baseline_models.ModelContractError, match="approved exact target pin"):
        baseline_models.load_model_contract(config)


def test_rejects_template_kwargs_drift() -> None:
    config = json.loads(CONFIGS[1].read_text(encoding="utf-8"))
    config["generation"]["chat_template_kwargs"] = {"enable_thinking": False}

    with pytest.raises(baseline_models.ModelContractError, match="chat template kwargs drifted"):
        baseline_models.load_model_contract(config)


def test_rejects_target_registry_pin_drift(tmp_path: Path) -> None:
    targets = json.loads(baseline_models.TARGETS.read_text(encoding="utf-8"))
    drifted = copy.deepcopy(targets)
    drifted["models"][0]["revision"] = "0" * 40
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    config = json.loads(CONFIGS[0].read_text(encoding="utf-8"))

    with pytest.raises(baseline_models.ModelContractError, match="target registry pin drifted"):
        baseline_models.load_model_contract(config, path)


def test_harness_identity_includes_model_contract() -> None:
    paths = {entry["path"] for entry in baseline._harness_identity()["files"]}

    assert "src/rdan_grpo/baseline_models.py" in paths
