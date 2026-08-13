from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rdan_grpo.roll_response_checkpoint import ArtifactIdentity, CheckpointIdentity, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/rdan_grpo/roll_response_pipeline.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_one_step_calls_exact_receipt_train_checkpoint_order(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_pipeline(monkeypatch)
    events: list[str] = []
    pipeline = module.ResponseTrainingPipeline.__new__(module.ResponseTrainingPipeline)
    pipeline._resume_manifest = None
    pipeline.completed_step = 0
    pipeline.stop_after_step = 1
    pipeline.pipeline_config = SimpleNamespace(num_return_sequences_in_group=8)
    pipeline.response_config = SimpleNamespace(method="rdan_scalar", quality_weight=0.5, mix_weight=None)
    pipeline.certificate = {"ready": True}
    pipeline.actor_train = SimpleNamespace(
        rdan_reset_cuda_peak=lambda **kwargs: events.append("reset_actor"),
    )
    pipeline.actor_infer = SimpleNamespace(
        rdan_reset_cuda_peak=lambda **kwargs: events.append("reset_infer"),
    )
    pipeline.scheduler = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: events.append("shutdown")))
    pipeline.tracker = SimpleNamespace(
        log=lambda **kwargs: events.append("log"),
        finish=lambda: events.append("finish"),
    )
    pipeline.state = SimpleNamespace(step=0, log_history=[])
    pipeline._transfer = lambda phase, step: events.append(f"receipt:{phase}:{step}") or {"phase": phase}
    pipeline._generate = lambda step: events.append(f"generate:{step}") or object()
    pipeline._save_step = lambda *args: events.append("checkpoint") or Path("step-000001")

    def train(**kwargs):
        events.append("train")
        kwargs["observe_training_state"]()
        kwargs["transfer_after_update"]()
        kwargs["observe_post_transaction_memory"]()
        return SimpleNamespace(promotion_ready=True, metrics={"actor/clipfrac": 0.0}, peak_memory_fraction=0.5)

    pipeline.actor_train.rdan_training_state = lambda **kwargs: events.append("state") or []
    pipeline.actor_train.rdan_cuda_memory = lambda **kwargs: events.append("memory") or []
    monkeypatch.setattr(module, "run_response_train_step", train)
    monkeypatch.setattr(
        module,
        "_group_diagnostics",
        lambda *args: {
            "selected_reward_variance_mean": 0.25,
            "response_active_group_rate": 1.0,
            "quality_active_group_rate": 0.5,
        },
    )

    assert pipeline.run() == [Path("step-000001")]
    assert events == [
        "receipt:initial:0",
        "reset_actor",
        "reset_infer",
        "generate:1",
        "train",
        "state",
        "receipt:post_update:1",
        "memory",
        "checkpoint",
        "log",
        "shutdown",
        "finish",
    ]


def test_cleanup_attempts_both_boundaries_and_preserves_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_pipeline(monkeypatch)
    events: list[str] = []

    class Remote:
        def remote(self) -> None:
            events.append("shutdown")
            raise RuntimeError("shutdown failed")

    tracker = SimpleNamespace(finish=lambda: events.append("finish"))
    error = module._cleanup(SimpleNamespace(shutdown=Remote()), tracker)

    assert isinstance(error, RuntimeError)
    assert events == ["shutdown", "finish"]

    pipeline = module.ResponseTrainingPipeline.__new__(module.ResponseTrainingPipeline)
    pipeline._resume_manifest = None
    pipeline.completed_step = 0
    pipeline.stop_after_step = 1
    pipeline.actor_train = SimpleNamespace(rdan_reset_cuda_peak=lambda **kwargs: None)
    pipeline.actor_infer = SimpleNamespace(rdan_reset_cuda_peak=lambda **kwargs: None)
    pipeline.scheduler = SimpleNamespace(shutdown=Remote())
    pipeline.tracker = SimpleNamespace(finish=lambda: events.append("finish-primary"))
    pipeline._transfer = lambda *args: {}

    def fail_generation(step: int) -> None:
        raise ValueError("primary generation failure")

    pipeline._generate = fail_generation
    with pytest.raises(ValueError, match="primary generation failure"):
        pipeline.run()
    assert events[-2:] == ["shutdown", "finish-primary"]


def test_pipeline_uses_custom_transaction_and_never_stock_checkpoint_upload() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert "run_response_train_step" in calls
    assert "create_checkpoint_stage" in calls
    assert "promote_checkpoint" in calls
    assert "load_checkpoint" in calls
    assert any(name.endswith("get_batch_opt_level_0.remote") for name in calls)
    assert "do_checkpoint" not in calls
    assert not any(name.endswith(".upload") for name in calls)


def test_constructor_resets_shared_lists_before_base_pipeline_construction() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    constructor = _constructor(tree)
    statements = constructor.body
    reset_model = _statement_index(statements, "BasePipeline.model_update_groups")
    reset_checkpoint = _statement_index(statements, "BasePipeline.checkpoint_clusters")
    base_init = next(index for index, node in enumerate(statements) if _contains_call(node, "super.__init__"))
    assert reset_model < base_init
    assert reset_checkpoint < base_init


def test_resume_restores_dcp_driver_inference_rng_before_resume_receipt() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    restore = source.index("self._restore_checkpoint()")
    run = source.index("def run(")
    resume_receipt = source.index('initial_phase = "resume_initial"', run)
    load_dcp = source.index("rdan_load_dcp", restore)
    driver_rng = source.index("WorkerState.load_rng_state", restore)
    inference_rng = source.index("rdan_load_rng.remote", restore)
    assert restore < load_dcp < driver_rng < inference_rng
    assert restore < run < resume_receipt


def test_receipts_bracket_generation_training_and_atomic_promotion() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    initial = source.index("receipt = self._transfer(initial_phase")
    generate = source.index("rewarded = self._generate(step)")
    train = source.index("result = run_response_train_step")
    post = source.index('value = self._transfer("post_update", step)')
    save = source.index("checkpoint = self._save_step")
    checkpoint_call = source.index("checkpoint = self._save_checkpoint", save)
    upload = source.index("self.tracker.log_artifact", checkpoint_call)
    promote = source.index("return promote_checkpoint", upload)
    assert initial < generate < train
    assert post < save < checkpoint_call < upload < promote


def test_mixed_hir_rubrichub_rows_reach_reward_contract_without_arrow_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_pipeline(monkeypatch)
    hir = {
        "id": 7,
        "source": "type4",
        "prompt": "HIR prompt",
        "rubrics": [{"id": 1, "category": "", "description": "Use two paragraphs.", "weight": 1}],
        "ground_truth": {"checker": ["[rule] Paragraphs"], "functions": ["def check_following(): pass"]},
    }
    rubrichub_rubrics = [
        {
            "id": 1,
            "category": "rule",
            "description": "Use exactly two paragraphs.",
            "weight": 2,
            "verifier": "rule",
            "function": "ParagraphChecker",
            "parameters": {"num_paragraphs": 2.0},
        },
        {
            "id": 2,
            "category": "llm",
            "description": "The answer is clear.",
            "weight": 1,
            "verifier": "llm",
            "function": "",
            "parameters": {},
        },
    ]
    rubrichub_truth = {
        "hard_mask": [True, False],
        "rubric_routes": [
            {"rubric_index": 0, "function": "ParagraphChecker", "parameters": {"num_paragraphs": 2.0}},
            {"rubric_index": 1, "function": "", "parameters": {}},
        ],
    }
    rubrichub = {
        "id": "rubrichub-1",
        "source": "rubrichub_instruction_following",
        "prompt": "RubricHub prompt",
        "rubrics": rubrichub_rubrics,
        "ground_truth": rubrichub_truth,
    }
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (hir, rubrichub)) + "\n", encoding="utf-8")

    dataset = module._load_response_dataset(SimpleNamespace(file_name=str(path), dataset_dir="."))
    assert dataset[0]["id"] == "7"
    assert dataset[0]["prompt"] == hir["prompt"]
    assert dataset[1]["prompt"] == rubrichub["prompt"]
    assert dataset[0]["source"] == hir["source"]
    assert dataset[1]["source"] == rubrichub["source"]
    assert dataset[0]["ground_truth"] == module._canonical_json(hir["ground_truth"])
    assert dataset[1]["ground_truth"] == module._canonical_json(rubrichub_truth)
    assert json.loads(dataset[0]["ground_truth"]) == hir["ground_truth"]
    assert json.loads(dataset[1]["ground_truth"]) == rubrichub_truth
    assert dataset[0]["rubrics"] == module._canonical_json(hir["rubrics"])
    restored = dataset.with_transform(module._restore_rubrics)
    assert restored[0]["rubrics"] == hir["rubrics"]
    assert restored[1]["rubrics"] == rubrichub_rubrics


def test_builder_preserves_json_canonical_worker_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_pipeline(monkeypatch)
    payload = {
        "actor_train": {"worker_cls": module.ACTOR_WORKER_PATH},
        "actor_infer": {"worker_cls": module.INFER_WORKER_PATH},
    }
    config = SimpleNamespace(
        actor_train=SimpleNamespace(worker_cls=module.ACTOR_WORKER_PATH),
        actor_infer=SimpleNamespace(worker_cls=module.INFER_WORKER_PATH),
        to_dict=lambda: payload,
    )
    before = module._canonical_json(config.to_dict())
    observed: dict[str, object] = {}

    def construct(value: object, **kwargs: object) -> object:
        observed.update(config=value, kwargs=kwargs)
        return value

    monkeypatch.setattr(module, "ResponseTrainingPipeline", construct)
    assert module.build_response_training_pipeline(config, marker=True) is config
    assert observed == {"config": config, "kwargs": {"marker": True}}
    assert module._canonical_json(config.to_dict()) == before
    assert config.actor_train.worker_cls == module.ACTOR_WORKER_PATH
    assert config.actor_infer.worker_cls == module.INFER_WORKER_PATH


def test_step_checkpoint_seals_redacted_rollout_and_logs_run_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_pipeline(monkeypatch)

    class Batch:
        def __init__(self) -> None:
            selected = torch.tensor([0.0, 1.0, 0.0, 1.0])
            quality = torch.tensor([0.0, 1.0, 1.0, 0.0])
            self.batch = {
                "input_ids": torch.tensor([[10, 11, 0], [10, 12, 0], [20, 21, 0], [20, 22, 0]]),
                "response_mask": torch.tensor([[0, 1, 0]] * 4, dtype=torch.bool),
                "rdan_raw_aon": selected,
                "rdan_raw_csr": selected,
                "rdan_raw_signed_csr": selected,
                "rdan_selected_reward": selected,
                "rdan_response_advantage": selected,
                "rdan_raw_quality": quality,
                "rdan_quality_eligible": torch.ones(4, dtype=torch.bool),
                "rdan_quality_advantage": quality,
                "rdan_scalar_advantage": selected + quality,
                "rdan_response_valid": torch.ones(4, dtype=torch.bool),
                "rdan_scores": torch.tensor([[1.0, 0.0]] * 4),
                "rdan_rubric_mask": torch.ones((4, 2), dtype=torch.bool),
                "rdan_eval_mask": torch.ones((4, 2), dtype=torch.bool),
                "rdan_hard_mask": torch.tensor([[True, False]] * 4),
                "rdan_judge_failed": torch.zeros(4, dtype=torch.bool),
                "rdan_unsupported_hard": torch.zeros(4, dtype=torch.bool),
            }
            rubrics = [[{"id": 1, "description": "hard"}, {"id": 2, "description": "soft"}]] * 4
            evidence = [[{"rubric_id": 1, "reason": "ok"}, {"rubric_id": 2, "reason": "ok"}]] * 4
            self.non_tensor_batch = {
                "prompt": ["credential sk-or-v1-abcdefghijklmnopqrstuvwxyz"] * 4,
                "rubrics": rubrics,
                "source": ["type4"] * 4,
                "ground_truth": [{}] * 4,
                "rdan_prompt_key": ["p0", "p0", "p1", "p1"],
                "rdan_rubric_evidence": evidence,
                "generation_id": ["g0", "g1", "g2", "g3"],
            }
            self.meta_info = {}

        def __len__(self) -> int:
            return 4

    checkpoint_root = tmp_path / "checkpoints"
    artifact_root = tmp_path / "run" / "artifacts"
    pipeline = module.ResponseTrainingPipeline.__new__(module.ResponseTrainingPipeline)
    pipeline.checkpoint_root = checkpoint_root.resolve()
    artifact_root.mkdir(parents=True)
    pipeline.artifact_root = artifact_root.resolve()
    pipeline.pipeline_config = SimpleNamespace(num_return_sequences_in_group=2, save_steps=20)
    pipeline.stop_after_step = 1
    pipeline.response_config = SimpleNamespace(method="rdan_scalar")

    def decode(tokens: list[int], **kwargs: object) -> str:
        if tokens == [11]:
            return "sk-or-v1-abcdefghijklmnopqrstuvwxyz"
        return "decoded " + " ".join(map(str, tokens))

    pipeline.tokenizer = SimpleNamespace(decode=decode)
    identity = CheckpointIdentity(
        planned_horizon=20,
        method="rdan_scalar",
        method_weight=0.5,
        resolved_config_sha256="a" * 64,
        certificate=ArtifactIdentity("cert", "b" * 64),
        data=ArtifactIdentity("merged-response-data-v1", "1" * 64),
        revisions={"code": "c" * 40, "rtt": "d" * 40, "model": "e" * 40},
        base_checkpoint_sha256="f" * 64,
        wandb={
            "entity": "RDAN-GRPO",
            "project": "rdan-grpo-qwen3-4b",
            "run_id": "run",
            "name": "name",
            "group": "group",
        },
    )
    pipeline.checkpoint_identity = identity

    def save_dcp(path: str, step: int, **kwargs: object) -> None:
        target = Path(path)
        target.mkdir(parents=True)
        (target / "rank.distcp").write_bytes(f"step-{step}".encode())

    pipeline.actor_train = SimpleNamespace(
        rdan_save_dcp=save_dcp,
        rdan_train_counters=lambda **kwargs: [
            {"rank": rank, "optimizer_steps": 2, "scheduler_steps": 2} for rank in range(2)
        ],
    )

    def remote() -> dict[str, torch.Tensor]:
        return {"cpu": torch.get_rng_state()}

    pipeline.actor_infer = SimpleNamespace(
        workers=[SimpleNamespace(rdan_save_rng=SimpleNamespace(remote=remote)) for _ in range(2)]
    )
    pipeline.scheduler = SimpleNamespace(get_scheduler_state=SimpleNamespace(remote=lambda: {"dataset_iter_count": 1}))
    logged: list[Path] = []
    pipeline.tracker = SimpleNamespace(log_artifact=lambda path, **kwargs: logged.append(Path(path)))
    monkeypatch.setattr(
        module,
        "WorkerState",
        SimpleNamespace(
            save_rng_state=lambda path, name: torch.save(torch.get_rng_state(), Path(path) / f"rng_state_{name}.pth")
        ),
    )
    result = SimpleNamespace(
        promotion_ready=True,
        metrics={"actor/clipfrac": 0.2, "actor/grad_norm": 1.0},
        peak_memory_fraction=0.8,
    )
    observations = {
        "post_receipt": {"status": "receipt_passed"},
        "memory": [{"rank": 0, "peak_bytes": 10}, {"rank": 1, "peak_bytes": 11}],
    }

    checkpoint = pipeline._save_step(1, Batch(), {"status": "receipt_passed"}, result, observations)

    manifest = load_checkpoint(checkpoint, identity=identity)
    assert manifest["group_diagnostics"]["response_active_group_rate"] == 1.0
    assert manifest["clipping_fraction"] == 0.2
    assert logged == [artifact_root / "step-000001"]
    rows = (logged[0] / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert "sk-or-v1" not in "".join(rows)
    assert json.loads(rows[0])["generation_id"] == "g0"
    assert json.loads(rows[0])["response_tokens"] == "[REDACTED]"
    assert json.loads(rows[1])["response_tokens"] == [12]
    assert (checkpoint / "artifacts/step.json").is_file()

    pipeline.stop_after_step = 20
    deferred = pipeline._save_step(2, Batch(), {"status": "receipt_passed"}, result, observations)
    assert deferred is None
    assert not (checkpoint_root / "step-000002").exists()
    deferred_manifest = json.loads((artifact_root / "step-000002/manifest.json").read_text(encoding="utf-8"))
    assert deferred_manifest["checkpoint"]["status"] == "not_scheduled"

    interleaved = Batch()
    interleaved.non_tensor_batch["rdan_prompt_key"] = ["p0", "p1", "p0", "p1"]
    with pytest.raises(RuntimeError, match="interleaved"):
        module._group_diagnostics(interleaved, 2)

    promoted = False

    def observe_promotion(*args: object, **kwargs: object) -> Path:
        nonlocal promoted
        promoted = True
        return Path("unexpected")

    def fail_upload(*args: object, **kwargs: object) -> None:
        raise RuntimeError("artifact upload failed")

    monkeypatch.setattr(module, "promote_checkpoint", observe_promotion)
    pipeline.tracker = SimpleNamespace(log_artifact=fail_upload)
    pipeline.stop_after_step = 2
    with pytest.raises(RuntimeError, match="artifact upload failed"):
        pipeline._save_step(2, Batch(), {"status": "receipt_passed"}, result, observations)
    assert promoted is True
    assert (checkpoint_root / ".incomplete-step-000002").is_dir()


def _constructor(tree: ast.AST) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ResponseTrainingPipeline":
            return next(
                child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"
            )
    raise AssertionError("ResponseTrainingPipeline.__init__ not found")


def _statement_index(statements: list[ast.stmt], target: str) -> int:
    for index, node in enumerate(statements):
        if isinstance(node, ast.Assign) and _name(node.targets[0]) == target:
            return index
    raise AssertionError(f"assignment not found: {target}")


def _contains_call(node: ast.AST, target: str) -> bool:
    return any(isinstance(child, ast.Call) and _name(child.func) == target for child in ast.walk(node))


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    return ""


def _load_pipeline(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    ray = types.ModuleType("ray")
    ray.get = lambda value, **kwargs: value
    ray.remote = lambda value: value
    scheduling = types.ModuleType("ray.util.scheduling_strategies")
    scheduling.NodeAffinitySchedulingStrategy = object
    modules = {
        "ray": ray,
        "ray.util": types.ModuleType("ray.util"),
        "ray.util.scheduling_strategies": scheduling,
    }
    names = {
        "roll.datasets.collator": {"DataCollatorWithPaddingForPaddedKeys": object},
        "roll.datasets.dataset": {"get_dataset": lambda value: None},
        "roll.distributed.executor.cluster": {"Cluster": object},
        "roll.distributed.scheduler.generate_scheduler": {"DynamicSamplingScheduler": object},
        "roll.distributed.scheduler.protocol": {"DataProto": object},
        "roll.models.model_providers": {"default_tokenizer_provider": lambda **kwargs: None},
        "roll.pipeline.base_pipeline": {"BasePipeline": object},
        "roll.pipeline.rlvr.rlvr_pipeline": {
            "get_encode_function": lambda *args: None,
            "preprocess_dataset": lambda *args, **kwargs: None,
            "update_dataset_domain": lambda *args: None,
        },
        "roll.utils.worker_state": {"WorkerState": object},
        "rdan_grpo.roll_response_workers": {"ResponseActorWorker": object, "ResponseInferWorker": object},
    }
    for name, attributes in names.items():
        fake = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(fake, key, value)
        modules[name] = fake
    for name, fake in modules.items():
        monkeypatch.setitem(sys.modules, name, fake)
    path = ROOT / "src/rdan_grpo/roll_response_pipeline.py"
    spec = importlib.util.spec_from_file_location("test_response_pipeline_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "torch", torch)
    return module
