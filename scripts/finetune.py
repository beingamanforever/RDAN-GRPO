#!/usr/bin/env python3
"""LoRA fine-tune the base policy on distilled data, as an SFT or a DPO baseline.

Both stages share one adapter shape and one sequence budget so their curves are comparable
against the RL runs. LoRA rather than full fine-tuning is forced by the hardware: a 4B model
with Adam state needs roughly 60GB, and it also lets DPO hold its reference policy by
disabling the adapter instead of loading a second copy of the weights.

Each stage writes its adapter and, under ``merged/``, the same weights folded back into the
base model. The merged copy is what the next stage and the evaluation harness consume: DPO
runs on top of the SFT policy, and vLLM loads full weights rather than an adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.tracking import METRICS_FILE, STEP_METRIC, plot_curves  # noqa: E402

# Prompts reach 1280 tokens and responses 2048, the same budget the RL rollouts use.
MAX_LENGTH = 3328
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# The RL runs report here too, so a baseline overlays on the same charts.
WANDB_PROJECT = "rdan-grpo-qwen3-4b"
WANDB_GROUP = "qwen3-4b-instruct-2507"


def main() -> int:
    """Train one stage, then save the adapter and the merged full weights."""

    args = _parse_args()
    _configure_wandb(args.out)
    import datasets
    from peft import LoraConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = datasets.load_dataset("json", data_files=str(args.data), split="train")
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    print(f"{args.stage}: {len(data)} rows from {args.data}")

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    metrics_path = args.out / "logs" / METRICS_FILE
    trainer = _build_trainer(args, data, tokenizer, peft_config)
    trainer.add_callback(_metric_mirror(metrics_path))
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"saved adapter to {args.out}")
    for plot in plot_curves(metrics_path):
        print(f"wrote {plot}")
    if not args.skip_merge:
        _merge(args.model, args.out, tokenizer)
    return 0


def _configure_wandb(out: Path) -> None:
    """Point the run at the RL project, and fall back to offline rather than failing.

    Credentials come from an API key or the machine's netrc, so an unauthenticated host is the
    only case that needs offline mode; reporting must never be what ends a training run.
    """

    import netrc

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    os.environ.setdefault("WANDB_RUN_GROUP", WANDB_GROUP)
    os.environ.setdefault("WANDB_DIR", str(out / "logs"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    if os.environ.get("WANDB_API_KEY"):
        return
    try:
        authenticated = netrc.netrc().authenticators("api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        authenticated = False
    if not authenticated:
        os.environ.setdefault("WANDB_MODE", "offline")
        print("W&B has no credentials; logging offline to the local mirror")


def _metric_mirror(path: Path) -> Any:
    """Build the callback that appends Trainer logs to the JSONL mirror the RL runs write.

    Rows land before W&B sees them, so a run whose connection dies still yields full curves.
    """

    from transformers import TrainerCallback

    class MetricMirror(TrainerCallback):
        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
        ) -> None:
            row = {name: value for name, value in (logs or {}).items() if isinstance(value, (int, float))}
            if not row:
                return
            row[STEP_METRIC] = state.global_step
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    return MetricMirror()


def _merge(base_model: str, adapter_dir: Path, tokenizer: Any) -> None:
    """Fold the adapter back into the base weights and save a standalone model."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    merged_dir = adapter_dir / "merged"
    # On the CPU, because the trainer still holds the GPU and a merge needs no accelerator.
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"saved merged weights to {merged_dir}")


def _build_trainer(args: argparse.Namespace, data: Any, tokenizer: Any, peft_config: Any) -> Any:
    """Construct the stage's trainer with the settings shared by both stages."""

    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    shared = {
        "output_dir": str(args.out),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": ["wandb"],
        "run_name": args.out.name,
        "seed": args.seed,
        "model_init_kwargs": {"dtype": "bfloat16", "attn_implementation": "flash_attention_2"},
    }
    if args.stage == "sft":
        config = SFTConfig(max_length=MAX_LENGTH, **shared)
        return SFTTrainer(
            model=args.model, args=config, train_dataset=data, processing_class=tokenizer, peft_config=peft_config
        )
    config = DPOConfig(max_length=MAX_LENGTH, beta=args.beta, **shared)
    return DPOTrainer(
        model=args.model, args=config, train_dataset=data, processing_class=tokenizer, peft_config=peft_config
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sft", "dpo"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("RDAN_MODEL_SNAPSHOT"), help="base model path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, help="default 1e-4 for SFT, 5e-6 for DPO")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL strength")
    parser.add_argument("--limit", type=int, help="train on the first N rows only")
    parser.add_argument("--skip-merge", action="store_true", help="save the adapter only")
    parser.add_argument("--seed", type=int, default=240520)
    args = parser.parse_args()
    if not args.model:
        raise ValueError("--model or RDAN_MODEL_SNAPSHOT is required")
    # DPO moves the policy against a fixed reference, so it needs a far smaller step than SFT.
    args.learning_rate = args.learning_rate or (1.0e-4 if args.stage == "sft" else 5.0e-6)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
