# RDAN-GRPO

**R**ubric **D**ecoupled **A**dvantage **N**ormalization GRPO: rubric-based RL for open-domain instruction following.

RDAN takes the rubric reward structure of [RTT](https://arxiv.org/abs/2604.02795) without its token-level relevance
discriminator, and applies the decoupled advantage normalization of [PAPO](https://arxiv.org/abs/2603.26535) to the
rubric channels. Per group of `G` responses to one prompt:

```
A_out  = normalize(outcome reward,  over responses with a well-defined outcome)
A_proc = normalize(soft rubric quality, over hard-passing responses only)
A      = A_out + quality_weight * A_proc
```

The outcome channel is the deterministic hard rubrics (IFEval-style checkers, MulDimIF constraints, sandboxed dataset
rules, RubricHub rule routes). The process channel is the LLM-judged soft rubrics. Normalizing the process channel over
the hard-passing subset is what stops a response from buying reward with good prose while violating the constraints,
which is the failure mode that collapses a naive process reward.

`rl_aon` and `rl_csr` are the RTT reward-union baselines: a single outcome channel over every rubric, all-or-nothing or
mean satisfaction respectively, with no process channel.

## Setup

Requires a checkout of RTT, which supplies the ROLL distributed RL runtime.

```bash
pip install -r requirements.txt && pip install -e .
export RTT_ROOT=/path/to/Rubrics-To-Tokens
export RDAN_MODEL_SNAPSHOT=/path/to/Qwen3-4B-Instruct-2507
export OPENROUTER_API_KEY=...
export WANDB_API_KEY=...        # optional, runs offline without it
```

## Preflight

Always run this before a long run. It generates on real prompts with the real model, scores them through the real
checkers and the real judge, and reports measured judge cost and latency projected over the full horizon.

```bash
python scripts/preflight.py
```

## Training

```bash
python scripts/train.py --config rdan
```

Baselines from the same base model:

```bash
python scripts/train.py --config rl_csr
```

```bash
python scripts/train.py --config rl_aon
```

Hydra overrides are positional, so scaling GPUs or shortening a run needs no config edit:

```bash
python scripts/train.py --config rdan num_gpus=4 max_steps=200
```

Resume from the latest checkpoint, or from a specific one:

```bash
python scripts/train.py --config rdan --resume
```

## Outputs

Under `output/<exp_name>/`:

- `checkpoints/step-NNNNNN/actor/` - Hugging Face safetensors plus tokenizer, loadable with `from_pretrained`, and the
  DCP shards and optimizer state needed to resume. The two most recent checkpoints are kept for resume; every 100th
  step is kept permanently so intermediate weights survive for evaluation.
- `logs/metrics.jsonl` - every metric row, written before it reaches W&B, so curves survive a lost connection.
- `logs/curves/*.png` - reward, advantage, policy, length, and judge panels rendered at the end of a run.
  Rebuild them any time with `plot_curves` from `rdan_grpo.tracking`.

## Layout

| Path | Role |
|---|---|
| `src/rdan_grpo/advantages.py` | masked group standardization, shared by both channels |
| `src/rdan_grpo/rewards.py` | rubric scoring and hard/soft channel extraction |
| `src/rdan_grpo/scalar.py` | RDAN advantage composition and the baseline methods |
| `src/rdan_grpo/judge.py` | concurrent OpenRouter judge with retries and cost accounting |
| `src/rdan_grpo/rules.py` | RubricHub checkers and the sandboxed dataset rule runner |
| `src/rdan_grpo/reward_worker.py` | ROLL reward worker: local checkers, one judge fan-out per batch |
| `src/rdan_grpo/bridge.py` | ROLL batch protocol to RDAN advantage adapter |
| `src/rdan_grpo/pipeline.py` | training loop, checkpointing, metrics |
| `src/rdan_grpo/train_step.py` | one optimizer transaction over a rewarded batch |
| `src/rdan_grpo/workers.py` | FSDP2 actor and seeded vLLM rollout workers |

## Failure handling

Long RL runs on paid APIs fail in specific ways, so these are handled rather than crashed on:

- A judge call that exhausts its retries clears the soft channel for that response only. The response keeps its outcome
  reward and its place in the group, and `judge/failure_rate` records it.
- A deterministic checker that cannot decide clears that rubric's evaluation bit; the response drops out of the group
  statistics instead of being scored zero.
- A non-finite gradient skips the update and increments `skipped_updates` rather than ending the run.
- W&B being unreachable degrades to offline mode and then to the JSONL mirror.

## Tests

```bash
pytest
```

Advantage expectations are recomputed in plain Python inside the tests, so they check the math rather than echo it.
Tests needing the ROLL runtime skip where it is unavailable and run on the training host.
