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

Requires a checkout of RTT, which supplies the ROLL distributed RL runtime. On a fresh CUDA
host the setup script installs everything, including the pieces whose absence produces
misleading errors (Python headers for Triton, a flash-attn wheel matching the local torch
build, and a wandb new enough for current API keys):

```bash
scripts/setup_env.sh /path/to/Rubrics-To-Tokens /path/to/Qwen3-4B-Instruct-2507
```

Then fill in `OPENROUTER_API_KEY` and `WANDB_API_KEY` in the generated `.env`.

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

Hydra overrides are positional, so scaling GPUs or shortening a run needs no config edit.
The global batch is `per_device_train_batch_size x gradient_accumulation_steps x world_size`
and must divide `rollout_batch_size x num_return_sequences_in_group`, so doubling the GPUs
means halving the accumulation:

```bash
python scripts/train.py --config rdan num_gpus=4 actor_train.training_args.gradient_accumulation_steps=16
```

Resume from the latest checkpoint, or from a specific one:

```bash
python scripts/train.py --config rdan --resume
```

## Cost and wall clock

Measured on 2x A100 80GB with Qwen3-4B-Instruct-2507, 64 prompts x 8 samples per step:

| phase | time per step |
|---|---|
| actor update | 195s |
| log-prob recompute | 54s |
| generation and reward | 66s |
| offload and weight sync | 14s |
| **total** | **~5.5 min** |

That is ~1.9 days for 500 steps on 2 GPUs, and roughly 1 day on 4. Judge spend is about
$0.02 per step, so around $10 for a full run against `qwen/qwen3.7-flash`.

Sequence length is the dominant cost, and both bounds are set from measurement rather than
convention: prompts are mean 225 and p99.5 1280 tokens, and 97.2 percent of responses finish
under 2048, so a row costs 3328 tokens instead of the 6144 the defaults implied. Watch
`length/cap_hit_rate` on the curves; if it climbs well above the measured 2 percent the
response cap is truncating real work and should go back up.

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
| `scripts/setup_env.sh` | reproducible environment build on a fresh CUDA host |
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
| `src/rdan_grpo/checkpoint.py` | atomic checkpoint promotion, resume lookup, retention |

## Failure handling

Long RL runs on paid APIs fail in specific ways, so these are handled rather than crashed on:

- A judge call that exhausts its retries clears the soft channel for that response only. The response keeps its outcome
  reward and its place in the group, and `judge/failure_rate` records it.
- A deterministic checker that cannot decide clears that rubric's evaluation bit; the response drops out of the group
  statistics instead of being scored zero.
- A non-finite gradient skips the update and increments `skipped_updates` rather than ending the run.
- W&B being unreachable degrades to offline mode and then to the JSONL mirror.
- The trainer and the rollout engine share the same GPUs, so each releases its cached blocks
  at the handoff. Offloading tensors alone leaves the pages with PyTorch's allocator, and
  vLLM's KV pool cannot be mapped around them.

## Tests

```bash
pytest
```

Advantage expectations are recomputed in plain Python inside the tests, so they check the math rather than echo it.
Tests needing the ROLL runtime skip where it is unavailable and run on the training host.
