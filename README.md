# RDAN-GRPO

**R**ubric **D**ecoupled **A**dvantage **N**ormalization GRPO: rubric-based RL for open-domain instruction following.

[Checkpoints](https://huggingface.co/beingamanforever/Qwen3-4B-RDAN-GRPO) ·
[Collection](https://huggingface.co/collections/beingamanforever/rdan-grpo-6a845a4225a35fc112217ae6) ·
[Experimental details](docs/EXPERIMENTS.md)

![RDAN-GRPO](docs/assets/rdan-grpo.png)

## Method

Instructions carry checkable requirements ("under 500 words") and matters of judgment (is it well
written). Reward judged quality directly and the policy learns that good prose earns credit while the
hard constraints become optional.

RDAN samples `G` responses per prompt, scores each twice, and normalizes the two scores over
**different populations**:

```
A_out  = (r_out  - mu)   / sigma      over all G responses
A_proc = (r_proc - mu_C) / sigma_C    over constraint-satisfying responses only
A      = A_out + quality_weight * A_proc
```

The second denominator is the whole method. A response failing any hard constraint leaves the process
statistics entirely, so quality is a tiebreaker among compliant responses and never a substitute for
compliance. It also keeps learning alive once the outcome channel saturates.

Outcome channel: deterministic checkers (IFEval-style, MulDimIF constraints, sandboxed dataset rules,
RubricHub routes). Process channel: LLM-judged soft rubrics. `rl_csr` and `rl_aon` are the
single-channel baselines.

## Results

Qwen3-4B-Instruct-2507, full official datasets, greedy decoding, each benchmark's own scorer.

| Checkpoint | IFEval | IFBench | MulDimIF | MATH-500 | GPQA |
|---|---|---|---|---|---|
| base | 82.99 | 30.95 | 57.17 | 87.60 | 55.05 |
| step 100 | 85.21 | 34.35 | 68.92 | 88.40 | **59.09** |
| **step 200** | **87.06** | **35.71** | 72.58 | 88.80 | 58.08 |
| step 300 | 86.69 | 34.69 | 73.25 | **89.60** | 57.58 |
| step 400 | 86.69 | 35.37 | 74.08 | 86.40 | 58.59 |
| step 420 | 86.88 | 33.67 | **74.67** | 87.20 | 55.56 |

Instruction following plateaus at step 200. MulDimIF climbs the whole way, +17.50 over base, and is
the benchmark closest to the training distribution. MATH-500 and GPQA are out of domain and decay
back to base by 420, so **step 200 is the checkpoint to use** unless MulDimIF is the target.

Scores and curves are in [`results/`](results/). Training stopped at step 430 of a planned 500.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "beingamanforever/Qwen3-4B-RDAN-GRPO"   # private, needs access
model = AutoModelForCausalLM.from_pretrained(repo, subfolder="step-000200", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="step-000200")
```

Steps 100, 200, 300, 320, 400, 420 are published. Trained with thinking disabled; evaluate the same way.

## Running it

Needs a checkout of RTT, which supplies the ROLL runtime. The setup script installs everything on a
fresh CUDA host, then fill in `OPENROUTER_API_KEY` and `WANDB_API_KEY` in the generated `.env`.

```bash
scripts/setup_env.sh /path/to/Rubrics-To-Tokens /path/to/Qwen3-4B-Instruct-2507
python scripts/preflight.py                    # real model, real judge, projects cost and latency
python scripts/train.py --config rdan          # or rl_csr / rl_aon, add --resume to continue
```

Evaluation caches generations, so reruns are free:

```bash
python scripts/eval_if.py  --model base=/path/to/model --rtt-root /path/to/Rubrics-To-Tokens --out results/if-eval
python scripts/eval_ood.py --model base=/path/to/model --data-root /path/to/benchmark-data --out results/ood-eval
```

### Changing the GPU count

`num_gpus` lives in [`configs/train/base.yaml`](configs/train/base.yaml), but Hydra overrides are
positional so no edit is needed. The global batch is
`per_device_train_batch_size x gradient_accumulation_steps x world_size` and must divide
`rollout_batch_size x num_return_sequences_in_group`, so doubling the GPUs means halving the
accumulation:

```bash
python scripts/train.py --config rdan num_gpus=4 actor_train.training_args.gradient_accumulation_steps=16
```

About 5.5 min per step on 2x A100 80GB, so ~1.9 days for 500 steps and ~1 day on 4 GPUs. Judge spend
is around $10 per run against `qwen/qwen3.7-flash`.

## Layout

| Path | Role |
|---|---|
| `src/rdan_grpo/advantages.py` | masked group standardization, shared by both channels |
| `src/rdan_grpo/scalar.py` | RDAN advantage composition and the baselines |
| `src/rdan_grpo/rewards.py`, `rules.py` | rubric scoring, checkers, sandboxed rule runner |
| `src/rdan_grpo/judge.py` | concurrent OpenRouter judge with retries and cost accounting |
| `src/rdan_grpo/reward_worker.py`, `bridge.py` | ROLL reward worker and batch adapter |
| `src/rdan_grpo/pipeline.py`, `train_step.py`, `workers.py` | training loop, FSDP2 actor, vLLM rollout |
| `src/rdan_grpo/checkpoint.py` | atomic promotion, resume lookup, retention, Hub upload |
| `scripts/` | setup, preflight, training, evaluation, distillation |

Long runs on paid APIs fail in specific ways, so these degrade rather than crash: a judge failure
clears the soft channel for that response only, an undecidable checker drops the response from group
statistics rather than scoring it zero, a non-finite gradient skips the update, and W&B being
unreachable falls back to the JSONL mirror.

## Tests

```bash
pytest
```

Advantage expectations are recomputed in plain Python inside the tests, so they check the math rather
than echo it. Tests needing the ROLL runtime skip where it is unavailable.

## License

Apache-2.0, see [`LICENSES/`](LICENSES/).
