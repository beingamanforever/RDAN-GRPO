# RDAN-GRPO

RTT rubric rewards plus PAPO process-aware advantage on Qwen3-4B-Instruct-2507.

## Method

ORM is the binary hard-rubric outcome in {0,1}, normalized over all 8 responses in a group.
PRM is the LLM-judge soft-rubric score in {0,0.5,1}, normalized over the correct subset only, requiring at least 2 correct.
`A_total = A_out + A_proc`, unweighted, matching PAPO Eq 5.
Both components are zero when a group has fewer than 2 usable responses or zero variance.
The token-level advantage from RTT is not implemented yet and arrives once the token discriminator is trained.
Judge is `openai/gpt-5.6-luna` at medium reasoning effort, scores validated against the enum before use.

## Experiments

| Run | Isolates | Config |
|---|---|---|
| `rtt_papo_response` | ORM + PRM, the full method | `configs/roll/qwen_rtt_papo_response_train.yaml` |
| RL-ORM | outcome only | `configs/roll/qwen_rl_aon_train.yaml` |
| RL-PRM | process only, PAPO negative control | `configs/roll/qwen_rl_csr_train.yaml` |

Each run is 500 steps at `rollout_batch_size: 64` with 8 responses per prompt and `save_steps: 20`.
Runs are sequential because each needs the whole node.
Data is `data/hybrid.jsonl`.

## Run it

```bash
export RTT_ROOT=/path/to/Rubrics-To-Tokens
export OPENROUTER_API_KEY=... WANDB_API_KEY=...
python scripts/train.py --config configs/roll/qwen_rtt_papo_response_train.yaml
```

Resume from a checkpoint with `--resume output/checkpoints/step-000020`.
Stop early with `--steps N`.

For 4x A100 change only these values in the train config:

- `num_gpus_per_node: 4`
- `actor_train.device_mapping` and `actor_infer.device_mapping` to `[0,1,2,3]`
- `actor_infer.world_size: 4`
- `gradient_accumulation_steps: 64`, keeping `rollout_batch_size * num_return_sequences` equal to `per_device_train_batch_size * gradient_accumulation_steps * num_gpus * 2`

## Layout

`src/rdan_grpo` holds the method.
`advantages.py` and `scalar.py` are the ORM/PRM math, `bridge.py` adapts it to the ROLL advantage seam, `reward_worker.py` and `judge.py` produce the rewards, `pipeline.py` and `train_step.py` and `workers.py` run the loop, `compat.py` patches symbols the pinned fork is missing.
`scripts` holds `train.py` plus the two evaluation entrypoints.
`tests` are end to end: `test_training_e2e.py` drives config to checkpoint and resume, `test_reward_e2e.py` drives rubric rows to advantages, `test_config_e2e.py` proves the shipped configs build.

## Known gaps

The judge aborts the run if a call still fails after its retries, because `training_ready` requires every response valid.
PAPO tolerates this per response and excluding only the failed response would be the more robust choice.

`test_config_e2e.py` constructs through a stand-in for the fork's `RLVRConfig` rather than the fork dataclass itself, so it proves our validator accepts the shipped config, not that the fork parses it.

The vLLM rollout worker's `generate` is exercised by its own test rather than inside the end-to-end training path.
