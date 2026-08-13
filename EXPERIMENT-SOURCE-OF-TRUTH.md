# RDAN-GRPO experiment source of truth

This file is the authoritative ledger for the Qwen-first experiment.
Every result claim must point to a run entry, immutable config, artifact manifest, and code revision recorded here.

## Scope

The first model is `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
The response-only training corpus merges the full Qwen-eligible HIR-16K inventory with the certified English-only instruction-following partition of `sojuL/RubricHub_v1`.
The HIR-16K source SHA-256 is `465a01c19dc29e2c8d1cf183ccf3135872f7ec94ef10b20b7eb35603164c183b`.
The RubricHub source revision is `3837d55971473a872e84879c88f708b8da3ec2ef` and its source SHA-256 is `4ba368a4c6f5b21dd25e43908a4cd8e9e577c176943df3768fbce5898a14f049`.
The physical HIR source contains 16,968 rows.
The Qwen response-training partition contains 16,962 HIR rows after excluding the six overlength rows `9604`, `9776`, `9854`, `9943`, `10531`, and `11279` under the frozen non-thinking tokenizer contract.
The main hybrid corpus therefore contains 16,962 HIR rows and 1,134 RubricHub rows for a total of 18,096 rows.
The separately frozen 5,699-row Type-4 HIR partition remains the strict deterministic scalar ablation and is not the main response-training corpus.
The RTT source is `TURLEing/Rubrics-To-Tokens` at `b1ab2fba9bece98674e5fa6e6c808d9d63235778`.
The PAPO source is `tanzelin430/PAPO` at `277dac1cfe29d465ad9aee0499b4869b698d2e97`.
Reference repositories remain separate checkouts and are never vendored into this repository.

## Evidence states

Use only these states in the run ledger.

- `queued` means the immutable inputs are known but execution has not started.
- `preparing` means the node, runtime, model, or data is being prepared.
- `running` means the recorded process identity is alive and producing artifacts.
- `passed` means every run gate and artifact check passed.
- `failed` means execution ended and a failure record was sealed.
- `blocked` means a named scientific or operational prerequisite is unresolved.

Planned estimates are not measurements.
Measurements require sealed raw outputs and a SHA-256 manifest.

## Fixed execution order

The immediate method order is base, RTT+PAPO response-only, RL-CSR, RL-AON, AON-CSR mix, token-label generation, discriminator training and certification, RTT-AON, RTT-CSR, and full RDAN.
RTT+PAPO response-only uses hard AON validity plus a separately normalized conditional soft-quality advantage, with token weight fixed to zero.
Soft-only rows use a vacuous hard gate, zero response advantage, and an independently normalized valid quality advantage.
RL-CSR and RL-AON start only after the RTT+PAPO response-only run has sealed evidence.
Token-label generation starts only after RTT+PAPO response-only, RL-CSR, RL-AON, and the AON-CSR mix have sealed evidence.
Token-level policy methods start only after the discriminator gates pass.
Reconstructed SFT and DPO remain independent control lanes and do not change the RL execution order.
Every trainable method starts independently from the same pinned base checkpoint.

## Run naming

W&B project: `rdan-grpo-qwen3-4b`.
W&B entity: `RDAN-GRPO`.
Training groups use `qwen-<method>` and run names use `qwen-<method>-<stage>-s<seed>`.
Examples are `qwen-rl-aon-pilot-s240520` and `qwen-rdan-full-confirm-s240521`.
Evaluation groups use `qwen-<method>-eval` and run names use `qwen-<method>-eval-<benchmark>-s<seed>`.
Examples are `qwen-base-eval-ifeval-s42` and `qwen-rdan-scalar-eval-math-500-s1701`.
Names must not contain credentials, host addresses, or mutable checkpoint paths.

## Hardware allocation

| Lane | GPU | Current role | State |
|---|---:|---|---|
| `baseline-l40s` | 1x L40S 48 GB | Completed IFEval sensitivity run | `passed_and_idle` |
| `baseline-a40` | 1x A40 48 GB | Completed certified MulDimIF run | `passed_and_idle` |
| `baseline-a30` | 1x A30 24 GB | Completed certified IFBench run | `passed_and_idle` |
| `baseline-a10g` | 1x A10G 24 GB | Completed base MATH-500, model server still loaded | `passed_and_idle` |
| `rl-a100` | 2x A100 80 GB | Not provisioned until the static response-only release passes | `blocked` |

The production response-only RL topology uses two FSDP2 Hugging Face actor workers and two colocated Hugging Face inference workers on the same two GPUs.
The external rubric judge is CPU and API based, and the token discriminator remains disabled.
Parallel benchmark clients may share a 48 GB model server after the memory and latency probe passes.
Two independent model engines on one 48 GB GPU are not part of the initial plan.
SFT and DPO on one 48 GB GPU are reconstructed QLoRA controls, not full-parameter RTT reproductions.

## Capacity and cost accounting

The cost-optimized Qwen program requires one 2x A100 80 GB node for serial full-parameter RL and one A10G-first QLoRA lane for reconstructed SFT and DPO.
The current A10G is certified for inference and OOD evaluation, while QLoRA training remains blocked on a reviewed one-step memory and fresh-process resume gate.
An identical 48 GB fallback is authorized only after a sealed A10G gate failure.
Full-parameter SFT or DPO is not planned on 24 GB.
An additional 48 GB discriminator lane is optional and may be provisioned only after the token checkpoint and its measured memory behavior are certified.
An 8x H100 or 8x A100 node is not required for the initial Qwen program.

The immutable matrix contains 18 scalar RL runs, 6 reconstructed QLoRA runs, and 12 token-level RL runs.
This gives 36 training runs and 37 evaluation suites including the base model.
The planned six-benchmark matrix contains 222 executions, while the five currently runnable benchmarks contain 185 executions because GPQA access is blocked.
The 30 RL runs imply 7,680,000 generated training responses under the released logical recipe before retries or failed calls.

Total project GPU-hours are `pending_measurement`.
They will be calculated as the sum of allocated GPU count multiplied by sealed wall-clock hours for each run, with a 20 percent reserve applied once after component measurement.
No H100-equivalent estimate is treated as measured evidence.

## Baseline queue

| Run | Benchmark | GPU lane | State | Required output |
|---|---|---|---|---|
| `qwen-base-ifeval-certified-s42` | IFEval | `baseline-a10g` | `passed` | 541 generations, strict and loose evaluator rows, compact metrics, manifest |
| `qwen-base-ifbench-certified-s42` | IFBench | `baseline-a30` | `passed` | 294 generations, strict and loose evaluator rows, compact metrics, manifest |
| `qwen-base-muldimif-certified-s42` | MulDimIF | `baseline-a40` | `passed` | 1,200 generations, native score breakdown, compact metrics, manifest |
| `qwen-base-ifeval-crosscheck-s42` | IFEval sensitivity | `baseline-l40s` | `passed` | Independent runtime result and sealed manifest |
| `qwen-base-math-500-s1701-1705` | MATH-500 | `baseline-a10g` | `passed` | 2,500 resumable completions, strict Math-Verify records, compact metrics, manifest |
| `qwen-base-mmlu-pro-s1701-1705` | MMLU-Pro | `baseline-a10g` | `queued` | 60,160 resumable completions, strict final-line records, compact metrics, manifest |
| `qwen-base-gpqa-s1701-1705` | GPQA | `baseline-a10g` | `blocked` | Gated dataset bytes, immutable revision, and access manifest |
| `qwen-base-advancedif-s42` | AdvancedIF | unassigned | `blocked` | Certified adapter, pinned evaluator, and license-safe artifact policy |

Generation is greedy with seed 42, one response, 4,096 maximum new tokens, and thinking disabled through the pinned chat template.
Any HTTP error, empty response, row-count mismatch, duplicate identity, or scorer failure fails closed.

## Training queue

The tuning seed is `240520`.
The confirmation seeds are `240521`, `240522`, and `240523`.
Tuning and confirmation seeds must remain disjoint.

| Stage | Runs | State | Start gate |
|---|---:|---|---|
| Base evaluation | 3 certified in-domain runs plus OOD runs | `running` | MATH-500 passed, MMLU-Pro is queued, and GPQA access is blocked |
| RTT+PAPO response-only pilot | 1 | `blocked` | Full hybrid data freeze, judge calibration, runtime parity, no-update preflight, and W&B secret |
| RTT+PAPO response-only tuning and confirmation | 6 | `blocked` | Passing pilot, three conditional-quality weights, then one frozen choice on three fresh seeds |
| RL-CSR confirmation | 3 | `blocked` | Sealed RTT+PAPO response-only evidence and the same runtime gates |
| RL-AON confirmation | 3 | `blocked` | Sealed RL-CSR evidence and the same runtime gates |
| AON-CSR mix tuning and confirmation | 6 | `blocked` | Three candidate weights followed by one frozen choice on three fresh seeds |
| Reconstructed SFT | 3 confirmation seeds | `blocked` | Versioned targets, selection ledger, and QLoRA config |
| Reconstructed DPO | 3 confirmation seeds | `blocked` | Versioned pairs, exclusion ledger, and QLoRA config |
| RTT-AON | 3 confirmation seeds | `blocked` | Token labels, discriminator checkpoint, and token parity |
| RTT-CSR | 3 confirmation seeds | `blocked` | Token labels, discriminator checkpoint, and token parity |
| Full RDAN tuning and confirmation | 6 | `blocked` | Three candidate weights followed by one frozen choice on three fresh seeds |
| Confirmation subtotal across nine trainable methods | 27 | `blocked` | This is a subtotal of the rows above, not an additional set of runs |

The exact run count is derived by the checked program config.
No tuning run may be relabeled as a confirmation run.

## Mandatory gates

- The model, tokenizer, chat template, dataset, evaluator, and code revisions must match their manifests.
- Behavior and training tokenization plus pre-update log probabilities must pass parity.
- All hard criteria must have an authoritative route or belong to a frozen exclusion set reported separately.
- Hard-invalid, unsupported-hard, and judge-failed responses must receive no positive conditional quality credit.
- A no-update preflight must measure reward variance, active groups, output lengths, judge failures, and memory.
- A 20-step pilot must have finite gradients, less than 100 percent clipping, a restorable checkpoint, and a fresh post-resume generation.
- Any flat reward, zero useful within-group variance, invalid output, or failed checkpoint resume stops the lane.

## Artifact contract

Compact evidence is committed under `results/` after secret scanning and hash verification.
Large generations, judge records, and logs are uploaded to `beingamanforever/RDAN-GRPO-Qwen3-4B-evidence`.
Model checkpoints are uploaded to `beingamanforever/RDAN-GRPO-Qwen3-4B` only after their run passes.
W&B stores dashboards and metric histories, but W&B is not the sole artifact store.
Every final run directory must contain the resolved config, environment lock, model and data identities, raw generations, decomposed reward fields, failures, evaluator outputs, metrics, and a SHA-256 manifest.
Secrets and host credentials are excluded from every artifact.

## Result table

Three base benchmark runs have passed their native scorers and manifest checks.

| Method | IFEval | IFBench | MulDimIF | AdvancedIF | Evidence |
|---|---:|---:|---:|---:|---|
| Base Qwen | 83.92% strict, 87.25% loose instruction accuracy | 31.29% strict, 33.67% loose instruction accuracy | 56.75% overall | blocked | `results/base/summary.json` |
| RL-AON | pending | pending | pending | blocked | not started |
| RL-CSR | pending | pending | pending | blocked | not started |
| AON-CSR mix | pending | pending | pending | blocked | not started |
| RTT+PAPO response-only | pending | pending | pending | blocked | not started |
| Reconstructed SFT | pending | pending | pending | blocked | data not frozen |
| Reconstructed DPO | pending | pending | pending | blocked | data not frozen |
| RTT-AON | pending | pending | pending | blocked | discriminator unavailable |
| RTT-CSR | pending | pending | pending | blocked | discriminator unavailable |
| Full RDAN | pending | pending | pending | blocked | token lane unavailable |

The base Qwen MATH-500 run passed 2,181 of 2,500 seeded completions for 87.24 percent micro accuracy.
Its sealed evidence is stored under `results/base/math-500/`.

## Operational log

### 2026-08-13

The L40S, A40, A30, and A10G nodes were reached and audited as idle before setup.
Dedicated task directories were created without modifying unrelated project environments.
The pinned Qwen snapshot download was started independently on all four baseline nodes.
The three new root nodes received isolated Python environments, while the A10G reused only its previously verified vLLM runtime for model download.
The L40S, A40, and A30 isolated runtimes passed CUDA imports with vLLM 0.10.0 and Transformers 4.55.2.
The first L40S, A40, and A30 server starts failed because Python development headers were absent.
The failure logs were preserved, Python development headers were installed on L40S, A40, and A30, and all three repaired servers passed model identity and deterministic request probes.
The A10G server passed the same probe on its previously verified CUDA 12.8 runtime.
IFEval generation started on A30 and IFBench generation started on A10G only after the pinned input and local server gates were wired into the fail-closed harness.
MulDimIF generation started on A40 after its native scorer dependencies and deterministic request probe passed.
An independent IFEval cross-check started on L40S after its native scorer dependencies and deterministic request probe passed.
All four active runs preserve partial generations in task-owned directories and will become `passed` only after native scoring and final manifest verification.
The A100 node was unreachable from the control host at the first connection attempt, so its RL lane remained blocked rather than being reported as running.
The A100 SSH port later became reachable and its host key was recorded before access.
The node audit found two idle A100 80 GB PCIe GPUs, 210 GiB host RAM, 2.7 TiB available root storage, CUDA driver capability 12.8, and a PIX PCIe path between the GPUs.
The pinned RTT checkout was created under the task directory and an isolated Python runtime bootstrap started.
No optimizer update was launched because the no-update adapter, hard-route policy, calibrated judge, and frozen data gates have not passed.
The certified A10G IFEval run passed at 454 of 541 strict and 472 of 541 loose instruction checks.
Its strict and loose instruction accuracies are 83.92 percent and 87.25 percent.
Its strict and loose rubric accuracies are 89.97 percent and 91.13 percent.
The certified A30 IFBench run passed at 92 of 294 strict and 99 of 294 loose instruction checks.
Its strict and loose instruction accuracies are 31.29 percent and 33.67 percent.
Its strict and loose rubric accuracies are 33.73 percent and 36.72 percent.
The certified A40 MulDimIF run passed at 681 of 1,200, or 56.75 percent.
Only 106 and 108 of 541 IFEval response texts matched exactly in two hardware sensitivity comparisons.
The pinned checkpoint and greedy controls therefore remain hardware-sensitive, so every trained comparison must use the same certified runtime protocol.
The compact certified summary and comparison evidence are stored under `results/base/`.
Raw certified generations remain outside Git.
The L40S, A40, and A30 workloads are complete, their model servers are stopped, and all three GPUs are safe to power off.
The A10G model server remains active for base OOD evaluation.
MATH-500 uses 500 pinned items, five independent completion seeds, strict boxed-answer parsing, and a clean pinned Math-Verify checkout.
MMLU-Pro is staged but has not started.
GPQA remains blocked because the active Hugging Face credential cannot access the gated public dataset.
The strict deterministic scalar ablation contains 5,700 certified Type-4 rows before tokenizer filtering.
Exact ROLL preprocessing excludes row `11279` from that ablation because its non-thinking prompt has 2,610 tokens against the 2,048-token limit.
The effective strict ablation therefore contains 5,699 Type-4 rows.
The main response-only lane separately preserves every Qwen-eligible HIR type and uses the RTT-compatible reward worker.
That worker evaluates supported hard rules deterministically, uses the pinned Luna judge for unsupported hard and soft rubrics, records route provenance, and fails closed on evaluator or judge errors.
This full-data compatibility lane is explicitly recorded as `full_rtt_compatible_not_authoritative` and must not be reported as strict deterministic certification.
The two-A100 training and preflight configurations construct with the exact pinned snapshot.
The preflight resolves to zero actor-training devices, while the current production topology resolves to two FSDP2 Hugging Face actor workers and two Hugging Face inference workers.
The Megatron and vLLM attempts below are historical diagnostics that were rejected and are not the production training path.
A live runtime-parity attempt loaded the exact Qwen snapshot into vLLM on both A100s and then stopped before an optimizer step because RTT's released runtime lacks one expected `mcore_adapter.patcher` module.
A revision and byte-gated compatibility repair passed local and A100 checks, including a copied-package identity check for the Ray runtime.
The next parity attempt advanced through all workers but stopped before model construction because Megatron selected `TESpecProvider` while Transformer Engine was absent.
The supported `transformer_impl: local` configuration replaced the unavailable Transformer Engine path without installing an unpinned binary dependency.
The local-spec parity attempt initialized both Megatron actor replicas, both vLLM engines, and the reward topology, then failed closed before any optimizer update because RTT forwarded a two-dimensional integer attention mask into a model path that expected a causal mask.
A revision and byte-gated dense Qwen3 compatibility patch now retains the original batch mask for log-probability boundaries while asking local Megatron to construct its causal model mask.
The first parity collector recomputed its recorded token boundaries outside the actual actor forward callback, so it was rejected as insufficient evidence.
The replacement parity-only actor returns the exact input IDs, attention mask, response mask, and log probabilities observed inside the actor forward callback, and the artifact is bound to the exact training YAML, RTT revision, rollout backend, actor backend, and local transformer implementation.
The next live launch stopped before model allocation because the remote shell could not resolve the pinned Ray executable.
After the pinned runtime path was supplied, the launch reached reward-worker construction and failed closed because the judge calibration certificate is intentionally still pending.
This exposed an unnecessary reward dependency in the parity-only topology, and the replacement constructor is being restricted to actor training, actor inference, and a zero-worker reward route used only with RTT's explicit `skip_rewards` path.
The completed default-backend no-update diagnostic compared 16,610 sampled tokens from 32 responses against the actor callback.
It failed the frozen maximum and mean absolute-error thresholds with `0.4873282313` maximum error and `0.02377806368` mean error.
The median error was `0.00292851`, the 99th percentile was `0.1973867`, and only 61.48 percent of compared tokens were within `0.01`.
The signed mean difference was `0.00144557`, while one-token shift controls had approximately `0.938` mean error, so simple boundary misalignment is not a supported explanation.
No passing parity artifact was emitted and no optimizer step occurred.
The failed default-backend log is sealed at `/root/aman/logs/qwen-runtime-parity.failed-default-backend-s240520.log` on the A100 node.
Both GPUs returned to zero allocated memory after the failed run.
A controlled no-update diagnostic now tests vLLM eager execution with prefix caching and chunked prefill disabled while leaving the production YAML and frozen thresholds unchanged.
Its first detached launch failed before GPU allocation because the Ray executable was absent from the detached process path, and that full error is preserved separately.
The diagnostic was relaunched with the pinned runtime path explicitly supplied.
The eager diagnostic then completed 32 responses and compared 16,592 sampled tokens.
It failed the same frozen gate with `0.9263907373` maximum error and `0.02532634031` mean error.
Only 60.60 percent of tokens were within `0.01`, the median error was `0.003349662`, and the signed mean difference was `0.001694791`.
The one-token shift controls remained approximately `0.95`, so eager execution did not repair the mismatch and did not support a token-boundary explanation.
The eager failure log is sealed at `/root/aman/logs/qwen-runtime-parity.failed-eager-backend-s240520.log` on the A100 node.
Both GPUs returned to zero allocated memory, no passing parity artifact was emitted, and no optimizer step occurred.
The next bounded diagnostic was defined to prove the exact actor-converted tensor stream matched every tensor received and loaded by both vLLM replicas before any training run could start.
A separate no-update reference comparison evaluated the untouched pinned Hugging Face checkpoint against vLLM on 859 identical sampled-token positions from eight fixed prompts.
With Hugging Face eager attention, the mean absolute error was `0.011948217`, the median was `0.000005006`, and the maximum was `1.596875936`.
With Hugging Face SDPA, the mean was `0.009792659`, the median was `0.000002157`, and the maximum was `0.660750657`.
The FP32 Hugging Face control still had `0.009670862` mean error and `1.538064986` maximum error against BF16 vLLM, so simple actor BF16 rounding is not a sufficient explanation.
With both prefix caching and chunked prefill disabled, the Hugging Face SDPA versus vLLM comparison still had `0.008703084` mean error and `0.245271683` maximum error across 832 sampled-token positions.
Replacing vLLM FlashAttention with FlexAttention did not improve the control: mean error was `0.010611353` and maximum error was `0.722115934` across 838 positions.
These results show substantial cross-backend tail differences even without Megatron conversion, but they do not unlock training or justify relaxing the frozen gate.
The clean-checkout weight receipt diagnostic passed on both A100 replicas before generation and with zero optimizer updates.
Each actor sent 398 tensors totaling 8,044,959,744 bytes, and each paired vLLM loader consumed the same ordered tensor-byte manifest and returned successfully.
This establishes transport consistency and real loader acceptance only.
It does not establish that the loader applied those tensors or that the observed names and bytes are identical to vLLM's packed internal parameter layout.
The receipt artifact is `results/diagnostics/qwen-a100-weight-receipt-r4-s240520.json` with SHA-256 `d32fdbc9ff2ffcd65ff538f0bb51164936ff27bf92514f4498b32e486bbb663a`.
The separately preserved parity diagnostic compared 16,679 sampled tokens from 32 responses and failed the unchanged parity thresholds with `0.5600969791` maximum error and `0.0244097257` mean error.
Its failure artifact is `results/diagnostics/qwen-a100-parity-failed-r4-s240520.json` with SHA-256 `a50fbb93e611a775ceba4f003486d64bffa2d137ad2a247d105f5902851f7b93`.
These historical artifacts predate the immutable transaction, resolved-config, and receipt-digest linkage now required by the parity runner, so they are preserved as independent diagnostics and are not accepted as a cryptographically paired artifact set.
Together they motivate the same-backend diagnostic, but they do not by themselves isolate internal model application as the remaining cause.
The next diagnostic is a supported same-backend generation and training topology, not a threshold relaxation or hidden importance correction.
The first independent review of the same-backend repair found two launch blockers: caller-supplied parity diagnostics could enter a failure artifact, and the downstream program gate accepted only the obsolete Megatron-vLLM backend identity.
The A100 launch remains stopped while failure evidence is sanitized and a separate exact FSDP2-HF diagnostic plus production-config lifecycle contract is implemented and retested.
An independent response-method review confirmed the four offline reward and advantage definitions but found that the live lifecycle gate, run identity, and artifact contract were still scalar-RDAN-specific.
The prioritized RTT+PAPO response-only lane now has a method-scoped config, immutable run identity, exact launch-backend parity linkage, and decomposed reward evidence contract.
RL-CSR, RL-AON, and AON-CSR mix configs are validated scaffolds and remain intentionally non-launchable until the preceding run evidence and their own method-scoped lifecycle artifacts are frozen.
The A100 node was deprovisioned after returning to zero allocated GPU memory, and no optimizer update occurred.
All failed distributed logs were preserved, and both A100s returned to zero allocated memory after every failed attempt.
An independent review rejected the first response-training transaction before launch.
The draft lost prompt and rubric metadata before reward evaluation, used full-width instead of shifted token fields, accepted a synthetic receipt schema, inferred one optimizer update from one RPC, lacked a durable post-update recovery point, and used caller-supplied pre-run memory observations.
The replacement production design uses RTT opt-level-zero scheduling for metadata-preserving generation and rewards, explicit actor log-probability recomputation, sequence-minus-one training fields, measured per-rank optimizer and scheduler counters, repeatable FSDP2-to-HF byte receipts, and atomic checkpoints.
The next A100 node must not be provisioned until the repaired transaction, workers, receipts, checkpoint store, method-scoped config, CLI, independent review, and local caller-boundary verification all pass.
No passing runtime-parity artifact, judge calibration, no-update certificate, pilot checkpoint, or optimizer evidence exists yet.
No optimizer update has been launched.
Fresh OpenRouter and W&B environment credentials are still required on the A100 node before the judge calibration and scalar pilot can run.
The reconstructed control-data pipeline now pins the canonical Luna revision, rejects unsupported temperature, preserves exact provider routing, supports immutable resume, and freezes program-compatible SFT and DPO rows and manifests.
No live Luna teacher call has been made because the human-reviewed judge certificate, frozen dev split, and protected runtime credential injection are still pending.
Creation of the private Hugging Face evidence dataset failed with HTTP 403 because the active local token lacks repository-write permission.
Git storage is the current lossless fallback for these small baseline artifacts, while larger future artifacts remain blocked on a replacement Hugging Face write token.
Credentials pasted into chat are treated as compromised and are not reused or stored.
Phase 1 was independently reviewed and verified, then pushed over SSH to `feature/rdan-grpo` at commit `a630b676dcb902a16f198a5685a12f347a5460f9`.

### 2026-08-14

The HIR inventory was corrected from the restricted Type-4 ablation to all four released HIR types for the main response-training lane.
The physical HIR source has 16,968 rows and the frozen Qwen tokenizer gate retains 16,962 rows.
The six overlength HIR rows remain explicitly recorded in the Qwen HIR manifest instead of being silently dropped.
The English RubricHub gate retains 1,134 certified rows, producing an 18,096-row main hybrid corpus.
The 5,699-row certified Type-4 corpus remains available only as a strict deterministic ablation.
The RTT-compatible reward worker evaluates supported hard routes locally and batches unsupported hard plus soft rubrics into one pinned Luna call per response.
Every rubric outcome retains route provenance, and evaluator or judge failures fail closed.
The new `rtt_papo_response` objective decouples the response advantage from the conditional quality advantage.
Soft-only groups use a vacuous hard gate, zero response advantage, and independently normalized valid quality.
Mixed and hard groups preserve hard AON validity and conditional quality.
Method-scoped FSDP2 Hugging Face configurations now exist for RTT+PAPO response-only, RL-CSR, and RL-AON with the same base checkpoint and training budget.
The immutable execution order is RTT+PAPO response-only, RL-CSR, RL-AON, AON-CSR mix, and only then token-label and discriminator work.
The A100 bootstrap probes the exact hybrid reward worker and does not authorize an optimizer update before runtime parity, judge calibration, and no-update gates pass.
The two-A100 node is not provisioned, no live Luna training-judge call has been made, and no optimizer update has occurred.
The A10G MATH-500 run completed all 2,500 generations and passed its corrected sealed artifact checks at 87.24 percent micro accuracy.
The initial scorer incorrectly parsed the selected bare LaTeX payload and reported 74.56 percent, so that score is superseded and must not be cited.
The corrected scorer wraps the last boxed payload and gold answer for pinned Math-Verify extraction, changing 361 scoring rows without issuing any new model requests.
The A10G benchmark client has exited, while the idle vLLM server remains loaded and reserves about 20.6 GiB at zero measured utilization.
The MATH-500 command, metrics, resume identity, rescoring provenance, records, and SHA-256 manifest were copied into `results/base/math-500/` and rehashed locally.
The A10G can be powered off after this Git checkpoint is confirmed upstream.
