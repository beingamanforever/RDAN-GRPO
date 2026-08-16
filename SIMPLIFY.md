# Simplification spec

## Size comparison

All "ours now" figures are `wc -l` against the current working tree.
Current totals: `src/rdan_grpo` 26,467 lines across 45 modules, `scripts/` 7,830 across 29 files, `configs/` 682, `tests/` 18,694 across 39 files.
RTT reference: 6,550 owned lines for a full rubric GRPO job, of which 2,362 are genuinely method-specific.
PAPO reference on the same ROLL runtime: 252 lines of delta, zero new files, zero new classes.

| concern | RTT lines | PAPO lines | ours now | ours target |
|---|---:|---:|---:|---:|
| pipeline / run loop | 911 (`rubircs2token_pipeline.py` 797 + `base_pipeline.py` 114) | 0 (reuses stock ROLL) | 716 (`roll_response_pipeline.py`) | 300 |
| advantage / GRPO | 207 (`functionals.py` spans) | 118 (advantage 101 + dispatch 14 + norm bypass 3) | 1,041 (`roll_bridge.py` 649 + `roll_scalar.py` 199 + `advantages.py` 141 + `loss.py` 52) | 412 |
| reward + judge | 662 (`rubrics_llm_judge_reward_worker.py` 592 + `reward_scheduler.py` 70) | 124 (inline in one existing worker) | 2,395 (`roll_reward.py` 979 + `judge.py` 519 + `evaluator_cert.py` 288 + `safe_rule.py` 242 + `rubrichub_rules.py` 141 + `hir.py` 129 + `rewards.py` 97) | 1,119 |
| workers + weight transfer | 984 (`base_worker.py` 750 + `actor_worker.py` 234) | 0 | 4,439 (`roll_response_workers.py` 670, `roll_response_train.py` 487, `roll_live.py` 721, `weight_receipt.py` 614, `fsdp_hf_receipt.py` 645, `roll_same_backend.py` 556, `roll_same_backend_live.py` 470, `roll_fsdp_hf_receipt.py` 148, `roll_weight_receipt.py` 128) | 280 |
| config / frozen contract | 1,437 (7 dataclass files) | 10 (2 dataclasses) | 5,588 (`program.py` 3,876 + `response_identity.py` 460 + `routes.py` 426 + `roll_response_config.py` 399 + `roll_compat.py` 427) | 300 |
| checkpoint | 227 (`base_pipeline` 25 + `base_worker` 23 + `checkpoint_manager` 102 + `worker_state` 77) | 0 | 2,216 (`roll_response_checkpoint.py` 751 + `response_pilot_lifecycle.py` 983 + `response_publish.py` 482) | 130 |
| tracking / metrics | 630 (`metrics_manager` 452 + `tracking` 126 + `reduce_metrics` 52) | 26 (a metrics dict) | 1,659 (`metrics.py` 917 + `wandb_tracking.py` 486 + `roll_response_receipt.py` 256) | 240 |
| launcher + preflight + parity + readiness | 246 (launcher 36 + sh 12 + yaml 198) | 171 yaml + bash overrides | 10,220 (all of `scripts/` 7,830 + `runtime_parity.py` 1,001 + `response_readiness.py` 701 + `vllm_runtime_parity.py` 384 + `roll_vllm_parity_live.py` 304) | 360 |
| certificates / parity / readiness / receipts | **0, verified** | **0, verified** | 5,100+ | **0** |
| data prep + eval (not a reference concern) | n/a | n/a | 6,023 | 1,219 |
| **total, src + scripts + configs** | **6,550** | **252** | **34,979** | **4,280** |
| tests | not shipped for the method | not shipped for the method | 18,694 | 1,500 |

The gap on the two directly comparable numbers.
Our `advantages.py` is 141 lines against PAPO's 101 for identical math with identical `count >= 2` and `std > eps` guards.
Our `roll_reward.py` plus `judge.py` is 1,498 lines against PAPO's 124 and RTT's 592.
Our `roll_response_pipeline.py` at 716 lines is 2.8x the entire PAPO ROLL delta, and `scripts/bootstrap_a100_response.py` alone at 2,118 lines is 8.4x it.
Both references ship a working multi-GPU RL trainer with exactly zero lines of certificate, parity, readiness, receipt, or frozen-contract code.

## Target file layout

This is the smallest set that still runs `rtt_papo_response` end to end on the pinned ROLL fork, plus the base-eval track that is currently in flight.

| path | lines (budget) | responsibility |
|---|---:|---|
| `src/rdan_grpo/__init__.py` | 30 | package exports |
| `src/rdan_grpo/roll_response_pipeline.py` | 300 | clusters, scheduler, train loop, checkpoint cadence, W&B logging |
| `src/rdan_grpo/roll_response_train.py` | 110 | one train step: advantage adapter, log-prob binding, `actor_train.train_step`, metrics |
| `src/rdan_grpo/roll_response_workers.py` | 170 | `ResponseActorWorker` DCP save/load RPC and optimizer priming, `ResponseVLLMInferWorker` seeded generate |
| `src/rdan_grpo/roll_response_checkpoint.py` | 130 | stage, promote by atomic rename, load, counter agreement check |
| `src/rdan_grpo/roll_response_config.py` | 120 | build the fork's `RLVRConfig` from our yaml, bind worker classes |
| `src/rdan_grpo/roll_compat.py` | 180 | vLLM sampling seed patch, RTT import shim, torch DCP planner patches |
| `src/rdan_grpo/roll_bridge.py` | 120 | `make_roll_compute_advantage`, `inject_roll_advantages`, `attach_roll_reward_fields`, `install_roll_adapter` |
| `src/rdan_grpo/roll_scalar.py` | 120 | `ScalarOutput`, `validate_groups`, `build_scalar_output`, reward selection |
| `src/rdan_grpo/advantages.py` | 120 | ORM group advantage, PRM correct-subset advantage, `A_out + A_proc` |
| `src/rdan_grpo/loss.py` | 52 | token-level policy loss |
| `src/rdan_grpo/rewards.py` | 97 | binary hard-rubric outcome and soft-score aggregation |
| `src/rdan_grpo/roll_reward.py` | 360 | `RTTCompatibleRubricRewardWorker`: hard/soft split, deterministic ORM, batched judge call, eval mask |
| `src/rdan_grpo/judge.py` | 150 | OpenRouter request build, call, parse, per-rubric validation |
| `src/rdan_grpo/rubrichub_rules.py` | 141 | certified RubricHub rule evaluators |
| `src/rdan_grpo/safe_rule.py` | 242 | AST-validated, forked, timeout-bounded type4 rule sandbox |
| `src/rdan_grpo/hir.py` | 129 | HIR row decode helpers |
| `src/rdan_grpo/response_dataset.py` | 77 | JSONL loader that keeps heterogeneous rubrics out of Arrow |
| `src/rdan_grpo/response_sampling.py` | 33 | rollout sampling parameters |
| `src/rdan_grpo/wandb_tracking.py` | 120 | tracker registration, secret redaction |
| `src/rdan_grpo/data_prep.py` | 300 | one module replacing `rubrichub_data.py` + `control_data.py` + `scalar_data.py` |
| `src/rdan_grpo/baseline.py` | 400 | base-model eval harness |
| `src/rdan_grpo/baseline_models.py` | 89 | base-model registry |
| `src/rdan_grpo/ood.py` | 200 | OOD eval |
| `scripts/run_response_train.py` | 60 | argparse, hydra compose, `install_rtt_compat`, build pipeline, run |
| `scripts/prepare_response_data.py` | 40 | CLI over `data_prep.py` |
| `scripts/run_base_eval.py` | 45 | base eval CLI |
| `scripts/run_ood_eval.py` | 45 | OOD eval CLI |
| `scripts/plot_training.py` | 120 | offline curves from W&B export, absorbs what is worth keeping from `metrics.py` |
| `configs/roll/qwen_rtt_papo_response_train.yaml` | 120 | the single run spec |
| `configs/roll/qwen_rl_csr_train.yaml` | 15 | method overlay |
| `configs/roll/qwen_rl_aon_train.yaml` | 15 | method overlay |
| `configs/judges/*` | 30 | judge endpoint config and rubric prompt |
| **total** | **4,280** | |

Tests get a separate budget of 1,500 lines, giving a repo total of 5,780 against 53,673 today.

## Cuts by file

### `src/rdan_grpo/roll_response_pipeline.py` 716 -> 300

- `_initialize_run_contract` (:90) and `_validate_inputs` (:579): frozen-contract pinning against certificate, runtime identity, model identity, planned horizon and resolved config sha256. Fold the 6 load-bearing lines (resume manifest load, `completed_step`, `_start_step`, `state.step`) into `__init__`.
- `_step_observers` (:227) and `_transfer` (:309): weight-receipt transaction plumbing. Replace both with a bare `self.model_update(step)`.
- `_validate_step_promotion` (:329) and `_clipping_fraction` (:694): abort gates on `promotion_ready`, clipping and `quality_active_group_rate`, none of which the loss reads.
- `_group_diagnostics` (:632) and `_validate_group_keys` (:677): shape and group-contiguity re-validation already done by `roll_scalar.validate_groups` and again by `advantages.py`. Keep only the two rates, inlined into `_reward_curve_metrics` as ~4 lines.
- `_promoted_step_dirs` (:457): symlink and name-length re-validation of directories this file created. Replace with one `sorted(root.glob("step-??????"))` inside `_prune_checkpoints`.
- `build_response_training_pipeline` (:495): re-validates `worker_cls` and `strategy_name` that `roll_response_config` sets and the fork fails loudly on. Collapse to the constructor call.
- `TRAINING_STATE_ARTIFACT` (:48), `CompletedResponseRun` (:52), `_row_value` (:714): dead once the observers and lifecycle receipt are gone.
- Inside `_write_checkpoint_payload` (:352): drop `step.json`, receipt and metrics writes, keep only DCP save, RNG save and scheduler position.
- Inside `_checkpoint_state` (:374): drop `reward_variance`, `group_diagnostics`, `clipping_fraction`, `receipt_links`.

### `src/rdan_grpo/roll_response_train.py` 487 -> 110

- `_validate_rewarded_batch` (:196), `_group_equal` (:259), `_values` (:270): 85 lines duplicating `roll_scalar.py:45 validate_groups` and `roll_bridge.py:325` field checks on the same batch in the same step.
- `_validate_topology` (:133), `_worker_devices` (:167), `_validate_cluster` (:180): 63 lines of device-mapping and world-size re-derivation. Keep one line reading `num_return_sequences` as `group_size`.
- `_pre_update_reward_gate` (:317): abort gate on pre-update reward statistics.
- `_bind_method_evidence` (:289): method evidence for the deleted artifact tree.
- `_validate_token_fields` (:354): re-checks alignment the fork enforces on its own tensors.
- `_training_state` (:378), `_validate_state_delta` (:409), `_validate_post_transaction_memory` (:425): optimizer-counter and memory-delta proof. Replaced by asserting a finite nonzero `actor_train/grad_norm` inside `_train_metrics`.
- `_execute_actor_update` (:92): fold the single `actor_train.train_step` line into `run_response_train_step`.
- `ResponseTrainResult` (:32): drop to four fields, `method`, `prompt_count`, `response_count`, `metrics`.

### `src/rdan_grpo/roll_response_workers.py` 670 -> 170

- `train_step` (:86), `_step_handlers` (:348), `handle_optimizer_step`, `handle_scheduler_step`, `_block_update` (:456), `_counters` (:462), `_train_state` (:475), `rdan_train_counters` (:43), `rdan_training_state` (:49), `_valid_counters` (:636), `_valid_checkpoint_state` (:652): the optimizer-step monkey-patch apparatus. The fork already skips `optimizer.step()` on non-finite grad at `fsdp2_strategy.py:1205-1216`, and it advances the scheduler on a skipped step, so `handle_scheduler_step` would kill the run on the first non-finite grad. Record the expected optimizer step by reading `optimizer.state[p]["step"]` at save time, ~3 lines inside `_save_dcp`.
- `loss_func` (:33), `_response_clip_fraction` (:389), `_current_log_probs` (:405), `_aligned_clip_tensors` (:418), `_importance_ratio` (:430), `_clip_bounds` (:446): recompute of a metric the fork already emits at `base_worker.py:255-259`, at the cost of a second full `op_compute_log_probs` per microbatch, and masked differently from the fork.
- `_all_ranks_grad_finite` (:493), `_grad_is_finite` (:509), `_all_ranks_fully_clipped` (:483): duplicate finiteness check that raises where the fork skips, converting a recoverable batch into a dead run.
- `start_model_update` (:123) and `_vllm_response_receipt` (:309), plus the `roll_weight_receipt` imports: the fork's `Worker.start_model_update` path is complete on its own.
- `rdan_save_rng` (:171), `rdan_load_rng` (:186): 42 lines persisting two integers and re-validating the dict this module wrote. vLLM is reseeded per request from `_generation_seed`, so this changes nothing on resume.
- `_vllm_engine_metrics` (:274) and the `_VLLM_METRICS_STATUS_*` constants: four distinct status codes for a rollout throughput chart.
- `rdan_reset_cuda_peak` (:74, :215), `rdan_cuda_memory` (:80, :221), `_cuda_memory` (:664): the fork logs GPU memory around every `train_step` and `generate` via `state_offload_manger`.
- Inside `generate`: the `strategy_name` assert, step-contiguity gate, ordinal derivation, `generation_config` isinstance gate, `vllm_metrics` attach and `generation_id` construction.

### `src/rdan_grpo/roll_response_checkpoint.py` 751 -> 130

- `_IDENTITY_KEYS`, `_OPTIONAL_IDENTITY_KEYS`, `_MANIFEST_KEYS`, `_WANDB_KEYS`, `_REVISION_KEYS`, `_GROUP_KEYS` (:23-61): exact-key-set literals that make any manifest field addition a breaking change.
- `ArtifactIdentity` (:68) and 7 of the 9 `CheckpointIdentity` fields: keep only `method` and `planned_horizon`.
- `_scan_files` (:404), `_inventory_entry` (:429), `_verify_inventory` (:696): SHA-256 of every checkpoint byte on write and again on every load, so a multi-GB DCP tree is fully re-read twice per step.
- `_artifact_links` (:443), `_validate_links` (:490): builds `rng_artifacts` and `receipt_links` tables that `_restore_checkpoint` never dereferences, it loads RNG from hardcoded paths.
- `_identity_value` (:298), `_state_value` (:349), `_validate_loaded_state` (:502), `_validate_manifest_linkage` (:463), `_validate_inventory` (:473), `_group_diagnostics` (:522): write-then-re-prove of JSON this module serialized itself, including the `reward_variance > 0` and `clipping < 1` gates already enforced in the pipeline before promotion.
- `_json_safe` (:617), `_number_mapping` (:585), `_string_mapping` (:603), `_rank` (:637), `_label` (:651), `_finite_number` (:657), `_positive_int` (:672), `_sha256` (:685), `_rank_counters` (:555): a hand-rolled JSON type system that `json.dumps(allow_nan=False)` already enforces.
- `_quarantine_stage` (:134), `_checkpoint_root` (:252), `_stage_path` (:265), `_promoted_path` (:278), `_relative_path` (:394), `_fsync_tree` (:705), `_step_lock` (:731): symlink and path-traversal hardening plus an O_EXCL lock on a single-writer path.

### `src/rdan_grpo/roll_reward.py` 979 -> 360

- `_load_rubrichub_contract` (:708), `_load_rubrichub_evidence` (:740), `_load_rubrichub_tokenizer` (:813), `_validate_rubrichub_row` (:578): 291 lines, 29.7% of the file, re-proving at every worker start and again per row that a frozen dataset still matches its own hashes. Replace with the module constant `CERTIFIED_FUNCTIONS` already in `rubrichub_rules.py:35` and a 4-line `json.loads` of `truth["rubric_routes"]`.
- `ScalarRubricRewardWorker` (:46-173): 128 lines, a second reward lane the production config never selects. Deleting it removes the only caller of `scalar_data.py` (640 lines).
- `_rubric_evidence` (:515): a 15-field per-rubric receipt whose only consumer was the deleted step-artifact writer.
- `_validate_hir_metadata` (:463): `_evaluate_hir_hard` already wraps every one of those index operations in a `try/except` that routes to `_failed_local_info`, so this changes the error message and not the outcome.
- `_load_type4_hashes` (:698): the type4 body allowlist. This is the one cut that loosens a code-execution control, so it needs `allowed_hashes` made optional at `safe_rule.py:167` and the sandbox itself kept intact.
- `_load_calibrated_effort` (:967) and `_load_program_seed` (:975): replace with two keys in the judge config.
- `_read_json_object`, `_same_json`, `_json_sha256`, `_canonical_json`, `_sha256_file`, `_is_sha256` (:930-964): unreferenced once the loaders go.
- The `evaluator_cert`, `scalar_data` and `wandb_tracking` imports plus the 8 certificate path constants.
- Dead output fields inside kept methods: `rdan_hybrid_hard_fallback` (:249), `rdan_evaluator_failed` (:250), `rdan_reward_lane` (:252), which have zero consumers anywhere in `src/` or `scripts/`.

### `src/rdan_grpo/roll_bridge.py` 649 -> 120

- `BatchAssessment` (:24), `PreflightCertificate` (:40), `as_dict` (:53), `assess_scalar_batch` (:73), `build_preflight_certificate` (:137), `write_certificate` (:377), `require_train_certificate` (:386), `_certificate_ready` (:604), `_certificate_keys` (:630): the whole preflight certificate subsystem.
- `_require_method_binding` (:430), `_resolve_method_parameters` (:456), `_check_method_weight` (:478): method binding re-validation that `roll_response_config` already performs.
- `sha256_file` (:486), `_validated_sources` (:585), `_check_sha256` (:597), `_canonical_json` (:648): source pinning.
- `_expected_hard_route` (:552), `_json_object` (:565): route re-derivation duplicating `roll_reward._hard_mask`.
- Keep `make_roll_compute_advantage` (:291), `compute_scalar_advantage` (:316), `inject_roll_advantages` (:237), `install_roll_adapter` (:354), `attach_roll_reward_fields` (:510), `_active_groups` (:496), `_response_mask` (:577).

### `src/rdan_grpo/roll_scalar.py` 199 -> 120

- Trim `build_scalar_output` (:69) internals to the reward-selection and normalization path.
- Merge `_method_weight` (:167) and `_method_weights` (:180) into one 8-line function.
- Keep `validate_groups` (:45) as the single group-shape check in the repo.

### `src/rdan_grpo/advantages.py` 141 -> 120

- No symbol deletions. Trim docstrings and the duplicate divisibility check at :137, which `validate_groups` already owns.
- This file is already at reference scale, it is 1.4x PAPO's 101-line equivalent for identical math.

### `src/rdan_grpo/judge.py` 519 -> 150

- `debug_canary` (:113), `preflight_snapshots` (:255), `calibration_plan` (:303), `select_reasoning_effort` (:314), `_percentile` (:503): the judge calibration and preflight subsystem, driven only by the deleted `scripts/calibrate_judge.py` (382 lines).
- `_generation_poll_contract` (:359), `_generation_data` (:372), `_selected_endpoint` (:400), `_validate_provenance` (:408), `_raw` (:512): OpenRouter generation-metadata provenance capture.
- `_dotted` (:445), `_field` (:488), `_sha256` (:498): support for the above.
- Keep `build_request` (:35), `OpenRouterJudge.judge` (:91), `_validate` (:190), `_validate_rows` (:430), `signed_process_score` (:455), `_invalid` (:461), `_zero` (:482), `TRANSPORT_RETRIES`.

### `src/rdan_grpo/roll_response_config.py` 399 -> 120

- `load_response_preflight_config` (:107), `_validate_preflight_payload` (:191), `_validate_preflight_config` (:362): the preflight config path, unused once preflight is gone.
- `_validate_payload` (:160), `_validate_recipe` (:225), `_validate_generation` (:246), `_validate_config` (:338), `_dotted_attr` (:394): 130 lines re-asserting a `FROZEN_PROFILE` the yaml already states and the fork already fails loudly on.
- `SCALAR_REWARD_WORKER_PATH`, `VLLM_RECEIPT_WORKER_PATH`, `SCALAR_DATA_PATH`, `UPDATES_PER_STEP`, `FROZEN_PROFILE`.
- `_verify_rtt` (:314): duplicate of `roll_compat._verify_rtt`.
- Keep `ResponseConfig`, `load_response_rlvr_config`, `_worker`, `_method_profile`, `_construct`, `_drop_cpu_device_mapping`.

### `src/rdan_grpo/roll_compat.py` 427 -> 180

- `_verify_rtt` (:269), `_run_git` (:288), `_roll_modules` (:297), `_preflight_roll_modules` (:304), `_module_spec_path` (:333), `_require_module_path` (:348), `_module_paths` (:408), `_is_pinned_package` (:412), `_same_python_tree` (:420): revision pinning of the fork checkout.
- `_install_mcore_patcher` (:354), `_reject_real_patcher` (:396), `_megatron_core_available` (:324): megatron is not used, the run is FSDP2 plus vLLM.
- `dump_batch_to_reward_system` (:196), `_validate_dense_text_batch` (:245), `_is_binary_2d_mask` (:263).
- Keep `install_vllm_sampling_seed_compat` (:32), `install_rtt_compat` (:71), `patch_torch_find_nd_overlapping_shards` (:89), `patch_torch_validate_global_plan` (:134), `_install_local_qwen_mask_patch` (:205), `_import_roll_modules` (:314).

### `src/rdan_grpo/wandb_tracking.py` 486 -> 120

- `verify_rtt_tracking` (:72), `deterministic_run_id` (:90), `canonical_config_sha256` (:98), `_validate_identity` (:297), `_validate_metadata` (:320), `_expected_names` (:364), `_full_hash` (:373), `_load_registry` (:465), `_git` (:476): run-identity pinning and the run registry.
- `_safe_run_dir` (:377), `_safe_artifact_path` (:386), `_safe_text` (:451), `_require_safe_name` (:455), `_safe_alias` (:460), `_normalize_runtime_paths` (:353), `_append_jsonl` (:401): artifact path hardening for the deleted artifact tree.
- Keep `register_wandb_tracker` (:58), `RdanWandbTracker` (:114), `redact_secrets` (:223), `_json_safe` (:411), `_secret_key` (:433), `_contains_secret` (:440).

### `scripts/run_response_train.py` 894 -> 60

Delete every symbol except `main`, `_parse_args` and the config load.
Specifically: `_LaunchPaths`, `_LaunchEvidence`, `_launch_evidence`, `_response_source_identity`, `_prepare_lifecycle`, `_stage_tracking`, `_validate_pipeline_config`, `_validate_method`, `_require_response_readiness`, `_require_current_launch_method`, `_validate_stage`, `_checkpoint_identity`, `_response_runtime_identity`, `_require_pilot_sequence`, `_require_recovery_sequence`, `_require_pilot_gate`, `_require_train_gate`, `_issue_lifecycle_outcome`, `_validate_stage_identities`, `_require_checkpoint`, `_runtime_parity`, `_vllm_runtime_parity`, `_model_identity`, `_regular_file`, `_real_directory`, `_prepare_directory`, `_checkpoint_path`, `_optional_regular_file`, `_snapshot`, `_bind_snapshot_environment`, `_bind_checkpoint_identity`, `_validate_snapshot_config`, `_json_object`, `_git_revision`, `_sha256`, `_print_outcome`.
Target shape is RTT's 36-line launcher: argparse, hydra compose, `install_rtt_compat`, `load_response_rlvr_config`, `build_response_training_pipeline(cfg).run()`.

### Whole-file deletions in `src/rdan_grpo` (14,000 lines)

`program.py` 3,876, `runtime_parity.py` 1,001, `response_pilot_lifecycle.py` 983, `metrics.py` 917, `roll_live.py` 721, `response_readiness.py` 701, `fsdp_hf_receipt.py` 645, `scalar_data.py` 640, `weight_receipt.py` 614, `roll_same_backend.py` 556, `response_publish.py` 482, `roll_same_backend_live.py` 470, `response_identity.py` 460, `routes.py` 426, `vllm_runtime_parity.py` 384, `roll_vllm_parity_live.py` 304, `evaluator_cert.py` 288, `roll_response_receipt.py` 256, `roll_fsdp_hf_receipt.py` 148, `roll_weight_receipt.py` 128.
Fold roughly 120 lines of `metrics.py` plotting into `scripts/plot_training.py` and discard the rest, which is a 300-line hand-rolled JSONL record validator.

### Whole-file deletions in `scripts` (5,926 lines)

`bootstrap_a100_response.py` 2,118, `certify_rubrichub_rules.py` 555, `certify_rubrichub_tokenizer.py` 463, `run_same_backend_parity.py` 430, `calibrate_judge.py` 382, `certify_hir_tokenizer.py` 358, `run_vllm_response_parity.py` 264, `run_roll_preflight.py` 253, `freeze_response_lifecycle.py` 192, `freeze_scalar_gate.py` 181, `run_roll_parity.py` 164, `capture_server_identity.py` 161, `publish_response_model.py` 103, `run_response_readiness.py` 70, `audit_hir.py` 67, `certify_evaluators.py` 45, `check_program.py` 43, `rescore_math_artifact.py` 39, `audit_routes.py` 38.

### Script merges

`prepare_a100_response_data.py` 485, `build_control_data.py` 104, `build_scalar_data.py` 88, `prepare_hir.py` 83, `build_merged_rl_data.py` 68, `fetch_hir.py` 63 collapse into `scripts/prepare_response_data.py` at 40 lines over `src/rdan_grpo/data_prep.py` at 300.
`rubrichub_data.py` 1,730, `control_data.py` 1,439 and `scalar_data.py` 640 collapse into that same `data_prep.py`.

### Config deletions

`qwen_rtt_papo_response_preflight.yaml`, `qwen_rtt_papo_response_parity.yaml`, `qwen_rtt_papo_response_vllm_parity.yaml`, `qwen_rl_csr_preflight.yaml`, `qwen_rl_aon_preflight.yaml`, `qwen_scalar_*.yaml` (5 files), plus every file under `configs/artifacts/` that only feeds a deleted certificate loader.

## What must NOT be cut

Each of these prevents a silent wrong result or is a hard requirement of the pinned fork.

- `_prime_optimizer_state` (`roll_response_workers.py:578`) and `_validate_restored_optimizer_state` (`:600`): the fork builds its DCP read plan from a never-stepped optimizer whose `state_dict()` is empty, so without priming a resume restores `param_groups` only and Adam silently restarts from zero moments with no error and no log line.
- `install_vllm_sampling_seed_compat` (`roll_compat.py:32`), `_base_seed` (`roll_response_workers.py:336`), `_generation_seed` (`:343`): the fork's `create_sampling_params_for_vllm` never passes `seed`, so rollouts are unreproducible and both engines emit identical samples, collapsing the 8-response group that ORM normalization depends on.
- `_rlvr_dataset_helpers` (`roll_response_pipeline.py:551`): the lazy RTT compat hook. Ray workers import this module to unpickle scheduler callables without running the hook, so hoisting the import to module scope kills every worker.
- `patch_torch_find_nd_overlapping_shards` (`roll_compat.py:89`) and `patch_torch_validate_global_plan` (`:134`): torch DCP planner patches required for FSDP2 checkpoint save and load to complete at all.
- `_bind_log_probs` (`roll_response_train.py:280`) and `_require_token_tensor` (`:373`): the fork's `loss_func` reads `ref_log_probs` unconditionally and `old_log_probs` whenever recompute is enabled, and with `enable_reference=False` nothing else populates either key. The shape check catches a vLLM/trainer sequence-length mismatch that would broadcast instead of erroring.
- The `final_response_mask` write in `_prepare_training_batch` (`roll_response_train.py:108`): the fork reads it unconditionally at `actor_worker.py:18`, and without it the loss silently falls back to the unshifted `response_mask`.
- `_valid_judgments` (`roll_reward.py:450`): asserts the judge returned exactly the expected rubric ids in order with scores in `SIGNED_PROCESS_SCORES`. Without it a truncated OpenRouter response resolves to 0.0 via a `.get` default and a judge outage is indistinguishable from a mediocre response inside the PRM channel.
- `_soft_rubric` id check (`roll_reward.py:503`): the reverse mapping `rubric_id = index + 1` at `:331` is only sound because of it. A drifted id lands judge scores on the wrong rubrics with no error.
- `_failed_local_info` (`roll_reward.py:439`) and the `eval_mask` it drives: separates "rubric genuinely failed" from "evaluator broke". This is what keeps PRM's correct-subset normalization honest.
- `_hard_mask` (`roll_reward.py:558`): the ORM/PRM router. A rubric that silently flips from hard to soft changes which advantage channel the objective reads.
- `_evaluate_rubrichub_hard` fail-closed raise (`roll_reward.py:677`): `rubrichub_rules` returns `RuleResult(valid=False)` rather than throwing, so without the raise an uncertified route scores -1 and looks like a failed response.
- `safe_rule.evaluate_rule` (`safe_rule.py:167-192`): AST validation, forked resource-limited child, 1.0s timeout kill. This is the actual code-execution sandbox and stays even though the hash allowlist above it goes.
- The `strict=True` zip in `_compute_rewards_impl` (`roll_reward.py:206`): prompt, response and rubric misalignment is a silent wrong result, not a crash.
- `count >= 2` and `std > eps` guards in `quality_advantages` (`advantages.py:76`): PAPO Eq 5 semantics, matching `core_algos.py:421-441`.
- `validate_groups` (`roll_scalar.py:45`): keep exactly one copy, on the advantage path, and delete the two duplicates.
- `_validate_training_counters` (`roll_response_checkpoint.py:567`): catches optimizer counters diverging across DP ranks, optimizer and scheduler out of phase, and optimizer steps not matching completed steps. None of these crash, all produce a run that trains and gives the wrong answer. Change the hardcoded `{"0","1"}` rank set to a length check.
- `create_checkpoint_stage` destination-exists check (`:120`), the atomic `os.rename` in `promote_checkpoint` (`:209`), and `_fsync_directory` (`:722`): POSIX rename durability requires the parent fsync, without it a crash right after promotion loses a checkpoint that was reported complete.
- `CheckpointIdentity.method` and `.planned_horizon`: three methods write into the same checkpoint root from the same base snapshot, so resuming an `rl_aon` checkpoint into an `rtt_papo_response` run is a live hazard, and a drifted horizon silently changes the LR schedule.
- `_restore_checkpoint` (`roll_response_pipeline.py:366`) driver and vLLM RNG restore: without it the sampling stream silently diverges on resume.
- The `offload_states`/`load_states` bracket in `_generate` (`roll_response_pipeline.py:282`) and its `finally`: on 2x A100 80GB the FSDP2 shards and the vLLM engine otherwise hold GPU memory simultaneously and OOM.
- `_restore_rubrics` (`roll_response_pipeline.py:573`): Arrow cannot hold heterogeneous rubric dicts, so rubrics are stored as JSON strings and this `with_transform` is what makes them readable at reward time.
- `BasePipeline.model_update_groups = []` and `checkpoint_clusters = []` in `__init__`: the fork declares both as mutable class attributes that leak across pipeline instances.
- `set_model_update_pair` in `_initialize_workers`: the only supported FSDP2 to vLLM weight-sync path in the fork.
- `_validate_finite_metrics` (`roll_response_pipeline.py:451`), `_train_metrics` and `_finite_metric` (`roll_response_train.py:473`, `:480`): a NaN loss that is merely logged is a silently wrong run, and the fork skips `optimizer.step()` on non-finite grad while still returning normally, so assert a finite nonzero `actor_train/grad_norm`.
- `_prune_checkpoints` (`roll_response_pipeline.py:475`) including its resume-path exemption: multi-GB DCP checkpoints exhaust disk mid-run, and pruning without the exemption can delete the only checkpoint you can restart from.

## Execution order

1. Delete the 19 standalone scripts listed above (5,926 lines) and their tests. Nothing in `src/rdan_grpo` imports any of them, so imports and the remaining suite stay clean. Verified by the import graph: `scripts/run_response_train.py` is the sole importer of `program`, `response_identity`, `response_pilot_lifecycle`, `response_readiness`, `runtime_parity` and `vllm_runtime_parity`.
2. Rewrite `scripts/run_response_train.py` to 60 lines and strip `roll_bridge.py` of the certificate subsystem and `roll_response_config.py` of the preflight path. Do these together, because the launcher is the only consumer of `require_train_certificate` and `load_response_preflight_config`. Drop the `PreflightCertificate` parameter from `run_response_train_step` and `ResponseTrainingPipeline.__init__` in the same commit.
3. Delete the 20 `src/rdan_grpo` modules listed above (14,000 lines) and the ~9,000 lines of tests that exercise only them. After step 2 nothing outside the deleted set imports them.
4. **First launchable point.** At the end of step 3 the tree runs `python scripts/run_response_train.py --config configs/roll/qwen_rtt_papo_response_train.yaml` to a real optimizer update on 2x A100 with no certificate, no readiness receipt, no parity artifact and no launch gate. Do a 3-step smoke run here and archive the W&B run before continuing, so every later step has a known-good baseline to diff against.
5. Strip `roll_response_workers.py` to 170 lines. Remove the optimizer-counter apparatus, clip recompute, grad-finiteness duplicate, receipt RPCs, RNG progress and CUDA telemetry. Land `_train_metrics`' finite-`grad_norm` assertion in `roll_response_train.py` in the same commit, since it is the replacement guarantee.
6. Strip `roll_response_train.py` to 110 lines and `roll_response_pipeline.py` to 300. The pipeline depends on the train result shape, so this order avoids a broken intermediate.
7. Strip `roll_response_checkpoint.py` to 130 lines. Run a save, kill, resume cycle and confirm `_validate_restored_optimizer_state` passes and the loss curve continues rather than restarting.
8. Strip `roll_reward.py` to 360 lines and `judge.py` to 150. Make `allowed_hashes` optional at `safe_rule.py:167` in the same commit. Re-score one frozen batch before and after and require byte-identical rewards.
9. Merge `rubrichub_data.py`, `control_data.py` and `scalar_data.py` into `data_prep.py` at 300 lines, regenerate `data/qwen_hir_rubrichub_if_hybrid.jsonl` and require it to hash identically to the frozen file.
10. Fold the useful 120 lines of `metrics.py` into `scripts/plot_training.py`, delete `metrics.py`, shrink `wandb_tracking.py` to 120.
11. Rewrite the test suite to a 1,500-line budget: one end-to-end test per surviving module, plus a dedicated test for each item in "What must NOT be cut". Delete the 6 largest test files, which target deleted subsystems.
12. Final verification: ruff, mypy, full pytest, then a 20-step training run with a mid-run kill and resume, comparing the resumed loss and reward curves against the step 4 baseline.