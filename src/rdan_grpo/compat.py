"""Compatibility repairs for the pinned RTT ROLL release."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import inspect
import logging
import math
import sys
from bisect import bisect_right, insort
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"

_PATCHER_MODULE = "mcore_adapter.patcher"
_PATCHER_OWNER = "rdan-grpo:rtt-b1ab2fb"
_MASK_PATCH_OWNER = "rdan-grpo:local-qwen3-mask"
_VLLM_SEED_PATCH_OWNER = "rdan-grpo:vllm-sampling-seed"


def install_vllm_sampling_seed_compat() -> None:
    """Add the request seed omitted by the pinned ROLL vLLM adapter."""

    from roll.distributed.strategy import vllm_strategy

    original = vllm_strategy.create_sampling_params_for_vllm
    if getattr(original, "__rdan_compat_owner__", None) == _VLLM_SEED_PATCH_OWNER:
        return

    def create_sampling_params_for_vllm(gen_kwargs: Mapping[str, Any]) -> Any:
        seed = gen_kwargs.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise RuntimeError("vLLM generation requires a nonnegative deterministic seed")
        output_kind = gen_kwargs.get("output_kind", vllm_strategy.RequestOutputKind.FINAL_ONLY)
        if output_kind != vllm_strategy.RequestOutputKind.FINAL_ONLY:
            if gen_kwargs["num_return_sequences"] != 1:
                raise RuntimeError("partial vLLM output requires one return sequence")
        return vllm_strategy.SamplingParams(
            max_tokens=gen_kwargs["max_new_tokens"],
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            top_k=gen_kwargs["top_k"],
            stop_token_ids=gen_kwargs["eos_token_id"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            n=gen_kwargs["num_return_sequences"],
            stop=gen_kwargs["stop_strings"],
            logprobs=gen_kwargs.get("logprobs", 0),
            output_kind=output_kind,
            include_stop_str_in_output=gen_kwargs.get("include_stop_str_in_output", True),
            seed=seed,
        )

    create_sampling_params_for_vllm.__rdan_compat_owner__ = _VLLM_SEED_PATCH_OWNER
    vllm_strategy.create_sampling_params_for_vllm = create_sampling_params_for_vllm


def install_rtt_compat(rtt_root: str | Path) -> None:
    """Install byte-gated repairs for the pinned RTT checkout."""

    root = Path(rtt_root).resolve()
    _preflight_roll_modules(root)
    _install_mcore_patcher(root)
    megatron_available = _megatron_core_available()
    modules = _import_roll_modules(root, include_megatron=megatron_available)
    utils = modules[0]

    if megatron_available:
        _install_local_qwen_mask_patch()

    if not hasattr(utils, "dump_batch_to_reward_system"):
        utils.dump_batch_to_reward_system = dump_batch_to_reward_system


def patch_torch_find_nd_overlapping_shards() -> None:
    """Install Alibaba ROLL's sweep-line overlap check for sharded tensors."""

    import torch
    from torch.distributed._shard.metadata import ShardMetadata
    from torch.distributed._shard.sharding_spec._internals import _check_shard_metadata_pair_overlap

    def _find_nd_overlapping_shards(shards: list[ShardMetadata], sharded_dims: list[int]) -> tuple[int, int] | None:
        if len(shards) <= 1 or not sharded_dims:
            return None

        sweep_index = 0
        if len(sharded_dims) > 1:
            max_size = 0
            for index, dim in enumerate(sharded_dims):
                dim_size = shards[0].shard_offsets[dim] + shards[0].shard_sizes[dim]
                if dim_size > max_size:
                    max_size = dim_size
                    sweep_index = index
        sweep_dim = sharded_dims[sweep_index]
        sorted_indices = sorted(
            range(len(shards)),
            key=lambda index: (
                shards[index].shard_offsets[sweep_dim],
                *(shards[index].shard_offsets[dim] for dim in sharded_dims if dim != sweep_dim),
            ),
        )
        active: list[tuple[int, int]] = []

        for index in sorted_indices:
            current = shards[index]
            start = current.shard_offsets[sweep_dim]
            end = start + current.shard_sizes[sweep_dim]
            cutoff = bisect_right(active, (start, sys.maxsize))
            if cutoff:
                del active[:cutoff]
            for _, other_index in active:
                if _check_shard_metadata_pair_overlap(current, shards[other_index]):
                    return other_index, index
            insort(active, (end, index))
        return None

    torch.distributed._shard.sharding_spec._internals._find_nd_overlapping_shards = _find_nd_overlapping_shards


def patch_torch_validate_global_plan() -> None:
    """Install Alibaba ROLL's sweep-line distributed checkpoint validation."""

    import torch
    from torch.distributed.checkpoint.default_planner import _check_box_bounds, _check_box_overlap
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata
    from torch.distributed.checkpoint.planner import SavePlan

    logger = logging.getLogger(_PATCHER_MODULE)

    def _validate_global_plan(global_plan: list[SavePlan], metadata: Metadata) -> bool:
        all_good = True
        for key, value in metadata.state_dict_metadata.items():
            if isinstance(value, BytesStorageMetadata) or len(value.size) == 0:
                continue
            chunks = value.chunks
            chunks_volume = 0
            for chunk in chunks:
                if not _check_box_bounds(value.size, chunk):
                    logger.warning("key:%s has out of bounds chunk: tensor-size:%s chunk:%s", key, value.size, chunk)
                    all_good = False
                chunks_volume += math.prod(chunk.sizes)

            if len(chunks) > 1:
                dims = len(value.size)
                sweep_dim = 0
                sorted_indices = sorted(
                    range(len(chunks)),
                    key=lambda index: (
                        chunks[index].offsets[sweep_dim],
                        *(chunks[index].offsets[dim] for dim in range(dims)),
                    ),
                )
                active: list[tuple[int, int]] = []
                for index in sorted_indices:
                    current = chunks[index]
                    start = current.offsets[sweep_dim]
                    end = start + current.sizes[sweep_dim]
                    cutoff = bisect_right(active, (start, sys.maxsize))
                    if cutoff:
                        del active[:cutoff]
                    for _, other_index in active:
                        other = chunks[other_index]
                        if _check_box_overlap(current, other):
                            logger.warning("key:%s has overlapping chunks: %s %s", key, current, other)
                            all_good = False
                    insort(active, (end, index))

            tensor_volume = math.prod(value.size)
            if len(global_plan) > 1 and chunks_volume != tensor_volume:
                logger.warning(
                    "key:%s invalid fill tensor-volume:%s chunks-volume:%s",
                    key,
                    tensor_volume,
                    chunks_volume,
                )
                all_good = False
        return all_good

    torch.distributed.checkpoint.default_planner._validate_global_plan = _validate_global_plan


def dump_batch_to_reward_system(batch: Any, tokenizer: Any) -> None:
    """Permit RTT's absent hook only when reward-system logging is disabled."""

    del tokenizer
    config = getattr(batch, "meta_info", {}).get("reward_system_config")
    if config:
        raise RuntimeError("pinned RTT reward-system logging hook is missing and cannot be emulated")


def _install_local_qwen_mask_patch() -> None:
    strategy_module = importlib.import_module("roll.distributed.strategy.megatron_strategy")
    strategy = strategy_module.MegatronInferStrategy
    original = strategy.inner_forward_step
    if getattr(original, "__rdan_compat_owner__", None) == _MASK_PATCH_OWNER:
        return
    if getattr(original, "__rdan_compat_owner__", None) is not None:
        raise RuntimeError("conflicting MegatronInferStrategy.inner_forward_step compatibility patch")

    def inner_forward_step(self: Any, loss_func: Any, data_iterator: Any, model: Any) -> Any:
        args = getattr(self, "megatron_train_args", None)
        config = getattr(model, "config", None)
        is_local_qwen = (
            getattr(args, "transformer_impl", None) == "local"
            and getattr(config, "transformer_impl", None) == "local"
            and getattr(config, "hf_model_type", None) == "qwen3"
            and not getattr(config, "num_moe_experts", None)
        )
        if not is_local_qwen or getattr(self, "use_sequence_packing", False):
            return original(self, loss_func, data_iterator, model)

        data = next(data_iterator)
        non_tensor_batch = getattr(data, "non_tensor_batch", None) or {}
        if "multi_modal_inputs" in non_tensor_batch:
            return original(self, loss_func, iter((data,)), model)
        _validate_dense_text_batch(data)

        def causal_model(*model_args: Any, **model_kwargs: Any) -> Any:
            attention_mask = model_kwargs.get("attention_mask")
            if not _is_binary_2d_mask(attention_mask):
                raise RuntimeError("pinned Megatron strategy did not forward a binary 2D attention mask")
            model_kwargs["attention_mask"] = None
            return model(*model_args, **model_kwargs)

        return original(self, loss_func, iter((data,)), causal_model)

    inner_forward_step.__rdan_compat_owner__ = _MASK_PATCH_OWNER
    strategy.inner_forward_step = inner_forward_step


def _validate_dense_text_batch(data: Any) -> None:
    import torch

    batch = getattr(data, "batch", None)
    if batch is None or not hasattr(batch, "get"):
        raise RuntimeError("local Qwen3 compatibility requires a tensor batch")
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise RuntimeError("local Qwen3 compatibility requires 2D input_ids")
    if not _is_binary_2d_mask(attention_mask) or attention_mask.shape != input_ids.shape:
        raise RuntimeError("local Qwen3 compatibility requires a shape-aligned binary 2D attention mask")
    if attention_mask.shape[0] == 0 or attention_mask.shape[1] == 0 or not bool(attention_mask.any(dim=1).all()):
        raise RuntimeError("local Qwen3 compatibility requires at least one valid token per row")
    if attention_mask.shape[1] > 1 and bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
        raise RuntimeError("local Qwen3 compatibility requires right-padded attention masks")


def _is_binary_2d_mask(value: Any) -> bool:
    import torch

    return isinstance(value, torch.Tensor) and value.ndim == 2 and bool(torch.logical_or(value == 0, value == 1).all())




def _roll_modules(root: Path) -> tuple[tuple[str, Path], ...]:
    return (
        ("roll.pipeline.rlvr.utils", root / "roll/pipeline/rlvr/utils.py"),
        ("roll.distributed.strategy.megatron_strategy", root / "roll/distributed/strategy/megatron_strategy.py"),
    )


def _preflight_roll_modules(root: Path) -> None:
    for name, path in _roll_modules(root):
        module = sys.modules.get(name)
        if module is not None:
            _require_module_path(name, module, path)
        spec_path = _module_spec_path(name)
        if spec_path != path.resolve():
            raise RuntimeError(f"resolved {name} from an unexpected path: {spec_path}")


def _import_roll_modules(root: Path, *, include_megatron: bool) -> tuple[ModuleType, ...]:
    modules = []
    module_specs = _roll_modules(root) if include_megatron else _roll_modules(root)[:1]
    for name, path in module_specs:
        module = importlib.import_module(name)
        _require_module_path(name, module, path)
        modules.append(module)
    return tuple(modules)


def _megatron_core_available() -> bool:
    if "megatron.core" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("megatron.core") is not None
    except ModuleNotFoundError:
        return False


def _module_spec_path(name: str) -> Path:
    search: list[str] | None = None
    current = ""
    spec = None
    for part in name.split("."):
        current = part if not current else f"{current}.{part}"
        spec = importlib.machinery.PathFinder.find_spec(current, search)
        if spec is None:
            raise RuntimeError(f"cannot resolve pinned module: {name}")
        search = list(spec.submodule_search_locations or ())
    if spec is None or not spec.origin:
        raise RuntimeError(f"cannot resolve pinned module source: {name}")
    return Path(spec.origin).resolve()


def _require_module_path(name: str, module: ModuleType, path: Path) -> None:
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != path.resolve():
        raise RuntimeError(f"loaded {name} from an unexpected path: {module_path}")


def _install_mcore_patcher(root: Path) -> None:
    source = root / "mcore_adapter" / "src"
    package = source / "mcore_adapter"
    if not (package / "__init__.py").is_file():
        raise RuntimeError(f"pinned RTT mcore_adapter source is missing: {source}")

    existing = sys.modules.get(_PATCHER_MODULE)
    if existing is not None:
        if (
            getattr(existing, "__rdan_compat_owner__", None) == _PATCHER_OWNER
            and getattr(existing, "patch_torch_find_nd_overlapping_shards", None)
            is patch_torch_find_nd_overlapping_shards
            and getattr(existing, "patch_torch_validate_global_plan", None) is patch_torch_validate_global_plan
        ):
            return
        raise RuntimeError(f"conflicting module is already loaded: {_PATCHER_MODULE}")

    parent = sys.modules.get("mcore_adapter")
    if parent is not None and not _is_pinned_package(parent, package):
        raise RuntimeError("a conflicting mcore_adapter package is already loaded")
    if parent is not None and hasattr(parent, "patcher"):
        raise RuntimeError("a conflicting mcore_adapter.patcher attribute is already loaded")
    _reject_real_patcher()

    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    patcher = ModuleType(_PATCHER_MODULE)
    patcher.__dict__.update(
        {
            "__doc__": "Pinned compatibility functions sourced from Alibaba ROLL.",
            "__package__": "mcore_adapter",
            "__rdan_compat_owner__": _PATCHER_OWNER,
            "patch_torch_find_nd_overlapping_shards": patch_torch_find_nd_overlapping_shards,
            "patch_torch_validate_global_plan": patch_torch_validate_global_plan,
        }
    )
    sys.modules[_PATCHER_MODULE] = patcher
    if parent is not None:
        parent.patcher = patcher


def _reject_real_patcher() -> None:
    parent_spec = importlib.machinery.PathFinder.find_spec("mcore_adapter", sys.path)
    if parent_spec is None:
        return
    locations = parent_spec.submodule_search_locations or []
    for location in locations:
        package = Path(location).resolve()
        patcher_spec = importlib.machinery.PathFinder.find_spec(_PATCHER_MODULE, [str(package)])
        if patcher_spec is not None:
            raise RuntimeError(f"conflicting real module exists: {patcher_spec.origin}")


def _module_paths(module: ModuleType) -> set[Path]:
    return {Path(path).resolve() for path in getattr(module, "__path__", ())}


def _is_pinned_package(module: ModuleType, package: Path) -> bool:
    paths = _module_paths(module)
    if not paths:
        return False
    pinned = package.resolve()
    return all(path == pinned or _same_python_tree(path, pinned) for path in paths)


def _same_python_tree(left: Path, right: Path) -> bool:
    if not left.is_dir() or not right.is_dir():
        return False
    left_files = {path.relative_to(left) for path in left.rglob("*.py")}
    right_files = {path.relative_to(right) for path in right.rglob("*.py")}
    return left_files == right_files and all(
        (left / path).read_bytes() == (right / path).read_bytes() for path in left_files
    )
