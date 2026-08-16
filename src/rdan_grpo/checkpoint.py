"""Checkpoint directory layout, atomic promotion, resume lookup, and retention.

A checkpoint is staged under ``.staging-step-NNNNNN`` and renamed into place only once every
artifact is written, so a crash mid-save never leaves a directory that resume would trust.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STATE_FILE = "rdan_state.json"
STAGING_PREFIX = ".staging-step-"
STEP_PREFIX = "step-"
STEP_DIGITS = 6


def stage_checkpoint(root: str | Path, step: int) -> Path:
    """Create an empty staging directory for one step, replacing any abandoned attempt."""

    staging = Path(root) / f"{STAGING_PREFIX}{step:0{STEP_DIGITS}d}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def promote_checkpoint(staging: Path, state: Mapping[str, Any]) -> Path:
    """Write the state file, flush the tree, and rename the staging directory into place."""

    step = int(state["completed_step"])
    target = staging.parent / f"{STEP_PREFIX}{step:0{STEP_DIGITS}d}"
    write_json(staging / STATE_FILE, state)
    _fsync_tree(staging)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    _fsync_directory(target.parent)
    return target


def read_state(checkpoint: str | Path) -> dict[str, Any]:
    """Read one checkpoint's state file."""

    return json.loads((Path(checkpoint) / STATE_FILE).read_text(encoding="utf-8"))


def latest_checkpoint(root: str | Path) -> Path | None:
    """Return the most recent promoted checkpoint, or None when there is none."""

    steps = promoted_checkpoints(root)
    return steps[-1] if steps else None


def promoted_checkpoints(root: str | Path) -> list[Path]:
    """Return promoted step directories, oldest first."""

    path = Path(root)
    if not path.is_dir():
        return []
    return sorted(
        (
            entry
            for entry in path.iterdir()
            if entry.is_dir()
            and not entry.is_symlink()
            and entry.name.startswith(STEP_PREFIX)
            and entry.name[len(STEP_PREFIX) :].isdigit()
        ),
        key=lambda entry: entry.name,
    )


def prune_checkpoints(root: str | Path, keep_recent: int, keep_every: int) -> list[Path]:
    """Delete checkpoints outside the retention window, keeping periodic milestones.

    ``keep_recent`` guarantees resume always has a recent checkpoint; ``keep_every`` keeps
    milestone steps permanently so intermediate weights survive for evaluation. Milestones
    outside the resume window shed their sharded optimizer state, which is an order of
    magnitude larger than the weights and is useless once the run has moved past them.
    """

    promoted = promoted_checkpoints(root)
    recent = set(promoted[-keep_recent:] if keep_recent > 0 else [])
    milestones = {entry for entry in promoted if keep_every > 0 and _step_of(entry) % keep_every == 0}
    removed = [entry for entry in promoted if entry not in recent | milestones]
    for entry in removed:
        shutil.rmtree(entry)
    for entry in milestones - recent:
        optimizer_state = entry / "actor" / "dcp"
        if optimizer_state.is_dir():
            shutil.rmtree(optimizer_state)
    for entry in Path(root).iterdir():
        if entry.is_dir() and not entry.is_symlink() and entry.name.startswith(STAGING_PREFIX):
            shutil.rmtree(entry)
    return removed


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to a new file, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _step_of(entry: Path) -> int:
    return int(entry.name[len(STEP_PREFIX) :])


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
