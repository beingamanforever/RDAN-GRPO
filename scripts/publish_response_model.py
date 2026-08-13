#!/usr/bin/env python3
"""Publish one validated final response checkpoint to its method revision."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdan_grpo.response_publish import PublishFile, load_publish_identity, publish_response_model

HUGGINGFACE_HUB_VERSION = "0.36.2"
_COMMIT = re.compile(r"[0-9a-f]{40}")


class HubUploader:
    """Upload an explicit file inventory through the pinned Hub client."""

    def __init__(self, token: str) -> None:
        import huggingface_hub

        if huggingface_hub.__version__ != HUGGINGFACE_HUB_VERSION:
            raise RuntimeError(f"huggingface-hub must be exactly {HUGGINGFACE_HUB_VERSION}")
        self._api = huggingface_hub.HfApi(token=token)
        self._add = huggingface_hub.CommitOperationAdd
        self._delete = huggingface_hub.CommitOperationDelete
        self._token = token

    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        method: str,
        completed_step: int,
        files: tuple[PublishFile, ...],
    ) -> str:
        """Create the method branch if needed and commit only validated files."""

        self._api.create_branch(
            repo_id,
            branch=revision,
            repo_type="model",
            token=self._token,
            exist_ok=True,
        )
        parent = self._api.model_info(repo_id, revision=revision, token=self._token).sha
        if not isinstance(parent, str) or not _COMMIT.fullmatch(parent):
            raise RuntimeError("Hub method revision returned an invalid parent commit")
        current = {file.path_in_repo for file in files}
        remote = set(self._api.list_repo_files(repo_id, revision=parent, repo_type="model", token=self._token))
        operations = [self._delete(path_in_repo=path) for path in sorted(remote - current)]
        operations.extend(self._add(path_in_repo=file.path_in_repo, path_or_fileobj=file.local_path) for file in files)
        result = self._api.create_commit(
            repo_id,
            operations=operations,
            commit_message=f"Publish {method} final step {completed_step}",
            repo_type="model",
            revision=revision,
            parent_commit=parent,
            token=self._token,
        )
        return result.oid


def main() -> int:
    """Load exact local identity, publish the final checkpoint, and print its receipt."""

    args = _parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("HF_TOKEN must be set in the environment")
    identity = load_publish_identity(args.identity)
    receipt = publish_response_model(
        args.checkpoint,
        identity=identity,
        receipt_path=args.receipt,
        uploader=HubUploader(token),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
