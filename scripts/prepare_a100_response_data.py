#!/usr/bin/env python3
"""Reconstruct and verify every ignored response-training data artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdan_grpo.response_identity import response_data_identity  # noqa: E402

RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
DATA_PYTHON = "3.11.15"
DATA_PACKAGES = {
    "absl-py": "2.5.0",
    "fasttext-wheel": "0.9.2",
    "huggingface-hub": "0.36.2",
    "immutabledict": "4.3.1",
    "jinja2": "3.1.6",
    "langdetect": "1.0.9",
    "markupsafe": "3.0.3",
    "nltk": "3.10.0",
    "numpy": "1.26.4",
    "pyarrow": "25.0.1",
    "transformers": "4.57.0",
}


class PreparationError(RuntimeError):
    """Raised when a training data artifact cannot be reproduced exactly."""


def main() -> int:
    """Prepare missing ignored artifacts or verify an already complete corpus."""

    args = _parse_args()
    config = _json(ROOT / "configs/data/rubrichub_instruction_following.json")
    manifest_path = ROOT / "configs/artifacts/qwen_merged_rl_data_manifest.json"
    expected_manifest = manifest_path.read_bytes()
    manifest = _json(manifest_path)
    _verify_rtt(args.rtt_root)
    _verify_data_python(args.data_python)

    if not args.check:
        _prepare_hir(args.data_python, args.snapshot)
        _prepare_rubrichub_source(config)
        _prepare_language(args.data_python, config, manifest)
        _prepare_rule_evidence(args.data_python, args.rtt_root)
        _prepare_tokenizer_evidence(args.data_python, args.snapshot)
        _prepare_merge(args.data_python, manifest, manifest_path, expected_manifest)

    _verify_all(args, manifest_path)
    identity = response_data_identity(ROOT / "configs/program/qwen_first.json")
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest_sha256": identity["manifest_sha256"],
                "output_sha256": identity["output_sha256"],
                "records": identity["records"],
                "checked_only": args.check,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-python", type=Path, required=True)
    parser.add_argument("--rtt-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _prepare_hir(python: Path, snapshot: Path) -> None:
    hir = _json(ROOT / "configs/data/hir.json")
    source = hir["source"]
    raw = ROOT / "data/HIR_trainv1.jsonl"
    _download(source["url"], raw, source["bytes"], source["sha256"])
    _run(python, "scripts/prepare_hir.py")
    certificate = ROOT / "configs/artifacts/hir_qwen_tokenizer_certificate.json"
    evidence_ref = _json(certificate)["evidence"]
    evidence = ROOT / evidence_ref["path"]
    if _matches(evidence, evidence_ref["bytes"], evidence_ref["sha256"]):
        _run(
            python,
            "scripts/certify_hir_tokenizer.py",
            "--check",
            "--model-path",
            str(snapshot),
        )
    elif evidence.exists() or evidence.is_symlink():
        raise PreparationError("existing HIR tokenizer evidence differs from its certificate")
    else:
        with tempfile.TemporaryDirectory(prefix="rdan-hir-tokenizer-") as directory:
            candidate = Path(directory) / certificate.name
            _run(
                python,
                "scripts/certify_hir_tokenizer.py",
                "--model-path",
                str(snapshot),
                "--certificate",
                str(candidate),
            )
            _same_bytes(candidate, certificate, "HIR tokenizer certificate")
    builds = (
        ("authoritative", "hir_scalar_certified_manifest.json"),
        ("rtt-full", "qwen_rtt_hir_data_manifest.json"),
    )
    with tempfile.TemporaryDirectory(prefix="rdan-hir-manifests-") as directory:
        for scope, name in builds:
            candidate = Path(directory) / name
            command = [
                "scripts/build_scalar_data.py",
                "--scope",
                scope,
                "--manifest",
                str(candidate),
            ]
            if scope == "authoritative":
                command.extend(("--output", "data/HIR_trainv1_rdan_scalar_certified.jsonl"))
            else:
                command.extend(("--output", "data/HIR_trainv1_rtt_qwen.jsonl"))
            _run(Path(sys.executable), *command)
            _same_bytes(candidate, ROOT / "configs/artifacts" / name, f"HIR {scope} manifest")


def _prepare_rubrichub_source(config: dict[str, Any]) -> None:
    source = config["source"]
    url = f"{source['url']}/resolve/{source['revision']}/{source['file']}?download=true"
    _download(url, ROOT / source["path"], source["bytes"], source["sha256"])
    detector = config["language_gate"]["detector"]
    _download(
        detector["model_url"],
        ROOT / "data/rubrichub-source/language-id/lid.176.bin",
        detector["model_bytes"],
        detector["model_sha256"],
    )
    for benchmark in config["benchmarks"]:
        if benchmark["id"] != "AdvancedIF":
            continue
        url = (
            f"https://huggingface.co/datasets/{benchmark['dataset']}/resolve/"
            f"{benchmark['dataset_revision']}/if_oss_full_data.csv?download=true"
        )
        _download(url, ROOT / benchmark["path"], benchmark["bytes"], benchmark["sha256"])


def _prepare_language(python: Path, config: dict[str, Any], manifest: dict[str, Any]) -> None:
    output = ROOT / config["language_certificate"]
    expected = manifest["language_gate"]["certificate"]
    if _matches(output, expected.get("bytes"), expected["sha256"]):
        return
    if output.exists() or output.is_symlink():
        raise PreparationError("existing language certificate differs from the frozen manifest")
    _run(
        python,
        "scripts/build_merged_rl_data.py",
        "language-certificate",
        "--model",
        "data/rubrichub-source/language-id/lid.176.bin",
        "--output",
        config["language_certificate"],
    )
    _verify_file(output, expected.get("bytes"), expected["sha256"], "language certificate")


def _prepare_rule_evidence(python: Path, rtt_root: Path) -> None:
    certificate = ROOT / "configs/artifacts/rubrichub_rule_certificate.json"
    expected = _json(certificate)["evidence"]
    evidence = ROOT / expected["path"]
    if _matches(evidence, expected["bytes"], expected["sha256"]):
        _run(python, "scripts/certify_rubrichub_rules.py", "--check", "--rtt-root", str(rtt_root))
        return
    if evidence.exists() or evidence.is_symlink():
        raise PreparationError("existing RubricHub rule evidence differs from its certificate")
    with tempfile.TemporaryDirectory(prefix="rdan-rule-certificate-") as directory:
        candidate = Path(directory) / certificate.name
        _run(
            python,
            "scripts/certify_rubrichub_rules.py",
            "--evidence",
            str(evidence),
            "--certificate",
            str(candidate),
            "--rtt-root",
            str(rtt_root),
        )
        _same_bytes(candidate, certificate, "RubricHub rule certificate")


def _prepare_tokenizer_evidence(python: Path, snapshot: Path) -> None:
    certificate = ROOT / "configs/artifacts/rubrichub_tokenizer_certificate.json"
    expected = _json(certificate)["evidence"]
    evidence = ROOT / expected["path"]
    if _matches(evidence, expected["bytes"], expected["sha256"]):
        _run(
            python,
            "scripts/certify_rubrichub_tokenizer.py",
            "--check",
            "--model-path",
            str(snapshot),
        )
        return
    if evidence.exists() or evidence.is_symlink():
        raise PreparationError("existing RubricHub tokenizer evidence differs from its certificate")
    with tempfile.TemporaryDirectory(prefix="rdan-token-certificate-") as directory:
        candidate = Path(directory) / certificate.name
        _run(
            python,
            "scripts/certify_rubrichub_tokenizer.py",
            "--model-path",
            str(snapshot),
            "--evidence",
            str(evidence),
            "--certificate",
            str(candidate),
        )
        _same_bytes(candidate, certificate, "RubricHub tokenizer certificate")


def _prepare_merge(python: Path, manifest: dict[str, Any], path: Path, expected: bytes) -> None:
    outputs = [ROOT / value["path"] for value in manifest["outputs"].values()]
    references = manifest["outputs"].values()
    complete = all(
        _matches(target, value["bytes"], value["sha256"]) for target, value in zip(outputs, references, strict=True)
    )
    if complete:
        return
    if any(target.exists() or target.is_symlink() for target in outputs):
        raise PreparationError("partial or changed merged response outputs require an empty data target")
    try:
        _run(python, "scripts/build_merged_rl_data.py", "build")
        if path.read_bytes() != expected:
            raise PreparationError("reconstructed response manifest differs from committed bytes")
    except BaseException:
        path.write_bytes(expected)
        for output in outputs:
            output.unlink(missing_ok=True)
        raise


def _verify_all(args: argparse.Namespace, manifest_path: Path) -> None:
    manifest = _json(manifest_path)
    hir_config = _json(ROOT / "configs/data/hir.json")
    hir_manifest = _json(ROOT / "configs/artifacts/hir_scalar_certified_manifest.json")
    rtt_hir_manifest = _json(ROOT / "configs/artifacts/qwen_rtt_hir_data_manifest.json")
    rubrichub_config = _json(ROOT / "configs/data/rubrichub_instruction_following.json")

    hir_source = hir_config["source"]
    _verify_file(
        ROOT / "data" / hir_source["path"],
        hir_source["bytes"],
        hir_source["sha256"],
        "HIR source",
    )
    for label in ("rtt_processed",):
        reference = hir_manifest["sources"][label]
        _verify_file(
            ROOT / reference["path"],
            reference.get("bytes"),
            reference["sha256"],
            f"HIR {label}",
        )
    derived = hir_manifest["derived"]
    _verify_file(
        ROOT / derived["path"],
        derived.get("bytes"),
        derived["sha256"],
        "HIR certified data",
    )
    rtt_derived = rtt_hir_manifest["derived"]
    _verify_file(
        ROOT / rtt_derived["path"],
        rtt_derived.get("bytes"),
        rtt_derived["sha256"],
        "full RTT-compatible HIR data",
    )
    hir_tokenizer = _json(ROOT / "configs/artifacts/hir_qwen_tokenizer_certificate.json")["evidence"]
    _verify_file(
        ROOT / hir_tokenizer["path"],
        hir_tokenizer["bytes"],
        hir_tokenizer["sha256"],
        "HIR tokenizer evidence",
    )

    rubrichub_source = rubrichub_config["source"]
    _verify_file(
        ROOT / rubrichub_source["path"],
        rubrichub_source["bytes"],
        rubrichub_source["sha256"],
        "RubricHub source",
    )
    detector = rubrichub_config["language_gate"]["detector"]
    _verify_file(
        ROOT / "data/rubrichub-source/language-id/lid.176.bin",
        detector["model_bytes"],
        detector["model_sha256"],
        "language detector",
    )
    for label, reference in manifest["outputs"].items():
        _verify_file(ROOT / reference["path"], reference["bytes"], reference["sha256"], label)
    language = manifest["language_gate"]["certificate"]
    _verify_file(
        ROOT / language["path"],
        language.get("bytes"),
        language["sha256"],
        "language certificate",
    )
    for label, certificate_path in (
        ("RubricHub rule evidence", ROOT / "configs/artifacts/rubrichub_rule_certificate.json"),
        ("RubricHub tokenizer evidence", ROOT / "configs/artifacts/rubrichub_tokenizer_certificate.json"),
    ):
        evidence = _json(certificate_path)["evidence"]
        _verify_file(ROOT / evidence["path"], evidence["bytes"], evidence["sha256"], label)
    _run(args.data_python, "scripts/certify_rubrichub_rules.py", "--check", "--rtt-root", str(args.rtt_root))
    _run(
        args.data_python,
        "scripts/certify_hir_tokenizer.py",
        "--check",
        "--model-path",
        str(args.snapshot),
    )
    _run(
        args.data_python,
        "scripts/certify_rubrichub_tokenizer.py",
        "--check",
        "--model-path",
        str(args.snapshot),
    )
    try:
        import datasets

        dataset = datasets.load_dataset(
            "json",
            data_files=[str(ROOT / manifest["outputs"]["merged_eligible"]["path"])],
            split="train",
        )
    except Exception as error:
        raise PreparationError(f"merged response data is not loadable by datasets: {error}") from error
    if len(dataset) != manifest["outputs"]["merged_eligible"]["records"]:
        raise PreparationError("loaded response dataset row count differs")


def _verify_data_python(python: Path) -> None:
    target = python.resolve()
    if python.is_symlink() or not target.is_file():
        raise PreparationError("data Python must be a regular executable")
    probe = (
        "import importlib.metadata,json,platform;"
        "print(json.dumps({'python':platform.python_version(),"
        "'packages':{n:importlib.metadata.version(n) for n in "
        f"{list(DATA_PACKAGES)!r}}}}}))"
    )
    result = subprocess.run(
        [str(target), "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:] or "no diagnostic output"
        raise PreparationError(f"data Python package probe failed with exit {result.returncode}: {detail}")
    observed = json.loads(result.stdout)
    if observed != {"python": DATA_PYTHON, "packages": DATA_PACKAGES}:
        raise PreparationError(f"data Python differs from the frozen detector runtime: {observed}")


def _verify_rtt(root: Path) -> None:
    target = root.resolve()
    revision = _git(target, "rev-parse", "HEAD")
    status = _git(target, "status", "--porcelain=v1", "--untracked-files=all")
    if Path(_git(target, "rev-parse", "--show-toplevel")).resolve() != target or revision != RTT_REVISION or status:
        raise PreparationError("RTT checkout must be exact, pinned, and clean")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(python: Path, script: str, *args: str) -> None:
    env = _clean_env()
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [str(python.resolve()), str(ROOT / script), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2_000:]
        raise PreparationError(f"{Path(script).name} failed with exit {result.returncode}: {detail}")


def _download(url: str, output: Path, size: int, sha256: str) -> None:
    if _matches(output, size, sha256):
        return
    if output.exists() or output.is_symlink():
        raise PreparationError(f"existing download target differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temp = Path(name)
    try:
        with urllib.request.urlopen(url, timeout=300) as response, temp.open("wb") as stream:
            while chunk := response.read(1 << 20):
                stream.write(chunk)
        _verify_file(temp, size, sha256, output.name)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def _verify_file(path: Path, size: int | None, sha256: str, label: str) -> None:
    wrong_size = size is not None and path.is_file() and path.stat().st_size != size
    if path.is_symlink() or not path.is_file() or wrong_size or _sha256(path) != sha256:
        raise PreparationError(f"{label} differs from its frozen bytes")


def _matches(path: Path, size: int | None, sha256: str) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and (size is None or path.stat().st_size == size)
        and _sha256(path) == sha256
    )


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in ("OPENROUTER_API_KEY", "WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(name, None)
    return env


def _same_bytes(candidate: Path, expected: Path, label: str) -> None:
    if candidate.read_bytes() != expected.read_bytes():
        raise PreparationError(f"{label} regeneration differs from committed bytes")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreparationError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PreparationError, subprocess.SubprocessError) as error:
        print(f"response data preparation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
