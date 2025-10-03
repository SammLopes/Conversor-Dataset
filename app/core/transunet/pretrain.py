"""
pretrain.py

Wrapper / helper to run a TransUNet pre-training step inside this project.

This file is intentionally framework-agnostic:
- It first tries to find an existing `treino_transunet.py` (or `train.py`) in the repo
  and import & call a `train(...)` function if available.
- If not available or import fails, it will execute the training script as a subprocess
  (`python train.py --dataset ...`).
- It also provides a helper to download ViT checkpoints used as backbones.

Usage examples (from project root):
    python -m app.core.transunet.pretrain --dataset BTCV --vit_name R50-ViT-B_16 --batch_size 6

The goal is to keep the orchestration here, and let the original training code do the
heavy-lifting (so minimal duplication).

"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    # prefer urllib for portability
    from urllib.request import urlretrieve
except Exception:  # pragma: no cover - fallback
    urlretrieve = None


LOG = logging.getLogger("transunet.pretrain")


def download_vit_checkpoint(model_name: str, dest_dir: str = "model/vit_checkpoint/imagenet21k") -> Path:
    """Download a ViT checkpoint from the Google storage bucket used by the original repo.

    This function attempts to download using urllib; if that fails it falls back to `wget`.

    Returns the Path to the downloaded file.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}.npz"
    url = f"https://storage.googleapis.com/vit_models/imagenet21k/{filename}"
    out_path = dest / filename

    if out_path.exists():
        LOG.info("ViT checkpoint already exists: %s", out_path)
        return out_path

    LOG.info("Downloading ViT checkpoint %s -> %s", url, out_path)
    try:
        if urlretrieve is not None:
            urlretrieve(url, str(out_path))
        else:
            raise RuntimeError("urllib not available")
    except Exception as exc:  # pragma: no cover - network behaviour
        LOG.warning("urllib failed to download checkpoint: %s", exc)
        # try wget as a fallback
        try:
            subprocess.run(["wget", url, "-O", str(out_path)], check=True)
        except Exception as e:  # pragma: no cover - network behaviour
            LOG.error("Failed to download ViT checkpoint via wget: %s", e)
            raise

    LOG.info("Downloaded ViT checkpoint to %s", out_path)
    return out_path


def _find_training_script_candidates(root: Path) -> List[Path]:
    candidates = []
    # Common names used in many forks
    for name in ("treino_transunet.py", "train.py", "train_transunet.py"):
        p = root / name
        if p.exists():
            candidates.append(p)
    # Also look under app/core (relative to this file)
    core_dir = (Path(__file__).parent.parent).resolve()
    for name in ("treino_transunet.py", "train.py"):
        p = core_dir / name
        if p.exists() and p not in candidates:
            candidates.append(p)
    return candidates


def run_training_subprocess(script_path: Path, dataset: str, vit_name: str, batch_size: int, epochs: int, extra_args: Optional[List[str]] = None):
    cmd = [sys.executable, str(script_path), "--dataset", dataset, "--vit_name", vit_name, "--batch_size", str(batch_size), "--epochs", str(epochs)]
    if extra_args:
        cmd.extend(extra_args)
    LOG.info("Running training script as subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_training_import(module_path: Path, dataset: str, vit_name: str, batch_size: int, epochs: int, extra_args: Optional[List[str]] = None):
    """Try to import a training module and call a standard `train` function if present.

    Expected signature (best-effort): train(dataset, vit_name, batch_size, epochs, **kwargs)
    """
    LOG.info("Attempting to import training module from %s", module_path)
    spec = importlib.util.spec_from_file_location("transunet_train_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # try common function names
    for fn in ("train", "main", "run"):
        if hasattr(module, fn):
            train_fn = getattr(module, fn)
            LOG.info("Calling %s() from imported module", fn)
            # call with conservative args; if function expects argparse, it should handle it
            try:
                return train_fn(dataset=dataset, vit_name=vit_name, batch_size=batch_size, epochs=epochs)
            except TypeError:
                try:
                    return train_fn(dataset, vit_name, batch_size, epochs)
                except TypeError:
                    LOG.warning("Imported function %s did not accept standard args; attempting no-arg call", fn)
                    return train_fn()

    raise AttributeError("No callable train-like function found in module")


def pretrain_model(dataset: str = "BTCV", vit_name: str = "R50-ViT-B_16", batch_size: int = 6, epochs: int = 100, save_dir: str = "models/transunet/pretrained", download_vit: bool = True, vit_dest: str = "model/vit_checkpoint/imagenet21k", use_import: bool = True, extra_args: Optional[List[str]] = None):
    """High-level orchestration to pretrain a TransUNet model.

    Steps:
    - Optionally download ViT checkpoint
    - Try to import local training module (preferred)
    - If import fails, run the training script as subprocess
    - Ensure save_dir exists
    """
    project_root = Path.cwd()
    save_dir_p = Path(save_dir)
    save_dir_p.mkdir(parents=True, exist_ok=True)

    if download_vit:
        try:
            download_vit_checkpoint(vit_name, dest_dir=vit_dest)
        except Exception as e:
            LOG.warning("Could not download ViT checkpoint: %s", e)

    candidates = _find_training_script_candidates(project_root)
    if not candidates:
        LOG.error("No training script found in repository. Please ensure train.py or treino_transunet.py exists.")
        raise FileNotFoundError("training script not found")

    # Prefer import when requested
    last_exc: Optional[Exception] = None
    for candidate in candidates:
        try:
            if use_import:
                run_training_import(candidate, dataset=dataset, vit_name=vit_name, batch_size=batch_size, epochs=epochs, extra_args=extra_args)
            else:
                run_training_subprocess(candidate, dataset=dataset, vit_name=vit_name, batch_size=batch_size, epochs=epochs, extra_args=extra_args)
            LOG.info("Pretraining finished (candidate: %s)", candidate)
            return
        except Exception as exc:
            LOG.warning("Candidate %s failed: %s", candidate, exc)
            last_exc = exc

    LOG.error("All candidates failed. Last error: %s", last_exc)
    raise last_exc


def _parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Pretrain TransUNet wrapper")
    p.add_argument("--dataset", type=str, default="BTCV", help="Dataset name (BTCV, ACDC, etc.)")
    p.add_argument("--vit_name", type=str, default="R50-ViT-B_16", help="ViT checkpoint name")
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--save_dir", type=str, default="models/transunet/pretrained")
    p.add_argument("--no_download_vit", action="store_true", help="Skip downloading the ViT checkpoint")
    p.add_argument("--use_subprocess", action="store_true", help="Run train script as subprocess instead of importing")
    p.add_argument("--vit_dest", type=str, default="model/vit_checkpoint/imagenet21k")
    p.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args passed to the training script")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    pretrain_model(dataset=args.dataset, vit_name=args.vit_name, batch_size=args.batch_size, epochs=args.epochs, save_dir=args.save_dir, download_vit=(not args.no_download_vit), vit_dest=args.vit_dest, use_import=(not args.use_subprocess), extra_args=args.extra_args)


if __name__ == "__main__":
    main()
