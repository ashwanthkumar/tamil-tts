"""Train a single-speaker Tamil VITS model with Coqui-TTS.

Reads a thin JSON config (see configs/tamil_female_vits.json), builds a Coqui ``VitsConfig``,
and launches training. Auto-detects Apple Silicon (MPS) so it uses the M1 Studio / M2 GPU.

    uv run python -m tamiltts.train --config configs/tamil_female_vits.json

Notes for Apple Silicon:
  * We set PYTORCH_ENABLE_MPS_FALLBACK=1 so ops without an MPS kernel fall back to CPU
    instead of crashing. Training works but some ops on CPU make it slower than CUDA.
  * VITS uses Monotonic Alignment Search (a CPU/Cython op) regardless of device.
  * Resume with --restore_path runs/<run>/<ckpt>.pth ; or --continue_path runs/<run>.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be set before torch is imported by Coqui/trainer.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _detect_device() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyTorch missing. Install training deps: `uv sync --extra train`.") from exc
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_and_train(cfg: dict, restore_path: str | None, continue_path: str | None) -> None:
    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    data_path = str(Path(cfg["data_path"]).resolve())
    output_path = str(Path(cfg["output_path"]).resolve())

    dataset = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_val.csv",
        path=data_path,
        language="ta",
    )

    audio = VitsAudioConfig(
        sample_rate=cfg["sample_rate"],
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    model_args = VitsArgs(hidden_channels=int(cfg.get("hidden_channels", 192)))

    config = VitsConfig(
        model_args=model_args,
        audio=audio,
        run_name=cfg["run_name"],
        batch_size=int(cfg["batch_size"]),
        eval_batch_size=int(cfg["eval_batch_size"]),
        batch_group_size=5,
        num_loader_workers=int(cfg.get("num_loader_workers", 4)),
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=int(cfg["epochs"]),
        use_phonemes=bool(cfg.get("use_phonemes", True)),
        phoneme_language=cfg.get("phoneme_language", "ta"),
        phonemizer=cfg.get("phonemizer", "espeak"),
        phoneme_cache_path=str(Path(output_path) / "phoneme_cache"),
        add_blank=bool(cfg.get("add_blank", True)),
        compute_input_seq_cache=True,
        print_step=int(cfg.get("print_step", 25)),
        print_eval=False,
        mixed_precision=bool(cfg.get("mixed_precision", False)),
        save_step=int(cfg.get("save_step", 1000)),
        save_n_checkpoints=int(cfg.get("save_n_checkpoints", 5)),
        output_path=output_path,
        datasets=[dataset],
        lr_gen=float(cfg.get("lr_gen", 2e-4)),
        lr_disc=float(cfg.get("lr_disc", 2e-4)),
        cudnn_benchmark=False,
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        dataset, eval_split=True, eval_split_size=config.eval_split_size
    )
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    device = _detect_device()
    print(f"[tamil-tts] training device: {device}")

    trainer = Trainer(
        TrainerArgs(restore_path=restore_path or "", continue_path=continue_path or ""),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


def main() -> None:
    p = argparse.ArgumentParser(description="Train Tamil VITS with Coqui-TTS.")
    p.add_argument("--config", type=Path, default=Path("configs/tamil_female_vits.json"))
    p.add_argument("--restore_path", default=None, help="checkpoint .pth to fine-tune/restore from")
    p.add_argument("--continue_path", default=None, help="run dir to resume training")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        build_and_train(cfg, args.restore_path, args.continue_path)
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies missing. Install them with:\n"
            "    uv sync --extra train\n"
            f"(import error: {exc})"
        )


if __name__ == "__main__":
    main()
