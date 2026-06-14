"""Export a trained Coqui VITS checkpoint to ONNX + tokenizer.json.

    uv run python -m tamiltts.export_onnx --run runs/tamil_female-<date> \
        --out models/tamil_female.onnx

Produces:
  * models/tamil_female.onnx      - the inference graph
  * models/tamil_female.tokenizer.json - phoneme vocab + flags for the SDKs

ONNX I/O contract (shared by the Python and Rust SDKs):
  inputs:
    input        int64   [1, T]   phoneme ids (from tamiltts.phonemize.Tokenizer)
    input_lengths int64  [1]      = T
    scales       float32 [3]      = [noise_scale, length_scale, noise_scale_w]
  output:
    output       float32 [1, 1, S] (or [1, S]) waveform in [-1, 1] at the model sample rate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_checkpoint(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("best_model*.pth")) or sorted(run_dir.glob("checkpoint_*.pth"))
    if not cands:
        cands = sorted(run_dir.glob("*.pth"))
    if not cands:
        raise SystemExit(f"No .pth checkpoint found under {run_dir}")
    return cands[-1]


def _build_id_map(tokenizer) -> dict[str, int]:
    chars = tokenizer.characters
    id_map: dict[str, int] = {}
    vocab = getattr(chars, "vocab", None)
    if vocab is None:
        raise SystemExit("Could not read tokenizer vocab; unexpected Coqui version.")
    for ch in vocab:
        try:
            id_map[ch] = int(chars.char_to_id(ch))
        except Exception:
            continue
    return id_map


def _token_id(chars, name: str):
    tok = getattr(chars, name, None)
    if not tok:
        return None
    try:
        return int(chars.char_to_id(tok))
    except Exception:
        return None


def export(run_dir: Path, out_path: Path) -> None:
    try:
        from TTS.tts.models.vits import Vits
        from TTS.config import load_config
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Training/export deps missing. Install: `uv sync --extra train`.\n"
            f"(import error: {exc})"
        )

    run_dir = run_dir.resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found in {run_dir}")
    ckpt = _find_checkpoint(run_dir)
    print(f"[export] config={config_path}")
    print(f"[export] checkpoint={ckpt}")

    config = load_config(str(config_path))
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, str(ckpt), eval=True)

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Coqui's Vits ships an ONNX exporter that builds the (input, input_lengths, scales) graph.
    model.export_onnx(output_path=str(out_path))
    print(f"[export] wrote {out_path}")

    # Emit the tokenizer contract next to the model.
    chars = model.tokenizer.characters
    tok = {
        "language": getattr(config, "phoneme_language", "ta") or "ta",
        "add_blank": bool(getattr(config, "add_blank", True)),
        "sample_rate": int(config.audio["sample_rate"]),
        "blank_id": _token_id(chars, "blank") or 0,
        "pad_id": _token_id(chars, "pad") or 0,
        "bos_id": _token_id(chars, "bos"),
        "eos_id": _token_id(chars, "eos"),
        "ignore": [" "],
        "id_map": _build_id_map(model.tokenizer),
    }
    tok_path = out_path.with_suffix(".tokenizer.json")
    tok_path.write_text(json.dumps(tok, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] wrote {tok_path}  (vocab size={len(tok['id_map'])})")


def main() -> None:
    p = argparse.ArgumentParser(description="Export Tamil VITS to ONNX + tokenizer.json")
    p.add_argument("--run", type=Path, required=True, help="training run dir (contains config.json)")
    p.add_argument("--out", type=Path, default=Path("models/tamil_female.onnx"))
    args = p.parse_args()
    export(args.run, args.out)


if __name__ == "__main__":
    main()
