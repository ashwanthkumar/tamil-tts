# Model card: tamil-tts (female, VITS)

- **Task:** Tamil text-to-speech (single-speaker, female)
- **Architecture:** VITS (Coqui-TTS), exported to ONNX
- **Sample rate:** 22.05 kHz mono
- **Inference:** CPU via onnxruntime (Python) / `ort` (Rust); no GPU required
- **Phonemization:** espeak-ng (`ta`), IPA, blank-interleaved ids — see `tamiltts/phonemize.py`

## Training data

[IndicTTS Tamil](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil) — the female speaker
subset (~10 hours, studio quality). License: CC-BY-4.0 + IIT Madras Indic TTS EULA. See
[`DATASET_LICENSE.md`](DATASET_LICENSE.md). Attribution is required in redistributions.

## Intended use & limitations

- Intended for Tamil speech synthesis from clean text input.
- Single voice; does not clone arbitrary speakers.
- Pronunciation quality depends on espeak-ng's Tamil rules; out-of-vocabulary words, code-mixed
  English, and unusual punctuation may degrade output.
- Not evaluated for safety-critical or biometric use.

## Inference knobs

| Param           | Default | Effect                          |
| --------------- | ------- | ------------------------------- |
| `noise_scale`   | 0.667   | voice variability               |
| `length_scale`  | 1.0     | speaking rate (>1 slower)       |
| `noise_scale_w` | 0.8     | duration/prosody variability    |

## Status

Pipeline complete and SDK-verified against a contract-compliant dummy model. A production voice
requires running the training step (`tamiltts/train.py`) to convergence on Apple Silicon.
