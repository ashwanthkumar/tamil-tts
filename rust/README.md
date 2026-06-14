# tamil-tts (Rust SDK)

ONNX inference for the Tamil VITS voice, on CPU. Mirrors the Python SDK's phonemization and
ONNX I/O contract, so it consumes the same `tamil_female.onnx` + `tamil_female.tokenizer.json`.

Requires `espeak-ng` on PATH (`brew install espeak-ng`).

```rust
use tamil_tts::{TamilTts, SynthesisOptions};

let mut tts = TamilTts::from_file("../models/tamil_female.onnx")?;
tts.save("வணக்கம், இது தமிழ் பேச்சு.", "hello.wav", &SynthesisOptions::default())?;
```

CLI example:

```bash
cargo run --release --example synthesize -- "வணக்கம்" hello.wav ../models/tamil_female.onnx
```

`ort` downloads a prebuilt onnxruntime on first build. Pinned to `ort = 2.0.0-rc.12`.
