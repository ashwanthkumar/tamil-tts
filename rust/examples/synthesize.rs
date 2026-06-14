//! Synthesize Tamil text to a WAV file.
//!
//!     cargo run --release --example synthesize -- "வணக்கம்" hello.wav
//!     cargo run --release --example synthesize -- "வணக்கம்" hello.wav ../models/tamil_female.onnx

use anyhow::{anyhow, Result};
use tamil_tts::{SynthesisOptions, TamilTts};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(anyhow!(
            "usage: synthesize <tamil-text> <out.wav> [model.onnx]\n\
             default model: ../models/tamil_female.onnx"
        ));
    }
    let text = &args[1];
    let out = &args[2];
    let model = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "../models/tamil_female.onnx".to_string());

    let mut tts = TamilTts::from_file(&model)?;
    tts.save(text, out, &SynthesisOptions::default())?;
    println!("wrote {out} ({} Hz)", tts.sample_rate());
    Ok(())
}
