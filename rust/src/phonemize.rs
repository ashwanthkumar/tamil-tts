//! Tamil phonemization — the Rust half of the cross-language contract.
//!
//! Mirrors `tamiltts/phonemize.py`: shell out to the same `espeak-ng` binary, then map IPA
//! characters to ids using the same `tokenizer.json`. Keeping both SDKs on the identical
//! espeak CLI invocation is what makes the exported ONNX model portable across languages.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Tokenizer {
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default = "default_true")]
    pub add_blank: bool,
    #[serde(default)]
    pub blank_id: i64,
    #[serde(default)]
    pub pad_id: i64,
    #[serde(default)]
    pub bos_id: Option<i64>,
    #[serde(default)]
    pub eos_id: Option<i64>,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_ignore")]
    pub ignore: Vec<String>,
    pub id_map: HashMap<String, i64>,
}

fn default_language() -> String {
    "ta".to_string()
}
fn default_true() -> bool {
    true
}
fn default_sample_rate() -> u32 {
    22050
}
fn default_ignore() -> Vec<String> {
    vec![" ".to_string()]
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading tokenizer {}", path.as_ref().display()))?;
        let tok: Tokenizer = serde_json::from_str(&data).context("parsing tokenizer.json")?;
        Ok(tok)
    }

    /// Run espeak-ng to convert Tamil text to an IPA phoneme string (`--ipa=3`).
    fn espeak(&self, text: &str) -> Result<String> {
        let out = Command::new("espeak-ng")
            .args(["-q", "--ipa=3", "-v", &self.language, text])
            .output()
            .context("running espeak-ng (is it installed? `brew install espeak-ng`)")?;
        if !out.status.success() {
            return Err(anyhow!(
                "espeak-ng failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        let raw = String::from_utf8_lossy(&out.stdout);
        Ok(raw.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    /// Encode Tamil text to the phoneme-id sequence the ONNX model expects.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let phonemes = self.espeak(text)?;
        let ignore: HashSet<&str> = self.ignore.iter().map(|s| s.as_str()).collect();

        let mut ids: Vec<i64> = Vec::new();
        for ch in phonemes.chars() {
            let key = ch.to_string();
            if ignore.contains(key.as_str()) {
                continue;
            }
            if let Some(&id) = self.id_map.get(&key) {
                ids.push(id);
            }
        }

        if self.add_blank {
            let mut interleaved = Vec::with_capacity(ids.len() * 2 + 1);
            interleaved.push(self.blank_id);
            for id in ids {
                interleaved.push(id);
                interleaved.push(self.blank_id);
            }
            ids = interleaved;
        }
        if let Some(bos) = self.bos_id {
            ids.insert(0, bos);
        }
        if let Some(eos) = self.eos_id {
            ids.push(eos);
        }

        if ids.is_empty() {
            return Err(anyhow!("text produced no phonemes; check input / espeak language"));
        }
        Ok(ids)
    }
}
