//! Print normalize(arg) for each CLI arg — used to verify Python/Rust normalizer parity.
use tamil_tts::normalize::normalize;

fn main() {
    for arg in std::env::args().skip(1) {
        println!("{}", normalize(&arg));
    }
}
