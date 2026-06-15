//! Text normalization (verbalization) — mirror of `tamiltts/mlx/normalize.py`. Keep in sync.
//!
//! Run before char tokenization: acronyms/Latin-letter runs -> Tamil letter-names (periods dropped),
//! symbols -> Tamil words, digit runs -> Tamil cardinal number-words (digit-by-digit for leading-zero
//! codes or huge numbers).

use regex::{Captures, Regex};

const DIGIT: [&str; 10] = ["பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"];
const TEENS: [&str; 10] = ["பத்து", "பதினொன்று", "பன்னிரண்டு", "பதிமூன்று", "பதினான்கு", "பதினைந்து", "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொன்பது"];
const TENS_ST: [&str; 10] = ["", "பத்து", "இருபது", "முப்பது", "நாற்பது", "ஐம்பது", "அறுபது", "எழுபது", "எண்பது", "தொண்ணூறு"];
const TENS_CO: [&str; 10] = ["", "", "இருபத்தி", "முப்பத்தி", "நாற்பத்தி", "ஐம்பத்தி", "அறுபத்தி", "எழுபத்தி", "எண்பத்தி", "தொண்ணூற்றி"];
const HUND_ST: [&str; 10] = ["", "நூறு", "இருநூறு", "முந்நூறு", "நானூறு", "ஐந்நூறு", "அறுநூறு", "எழுநூறு", "எண்ணூறு", "தொள்ளாயிரம்"];
const HUND_CO: [&str; 10] = ["", "நூற்றி", "இருநூற்று", "முந்நூற்று", "நானூற்று", "ஐந்நூற்று", "அறுநூற்று", "எழுநூற்று", "எண்ணூற்று", "தொள்ளாயிரத்து"];
const THOU_ST: [&str; 10] = ["", "ஆயிரம்", "இரண்டாயிரம்", "மூவாயிரம்", "நான்காயிரம்", "ஐந்தாயிரம்", "ஆறாயிரம்", "ஏழாயிரம்", "எட்டாயிரம்", "ஒன்பதாயிரம்"];
const THOU_CO: [&str; 10] = ["", "ஆயிரத்து", "இரண்டாயிரத்து", "மூவாயிரத்து", "நான்காயிரத்து", "ஐந்தாயிரத்து", "ஆறாயிரத்து", "ஏழாயிரத்து", "எட்டாயிரத்து", "ஒன்பதாயிரத்து"];

const SYMBOLS: [(&str, &str); 8] = [
    ("%", " சதவீதம் "), ("₹", " ரூபாய் "), ("$", " டாலர் "), ("&", " மற்றும் "),
    ("+", " கூட்டல் "), ("=", " சமம் "), ("@", " அட்டு "), ("°", " டிகிரி "),
];

fn letter_name(c: char) -> &'static str {
    match c.to_ascii_lowercase() {
        'a' => "ஏ", 'b' => "பி", 'c' => "சி", 'd' => "டி", 'e' => "ஈ", 'f' => "எஃப்", 'g' => "ஜி",
        'h' => "ஹெச்", 'i' => "ஐ", 'j' => "ஜே", 'k' => "கே", 'l' => "எல்", 'm' => "எம்", 'n' => "என்",
        'o' => "ஓ", 'p' => "பி", 'q' => "கியூ", 'r' => "ஆர்", 's' => "எஸ்", 't' => "டி", 'u' => "யூ",
        'v' => "வி", 'w' => "டபிள்யூ", 'x' => "எக்ஸ்", 'y' => "வை", 'z' => "செட்", _ => "",
    }
}

fn two(n: u64) -> String {
    if n < 10 { return DIGIT[n as usize].into(); }
    if n < 20 { return TEENS[(n - 10) as usize].into(); }
    let (t, u) = (n / 10, n % 10);
    if u == 0 { TENS_ST[t as usize].into() } else { format!("{} {}", TENS_CO[t as usize], DIGIT[u as usize]) }
}

fn three(n: u64) -> String {
    let (h, r) = (n / 100, n % 100);
    if h == 0 { return two(r); }
    if r == 0 { return HUND_ST[h as usize].into(); }
    format!("{} {}", HUND_CO[h as usize], two(r))
}

fn cardinal(n: u64) -> String {
    if n == 0 { return "பூஜ்ஜியம்".into(); }
    let crore = n / 10_000_000;
    let n = n % 10_000_000;
    let lakh = n / 100_000;
    let n = n % 100_000;
    let (thou, hund) = (n / 1000, n % 1000);
    let mut parts: Vec<String> = Vec::new();
    if crore > 0 { parts.push(format!("{} கோடி", if crore == 1 { "ஒரு".into() } else { cardinal(crore) })); }
    if lakh > 0 { parts.push(format!("{} லட்சம்", if lakh == 1 { "ஒரு".into() } else { three(lakh) })); }
    if thou > 0 {
        if thou < 10 {
            parts.push((if hund > 0 { THOU_CO[thou as usize] } else { THOU_ST[thou as usize] }).into());
        } else {
            parts.push(format!("{}{}", three(thou), if hund > 0 { " ஆயிரத்து" } else { " ஆயிரம்" }));
        }
    }
    if hund > 0 { parts.push(three(hund)); }
    parts.join(" ")
}

fn spell_latin(run: &str) -> String {
    let letters: Vec<char> = run.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() { return run.to_string(); }
    let dotted = run.contains('.');
    let all_caps = letters.iter().all(|c| c.is_ascii_uppercase());
    if !(dotted || all_caps) || letters.len() > 6 { return run.to_string(); }
    letters.iter().map(|&c| letter_name(c)).collect()
}

fn spell_digits(s: &str) -> String {
    let n: u64 = s.parse().unwrap_or(u64::MAX);
    if (s.len() > 1 && s.starts_with('0')) || n > 99_99_99_999 {
        let d: Vec<&str> = s.chars().map(|c| DIGIT[c as usize - '0' as usize]).collect();
        return format!(" {} ", d.join(" "));
    }
    format!(" {} ", cardinal(n))
}

pub fn normalize(text: &str) -> String {
    let mut s = text.to_string();
    for (sym, word) in SYMBOLS { s = s.replace(sym, word); }
    let latin = Regex::new(r"[A-Za-z](?:\.[A-Za-z])*").unwrap();
    s = latin.replace_all(&s, |c: &Captures| spell_latin(&c[0])).into_owned();
    let digits = Regex::new(r"\d+").unwrap();
    s = digits.replace_all(&s, |c: &Captures| spell_digits(&c[0])).into_owned();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
