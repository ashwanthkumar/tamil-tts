"""Text normalization (verbalization) front-end — run BEFORE char tokenization.

Converts written forms the char-level model never saw into pronounceable, in-vocab Tamil:
  - acronyms / Latin-letter runs -> spelled out as Tamil letter-names, with periods removed
    (periods otherwise read as sentence-final pauses, making "I.I.T." sound choppy)
  - common symbols -> Tamil words
  - digit runs -> spoken digit-by-digit in Tamil (full cardinal number-words = future work)

Mirrored 1:1 by the Rust SDK (rust/src/normalize.rs); keep the two in sync.
"""
from __future__ import annotations

import re

# English letter names rendered in Tamil script (for spelling acronyms like IIT, MLX, TTS).
LETTER = {
    "a": "ஏ", "b": "பி", "c": "சி", "d": "டி", "e": "ஈ", "f": "எஃப்", "g": "ஜி",
    "h": "ஹெச்", "i": "ஐ", "j": "ஜே", "k": "கே", "l": "எல்", "m": "எம்", "n": "என்",
    "o": "ஓ", "p": "பி", "q": "கியூ", "r": "ஆர்", "s": "எஸ்", "t": "டி", "u": "யூ",
    "v": "வி", "w": "டபிள்யூ", "x": "எக்ஸ்", "y": "வை", "z": "செட்",
}

DIGIT = ["பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]

# --- Tamil cardinal number words (Indian place-value grouping) ---
_TEENS = ["பத்து", "பதினொன்று", "பன்னிரண்டு", "பதிமூன்று", "பதினான்கு", "பதினைந்து",
          "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொன்பது"]
_TENS_ST = ["", "பத்து", "இருபது", "முப்பது", "நாற்பது", "ஐம்பது", "அறுபது", "எழுபது", "எண்பது", "தொண்ணூறு"]
_TENS_CO = ["", "", "இருபத்தி", "முப்பத்தி", "நாற்பத்தி", "ஐம்பத்தி", "அறுபத்தி", "எழுபத்தி", "எண்பத்தி", "தொண்ணூற்றி"]
_HUND_ST = ["", "நூறு", "இருநூறு", "முந்நூறு", "நானூறு", "ஐந்நூறு", "அறுநூறு", "எழுநூறு", "எண்ணூறு", "தொள்ளாயிரம்"]
_HUND_CO = ["", "நூற்றி", "இருநூற்று", "முந்நூற்று", "நானூற்று", "ஐந்நூற்று", "அறுநூற்று", "எழுநூற்று", "எண்ணூற்று", "தொள்ளாயிரத்து"]
_THOU_ST = ["", "ஆயிரம்", "இரண்டாயிரம்", "மூவாயிரம்", "நான்காயிரம்", "ஐந்தாயிரம்", "ஆறாயிரம்", "ஏழாயிரம்", "எட்டாயிரம்", "ஒன்பதாயிரம்"]
_THOU_CO = ["", "ஆயிரத்து", "இரண்டாயிரத்து", "மூவாயிரத்து", "நான்காயிரத்து", "ஐந்தாயிரத்து", "ஆறாயிரத்து", "ஏழாயிரத்து", "எட்டாயிரத்து", "ஒன்பதாயிரத்து"]


def _two(n):  # 1..99
    if n < 10:
        return DIGIT[n]
    if n < 20:
        return _TEENS[n - 10]
    t, u = divmod(n, 10)
    return _TENS_ST[t] if u == 0 else f"{_TENS_CO[t]} {DIGIT[u]}"


def _three(n):  # 1..999
    h, r = divmod(n, 100)
    if h == 0:
        return _two(r)
    if r == 0:
        return _HUND_ST[h]
    return f"{_HUND_CO[h]} {_two(r)}"


def cardinal(n: int) -> str:
    if n == 0:
        return "பூஜ்ஜியம்"
    if n < 0:
        return "கழித்தல் " + cardinal(-n)
    crore, n = divmod(n, 10_000_000)
    lakh, n = divmod(n, 100_000)
    thou, hund = divmod(n, 1000)
    parts = []
    if crore:
        parts.append(("ஒரு" if crore == 1 else cardinal(crore)) + " கோடி")
    if lakh:
        parts.append(("ஒரு" if lakh == 1 else _three(lakh)) + " லட்சம்")
    if thou:
        if thou < 10:
            parts.append(_THOU_CO[thou] if hund else _THOU_ST[thou])
        else:
            parts.append(_three(thou) + (" ஆயிரத்து" if hund else " ஆயிரம்"))
    if hund:
        parts.append(_three(hund))
    return " ".join(parts)

SYMBOL = {
    "%": " சதவீதம் ", "₹": " ரூபாய் ", "$": " டாலர் ", "&": " மற்றும் ",
    "+": " கூட்டல் ", "=": " சமம் ", "@": " அட்டு ", "°": " டிகிரி ",
}

# a run of latin letters with optional INTERIOR dots: "IIT", "I.I.T", "MLX".
# (a trailing sentence period is left out, so it stays as end-of-sentence punctuation)
_LATIN_RUN = re.compile(r"[A-Za-z](?:\.[A-Za-z])*")
_DIGITS = re.compile(r"\d+")


def _spell_latin(run: str) -> str:
    letters = [c for c in run if c.isalpha()]
    if not letters:
        return run
    # only spell acronym-like runs (all-caps, or originally dotted); leave real words alone
    dotted = "." in run
    all_caps = run.replace(".", "").isupper()
    if not (dotted or all_caps) or len(letters) > 6:
        return run
    return "".join(LETTER.get(c.lower(), "") for c in letters)


def _spell_digits(m: re.Match) -> str:
    s = m.group(0)
    # leading-zero runs (codes, phone numbers) or huge numbers -> read digit by digit
    if (len(s) > 1 and s[0] == "0") or int(s) > 99_99_99_999:
        return " " + " ".join(DIGIT[int(d)] for d in s) + " "
    return " " + cardinal(int(s)) + " "


def normalize(text: str) -> str:
    for sym, word in SYMBOL.items():
        text = text.replace(sym, word)
    text = _LATIN_RUN.sub(lambda m: _spell_latin(m.group(0)), text)
    text = _DIGITS.sub(_spell_digits, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    for t in ["IIT", "I.I.T.", "MLX model", "காலை உணவு", "100% சரி", "2026 ஆம் ஆண்டு", "₹50"]:
        print(f"{t!r:30} -> {normalize(t)!r}")
