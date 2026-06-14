"""Tamil phonemization — the single source of truth shared by training and inference.

The trained VITS model maps a sequence of *phoneme ids* to audio. To run the exported ONNX
model outside Coqui-TTS (in our Python and Rust SDKs) we must reproduce the exact same
text -> phoneme-id pipeline used at training time. That contract is:

    1. espeak-ng converts Tamil graphemes -> IPA phonemes  (`espeak-ng -q --ipa=3 -v <lang>`)
    2. each IPA character is mapped to an id via the vocabulary in `tokenizer.json`
    3. if `add_blank` is set, a blank id is interleaved between phonemes and at both ends
    4. `bos`/`eos` ids are prepended/appended if present

`tokenizer.json` (written by `tamiltts.export_onnx`) is the authoritative description of the
vocabulary and these flags, so the Python and Rust SDKs stay byte-for-byte consistent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LANGUAGE = "ta"


def espeak_phonemes(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Return the IPA phoneme string for `text` using the espeak-ng CLI.

    Uses ``--ipa=3`` (phonemes separated, no stress/tie ties) for stable, parseable output.
    """
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        raise RuntimeError(
            "espeak-ng not found on PATH. Install it: `brew install espeak-ng`."
        )
    out = subprocess.run(
        [exe, "-q", "--ipa=3", "-v", language, text],
        capture_output=True,
        text=True,
        check=True,
    )
    # espeak emits one line per sentence; join and normalize whitespace.
    return " ".join(out.stdout.split())


@dataclass
class Tokenizer:
    """Maps Tamil text -> phoneme ids, per an exported `tokenizer.json` contract."""

    id_map: dict[str, int]
    language: str = DEFAULT_LANGUAGE
    add_blank: bool = True
    blank_id: int = 0
    pad_id: int = 0
    bos_id: int | None = None
    eos_id: int | None = None
    # characters that espeak may emit but the model never saw; dropped silently.
    ignore: set[str] = field(default_factory=lambda: {" "})

    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            id_map={str(k): int(v) for k, v in data["id_map"].items()},
            language=data.get("language", DEFAULT_LANGUAGE),
            add_blank=bool(data.get("add_blank", True)),
            blank_id=int(data.get("blank_id", 0)),
            pad_id=int(data.get("pad_id", 0)),
            bos_id=data.get("bos_id"),
            eos_id=data.get("eos_id"),
            ignore=set(data.get("ignore", [" "])),
        )

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "add_blank": self.add_blank,
            "blank_id": self.blank_id,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "ignore": sorted(self.ignore),
            "id_map": self.id_map,
        }

    def encode(self, text: str) -> list[int]:
        phonemes = espeak_phonemes(text, self.language)
        ids: list[int] = []
        for ch in phonemes:
            if ch in self.ignore:
                continue
            tok = self.id_map.get(ch)
            if tok is not None:
                ids.append(tok)

        if self.add_blank:
            interleaved = [self.blank_id]
            for tok in ids:
                interleaved.append(tok)
                interleaved.append(self.blank_id)
            ids = interleaved

        if self.bos_id is not None:
            ids = [self.bos_id, *ids]
        if self.eos_id is not None:
            ids = [*ids, self.eos_id]
        return ids
