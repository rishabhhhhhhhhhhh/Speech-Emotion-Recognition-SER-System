"""Repair cp1252-mojibake in MELD transcripts.

MELD's CSVs are UTF-8 but carry C1 CONTROL characters (U+0080-U+009F) where Windows-1252
punctuation belongs - an artifact baked into the original release. e.g. "companys"
should be "company's". 2,558 / 9,989 train rows (26%) are affected.

This matters more here than in a normal pipeline: these strings are rendered verbatim into
the PC-ESR prompt's <content> block, and control characters in a prompt are at best noise
and at worst tokeniser garbage.
"""
from __future__ import annotations

# C1 control codepoint -> the Windows-1252 character actually intended
CP1252_C1 = {
    0x80: "€", 0x82: "‚", 0x83: "ƒ", 0x84: "„",
    0x85: "…", 0x86: "†", 0x87: "‡", 0x88: "ˆ",
    0x89: "‰", 0x8a: "Š", 0x8b: "‹", 0x8c: "Œ",
    0x8e: "Ž", 0x91: "‘", 0x92: "’", 0x93: "“",
    0x94: "”", 0x95: "•", 0x96: "–", 0x97: "—",
    0x98: "˜", 0x99: "™", 0x9a: "š", 0x9b: "›",
    0x9c: "œ", 0x9e: "ž", 0x9f: "Ÿ",
}

# ASCII-fold curly punctuation so prompts stay tokeniser-friendly
ASCII_FOLD = {
    0x2018: "'", 0x2019: "'", 0x201a: "'",
    0x201c: '"', 0x201d: '"', 0x201e: '"',
    0x2013: "-", 0x2014: "-",
    0x2026: "...", 0x2022: "*",
    0x00a0: " ",   # non-breaking space
}


def clean_transcript(s, ascii_fold: bool = True) -> str:
    """Repair C1 mojibake, optionally ASCII-fold punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).translate(CP1252_C1)
    if ascii_fold:
        s = s.translate(ASCII_FOLD)
    # drop any surviving C0/C1 control characters
    s = "".join(ch for ch in s if ch == "\n" or ord(ch) >= 0x20)
    return " ".join(s.split()).strip()


def has_control(s) -> bool:
    """True if the string still contains C1 control characters."""
    return any(0x80 <= ord(c) <= 0x9f for c in str(s))
