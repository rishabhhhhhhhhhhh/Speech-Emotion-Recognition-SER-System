"""Canonical 7-class label space and per-corpus mappings.

Fixed forever. Everything downstream (CNN head, prior vector, LALM prompt options,
Plutchik matrix) indexes into CANONICAL in this exact order.
"""
from __future__ import annotations

CANONICAL = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
LABEL2ID = {lab: i for i, lab in enumerate(CANONICAL)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}
NUM_CLASSES = len(CANONICAL)

# Synonyms the LALM may emit in <answer>; normalised before scoring.
SYNONYMS = {
    "happy": "joy", "happiness": "joy", "joyful": "joy", "excited": "joy",
    "sad": "sadness", "angry": "anger", "mad": "anger",
    "fearful": "fear", "afraid": "fear", "scared": "fear",
    "surprised": "surprise", "disgusted": "disgust",
    "calm": "neutral", "neutral state": "neutral",
}

def canonicalize(raw: str) -> str | None:
    """Normalise a free-text emotion string to CANONICAL, or None if unmappable."""
    if raw is None:
        return None
    s = raw.strip().strip(".<>[]()\"'").lower()
    if s in LABEL2ID:
        return s
    return SYNONYMS.get(s)

# --- RAVDESS: 03-01-05-01-02-01-16.wav -----------------------------------
# modality-vocalChannel-EMOTION-intensity-statement-repetition-ACTOR
# NOTE: 'calm' is merged into neutral (Paper A does the same). Original kept
# in the manifest's `orig_label` column so the merge stays auditable.
RAVDESS_EMOTION = {
    "01": "neutral", "02": "neutral", "03": "joy", "04": "sadness",
    "05": "anger", "06": "fear", "07": "disgust", "08": "surprise",
}
RAVDESS_ORIG = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
RAVDESS_STATEMENT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
# RAVDESS actor id: odd = male, even = female.

# --- CREMA-D: 1022_ITS_ANG_XX.wav ----------------------------------------
# actorID_SENTENCE_EMOTION_intensity.  No 'surprise' class.
CREMAD_EMOTION = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "joy", "NEU": "neutral", "SAD": "sadness",
}
CREMAD_SENTENCE = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

# --- SAVEE: DC_a01.wav ----------------------------------------------------
SAVEE_EMOTION = {
    "a": "anger", "d": "disgust", "f": "fear", "h": "joy",
    "n": "neutral", "sa": "sadness", "su": "surprise",
}

# --- MELD: labels already canonical, just lowercase ----------------------
MELD_EMOTION = {lab: lab for lab in CANONICAL}

CORPORA = ("ravdess", "cremad", "savee", "meld")
