"""Emotion-state-transition matrix from Plutchik's wheel + the ESWR reward.

Paper B (EMO-RL, EMNLP 2025 Findings) Eq. 7-8:

    S(i,j) = 1/2                          if y_i or y_j is 'neutral'
           = 1/2 * (cos(Pl(y_i,y_j)) + 1) otherwise

    R_ESWR = 1              if S(yhat,y) == 1
           = alpha * S      if S(yhat,y) >  gamma
           = 0              if S(yhat,y) <= gamma      (gamma = 0.7)

TWO DOCUMENTED DEVIATIONS - both are bugs in the paper as literally written:

 1. Eq. 7 reads "if y_i or y_i = 'neutral'". The second subscript is clearly y_j.

 2. Taken literally, a CORRECT neutral prediction scores S = 0.5 <= gamma = 0.7 and
    therefore earns reward 0. That cannot be intended - it would actively punish the
    single largest class in MELD (~48% neutral). We short-circuit exact matches to 1.0.

Consequence worth reporting on MELD
-----------------------------------
Plutchik's 8 primaries sit 45 degrees apart. MELD's labels land at:
    joy 0 | fear 90 | surprise 135 | sadness 180 | disgust 225 | anger 270
With gamma = 0.7 only pairs exactly 45 degrees apart clear the threshold (S = 0.854):
    (fear,surprise) (surprise,sadness) (sadness,disgust) (disgust,anger)
`joy` has NO 45-degree neighbour in MELD's label set - its Plutchik neighbours are Trust
and Anticipation, which MELD does not use - so joy is always scored binary. `neutral` is
binary too. So ESWR densifies the negative-emotion chain only, leaving ~55% of MELD's
mass (neutral + joy) on a binary reward. Report this; it explains part of any gap vs the
paper's headline numbers.
"""
from __future__ import annotations
import math
import numpy as np

# Plutchik's wheel, 8 primaries, 45 degrees apart, opposites 180 apart.
PLUTCHIK_ANGLE = {
    "joy": 0.0, "trust": 45.0, "fear": 90.0, "surprise": 135.0,
    "sadness": 180.0, "disgust": 225.0, "anger": 270.0, "anticipation": 315.0,
}
NEUTRAL_S = 0.5
GAMMA = 0.7


def angle_between(a: str, b: str) -> float:
    """Smaller of the two arc distances on the wheel, in degrees (0..180)."""
    d = abs(PLUTCHIK_ANGLE[a] - PLUTCHIK_ANGLE[b]) % 360.0
    return min(d, 360.0 - d)


def similarity(a: str, b: str) -> float:
    """S(a,b) in [0,1]. Exact match short-circuits to 1.0 (DEVIATION 2)."""
    if a == b:
        return 1.0
    if a == "neutral" or b == "neutral":      # DEVIATION 1: y_i or y_j
        return NEUTRAL_S
    return 0.5 * (math.cos(math.radians(angle_between(a, b))) + 1.0)


def similarity_matrix(labels) -> np.ndarray:
    labels = list(labels)
    return np.array([[similarity(a, b) for b in labels] for a in labels], dtype=np.float64)


def eswr_reward(pred: str | None, true: str, alpha: float = 1.0,
                gamma: float = GAMMA) -> float:
    """Emotion Similarity-Weighted Reward. `pred=None` (unparseable answer) -> 0."""
    if pred is None:
        return 0.0
    s = similarity(pred, true)
    if s >= 1.0:
        return 1.0
    if s > gamma:
        return alpha * s
    return 0.0


def alpha_schedule(step: int, total_steps: int) -> float:
    """Curriculum: partial credit fades 1.0 -> 0.0 across training (Paper B section 3.3.2).
    Early on 'close enough' pays, so the policy first learns the valence/arousal
    direction; late in training only exact answers score."""
    return max(0.0, 1.0 - step / max(total_steps, 1))
