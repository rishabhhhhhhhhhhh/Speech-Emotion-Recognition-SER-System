"""Prompt schemas for the LALM stage, plus their format-reward regexes.

Four modes. IR / EUR / ESR reproduce Paper B's three reasoning conditions so we can verify
the ESR > EUR > IR ordering before spending money on RL. PC_ESR is ours.

    IR      answer only, in <answer> tags
    EUR     free-form <think> then <answer>
    ESR     Paper B's structured schema: content / acoustics / summary
    PC_ESR  ESR + a <prior_check> block that must adjudicate the CNN prior

TWO SCHEMAS, NEVER ONE
----------------------
When no prior is supplied, BOTH the prior paragraph and the <prior_check> tag are dropped
and the format reward validates against ESR instead. Asking a model to "check the prior"
when there is no prior would train it to hallucinate one.

WHY <prior_check> EXISTS
------------------------
Injecting "the CNN says anger 0.41" into a prompt has two failure modes: the model ignores
it, or it copies it. Neither is a result. Forcing an explicit AGREE/CONFLICT verdict, and
rewarding the format, makes engagement structural rather than optional - and it makes the
behaviour measurable afterwards (override precision: when the model says CONFLICT, how
often is it right?).
"""
from __future__ import annotations
import re

from data.labels import CANONICAL

OPTIONS = "[" + ", ".join(f"'{c}'" for c in CANONICAL) + "]"

# --------------------------------------------------------------------------- IR
IR_INSTRUCTION = (
    "Please think step by step. Output the final answer in <answer> </answer>."
)

# -------------------------------------------------------------------------- EUR
EUR_INSTRUCTION = (
    "Please think step by step. Output the thinking process in <think> </think> "
    "and the final answer in <answer> </answer>."
)

# -------------------------------------------------------------------------- ESR
ESR_INSTRUCTION = """Please think step by step. Your reply must strictly follow this format:
<think>
<content> recognise and analyse the speaker's speech content and any emotionally salient words </content>
<acoustics> analyse the speaker's acoustic features: pitch, rhythm, speed, volume, voice quality </acoustics>
<summary> summarise all views </summary>
</think><answer> the selected emotion, one of: neutral, joy, sadness, anger, surprise, fear, disgust </answer>"""

# ----------------------------------------------------------------------- PC-ESR
PC_ESR_INSTRUCTION = """Please think step by step. Your reply must strictly follow this format:
<think>
<content> recognise and analyse the speaker's speech content and any emotionally salient words </content>
<acoustics> analyse the speaker's acoustic features: pitch, rhythm, speed, volume, voice quality </acoustics>
<prior_check> write exactly the word AGREE or the word CONFLICT, then one sentence of justification </prior_check>
<summary> summarise all views </summary>
</think><answer> the selected emotion, one of: neutral, joy, sadness, anger, surprise, fear, disgust </answer>"""

PRIOR_BLOCK = """Acoustic classifier prior - a CNN over the mel spectrogram, scoring {ua:.1f}% UA / {wa:.1f}% WA on this corpus when used alone. Treat this as evidence, not as the answer:
  {top3}   (speaker: {gender})"""

ONESHOT = '\n\nRespond in English only.\n\nHere is a correctly formatted example (for a different audio clip):\n<think>\n<content> The speaker says "I can\'t believe you did that" - the words carry reproach. </content>\n<acoustics> Raised pitch, clipped rhythm, elevated volume, tense voice quality. </acoustics>\n<prior_check> AGREE. The classifier favours anger and both the wording and the sharp, loud delivery support that. </prior_check>\n<summary> Reproachful wording plus sharp loud delivery indicate anger. </summary>\n</think><answer> anger </answer>\n\nNow analyse the audio above, filling in every tag. Use only these emotion words: neutral, joy, sadness, anger, surprise, fear, disgust.'
ONESHOT_NOPRIOR = '\n\nRespond in English only.\n\nHere is a correctly formatted example (for a different audio clip):\n<think>\n<content> The speaker says "I can\'t believe you did that" - the words carry reproach. </content>\n<acoustics> Raised pitch, clipped rhythm, elevated volume, tense voice quality. </acoustics>\n<summary> Reproachful wording plus sharp loud delivery indicate anger. </summary>\n</think><answer> anger </answer>\n\nNow analyse the audio above, filling in every tag. Use only these emotion words: neutral, joy, sadness, anger, surprise, fear, disgust.'

QUESTION = ("What is the emotion of the speaker in the audio? "
            f"Choose exactly one of: {OPTIONS}")


def build_prompt(mode: str, transcript: str | None = None,
                 prior_top3: str | None = None, gender: str | None = None,
                 prior_ua: float = 0.0, prior_wa: float = 0.0,
                 include_transcript: bool = True) -> str:
    """Assemble the text half of the prompt. Audio is attached separately by the caller."""
    mode = mode.upper()
    parts = ["You are an expert speech-emotion analyst."]

    if include_transcript and transcript:
        parts.append(f'Transcript: "{transcript}"')

    if mode == "PC_ESR":
        if not prior_top3:
            raise ValueError("PC_ESR requires prior_top3 - use ESR when there is no prior")
        parts.append(PRIOR_BLOCK.format(ua=prior_ua, wa=prior_wa,
                                        top3=prior_top3, gender=gender or "unknown"))

    parts.append(QUESTION)
    parts.append({
        "IR": IR_INSTRUCTION,
        "EUR": EUR_INSTRUCTION,
        "ESR": ESR_INSTRUCTION,
        "PC_ESR": PC_ESR_INSTRUCTION,
    }[mode])
    if mode == "PC_ESR":
        parts.append(ONESHOT.strip())
    elif mode == "ESR":
        parts.append(ONESHOT_NOPRIOR.strip())
    else:
        parts.append("Respond in English only.")
    return "\n\n".join(parts)


# ------------------------------------------------------- format-reward regexes
# Binary scoring, Paper B section 3.3.1: exact structural match -> 1, else 0.
# DOTALL so reasoning may span lines; each block must be NON-EMPTY (\S) so the model
# cannot satisfy the schema with empty tags.
_A = r"<answer>\s*(?P<answer>[^<>]*?\S[^<>]*?)\s*</answer>"

# Schema block sequences. Checked with SEPARATE simple regexes, never one giant pattern.
#
# The previous version used `[\s\S]*?\S[\s\S]*?` per block - two lazy quantifiers around a
# single non-space char, five times in sequence. On a long response that does NOT match,
# that backtracks exponentially and HANGS (observed: the scoring step wedged for minutes
# on 32 responses). Sequential simple matches are linear and do the same job.
SCHEMA_BLOCKS = {
    "IR": [],
    "EUR": ["think"],
    "ESR": ["think", "content", "acoustics", "summary"],
    "PC_ESR": ["think", "content", "acoustics", "prior_check", "summary"],
}

# Blocks that must be ABSENT. Without these, a schema accepts any superset of itself:
# ESR would score 1.0 on a PC-ESR response, and IR on anything ending in </answer>.
# ESR vs PC_ESR is exactly the comparison this project rests on, so they must be
# mutually exclusive, not nested. EUR stays permissive by design - Paper B defines it
# as free-form reasoning, so internal structure is allowed.
FORBIDDEN_BLOCKS = {
    "IR": ["think", "content", "acoustics", "prior_check", "summary"],
    "EUR": [],
    "ESR": ["prior_check"],
    "PC_ESR": [],
}

_BLOCK = {}
for _tags in SCHEMA_BLOCKS.values():
    for _t in _tags:
        _BLOCK.setdefault(_t, re.compile(rf"<{_t}>(.*?)</{_t}>", re.DOTALL))

_ANSWER_STRICT = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def format_reward(response: str, mode: str) -> float:
    """1.0 iff `response` matches the schema `mode` requested. Binary, Paper B sec 3.3.1.

    Requires: every block present exactly once and non-empty, blocks in the specified
    order, exactly one properly closed <answer>, and nothing but whitespace outside the
    <think>...</think><answer>...</answer> envelope.
    """
    r = (response or "").strip()
    mode = mode.upper()
    if mode not in SCHEMA_BLOCKS:
        raise KeyError(mode)

    ans = _ANSWER_STRICT.findall(r)
    if len(ans) != 1 or not ans[0].strip():
        return 0.0
    if not r.endswith("</answer>"):
        return 0.0

    for tag in FORBIDDEN_BLOCKS[mode]:
        if f"<{tag}>" in r:
            return 0.0

    pos = 0
    for tag in SCHEMA_BLOCKS[mode]:
        m = _BLOCK[tag].search(r, pos)
        if m is None or not m.group(1).strip():
            return 0.0
        if r.count(f"<{tag}>") != 1:
            return 0.0
        pos = m.start() + 1 if tag == "think" else m.end()
    if SCHEMA_BLOCKS[mode] and not r.startswith("<think>"):
        return 0.0
    return 1.0


# Lenient answer extraction for ACCURACY scoring - deliberately separate from the format
# reward. A response may be malformed (format reward 0) yet still contain a correct answer,
# and the two rewards must be able to disagree; collapsing them would hide that signal.
# Accept an unclosed <answer> too: generation can hit max_new_tokens mid-tag, and a
# truncated-but-present answer is still a real prediction. The FORMAT reward stays strict
# (it requires the closing tag); only accuracy parsing is lenient. The two must be able
# to disagree - that difference is itself a measurement.
ANSWER_RE = re.compile(r"<answer>\s*([^<>]+?)\s*(?:</answer>|$)", re.DOTALL | re.IGNORECASE)

VERDICT_RE = re.compile(r"<prior_check>\s*[\s\S]*?\b(AGREE|CONFLICT)\b", re.IGNORECASE)


def extract_answer(response: str) -> str | None:
    """Last <answer> block, raw text. None if absent."""
    m = ANSWER_RE.findall(response or "")
    return m[-1].strip() if m else None


def extract_verdict(response: str) -> str | None:
    """'AGREE' / 'CONFLICT' from <prior_check>, or None. Drives override-precision analysis."""
    m = VERDICT_RE.search(response or "")
    return m.group(1).upper() if m else None
