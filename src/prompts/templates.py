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
</think><answer> the selected emotion </answer>"""

# ----------------------------------------------------------------------- PC-ESR
PC_ESR_INSTRUCTION = """Please think step by step. Your reply must strictly follow this format:
<think>
<content> recognise and analyse the speaker's speech content and any emotionally salient words </content>
<acoustics> analyse the speaker's acoustic features: pitch, rhythm, speed, volume, voice quality </acoustics>
<prior_check> state AGREE or CONFLICT with the acoustic classifier prior, and explain why </prior_check>
<summary> summarise all views </summary>
</think><answer> the selected emotion </answer>"""

PRIOR_BLOCK = """Acoustic classifier prior - a CNN over the mel spectrogram, scoring {ua:.1f}% UA / {wa:.1f}% WA on this corpus when used alone. Treat this as evidence, not as the answer:
  {top3}   (speaker: {gender})"""

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
    return "\n\n".join(parts)


# ------------------------------------------------------- format-reward regexes
# Binary scoring, Paper B section 3.3.1: exact structural match -> 1, else 0.
# DOTALL so reasoning may span lines; each block must be NON-EMPTY (\S) so the model
# cannot satisfy the schema with empty tags.
_A = r"<answer>\s*(?P<answer>[^<>]*?\S[^<>]*?)\s*</answer>"

FORMAT_PATTERNS = {
    "IR": re.compile(rf"^\s*{_A}\s*$", re.DOTALL),
    "EUR": re.compile(rf"^\s*<think>\s*(?=[\s\S]*?\S)[\s\S]*?</think>\s*{_A}\s*$", re.DOTALL),
    "ESR": re.compile(
        r"^\s*<think>\s*"
        r"<content>\s*[\s\S]*?\S[\s\S]*?</content>\s*"
        r"<acoustics>\s*[\s\S]*?\S[\s\S]*?</acoustics>\s*"
        r"<summary>\s*[\s\S]*?\S[\s\S]*?</summary>\s*"
        rf"</think>\s*{_A}\s*$", re.DOTALL),
    "PC_ESR": re.compile(
        r"^\s*<think>\s*"
        r"<content>\s*[\s\S]*?\S[\s\S]*?</content>\s*"
        r"<acoustics>\s*[\s\S]*?\S[\s\S]*?</acoustics>\s*"
        r"<prior_check>\s*[\s\S]*?\S[\s\S]*?</prior_check>\s*"
        r"<summary>\s*[\s\S]*?\S[\s\S]*?</summary>\s*"
        rf"</think>\s*{_A}\s*$", re.DOTALL),
}

# Lenient answer extraction for ACCURACY scoring - deliberately separate from the format
# reward. A response may be malformed (format reward 0) yet still contain a correct answer,
# and the two rewards must be able to disagree; collapsing them would hide that signal.
ANSWER_RE = re.compile(r"<answer>\s*([^<>]+?)\s*</answer>", re.DOTALL | re.IGNORECASE)

VERDICT_RE = re.compile(r"<prior_check>\s*[\s\S]*?\b(AGREE|CONFLICT)\b", re.IGNORECASE)


def format_reward(response: str, mode: str) -> float:
    """1.0 iff `response` exactly matches the schema `mode` requested."""
    return 1.0 if FORMAT_PATTERNS[mode.upper()].match(response or "") else 0.0


def extract_answer(response: str) -> str | None:
    """Last <answer> block, raw text. None if absent."""
    m = ANSWER_RE.findall(response or "")
    return m[-1].strip() if m else None


def extract_verdict(response: str) -> str | None:
    """'AGREE' / 'CONFLICT' from <prior_check>, or None. Drives override-precision analysis."""
    m = VERDICT_RE.search(response or "")
    return m.group(1).upper() if m else None
