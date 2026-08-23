"""Tests for the prompt schemas and format-reward regexes.
    .venv/bin/python src/prompts/test_templates.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from prompts.templates import (build_prompt, format_reward, extract_answer,
                               extract_verdict, OPTIONS)

ok = True
def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name:<52} got {got!r}")

GOOD_PC = """<think>
<content> She says "let's enjoy that party" - anticipation of a celebration. </content>
<acoustics> Higher pitch, faster pace, louder volume. </acoustics>
<prior_check> CONFLICT. The prior favours anger, but the raised pitch here reads as
excitement, and the words are celebratory rather than hostile. </prior_check>
<summary> Content and prosody both point to elevated positive affect. </summary>
</think><answer> joy </answer>"""

GOOD_ESR = """<think>
<content> Neutral statement about the time. </content>
<acoustics> Flat pitch, even pace. </acoustics>
<summary> No emotional markers. </summary>
</think><answer> neutral </answer>"""

# --- format reward accepts well-formed responses ---
check("PC_ESR accepts valid PC-ESR", format_reward(GOOD_PC, "PC_ESR"), 1.0)
check("ESR accepts valid ESR", format_reward(GOOD_ESR, "ESR"), 1.0)
check("EUR accepts <think>+<answer>",
      format_reward("<think> reasoning here </think><answer> anger </answer>", "EUR"), 1.0)
check("IR accepts bare <answer>", format_reward("<answer> anger </answer>", "IR"), 1.0)

# --- the two schemas must NOT accept each other ---
check("ESR REJECTS a PC-ESR response", format_reward(GOOD_PC, "ESR"), 0.0)
check("PC_ESR REJECTS an ESR response (no prior_check)",
      format_reward(GOOD_ESR, "PC_ESR"), 0.0)

# --- malformed cases ---
check("rejects empty <content>",
      format_reward(GOOD_ESR.replace("Neutral statement about the time.", "  "), "ESR"), 0.0)
check("rejects missing </think>",
      format_reward(GOOD_ESR.replace("</think>", ""), "ESR"), 0.0)
check("rejects wrong block order", format_reward(
    "<think><acoustics> a </acoustics><content> c </content><summary> s </summary></think>"
    "<answer> joy </answer>", "ESR"), 0.0)
check("rejects trailing text after </answer>",
      format_reward(GOOD_ESR + "\nAlso I think...", "ESR"), 0.0)
check("rejects empty <answer>",
      format_reward("<answer>   </answer>", "IR"), 0.0)
check("IR rejects a full ESR response", format_reward(GOOD_ESR, "IR"), 0.0)

# --- answer + verdict extraction (independent of format reward) ---
check("extract_answer from valid", extract_answer(GOOD_PC), "joy")
check("extract_answer from MALFORMED still works",
      extract_answer("blah <answer> anger </answer> trailing"), "anger")
check("extract_answer none when absent", extract_answer("no tags here"), None)
check("extract_verdict CONFLICT", extract_verdict(GOOD_PC), "CONFLICT")
check("extract_verdict none in plain ESR", extract_verdict(GOOD_ESR), None)

# --- prompt assembly ---
p_pc = build_prompt("PC_ESR", transcript="So it was MY fault?",
                    prior_top3="anger 0.41 | disgust 0.22 | neutral 0.18",
                    gender="male", prior_ua=28.4, prior_wa=52.1)
check("PC_ESR prompt mentions prior_check", "<prior_check>" in p_pc, True)
check("PC_ESR prompt carries the reliability line", "28.4% UA / 52.1% WA" in p_pc, True)
check("PC_ESR prompt carries the transcript", "So it was MY fault?" in p_pc, True)

p_esr = build_prompt("ESR", transcript="So it was MY fault?")
check("ESR prompt has NO prior_check", "<prior_check>" in p_esr, False)
check("ESR prompt has NO prior paragraph", "Acoustic classifier prior" in p_esr, False)

try:
    build_prompt("PC_ESR", transcript="x")
    check("PC_ESR without prior raises", False, True)
except ValueError:
    check("PC_ESR without prior raises", True, True)

check("all 7 options offered", all(c in OPTIONS for c in
      ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]), True)

print("\n--- example PC-ESR prompt ---")
print(p_pc)
print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
sys.exit(0 if ok else 1)
