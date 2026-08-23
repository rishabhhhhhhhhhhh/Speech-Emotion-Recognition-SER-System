"""Unit tests for the Plutchik similarity + ESWR reward. Run:
    .venv/bin/python src/rl/test_plutchik.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rl.plutchik import similarity, angle_between, eswr_reward, alpha_schedule, GAMMA
from data.labels import CANONICAL

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(got - want) < tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name:<44} got {got:.4f}  want {want:.4f}")

# --- geometry ---
check("angle(joy, sadness) opposites",     angle_between("joy", "sadness"), 180.0)
check("angle(anger, disgust) adjacent",    angle_between("anger", "disgust"), 45.0)
check("angle(fear, surprise) adjacent",    angle_between("fear", "surprise"), 45.0)
check("angle(anger, joy)",                 angle_between("anger", "joy"), 90.0)

# --- similarity ---
check("S(anger, anger) exact",             similarity("anger", "anger"), 1.0)
check("S(neutral, neutral) exact  [DEV 2]", similarity("neutral", "neutral"), 1.0)
check("S(neutral, anger)",                 similarity("neutral", "anger"), 0.5)
check("S(anger, disgust) 45deg",           similarity("anger", "disgust"), 0.8536, 1e-4)
check("S(joy, sadness) opposite",          similarity("joy", "sadness"), 0.0, 1e-9)
check("S(anger, fear) 180deg",             similarity("anger", "fear"), 0.0, 1e-9)

# --- reward ---
check("R exact",                           eswr_reward("anger", "anger"), 1.0)
check("R neutral exact  [DEV 2]",          eswr_reward("neutral", "neutral"), 1.0)
check("R adjacent, alpha=1",               eswr_reward("disgust", "anger", 1.0), 0.8536, 1e-4)
check("R adjacent, alpha=0.5",             eswr_reward("disgust", "anger", 0.5), 0.4268, 1e-4)
check("R 90deg (below gamma)",             eswr_reward("joy", "anger"), 0.0)
check("R neutral vs anger (S=0.5<gamma)",  eswr_reward("neutral", "anger"), 0.0)
check("R unparseable",                     eswr_reward(None, "anger"), 0.0)

# --- curriculum ---
check("alpha at step 0",                   alpha_schedule(0, 250), 1.0)
check("alpha at half",                     alpha_schedule(125, 250), 0.5)
check("alpha at end",                      alpha_schedule(250, 250), 0.0)

# --- which MELD pairs actually earn partial credit ---
print(f"\nMELD pairs clearing gamma={GAMMA} (i.e. where ESWR is denser than binary):")
found = []
for i, a in enumerate(CANONICAL):
    for b in CANONICAL[i+1:]:
        s = similarity(a, b)
        if s > GAMMA:
            found.append((a, b, s))
for a, b, s in found:
    print(f"    ({a}, {b})  S = {s:.4f}")
print(f"  -> {len(found)} pairs. joy and neutral appear in NONE: always binary.")
assert not any("joy" in (a, b) or "neutral" in (a, b) for a, b, _ in found), "joy/neutral leaked"
print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
sys.exit(0 if ok else 1)
