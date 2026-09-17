"""Stage 1 of RFT: sample k completions per training prompt and score them. RUNS ON POD.

    python src/rl/rft_generate.py --prompts artifacts/prompts/meld_train_train.jsonl \
        --out artifacts/rl/rft_pool.jsonl --n 1200 --k 4

WHY RFT AND NOT GRPO
--------------------
Measured on the pod: TRL 1.13.0's GRPOTrainer has NO audio support at all
(`grep audio` -> 0 hits, `input_features` -> 0 hits, while `image` -> 294 hits; vision
LMs are supported, audio LMs are not). Paper B built on a Qwen2-Audio-specific fork.
Rejection-sampling fine-tuning reaches the same questions with code that already works,
and Paper B itself cites RFT (Yuan et al. 2023) as a legitimate training baseline.

ESWR IS PRESERVED, NOT DROPPED
------------------------------
Plain RFT keeps only exactly-correct samples. That would throw away Paper B's Emotion
Similarity-Weighted Reward entirely. Instead each kept sample carries a WEIGHT equal to
its total reward (format + ESWR), so a Plutchik-adjacent answer (e.g. predicting disgust
for anger, 45 degrees apart, S=0.854) still contributes, at reduced strength. That is a
direct translation of eq. 8 into a supervised regime.

Sampling is stochastic (temperature 1.0) precisely so the k completions differ - greedy
decoding would give k identical samples and there would be nothing to select over.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from prompts.templates import format_reward, extract_answer
from data.labels import canonicalize
from rl.plutchik import eswr_reward

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


def load_audio(path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return y.mean(axis=1) if y.ndim > 1 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="artifacts/rl/rft_pool.jsonl")
    ap.add_argument("--n", type=int, default=1200, help="training prompts to sample from")
    ap.add_argument("--k", type=int, default=4, help="completions per prompt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    rows = [json.loads(l) for l in open(ROOT / a.prompts)]
    rng = np.random.default_rng(a.seed)
    # Stratify by prior_condition so the 65/20/15 mix survives subsampling - the whole
    # point of the mix is to teach when to trust, ignore, or override the prior.
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["prior_condition"], []).append(r)
    picked = []
    for cond, grp in by_cond.items():
        take = max(1, round(a.n * len(grp) / len(rows)))
        idx = rng.choice(len(grp), size=min(take, len(grp)), replace=False)
        picked += [grp[i] for i in idx]
    rng.shuffle(picked)
    from collections import Counter
    print(f"sampling {len(picked)} prompts x k={a.k}  mix={dict(Counter(r['prior_condition'] for r in picked))}")

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    proc.tokenizer.padding_side = "left"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    im_start = proc.tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")

    outp = ROOT / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        done = {json.loads(l)["uid"] for l in open(outp)}
        print(f"resuming: {len(done)} prompts already sampled")
    todo = [r for r in picked if r["uid"] not in done]

    t0, kept, total = time.time(), 0, 0
    with open(outp, "a") as fh:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            convs = [[{"role": "user", "content": [
                {"type": "audio", "audio_url": r["audio"]},
                {"type": "text", "text": r["prompt"]}]}] for r in chunk]
            texts = [proc.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                     for c in convs]
            audios = [load_audio(ROOT / r["audio"]) for r in chunk]
            inputs = proc(text=texts, audio=audios, sampling_rate=16000,
                          return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=a.max_new, do_sample=True,
                    temperature=a.temperature, top_p=0.95,
                    num_return_sequences=a.k,
                    bad_words_ids=[[im_start]], eos_token_id=[im_end],
                    pad_token_id=proc.tokenizer.pad_token_id)
            gen = out[:, inputs["input_ids"].shape[1]:]
            decoded = proc.batch_decode(gen, skip_special_tokens=True)

            for j, r in enumerate(chunk):
                cands = decoded[j * a.k:(j + 1) * a.k]
                scored = []
                for c in cands:
                    fr = format_reward(c, r["schema"])
                    pred = canonicalize(extract_answer(c) or "")
                    er = eswr_reward(pred, r["label"], alpha=1.0)
                    scored.append({"text": c, "format": fr, "eswr": er,
                                   "reward": fr + er, "pred": pred})
                total += len(scored)
                kept += sum(1 for s in scored if s["reward"] > 0)
                fh.write(json.dumps({**r, "candidates": scored}) + "\n")
            fh.flush()
            n = i + len(chunk)
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el:.0f}s  {el/max(n,1):.2f}s/prompt  "
                  f"kept {kept}/{total} ({100*kept/max(total,1):.1f}%)  "
                  f"eta {(len(todo)-n)*el/max(n,1)/60:.1f}min", flush=True)

    print(f"\npool -> {a.out}   usable candidates {kept}/{total} ({100*kept/max(total,1):.1f}%)")


if __name__ == "__main__":
    main()
