"""Zero-shot / post-training evaluation of Qwen2-Audio on a prompt jsonl. RUNS ON THE POD.

    python src/eval/zeroshot.py --prompts artifacts/prompts/meld_dev1k_true.jsonl \
        --out artifacts/results/zs_dev1k_true.json [--adapter artifacts/rl/lora_grpo_final]

Deliberately uses HF `generate` with left-padded batching, NOT vLLM. vLLM hard-pins torch
and would rewrite the pod image's CUDA stack; the extra ~$1.50 of GPU time is far cheaper
than a broken environment mid-session. Likewise `attn_implementation="sdpa"` rather than
flash-attn, which would cost 30-60 min of paid compute to compile.

Every row is written incrementally to a .jsonl sidecar so a crash or a pod eviction never
loses completed work - resume just skips uids already present.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from prompts.templates import format_reward, extract_answer, extract_verdict
from eval.metrics import (core_metrics, eswr_soft_accuracy, format_compliance,
                          parse_rate, canon_predictions, confusion, format_confusion)

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


def load_model(adapter: str | None):
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"loaded LoRA adapter: {adapter}")
    model.eval()
    proc.tokenizer.padding_side = "left"          # required for batched generate
    return proc, model


def generation_controls(proc):
    """Stop the model opening a NEW chat turn instead of answering.

    Diagnosed at token level: for many clips the model generated exactly
    [<|im_start|>, \n, <|endoftext|>] and nothing else - an empty response. The one-shot
    example inside the user message ends with `<answer> anger </answer>`, which reads as a
    COMPLETED exchange, so the model politely starts the next turn instead of replying.

    Banning <|im_start|> removes that escape route, and eos is pinned to <|im_end|> (the
    chat turn terminator) rather than whatever generate() infers.
    """
    tok = proc.tokenizer
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    bad = [[i] for i in (im_start,) if i is not None and i >= 0]
    eos = [i for i in (im_end, tok.eos_token_id) if i is not None and i >= 0]
    return bad, sorted(set(eos))


def load_audio(path, target_sr=16000):
    """Read a MELD FLAC as 16 kHz mono float32.

    Uses `soundfile` rather than `librosa` on purpose: MELD ships at exactly 16 kHz
    (verified in the manifests), so no resampling is needed, and dropping librosa
    removes numba + llvmlite from the pod's dependency chain entirely - a whole class
    of Python-version breakage avoided (numba 0.60 caps at Python 3.12, and the
    ubuntu2404 image may ship 3.12 or 3.13).
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:                       # defensive: should never fire on MELD
        import math
        idx = np.linspace(0, len(y) - 1, int(math.ceil(len(y) * target_sr / sr)))
        y = np.interp(idx, np.arange(len(y)), y).astype(np.float32)
    return y


def strip_transcript(prompt: str) -> str:
    """Remove the 'Transcript: "..."' line, leaving prior + question + instructions."""
    return "\n\n".join(b for b in prompt.split("\n\n")
                        if not b.lstrip().startswith("Transcript:"))


def build_batch(proc, rows, sr=16000, silent=False, no_transcript=False):
    convs, audios = [], []
    for r in rows:
        text = strip_transcript(r["prompt"]) if no_transcript else r["prompt"]
        convs.append([{"role": "user", "content": [
            {"type": "audio", "audio_url": r["audio"]},
            {"type": "text", "text": text}]}])
        y = load_audio(ROOT / r["audio"], sr)
        if silent:
            y = np.zeros_like(y)          # same length, no signal
        audios.append(y)
    texts = [proc.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
             for c in convs]
    return proc(text=texts, audio=audios, sampling_rate=sr,
                return_tensors="pt", padding=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=700)  # ESR needs ~600; 400 truncated mid-<prior_check>
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--silent-audio", action="store_true",
                    help="ABLATION: replace the waveform with zeros, keep everything else. "
                         "If accuracy is unchanged, the model is NOT using the audio and we "
                         "are measuring transcript-only performance.")
    ap.add_argument("--no-transcript", action="store_true",
                    help="ABLATION: strip the Transcript line from the prompt, keep the audio. "
                         "Measures what the model gets from sound alone.")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / a.prompts)]
    if a.limit:
        rows = rows[:a.limit]

    side = pathlib.Path(str(ROOT / a.out).replace(".json", "_raw.jsonl"))
    side.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if side.exists():
        done = {json.loads(l)["uid"] for l in open(side)}
        print(f"resuming: {len(done)} rows already generated")
    todo = [r for r in rows if r["uid"] not in done]

    proc, model = load_model(a.adapter)
    bad_words, eos_ids = generation_controls(proc)
    print(f"generation: bad_words={bad_words}  eos_ids={eos_ids}  max_new={a.max_new}")
    t0 = time.time()
    with open(side, "a") as fh:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            inputs = build_batch(proc, chunk, silent=a.silent_audio,
                                 no_transcript=a.no_transcript).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=a.max_new,
                                     do_sample=False,
                                     bad_words_ids=bad_words,
                                     eos_token_id=eos_ids,
                                     pad_token_id=proc.tokenizer.pad_token_id)
            gen = out[:, inputs["input_ids"].shape[1]:]
            texts = proc.batch_decode(gen, skip_special_tokens=True)
            for r, resp in zip(chunk, texts):
                fh.write(json.dumps({**r, "response": resp}) + "\n")
            fh.flush()
            n = i + len(chunk)
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el:.0f}s  {el/max(n,1):.2f}s/item  "
                  f"eta {(len(todo)-n)*el/max(n,1)/60:.1f}min", flush=True)

    # ---- score ----
    all_rows = [json.loads(l) for l in open(side)]
    keep = {r["uid"] for r in rows}
    all_rows = [r for r in all_rows if r["uid"] in keep]

    raw = [extract_answer(r["response"]) for r in all_rows]
    preds = canon_predictions(raw, [r["response"] for r in all_rows])
    trues = [r["label"] for r in all_rows]
    fmt = [format_reward(r["response"], r["schema"]) for r in all_rows]
    verds = [extract_verdict(r["response"]) for r in all_rows]

    ok = [(p, t) for p, t in zip(preds, trues) if p is not None]
    res = {
        "prompts": a.prompts, "adapter": a.adapter, "n": len(all_rows),
        "silent_audio": a.silent_audio, "no_transcript": a.no_transcript,
        **{k: round(v, 2) for k, v in core_metrics([t for _, t in ok], [p for p, _ in ok]).items()},
        "eswr_soft": round(eswr_soft_accuracy(preds, trues), 2),
        "format_compliance": round(format_compliance(fmt), 2),
        "parse_rate": round(parse_rate(preds), 2),
        "verdict_counts": {v: verds.count(v) for v in set(verds)},
    }
    print("\n" + json.dumps(res, indent=2))
    print("\n" + format_confusion(confusion(trues, [p or "neutral" for p in preds])))
    json.dump(res, open(ROOT / a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
