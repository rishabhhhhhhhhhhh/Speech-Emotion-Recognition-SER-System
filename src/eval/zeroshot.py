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
import librosa

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


def build_batch(proc, rows, sr=16000):
    convs, audios = [], []
    for r in rows:
        convs.append([{"role": "user", "content": [
            {"type": "audio", "audio_url": r["audio"]},
            {"type": "text", "text": r["prompt"]}]}])
        y, _ = librosa.load(ROOT / r["audio"], sr=sr, mono=True)
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
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
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
    t0 = time.time()
    with open(side, "a") as fh:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            inputs = build_batch(proc, chunk).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=a.max_new,
                                     do_sample=False,
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
    preds = canon_predictions(raw)
    trues = [r["label"] for r in all_rows]
    fmt = [format_reward(r["response"], r["schema"]) for r in all_rows]
    verds = [extract_verdict(r["response"]) for r in all_rows]

    ok = [(p, t) for p, t in zip(preds, trues) if p is not None]
    res = {
        "prompts": a.prompts, "adapter": a.adapter, "n": len(all_rows),
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
