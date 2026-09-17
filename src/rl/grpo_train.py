"""GRPO fine-tuning of Qwen2-Audio-7B-Instruct with ESWR + format rewards. RUNS ON THE POD.

    python src/rl/grpo_train.py --prompts artifacts/prompts/meld_train_train.jsonl \
        --out artifacts/rl/lora_grpo --steps 250

WHAT THIS TESTS
---------------
Zero-shot measurement (session 1, 1108 MELD dev clips) found:
    IR 48.52 UA > EUR 49.05 > ESR 33.71 > PC-ESR 26.81
i.e. structure HURTS an untrained model, exactly INVERTED from Paper B's post-RL
ordering (ESR > EUR > IR). And the injected CNN prior was ignored outright:
true prior 26.81 vs SHUFFLED prior 26.72 - 0.09 apart, pure noise.

So this run asks two things at once, from ONE training run:
  1. Does RL flip the ordering back? If yes, Paper B's gain is not "structure helps"
     but "RL teaches a model to exploit structure it otherwise cannot use".
  2. Does RL teach the model to USE the prior? The training mix is 65% true /
     20% absent / 15% corrupted prior (built offline in build_dataset.py), with the
     TRUE label as target in all three cases. That is what would teach calibrated
     trust rather than either blind copying or blanket ignoring.

Evaluating the trained model on true / none / shuffled answers (2) directly.

DESIGN NOTES
------------
* Audio encoder + multimodal projector are FROZEN. Only the LM gets LoRA. Saves ~8 GB of
  optimiser/activation memory and stops the acoustic front-end drifting in 250 steps.
* Reference model = the policy with the adapter disabled, so no second 17 GB copy in VRAM.
* No vLLM: it hard-pins torch and would rewrite the pod's CUDA stack.
* bf16 throughout; A40 is Ampere so bf16 is native.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from prompts.templates import format_reward, extract_answer
from data.labels import canonicalize
from rl.plutchik import eswr_reward, alpha_schedule

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

# Set by main() so the reward functions can read the annealing position.
_STATE = {"step": 0, "total": 250}


def load_audio(path, target_sr=16000):
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


# --------------------------------------------------------------------- rewards
def reward_format(completions, schema, **kwargs):
    """Binary structural compliance against the schema THIS prompt requested.

    Paper B section 3.3.1. Note the prompt mix contains two schemas (PC_ESR when a prior
    is shown, ESR when it is not), so the reward must be scored per-row against the
    requested one - scoring everything against PC_ESR would punish the 20% no-prior rows
    for correctly omitting <prior_check>.
    """
    out = []
    for c, sc in zip(completions, schema):
        text = c[0]["content"] if isinstance(c, list) else c
        out.append(format_reward(text, sc))
    return out


def reward_eswr(completions, label, **kwargs):
    """Emotion Similarity-Weighted Reward with the alpha curriculum (Paper B eq. 8)."""
    alpha = alpha_schedule(_STATE["step"], _STATE["total"])
    out = []
    for c, gold in zip(completions, label):
        text = c[0]["content"] if isinstance(c, list) else c
        pred = canonicalize(extract_answer(text) or "")
        out.append(eswr_reward(pred, gold, alpha=alpha))
    return out


class AlphaCallback:
    """Advances the ESWR alpha curriculum; TRL calls this each step."""

    def __init__(self, total):
        _STATE["total"] = total

    def on_step_end(self, args, state, control, **kwargs):
        _STATE["step"] = int(state.global_step)
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="artifacts/rl/lora_grpo")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--num-generations", type=int, default=4)   # paper used 6
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)           # LoRA needs >> paper's 1e-6
    ap.add_argument("--beta", type=float, default=0.04)         # KL coefficient
    ap.add_argument("--max-completion", type=int, default=700)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    rows = [json.loads(l) for l in open(ROOT / a.prompts)]
    if a.limit:
        rows = rows[:a.limit]
    print(f"train rows: {len(rows)}")
    from collections import Counter
    print("  prior mix:", dict(Counter(r["prior_condition"] for r in rows)))
    print("  schemas  :", dict(Counter(r["schema"] for r in rows)))

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    proc.tokenizer.padding_side = "left"

    def to_example(r):
        return {
            "prompt": [{"role": "user", "content": [
                {"type": "audio", "audio_url": r["audio"]},
                {"type": "text", "text": r["prompt"]}]}],
            "audio": load_audio(ROOT / r["audio"]),
            "label": r["label"],
            "schema": r["schema"],
        }

    ds = Dataset.from_list([to_example(r) for r in rows])

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")

    # FREEZE the audio tower + projector: only the language model is trained.
    frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("audio_tower") or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    print(f"frozen audio encoder + projector: {frozen/1e6:.1f}M params")

    peft_cfg = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = GRPOConfig(
        output_dir=str(ROOT / a.out),
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_generations=a.num_generations,
        max_completion_length=a.max_completion,
        learning_rate=a.lr,
        beta=a.beta,
        temperature=1.0,
        max_steps=a.steps,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,   # we need `label` and `schema` in the reward fns
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=proc,
        reward_funcs=[reward_format, reward_eswr],
        args=cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )
    trainer.add_callback(AlphaCallback(a.steps))
    print("starting GRPO ...", flush=True)
    trainer.train()
    trainer.save_model(str(ROOT / a.out / "final"))
    print(f"saved -> {a.out}/final")


if __name__ == "__main__":
    main()
