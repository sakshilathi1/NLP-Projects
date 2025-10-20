#!/usr/bin/env python3
# train_vanilla_sft.py
import os, re, json, inspect, torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)
from transformers.utils import is_bitsandbytes_available

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUT_DIR  = "outputs"
CKPT_DIR = "checkpoints/vanilla_sft"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ===== Prompt used only for evaluation =====
COT_PROMPT = """You are a careful math tutor. Solve the problem step by step.
Put the final answer on a new line that begins with '#### '.

{question}
"""
NUM_RE = re.compile(r"####\s*(-?\d+)")

def _auth_kwargs():
    token = os.environ.get("HF_TOKEN", None)
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if token and "token" in sig.parameters: return {"token": token}
    if token and "use_auth_token" in sig.parameters: return {"use_auth_token": token}
    return {}

def load_llama_for_training(model_id: str):
    ak = _auth_kwargs()
    use_cuda = torch.cuda.is_available()
    use_4bit = is_bitsandbytes_available(check_library_only=True) and use_cuda
    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16 if use_cuda else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto", **ak)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16 if use_cuda else torch.float32, device_map="auto", **ak)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **ak)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model.config.use_cache = False
    return model, tok

def extract_final(text: str):
    m = NUM_RE.findall(text)
    return m[-1] if m else None

@torch.inference_mode()
def batched_generate(model, tok, prompts, max_new_tokens=128, temperature=0.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    gen = model.generate(**ids, do_sample=(temperature>0.0), temperature=max(temperature,1e-5),
                         max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    trimmed = []
    for p,o in zip(prompts, outs):
        trimmed.append(o[len(p):] if o.startswith(p) else o)
    return trimmed

def eval_em_full(model, tok, outpath):
    ds = load_dataset("openai/gsm8k","main")["test"]
    em, total, rows = 0, 0, []
    for i in tqdm(range(0, len(ds), 6), desc=f"[Vanilla SFT] Eval test ({len(ds)} ex)"):
        batch = ds[i:i+6]
        Q, A = batch["question"], batch["answer"]
        prompts = [COT_PROMPT.format(question=q) for q in Q]
        gens = batched_generate(model, tok, prompts, max_new_tokens=128, temperature=0.0)
        for q,a,g in zip(Q,A,gens):
            gm = NUM_RE.search(a); gold = gm.group(1) if gm else None
            pred = extract_final(g)
            ok = (gold is not None) and (pred == gold)
            em += int(ok); total += 1
            rows.append({"question": q, "gold": gold, "pred": pred, "rationale": g.strip(), "correct": bool(ok)})
    score = em/max(1,total)
    with open(outpath, "w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(f"[Vanilla SFT] EM: {score:.4f} ({em}/{total}) -> {outpath}")
    return score

def main():
    # 1) Load model/tokenizer (QLoRA if available)
    model, tokenizer = load_llama_for_training(MODEL_ID)

    # 2) Prepare GSM8K train for LM (train on gold rationale+answer text)
    train_full = load_dataset("openai/gsm8k","main")["train"]
    def fmt(ex):
        q = ex["question"].strip()
        a = ex["answer"].strip()
        text = f"### Question:\n{q}\n\n### Answer:\n{a}\n"
        t = tokenizer(text, truncation=True, max_length=768)
        t["labels"] = t["input_ids"].copy()
        return t
    train_proc = train_full.map(fmt, remove_columns=train_full.column_names, desc="[Vanilla SFT] Tokenizing")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 3) Training args (epoch across full train)
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_flag = torch.cuda.is_available() and not supports_bf16
    bf16_flag = supports_bf16

    args = TrainingArguments(
        output_dir = CKPT_DIR,
        num_train_epochs = 1,                 # increase to 2 if you have time
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        learning_rate = 2e-4,
        warmup_ratio = 0.03,
        lr_scheduler_type = "cosine",
        logging_steps = 50,
        save_steps = 1000,
        save_total_limit = 2,
        fp16 = fp16_flag,
        bf16 = bf16_flag,
        optim = "paged_adamw_8bit" if is_bitsandbytes_available() else "adamw_torch",
        report_to = [],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_proc, data_collator=collator)
    print("[Vanilla SFT] Training...")
    trainer.train()
    model.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)
    print("Saved Vanilla SFT checkpoint:", CKPT_DIR)

    # 4) Evaluate on FULL test set
    outpath = f"{OUT_DIR}/vanilla_test_full.jsonl"
    eval_em_full(model, tokenizer, outpath)

if __name__ == "__main__":
    main()
