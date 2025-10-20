#!/usr/bin/env python3
# zero_shot_cot.py
import os, re, json, inspect, torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Prompts (include in your report) =====
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

def load_llama(model_id: str):
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
    for i in tqdm(range(0, len(ds), 6), desc=f"Eval test ({len(ds)} ex)"):
        batch = ds[i:i+6]
        Q, A = batch["question"], batch["answer"]
        prompts = [COT_PROMPT.format(question=q) for q in Q]
        gens = batched_generate(model, tok, prompts, max_new_tokens=128, temperature=0.0)
        for q,a,g in zip(Q,A,gens):
            gold_m = NUM_RE.search(a); gold = gold_m.group(1) if gold_m else None
            pred = extract_final(g)
            ok = (gold is not None) and (pred == gold)
            em += int(ok); total += 1
            rows.append({"question": q, "gold": gold, "pred": pred, "rationale": g.strip(), "correct": bool(ok)})
    score = em/max(1,total)
    with open(outpath, "w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(f"[Zero-Shot CoT] EM: {score:.4f} ({em}/{total}) -> {outpath}")
    return score

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, tok = load_llama(MODEL_ID)
    outpath = f"{OUT_DIR}/baseline_test_full.jsonl"
    eval_em_full(model, tok, outpath)

if __name__ == "__main__":
    main()
