import math
import os
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".hf_cache"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test():
    df = pd.read_parquet("dataset_output/parquet/sft_dialogues.parquet")
    sample_texts = []
    for _, row in df.sample(5, random_state=42).iterrows():
        msgs = row.get("messages", [])
        text = "\n".join([f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}" for m in msgs])
        sample_texts.append(text)

    print(f"Loaded {len(sample_texts)} samples.")
    base_name = "EleutherAI/pythia-70m"
    adapter_path = "lora_adapters/pythia_70m"

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16, device_map="cuda")
    base_model.eval()

    base_losses = []
    with torch.no_grad():
        for t in sample_texts:
            enc = tok(t, return_tensors="pt", max_length=256, truncation=True).input_ids.to(base_model.device)
            if enc.shape[1] < 4:
                continue
            loss = base_model(enc, labels=enc).loss.item()
            base_losses.append(loss)

    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_model.eval()

    lora_losses = []
    with torch.no_grad():
        for t in sample_texts:
            enc = tok(t, return_tensors="pt", max_length=256, truncation=True).input_ids.to(lora_model.device)
            if enc.shape[1] < 4:
                continue
            loss = lora_model(enc, labels=enc).loss.item()
            lora_losses.append(loss)

    base_ppl = [math.exp(x) for x in base_losses]
    lora_ppl = [math.exp(x) for x in lora_losses]
    print("Base losses:", [round(x, 4) for x in base_losses])
    print("LoRA losses:", [round(x, 4) for x in lora_losses])
    print("Base PPLs:", [round(x, 2) for x in base_ppl])
    print("LoRA PPLs:", [round(x, 2) for x in lora_ppl])
    print("Base Mean PPL:", round(sum(base_ppl)/len(base_ppl), 2))
    print("LoRA Mean PPL:", round(sum(lora_ppl)/len(lora_ppl), 2))

if __name__ == "__main__":
    test()
