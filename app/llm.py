from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    low_cpu_mem_usage=True
)

model.eval()
torch.set_grad_enabled(False)

def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
