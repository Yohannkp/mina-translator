#!/usr/bin/env python3
"""Ultra-minimal test to find segfault source"""
import sys
print("START")

# Test 1: Import only
print("1. Import torch...")
import torch
print(f"   torch OK: {torch.__version__}")

# Test 2: CUDA check
print("2. CUDA check...")
if torch.cuda.is_available():
    print(f"   CUDA OK: {torch.cuda.get_device_name(0)}")
else:
    print("   No CUDA!")

# Test 3: Import transformers
print("3. Import transformers...")
import transformers
print(f"   transformers OK: {transformers.__version__}")

# Test 4: Import peft
print("4. Import peft...")
import peft
print(f"   peft OK: {peft.__version__}")

# Test 5: Import datasets
print("5. Import datasets...")
import datasets
print(f"   datasets OK: {datasets.__version__}")

print("ALL IMPORTS SUCCESS - now testing model load")
sys.stdout.flush()

# Test 6: Load tokenizer
print("6. Load tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
print("   Tokenizer OK")

# Test 7: Load model
print("7. Load model...")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"   Model OK: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Test 8: LoRA
print("8. Apply LoRA...")
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(r=4, target_modules=["q_proj", "k_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)
print("   LoRA OK")

# Test 9: Forward pass
print("9. Forward pass test...")
inputs = tokenizer("Bonjour", return_tensors="pt").to("cuda")
outputs = model(**inputs)
print(f"   Forward OK: {outputs.logits.shape}")

# Test 10: Backward pass (the real test)
print("10. Backward pass test...")
loss = outputs.loss
loss.backward()
print("    Backward OK!")

print("\nALL TESTS PASSED!")
