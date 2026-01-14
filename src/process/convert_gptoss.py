import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

model_id = "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
output_dir = "/mnt/shared-storage-user/ailab-hs/zouyicheng/models/gpt-oss-120b-bf16"

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

# Patch config with custom attribute before saving
model.config.attn_implementation = "eager"

model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_dir)
