from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer
import os

# ================= CONFIG =================
HF_USERNAME = "teohyc"   
REPO_NAME = "QwigLip-VQA"

LLM_NAME = "Qwen/Qwen2-0.5B-Instruct"

LORA_PATH = "vqa_lora_adapter"
PROJECTOR_PATH = "projector.pt"

TEMP_DIR = "hf_upload"

#prepare folder
os.makedirs(TEMP_DIR, exist_ok=True)

#projector
import shutil
shutil.copy(PROJECTOR_PATH, os.path.join(TEMP_DIR, "projector.pt"))

#LoRA adapter
shutil.copytree(LORA_PATH, os.path.join(TEMP_DIR, "lora_adapter"), dirs_exist_ok=True)

#inference script
shutil.copy("inference.py", os.path.join(TEMP_DIR, "inference.py"))

#model file
shutil.copy("vlm_model.py", os.path.join(TEMP_DIR, "vlm_model.py"))

#save tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
tokenizer.save_pretrained(os.path.join(TEMP_DIR, "tokenizer"))


#readme
readme = f"""
---
license: mit
datasets:
- phiyodr/coco2017
- landersanmi/VQAv2
language:
- en
metrics:
- accuracy
base_model:
- Qwen/Qwen2-0.5B-Instruct
- google/siglip-base-patch16-224
library_name: transformers
pipeline_tag: image-text-to-text
---

# Qwiglip VQA (Qwen2 + SigLIP)

Custom Vision-Question-Answering Model built from scratch. Built upon earlier QwigLip-VLM with additional 5k data from, but with a custom MLP projector and LoRA fine-tuning for efficient training.
Training data from https://huggingface.co/datasets/landersanmi/VQAv2 and https://huggingface.co/datasets/phiyodr/coco2017

## Components
- Base LLM: {LLM_NAME}
- Vision Encoder: SigLIP
- LoRA fine-tuning
- Custom MLP projector

## Usage 
***** CHECK OUT inference.py FOR DETAILED INFERENCE EXAMPLE *****

```python
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel, Qwen2ForCausalLM
from peft import PeftModel

from vlm_model import MLPProjector, SiglipQwenVLM

#configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LLM_NAME = "Qwen/Qwen2-0.5B-Instruct"
VISION_NAME = "google/siglip-base-patch16-224"

LORA_PATH = "vqa_lora_adapter"
PROJECTOR_PATH = "projector.pt"

NUM_IMAGE_TOKENS = 196

#refer to inference.py for full code
```"""

with open(os.path.join(TEMP_DIR, "README.md"), "w") as f:
    f.write(readme)

#create repo and upload

api = HfApi()

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

create_repo(repo_id, exist_ok=True)

print(f"Uploading to {repo_id}...")

upload_folder(
folder_path=TEMP_DIR,
repo_id=repo_id,
repo_type="model"
)

print("\nUpload complete!")
print(f"https://huggingface.co/{repo_id}")
