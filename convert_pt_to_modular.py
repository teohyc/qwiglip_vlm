import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from vlm_model import SiglipQwenVLM

#configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "qwiglip_vqa.pt"

LLM_NAME = "Qwen/Qwen2-0.5B-Instruct"
VISION_NAME = "google/siglip-base-patch16-224"

OUTPUT_LORA_DIR = "vqa_lora_adapter"
OUTPUT_PROJECTOR_PATH = "projector.pt"

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids("<image>")

#load models
vision_model = AutoModel.from_pretrained(VISION_NAME).to(DEVICE)

llm = Qwen2ForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)
llm.resize_token_embeddings(len(tokenizer))

# lora
print("Rebuilding LoRA structure...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

llm = get_peft_model(llm, lora_config)

#load model
model = SiglipQwenVLM(vision_model, llm, IMAGE_TOKEN_ID).to(DEVICE)

#load state dict
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.eval()

'''
#extract projector
print("Saving projector")
torch.save(model.projector.state_dict(), OUTPUT_PROJECTOR_PATH)
'''

#extract lora
print("Saving LoRA adapter...")
model.llm.save_pretrained(OUTPUT_LORA_DIR)

print("\nConversion complete!")
print(f"LoRA saved to: {OUTPUT_LORA_DIR}/")
print(f"Projector saved to: {OUTPUT_PROJECTOR_PATH}")
