import torch
import torch.nn as nn
import random
from datasets import load_from_disk
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoProcessor, AutoModel, Qwen2ForCausalLM
from vlm_model import MLPProjector, SiglipQwenVLM
from peft import LoraConfig, get_peft_model


#configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "coco_chat_dataset"
MODEL_PATH = "qwiglip_vlm.pt"

LLM_NAME = "Qwen/Qwen2-0.5B-Instruct"
VISION_NAME = "google/siglip-base-patch16-224"

NUM_IMAGE_TOKENS = 196

#load tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
processor = AutoProcessor.from_pretrained(VISION_NAME)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids("<image>")

#load models
vision_model = AutoModel.from_pretrained(VISION_NAME).to(DEVICE)
llm = Qwen2ForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)
llm.resize_token_embeddings(len(tokenizer))

#attach lora
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
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
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

#load data
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
val_dataset = dataset["test"]

sample = random.choice(val_dataset)
#print caption and image path
print("Caption:", sample["messages"][1]["content"])
print("Image Path:", sample["image_path"])

image = Image.open(sample["image_path"]).convert("RGB")

# show image
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()

#input preparation
image_block = " ".join(["<image>"] * NUM_IMAGE_TOKENS)

prompt = f"USER: {image_block}\nDescribe the image in 2-3 short sentences. Be accurate and only mention clearly visible details.\nASSISTANT:"

inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(DEVICE)

tokenized = tokenizer(prompt, return_tensors="pt").to(DEVICE)

#generate
with torch.no_grad():
    output_ids = model.generate(
        pixel_values=pixel_values,
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        max_new_tokens=60,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#only keep the first 3 sentences
output_text = output_text.split(".")[:3]
output_text = ".".join(output_text) + "."

print("\n=== Generated Caption ===")
print(output_text)