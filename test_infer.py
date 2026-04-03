import torch
import random
from datasets import load_from_disk
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoProcessor, AutoModel, Qwen2ForCausalLM
from peft import PeftModel

from vlm_model import ResMLPProjector, SiglipQwenVLM

#configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "coco_chat_dataset"

LLM_NAME = "Qwen/Qwen2-0.5B-Instruct"
VISION_NAME = "google/siglip-base-patch16-224"

LORA_PATH = "lora_adapter"
PROJECTOR_PATH = "projector.pt"

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

#load lora adapter
llm = PeftModel.from_pretrained(llm, LORA_PATH)

#load projector
projector = ResMLPProjector(vision_model.config.vision_config.hidden_size, llm.config.hidden_size)  
projector.load_state_dict(torch.load(PROJECTOR_PATH, map_location=DEVICE))
projector.to(DEVICE)

#assemble QwigLip VLM
model = SiglipQwenVLM(vision_model, llm, IMAGE_TOKEN_ID).to(DEVICE)
model.projector = projector
model.eval()


#load test data
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
val_dataset = dataset["test"]

sample = random.choice(val_dataset)

print("Caption:", sample["messages"][1]["content"])
print("Image Path:", sample["image_path"])

image = Image.open(sample["image_path"]).convert("RGB")

#show image
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()


#input preparation
image_block = " ".join(["<image>"] * NUM_IMAGE_TOKENS)

prompt = (
    f"USER: {image_block}\n"
    "Describe the image in 2-3 short sentences. "
    "Be accurate and only mention clearly visible details.\n"
    "ASSISTANT:"
)

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
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#cleaning output
sentences = output_text.split(".")
output_text = ".".join(sentences[:3]).strip() + "."

print("\n=== Generated Caption ===")
print(output_text)