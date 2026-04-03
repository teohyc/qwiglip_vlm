import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor, AutoModel, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

from vlm_model import SiglipQwenVLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

LLM_NAME = "Qwen/Qwen2-0.5B-Instruct"
VISION_NAME = "google/siglip-base-patch16-224"

#load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
processor = AutoProcessor.from_pretrained(VISION_NAME)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids("<image>")

vision_model = AutoModel.from_pretrained(VISION_NAME)
llm = Qwen2ForCausalLM.from_pretrained(LLM_NAME)
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
llm.print_trainable_parameters()

#load dataset
dataset = load_from_disk("coco_chat_dataset")

#train test split
dataset = dataset.train_test_split(test_size = 0.05, seed=42)

train_dataset = dataset["train"]
val_dataset = dataset["test"]

print(f"Datasets:\n{train_dataset[0:8]}") #check dataset

#function to convert messages into string
NUM_IMAGE_TOKENS = 196

def format_chat_with_image_tokens(messages, num_image_tokens):
    image_block = " ".join(["<image>"] * num_image_tokens)

    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            content = content.replace("<image>", image_block)
            text += f"USER: {content}\n"
        elif role == "assistant":
            text += f"ASSISTANT: {content}\n"

    return text

#label masking only train on assistant response
def create_labels(input_ids, text, tokenizer):
    labels = input_ids.clone()

    #seach starting position of assistant response
    assistant_prefix = "ASSISTANT:"
    assistant_start = text.find(assistant_prefix)

    #tokenize prefix
    prefix = text[:assistant_start + len(assistant_prefix)]
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids[0]

    prefix_len = len(prefix_ids)

    #mask prefix tokens
    labels[:prefix_len] = -100

    return labels

#image loader
def load_image(path):
    return Image.open(path).convert("RGB")

#collate function to process batch of data
def collate_fn(batch):

    images = []
    texts = []

    for sample in batch:

        #load image
        image = load_image(sample["image_path"])
        images.append(image)

        #format chat
        text = format_chat_with_image_tokens(sample["messages"], NUM_IMAGE_TOKENS)
        texts.append(text)

    #process images
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]

    #tokenize text
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    #create labels
    labels = []

    for i in range(len(texts)):
        label = create_labels(input_ids[i], texts[i], tokenizer)
        labels.append(label)

    labels = torch.stack(labels)
    image_token_mask = (input_ids == IMAGE_TOKEN_ID)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_token_mask": image_token_mask,
        "texts": texts,
    }

#define dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
)

#clear cuda memory
torch.cuda.empty_cache()

#define model
model = SiglipQwenVLM(
    vision_model=vision_model,
    llm=llm,
    image_token_id=IMAGE_TOKEN_ID,
).to(DEVICE)

#freeze the vision model
for param in model.vision_model.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

#print trainable parameters
print(f"Trainable params: {trainable:,}")
print(f"Total params: {total:,}")
print(f"Percent trainable: {100 * trainable / total:.4f}%")

#optimizer
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5,
    weight_decay=0.01,
)

#validation function
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):

            batch = {
                k: v.to(DEVICE) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k != "texts"
            }

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

    model.train()
    return avg_loss

#training loop
EPOCHS = 3
GRAD_ACCUM_STEPS = 8
MAX_GRAD_NORM = 1.0

best_val_loss = float("inf")

#record
rec_train_loss = []
rec_val_loss = []

model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    optimizer.zero_grad()

    print("Using Device:", DEVICE)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(pbar):

        batch = {
            k: v.to(DEVICE) if torch.is_tensor(v) else v
            for k, v in batch.items()
            if k != "texts"
        }
        
        outputs = model(
            pixel_values = batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        total_loss += loss.item()

        #gradient accumulation
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    rec_train_loss.append(avg_loss)
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

    val_loss = evaluate(model, val_loader)
    rec_val_loss.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss    
        torch.save(model.state_dict(), "qwiglip_vlm.pt")
        print("Best VLM saved.")
