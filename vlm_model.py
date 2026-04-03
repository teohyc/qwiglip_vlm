import torch
import torch.nn as nn

class ResMLPProjector(nn.Module):
    def __init__(self, vision_hidden_size=768, llm_hidden_size=1024):
        super().__init__()

        self.input_proj = nn.Linear(vision_hidden_size, llm_hidden_size)

        self.block1 = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, llm_hidden_size * 2),
            nn.GELU(),
            nn.Linear(llm_hidden_size * 2, llm_hidden_size),
            nn.Dropout(0.1)
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.input_proj(x)

        x = x + self.block1(x)
        x = x + self.block2(x)

        return x



class SiglipQwenVLM(nn.Module):
    def __init__(self, vision_model, llm, image_token_id):
        super().__init__()
        self.vision_model = vision_model
        self.llm = llm
        self.image_token_id = image_token_id

        vision_hidden = self.vision_model.config.vision_config.hidden_size
        llm_hidden = self.llm.config.hidden_size

        self.projector = ResMLPProjector(vision_hidden, llm_hidden)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        #encode image into patch tokens
        vision_outputs = self.vision_model.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state #[B, no image patches, hidden(vm)]

        #project image features to LLM space
        projected_image_features = self.projector(image_features) #[B, no image patches, hidden(llm)]

        #get normal text embeddings from LLM
        inputs_embeds = self.llm.get_input_embeddings()(input_ids) #[B, seq_len, hidden(llm)]

        #matching dtype
        projected_image_features = projected_image_features.to(
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device
        )

        batch_size = input_ids.size(0)
        num_image_tokens = projected_image_features.size(1)

        #replace each <image> with one projected image token
        for b in range(batch_size):
            image_positions = torch.where(input_ids[b] == self.image_token_id)[0]

            if len(image_positions) != num_image_tokens:
                raise ValueError(
                    f"Sample {b}: found {len(image_positions)} <image> tokens, "
                    f"but vision encoder produced {num_image_tokens} image tokens."
                )
            
            inputs_embeds[b, image_positions, :] = projected_image_features[b]
        
        #run llm using inputs_embeds
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=60, temperature=0.7, top_p=0.9, no_repeat_ngram_size=3, repetition_penalty=1.2):
        vision_outputs = self.vision_model.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state

        projected = self.projector(image_features)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        projected = projected.to(inputs_embeds.dtype)

        for b in range(input_ids.size(0)):
            pos = torch.where(input_ids[b] == self.image_token_id)[0]
            inputs_embeds[b, pos, :] = projected[b]

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )