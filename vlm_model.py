import torch
import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, vision_hidden_size, llm_hidden_size):
        super().__init__()
        self.pre_norm = nn.LayerNorm(vision_hidden_size)
        self.net = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.post_norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, x):
        return self.post_norm(self.net(self.pre_norm(x)))


# ===== VLM MODEL =====
class SiglipQwenVLM(nn.Module):
    def __init__(self, vision_model, llm, image_token_id):
        super().__init__()
        self.vision_model = vision_model
        self.llm = llm
        self.image_token_id = image_token_id

        vision_hidden = self.vision_model.config.vision_config.hidden_size
        llm_hidden = self.llm.config.hidden_size

        self.projector = MLPProjector(vision_hidden, llm_hidden)

    def forward(self, pixel_values, input_ids, attention_mask=None):
        vision_outputs = self.vision_model.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state

        projected = self.projector(image_features)

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        projected = projected.to(inputs_embeds.dtype)

        for b in range(input_ids.size(0)):
            pos = torch.where(input_ids[b] == self.image_token_id)[0]
            inputs_embeds[b, pos, :] = projected[b]

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        return outputs

    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=50):
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
        )