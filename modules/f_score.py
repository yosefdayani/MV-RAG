import torch
from PIL import Image
from torch import nn
from torchvision.transforms.v2.functional import to_pil_image
from transformers import AutoProcessor, AutoModel
from torch.nn.functional import cosine_similarity


class SimilarityModel(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_embeddings(self, images):
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            images = [Image.open(p).convert("RGB") for p in images]
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
            pass
        elif isinstance(images, torch.Tensor):
            images = [to_pil_image(img) for img in images]
        else:
            raise ValueError(f"images should be list of PIL.Image")
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def get_similarity_score(self, evaled_embs, ret_embs):
        scores = []
        for i in range(evaled_embs.size(0)):
            sim = cosine_similarity(evaled_embs[i].unsqueeze(0), ret_embs)  # shape: (k,)
            mean_sim = sim.mean().item()
            scores.append(mean_sim)
        return max(scores)
