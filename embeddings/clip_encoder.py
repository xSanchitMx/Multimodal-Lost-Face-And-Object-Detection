import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.cpu().numpy()

def embed_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)
    return embedding.cpu().numpy()
