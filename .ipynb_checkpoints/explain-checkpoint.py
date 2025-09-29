import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms

# Preprocessing pipeline for CLIP
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

def explain_semantic_match(query_path, matched_path, model_name='ViT-B/32', device='cpu'):
    """
    Explain why two images match semantically using Grad-CAM on CLIP's visual encoder.
    Displays attention heatmaps over important regions.
    """

    # Load CLIP
    model, _ = clip.load(model_name, device=device)
    model.eval()

    def process_image(path):
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        return img, tensor

    q_img, q_tensor = process_image(query_path)
    m_img, m_tensor = process_image(matched_path)

    # Get embeddings
    with torch.no_grad():
        q_embed = model.encode_image(q_tensor)
        m_embed = model.encode_image(m_tensor)

    q_embed /= q_embed.norm(dim=-1, keepdim=True)
    m_embed /= m_embed.norm(dim=-1, keepdim=True)

    similarity = torch.cosine_similarity(q_embed, m_embed).item()

    # -------------------------
    # Simple Grad-CAM for ViT patch embeddings
    # -------------------------
    q_tensor.requires_grad_(True)
    m_tensor.requires_grad_(True)

    # Forward through visual encoder
    q_features = model.visual(q_tensor)  # (1, 768)
    m_features = model.visual(m_tensor)

    # Cosine similarity score
    score = torch.cosine_similarity(q_features, m_features)
    score.backward()

    # Gradients
    q_grad = q_tensor.grad[0].cpu().numpy()  # (3, 224, 224)
    m_grad = m_tensor.grad[0].cpu().numpy()

    # Compute simple heatmaps (sum of absolute gradients across channels)
    q_heatmap = np.sum(np.abs(q_grad), axis=0)
    m_heatmap = np.sum(np.abs(m_grad), axis=0)

    # Normalize heatmaps
    q_heatmap = (q_heatmap - q_heatmap.min()) / (q_heatmap.max() - q_heatmap.min() + 1e-8)
    m_heatmap = (m_heatmap - m_heatmap.min()) / (m_heatmap.max() - m_heatmap.min() + 1e-8)

    # Overlay heatmap function
    def overlay(image, heatmap):
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        overlay_img = cv2.addWeighted(image_cv, 0.7, heatmap, 0.3, 0)
        return cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    q_overlay = overlay(q_img, q_heatmap)
    m_overlay = overlay(m_img, m_heatmap)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].imshow(q_img)
    axs[0].set_title("Query Image")
    axs[1].imshow(m_img)
    axs[1].set_title("Matched Image")
    axs[2].imshow(q_overlay)
    axs[2].set_title(f"Similarity: {similarity:.2f}")
    for ax in axs: ax.axis("off")
    plt.show()

    return {
        "similarity": similarity,
        "query_embedding": q_embed.cpu().numpy(),
        "matched_embedding": m_embed.cpu().numpy()
    }
