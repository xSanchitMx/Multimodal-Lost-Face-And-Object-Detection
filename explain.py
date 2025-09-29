import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms, models

# ---------------------------
# Preprocessing
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# Grad-CAM Class
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def save_activations(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        grads = self.gradients.cpu().data.numpy()[0]
        fmap = self.activations.cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1,2))
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmap[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def overlay_cam(img: Image.Image, cam: np.ndarray):
    """
    Overlay Grad-CAM on original image
    """
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cam_resized = cv2.resize(cam, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)
    return overlay

# ---------------------------
# Main Grad-CAM function
# ---------------------------
def explain(query_path, matched_path, device='cpu'):
    # Load models
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    resnet_model.eval()
    gradcam = GradCAM(resnet_model, target_layer=resnet_model.layer4[2].conv3)

    # Load images
    q_img = Image.open(query_path).convert("RGB")
    m_img = Image.open(matched_path).convert("RGB")

    # Preprocess for Grad-CAM
    preprocess_tensor = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    q_tensor = preprocess_tensor(q_img).unsqueeze(0).to(device)
    m_tensor = preprocess_tensor(m_img).unsqueeze(0).to(device)

    # Generate Grad-CAM
    cam_q = gradcam.generate(q_tensor)
    cam_m = gradcam.generate(m_tensor)

    # Overlay heatmaps
    q_overlay = overlay_cam(q_img, cam_q)
    m_overlay = overlay_cam(m_img, cam_m)

    # Display side by side
    q_h, q_w = q_overlay.shape[:2]
    m_overlay_resized = cv2.resize(m_overlay, (int(q_w * m_overlay.shape[1]/q_overlay.shape[1]), q_h))
    combined_img = np.concatenate([q_overlay, m_overlay_resized], axis=1)
    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14,7))
    plt.imshow(combined_img)
    plt.axis('off')
    plt.title("Grad-CAM Heatmaps")
    plt.show()
