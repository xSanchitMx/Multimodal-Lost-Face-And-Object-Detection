import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# ---------------------------
# Preprocessing
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def save_activations(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        # Backprop with respect to embedding norm (to get meaningful gradients)
        target = output.norm()
        self.model.zero_grad()
        target.backward(retain_graph=True)

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
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cam_resized = cv2.resize(cam, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    return overlay

# ---------------------------
# Main function
# ---------------------------
def explain(query_path, matched_path, save_path=None, device="cpu"):
    model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    gradcam = GradCAM(model, model.last_bn)  # attach Grad-CAM to last layer

    # Load images
    q_img = Image.open(query_path).convert("RGB")
    m_img = Image.open(matched_path).convert("RGB")

    # Preprocess
    q_tensor = preprocess(q_img).unsqueeze(0).to(device)
    m_tensor = preprocess(m_img).unsqueeze(0).to(device)

    # Grad-CAM
    cam_q = gradcam.generate(q_tensor)
    cam_m = gradcam.generate(m_tensor)

    # Overlay
    q_overlay = overlay_cam(q_img, cam_q)
    m_overlay = overlay_cam(m_img, cam_m)

    # Combine
    combined_img = np.concatenate([q_overlay, m_overlay], axis=1)

    if save_path:
        cv2.imwrite(save_path, combined_img)

    return combined_img
