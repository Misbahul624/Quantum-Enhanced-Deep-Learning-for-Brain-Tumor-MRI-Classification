"""
Inference & Prediction utility for Brain Tumor MRI Classification.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_inference_model(weights_path: str, arch: str = 'convnext_small', device='cuda'):
    from model import build_model
    model = build_model(arch, pretrained=False, dropout=0.0)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    state = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()
    return model, dev

def predict_single_image(image_path: str, model, device, img_size: int = 288, crop_brain: bool = True):
    from dataset import crop_brain_contour, get_transforms
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        pil_img = Image.open(image_path).convert('RGB')
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
    if crop_brain:
        img_bgr = crop_brain_contour(img_bgr)
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tf = get_transforms(img_size=img_size, is_training=False)
    tensor = tf(image=img_rgb)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    
    class_probs = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'all_probabilities': class_probs
    }

if __name__ == '__main__':
    import sys
    model_path = os.path.join('output', 'convnext_small_best.pth')
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        test_img = os.path.join('dataset', 'Testing', 'glioma', 'Te-gl_1.jpg')
        
    if os.path.exists(model_path) and os.path.exists(test_img):
        m, dev = load_inference_model(model_path)
        res = predict_single_image(test_img, m, dev)
        print(f"Sample: {test_img}")
        print(f"Result: {res}")
