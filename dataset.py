import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}


def crop_brain_contour(image_bgr: np.ndarray, pad: int = 6) -> np.ndarray:
    """
    Locates the brain region within an MRI slice by thresholding out the black
    scanner background and cropping to the tight bounding box with padding.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 18, 255, cv2.THRESH_BINARY)
    
    # Clean up small noise and close gaps in skull boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_bgr

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Sanity check: Ensure detected bounding box covers at least 15% of image area
    h_orig, w_orig = image_bgr.shape[:2]
    if (w * h) < 0.15 * (w_orig * h_orig):
        return image_bgr

    x_min = max(0, x - pad)
    y_min = max(0, y - pad)
    x_max = min(w_orig, x + w + pad)
    y_max = min(h_orig, y + h + pad)

    cropped = image_bgr[y_min:y_max, x_min:x_max]
    return cropped if cropped.size > 0 else image_bgr


def get_transforms(img_size: int = 288, is_training: bool = True):
    """
    Returns high-performance Albumentations transform pipelines.
    Carefully designed for MRI scans:
    - Horizontal flips (anatomically valid brain bilateral symmetry)
    - Gentle rotation & scaling (MRIs have standard canonical axes, but slight tilt exists)
    - Brightness & contrast shifts (accounts for 1.5T vs 3.0T scanner differences)
    - Cutout/CoarseDropout (prevents overfitting to local scanner artifacts)
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if is_training:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            # Albumentations 2.x: use Affine instead of ShiftScaleRotate
            A.Affine(
                translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                scale=(0.92, 1.08),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.GaussNoise(std_range=(0.03, 0.12), p=0.5),  # v2: std_range instead of var_limit
            ], p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(int(img_size * 0.04), int(img_size * 0.12)),
                hole_width_range=(int(img_size * 0.04), int(img_size * 0.12)),
                fill=0,
                p=0.4
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


class BrainTumorMRIDataset(Dataset):
    def __init__(self, file_paths, labels, img_size=288, is_training=True, crop_brain=True):
        self.file_paths = file_paths
        self.labels = labels
        self.crop_brain = crop_brain
        self.transform = get_transforms(img_size=img_size, is_training=is_training)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Load image via OpenCV
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Fallback for rare color/palette types via PIL
            pil_img = Image.open(path).convert('RGB')
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if self.crop_brain:
            img_bgr = crop_brain_contour(img_bgr)

        # Convert to RGB for torchvision / timm models
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply augmentations / normalization
        augmented = self.transform(image=img_rgb)
        tensor_img = augmented['image']

        return tensor_img, torch.tensor(label, dtype=torch.long)
