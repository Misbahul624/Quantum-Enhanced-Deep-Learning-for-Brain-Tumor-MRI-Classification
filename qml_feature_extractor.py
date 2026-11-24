"""
Step 2 & 3 from Diagram: Classical Feature Extraction & Reduction
Extracts deep features using ResNet18 and EfficientNet backbones, then saves
high-dimensional features along with PCA/linear projections for Quantum Neural Networks (QNN).
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BrainTumorMRIDataset, CLASSES, CLASS_TO_IDX

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collect_paths_and_labels(split_root):
    paths, labels = [], []
    for cls_name in CLASSES:
        cls_dir = os.path.join(split_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(CLASS_TO_IDX[cls_name])
    return paths, labels

def extract_features(model_name: str = 'resnet18', img_size: int = 224, batch_size: int = 64):
    print(f"[*] Initializing feature extractor: {model_name} (Resolution: {img_size}x{img_size})...")
    # Using timm feature extractor with num_classes=0 gives global pooled feature vectors
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.to(DEVICE).eval()

    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    train_paths, train_labels = collect_paths_and_labels(os.path.join(data_root, 'Training'))
    test_paths, test_labels = collect_paths_and_labels(os.path.join(data_root, 'Testing'))

    train_ds = BrainTumorMRIDataset(train_paths, train_labels, img_size=img_size, is_training=False, crop_brain=True)
    test_ds = BrainTumorMRIDataset(test_paths, test_labels, img_size=img_size, is_training=False, crop_brain=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    def run_extractor(loader):
        feats_list = []
        labels_list = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    feats = model(imgs)
                feats_list.append(feats.cpu().numpy())
                labels_list.append(lbls.numpy())
        return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)

    print("[*] Extracting training features (5600 images)...")
    X_train, y_train = run_extractor(train_loader)
    print(f"    Train feature shape: {X_train.shape}")

    print("[*] Extracting testing features (1600 images)...")
    X_test, y_test = run_extractor(test_loader)
    print(f"    Test feature shape: {X_test.shape}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output_qml')
    os.makedirs(output_dir, exist_ok=True)

    # Dimensionality Reduction: PCA to 8 features (for 8-qubit variational circuit)
    print("[*] Performing PCA dimensionality reduction (512 -> 8 features for 8 qubits)...")
    pca = PCA(n_components=8, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"    Explained variance ratio sum (8 features): {np.sum(pca.explained_variance_ratio_):.4f}")

    # Min-max scale to [0, pi] for angle embedding in quantum gates (Ry rotations)
    min_val = X_train_pca.min(axis=0)
    max_val = X_train_pca.max(axis=0)
    range_val = np.where(max_val - min_val == 0, 1e-6, max_val - min_val)
    
    X_train_angles = (X_train_pca - min_val) / range_val * np.pi
    X_test_angles = (X_test_pca - min_val) / range_val * np.pi

    feature_file = os.path.join(output_dir, f'qml_features_{model_name}.npz')
    np.savez_compressed(
        feature_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_pca=X_train_angles,
        X_test_pca=X_test_angles,
        pca_explained=pca.explained_variance_ratio_
    )
    print(f"[+] Saved quantum feature representations to {feature_file}")
    return feature_file

if __name__ == '__main__':
    extract_features('resnet18', img_size=224)
