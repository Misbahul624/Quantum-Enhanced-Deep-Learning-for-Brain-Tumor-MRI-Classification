"""
Extract features from our top-performing fine-tuned ConvNeXt model (output/convnext_small_best.pth),
which already achieved 95.25% test accuracy, and combine it with the Quantum Variational Circuit!
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BrainTumorMRIDataset, CLASSES, CLASS_TO_IDX
from model import build_model
from trainer import DEVICE

def extract_convnext_features():
    print("[*] Loading fine-tuned ConvNeXt-Small model...")
    model = build_model('convnext_small', pretrained=False, dropout=0.0)
    ckpt = torch.load('output/convnext_small_best.pth', map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()

    # We extract features right before the final classification head
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    
    def collect(split):
        p, l = [], []
        for c in CLASSES:
            cd = os.path.join(data_root, split, c)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    p.append(os.path.join(cd, f))
                    l.append(CLASS_TO_IDX[c])
        return p, l

    train_paths, train_labels = collect('Training')
    test_paths, test_labels = collect('Testing')

    train_loader = DataLoader(BrainTumorMRIDataset(train_paths, train_labels, img_size=288, is_training=False, crop_brain=True), batch_size=32, shuffle=False)
    test_loader = DataLoader(BrainTumorMRIDataset(test_paths, test_labels, img_size=288, is_training=False, crop_brain=True), batch_size=32, shuffle=False)

    # Hook forward features
    def extract_from_loader(loader):
        feats_list, logits_list, labels_list = [], [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    # Feature representation from backbone
                    feat = model.forward_features(imgs)
                    if feat.dim() == 4:
                        feat = feat.mean(dim=[2, 3])
                    logits = model(imgs)
                feats_list.append(feat.cpu().float().numpy())
                logits_list.append(logits.cpu().float().numpy())
                labels_list.append(lbls.numpy())
        return np.concatenate(feats_list, axis=0), np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)

    print("[*] Extracting fine-tuned representations for Training Set...")
    X_train, logits_train, y_train = extract_from_loader(train_loader)
    print("[*] Extracting fine-tuned representations for Testing Set...")
    X_test, logits_test, y_test = extract_from_loader(test_loader)

    out_file = os.path.join('output_qml', 'qml_features_convnext_finetuned.npz')
    np.savez_compressed(
        out_file,
        X_train=X_train,
        logits_train=logits_train,
        y_train=y_train,
        X_test=X_test,
        logits_test=logits_test,
        y_test=y_test
    )
    print(f"[+] Saved fine-tuned deep features to {out_file}: shape {X_train.shape}")
    return out_file

if __name__ == '__main__':
    extract_convnext_features()
