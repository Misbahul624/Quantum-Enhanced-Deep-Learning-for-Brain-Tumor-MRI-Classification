"""
MobileNetV2 + Quantum Neural Network (QNN) Hybrid Model
======================================================
Architecture:
1. Classical Backbone: MobileNetV2 (Pretrained on ImageNet-1K)
   - Inverted Residual Blocks (MobileNet architecture)
   - Extracts 1,280 high-level deep spatial features
2. Dimensionality Compression:
   - Linear(1280 -> 64) -> GELU -> Linear(64 -> 8) -> Tanh * pi
   - Maps rich representations to 8 continuous quantum rotation angles
3. Quantum Neural Network (QNN):
   - 8-Qubit Variational Quantum Circuit (VQC) via PennyLane
   - Ry Angle Embedding: |psi_0> = (X) Ry(theta_i) |0>
   - Variational Parameterized Rotations (Rx, Ry, Rz)
   - Circular CNOT Entangling Gates (L=3 layers)
   - Pauli-Z Expectation Measurements: <sigma_z^(i)> on all 8 wires
4. Hybrid Classification Head:
   - Interprets quantum state measurements to classify 4 brain tumor types:
     (Glioma, Meningioma, No Tumor, Pituitary)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml

# UTF-8 stdout encoding for Windows
if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BrainTumorMRIDataset, CLASSES, CLASS_TO_IDX

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_QUBITS = 8
N_LAYERS = 3
N_CLASSES = 4

# Initialize Quantum Simulator Device
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnn_circuit(inputs, weights):
    # 1. Ry Angle Embedding
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    # 2. Variational Quantum Circuit with CNOT Entanglement
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    # 3. Pauli-Z Expectation Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class MobileNetV2_QNN_Hybrid(nn.Module):
    def __init__(self, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1. Classical Backbone: MobileNetV2 Feature Extractor
        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2. Feature Reduction: 1280 -> 8 Quantum Angles
        self.feature_reduction = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 64),
            nn.GELU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()  # Scale to [-pi, pi]
        )
        
        # 3. Quantum Neural Network (QNN) Layer
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnn_layer = qml.qnn.TorchLayer(qnn_circuit, weight_shapes)
        
        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        # MobileNet features: [B, 1280, 7, 7] -> [B, 1280]
        feats = self.features(x)
        feats = self.pool(feats)
        feats = torch.flatten(feats, 1)
        
        # Map to 8 quantum angles
        quantum_angles = self.feature_reduction(feats) * np.pi
        
        # PennyLane QNode execution (CPU sync)
        q_in_cpu = quantum_angles.cpu()
        q_expvals_cpu = self.qnn_layer(q_in_cpu)
        q_expvals = q_expvals_cpu.to(x.device)
        
        # Output prediction logits
        logits = self.classifier(q_expvals)
        return logits

def collect_paths(split_root):
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

def train_mobilenet_qnn():
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output_mobilenet_qnn')
    os.makedirs(output_dir, exist_ok=True)

    print("[*] Collecting MRI Dataset...")
    train_all_paths, train_all_labels = collect_paths(os.path.join(data_root, 'Training'))
    test_paths, test_labels = collect_paths(os.path.join(data_root, 'Testing'))

    # Stratified validation split (15%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(train_all_paths, train_all_labels))

    train_paths = [train_all_paths[i] for i in train_idx]
    train_labels = [train_all_labels[i] for i in train_idx]
    val_paths = [train_all_paths[i] for i in val_idx]
    val_labels = [train_all_labels[i] for i in val_idx]

    img_size = 224
    batch_size = 64
    epochs = 20
    lr = 0.001

    print(f"[*] Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print(f"[*] Model: Pretrained MobileNetV2 + 8-Qubit QNN (Variational Quantum Circuit)")

    train_ds = BrainTumorMRIDataset(train_paths, train_labels, img_size=img_size, is_training=True, crop_brain=True)
    val_ds = BrainTumorMRIDataset(val_paths, val_labels, img_size=img_size, is_training=False, crop_brain=True)
    test_ds = BrainTumorMRIDataset(test_paths, test_labels, img_size=img_size, is_training=False, crop_brain=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MobileNetV2_QNN_Hybrid(n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES)
    model.to(DEVICE)

    # Freeze base MobileNet weights for first 3 epochs, then fine-tune
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print("\n[*] Starting MobileNetV2 + QNN Hybrid Training...")
    for epoch in range(epochs):
        # Unfreeze MobileNet backbone at epoch 4 for full fine-tuning
        if epoch == 3:
            print("    >> Unfreezing MobileNetV2 backbone for end-to-end fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam([
                {'params': model.features.parameters(), 'lr': lr * 0.1},
                {'params': model.feature_reduction.parameters(), 'lr': lr},
                {'params': model.qnn_layer.parameters(), 'lr': lr},
                {'params': model.classifier.parameters(), 'lr': lr}
            ], weight_decay=1e-4)

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(lbls)
            correct += (logits.argmax(dim=1) == lbls).sum().item()
            total += len(lbls)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                vo = model(imgs)
                vl = criterion(vo, lbls)
                v_loss += vl.item() * len(lbls)
                v_corr += (vo.argmax(dim=1) == lbls).sum().item()
                v_tot += len(lbls)

        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'mobilenet_qnn_best.pth'))
            print(f"    >> New Best Val Acc: {best_val_acc*100:.2f}% (Saved checkpoint)")

    # Final Evaluation on 1600 Held-Out Test Scans
    print("\n[*] Evaluating Best MobileNetV2 + QNN on 1600 Held-Out Test Images...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'mobilenet_qnn_best.pth')))
    model.eval()

    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_targets.extend(lbls.numpy().tolist())

    y_true = np.array(all_targets)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average='weighted')
    rec = recall_score(y_true, preds, average='weighted')
    f1 = f1_score(y_true, preds, average='weighted')
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    roc_auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='weighted')

    print("=" * 60)
    print("HYBRID MOBILENETV2 + QNN FINAL TEST RESULTS:")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Precision     : {prec*100:.2f}%")
    print(f"  Recall        : {rec*100:.2f}%")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")
    print("=" * 60)
    print(classification_report(y_true, preds, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'MobileNetV2 + QNN - Test Accuracy: {acc*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mobilenet_qnn_confusion_matrix.png'), dpi=150)
    plt.close()

    # Convergence Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('MobileNetV2 + QNN Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='coral')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('MobileNetV2 + QNN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mobilenet_qnn_convergence.png'), dpi=150)
    plt.close()

    print(f"[+] All artifacts successfully saved to {output_dir}")

if __name__ == '__main__':
    train_mobilenet_qnn()
