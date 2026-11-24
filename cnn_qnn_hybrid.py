"""
End-to-End Hybrid CNN + QNN (VQC) Model
As depicted in Box 5 (Hybrid Models -> CNN + VQC Baseline / Optimized QNN) of the diagram:

Architecture:
1. Classical CNN Backbone:
   Input Image (224x224x3) -> Conv2D -> BatchNorm -> ReLU -> MaxPool
   -> Conv2D blocks extracting hierarchical medical imaging features.
   -> Global Average Pooling -> Linear projection to 8 features.
2. Quantum Neural Network (QNN):
   8 Features -> Ry Angle Embedding across 8 Qubits (|0> state)
   -> Variational Quantum Circuit (VQC) with Parameterized Rotations (Rx, Ry, Rz)
   -> CNOT Circular Entangling Gates (L layers)
   -> Quantum Measurement: PauliZ expectation values <Z_i> on all 8 qubits.
3. Hybrid Output Layer:
   8 Quantum Expectation measurements -> Linear Classifier -> 4 Brain Tumor Classes
   (Glioma, Meningioma, No Tumor, Pituitary).

Trained end-to-end with backpropagation using PennyLane and PyTorch!
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml

# Force UTF-8 encoding on Windows
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
    """
    Step 4 in Diagram:
    - Data Encoding: Ry Angle Embedding of 8 Classical CNN Features
    - Variational Quantum Circuit: Parameterized Rotations + CNOT Entanglement
    - Measurement: PauliZ expectation values
    """
    # 1. Ry Angle Embedding
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    
    # 2. Variational Quantum Layers with Entanglement
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    
    # 3. Quantum Expectation Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class CNN_QNN_HybridModel(nn.Module):
    def __init__(self, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1. Classical CNN Feature Extractor (Custom Deep CNN Backbone)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), # 256
            nn.Flatten()
        )
        
        # Feature Reduction: 256 -> 8 features for 8 Qubits
        self.feature_reduction = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_qubits),
            nn.Tanh() # Scale angles to [-pi, pi]
        )
        
        # 2. Quantum Neural Network (QNN) Layer
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnn_layer = qml.qnn.TorchLayer(qnn_circuit, weight_shapes)
        
        # 3. Final Classical Classifier Head (interprets Quantum Measurements)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        # Pass through CNN on GPU
        feats = self.cnn(x)
        # Reduce to 8 quantum angles
        quantum_angles = self.feature_reduction(feats) * np.pi
        
        # PennyLane default.qubit simulator executes on CPU
        q_in_cpu = quantum_angles.cpu()
        q_expvals_cpu = self.qnn_layer(q_in_cpu)
        q_expvals = q_expvals_cpu.to(x.device)
        
        # Classify based on quantum measurements
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

def train_cnn_qnn():
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output_cnn_qnn')
    os.makedirs(output_dir, exist_ok=True)

    print("[*] Collecting MRI Dataset...")
    train_all_paths, train_all_labels = collect_paths(os.path.join(data_root, 'Training'))
    test_paths, test_labels = collect_paths(os.path.join(data_root, 'Testing'))

    # Stratified validation split
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
    print(f"[*] Architecture: Hybrid End-to-End CNN + QNN (8 Qubits, {N_LAYERS} Layers)")

    train_ds = BrainTumorMRIDataset(train_paths, train_labels, img_size=img_size, is_training=True, crop_brain=True)
    val_ds = BrainTumorMRIDataset(val_paths, val_labels, img_size=img_size, is_training=False, crop_brain=True)
    test_ds = BrainTumorMRIDataset(test_paths, test_labels, img_size=img_size, is_training=False, crop_brain=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNN_QNN_HybridModel(n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print("\n[*] Starting End-to-End CNN + QNN Training...")
    for epoch in range(epochs):
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

        scheduler.step()
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
            torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_qnn_best.pth'))
            print(f"    >> New Best Val Acc: {best_val_acc*100:.2f}% (Saved checkpoint)")

    # Final Evaluation on Held-Out Test Set (1600 images)
    print("\n[*] Evaluating Best CNN+QNN on 1600 Held-Out Test Images...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'cnn_qnn_best.pth')))
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
    print("HYBRID CNN + QNN FINAL TEST RESULTS:")
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
    plt.title(f'Hybrid CNN+QNN - Test Accuracy: {acc*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnn_qnn_confusion_matrix.png'), dpi=150)
    plt.close()

    # Training Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('CNN+QNN Accuracy Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='coral')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('CNN+QNN Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnn_qnn_convergence.png'), dpi=150)
    plt.close()

    print(f"[+] All artifacts successfully saved to {output_dir}")

if __name__ == '__main__':
    train_cnn_qnn()
