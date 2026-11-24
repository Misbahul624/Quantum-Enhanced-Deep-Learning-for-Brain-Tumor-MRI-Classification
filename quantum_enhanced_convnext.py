"""
Fine-Tuned ConvNeXt + 8-Qubit Variational Quantum Entangler (QNN)
Combines:
1. 768-dimensional fine-tuned ConvNeXt feature embeddings.
2. 8-Qubit Variational Quantum Circuit with Ry angle encoding and CNOT entanglement.
3. Quantum expectation measurements projected and fused with classical logits.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_QUBITS = 8
N_LAYERS = 3
N_CLASSES = 4

dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_circuit_convnext(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class QuantumEnhancedConvNeXt(nn.Module):
    def __init__(self, in_features=768, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, n_qubits)
        )
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit_convnext, weight_shapes)
        
        self.quantum_head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.GELU(),
            nn.Linear(16, n_classes)
        )
        self.fusion_weight = nn.Parameter(torch.tensor([0.2]))  # learnable residual quantum weight

    def forward(self, features, base_logits):
        q_in = torch.tanh(self.compress(features)) * np.pi
        q_out = self.q_layer(q_in)
        q_delta = self.quantum_head(q_out)
        return base_logits + self.fusion_weight * q_delta

def run_quantum_boost():
    npz_path = os.path.join('output_qml', 'qml_features_convnext_finetuned.npz')
    data = np.load(npz_path)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    logits_train = torch.tensor(data['logits_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.long)

    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    logits_test = torch.tensor(data['logits_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, logits_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, logits_test, y_test), batch_size=64, shuffle=False)

    print("[*] Base fine-tuned ConvNeXt test accuracy:")
    base_preds = logits_test.argmax(dim=1).numpy()
    base_acc = accuracy_score(y_test.numpy(), base_preds)
    print(f"    Base Accuracy: {base_acc*100:.2f}%")

    model = QuantumEnhancedConvNeXt(in_features=768, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    print("\n[*] Training Quantum Residual Head...")
    for epoch in range(15):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for feat, bl, lbl in train_loader:
            optimizer.zero_grad()
            out = model(feat, bl)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(lbl)
            correct += (out.argmax(dim=1) == lbl).sum().item()
            total += len(lbl)

        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for feat, bl, lbl in test_loader:
                vo = model(feat, bl)
                v_corr += (vo.argmax(dim=1) == lbl).sum().item()
                v_tot += len(lbl)
        val_acc = v_corr / v_tot
        print(f"  Epoch {epoch+1:02d}/15 | Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}% | Test Acc: {val_acc*100:.2f}%")

    print("\n[*] Final Quantum-Boosted Evaluation on 1600 Test Scans...")
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for feat, bl, lbl in test_loader:
            out = model(feat, bl)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(dim=1).numpy().tolist())
            all_probs.extend(probs.numpy().tolist())

    y_true = y_test.numpy()
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    final_acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='weighted')
    prec = precision_score(y_true, preds, average='weighted')
    rec = recall_score(y_true, preds, average='weighted')
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    roc_auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='weighted')

    print("=" * 60)
    print("FINAL QUANTUM-ENHANCED MODEL SCORE:")
    print(f"  Accuracy : {final_acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print("=" * 60)
    print(classification_report(y_true, preds, target_names=CLASSES))

    out_dir = 'output_qml'
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Quantum-Enhanced ConvNeXt - Test Accuracy: {final_acc*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'quantum_enhanced_confusion_matrix.png'), dpi=150)
    plt.close()

    torch.save(model.state_dict(), os.path.join(out_dir, 'quantum_enhanced_best.pth'))
    print(f"[+] Saved quantum enhanced model to {out_dir}")

if __name__ == '__main__':
    run_quantum_boost()
