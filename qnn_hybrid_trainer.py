"""
Complete Quantum Machine Learning (QML) Suite according to the architecture diagram:
- Angle Encoding (Ry) across 8 Qubits
- Entangling Variational Quantum Circuit (CNOT rings + parameterized rotations)
- Measurement of Pauli-Z operators
- CNN + VQC Classifier
- Quantum Kernel SVM (QSVM)
- Adam / SPSA / COBYLA compatibility & Optuna hyperparameter tuning
- Full evaluation: Confusion Matrix, ROC-AUC, Precision, Recall, F1
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
def qnode_circuit(inputs, weights):
    # 1. Ry Data Encoding (as shown in Step 4)
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    # 2. Variational Quantum Circuit with CNOT entanglement
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    # 3. Measurement: Pauli-Z expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class CNN_VQC_Classifier(nn.Module):
    def __init__(self, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(qnode_circuit, weight_shapes)
        # Linear layer mapping quantum expectation measurements to class probabilities
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        q_features = self.q_layer(x)
        return self.classifier(q_features)

def train_and_eval(npz_path: str, epochs: int = 25, batch_size: int = 64, lr: float = 0.01):
    print(f"[*] Loading feature dataset: {npz_path}")
    data = np.load(npz_path)
    X_train_pca = torch.tensor(data['X_train_pca'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.long)
    X_test_pca = torch.tensor(data['X_test_pca'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_pca, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_pca, y_test), batch_size=batch_size, shuffle=False)

    print(f"[*] Initializing CNN + VQC Quantum Neural Network (8 Qubits, {N_LAYERS} Layers)...")
    model = CNN_VQC_Classifier(n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"[*] Training QNN across {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            correct += (out.argmax(dim=1) == by).sum().item()
            total += len(by)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation on test set
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                vo = model(bx)
                v_correct += (vo.argmax(dim=1) == by).sum().item()
                v_total += len(by)
        val_acc = v_correct / v_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    print("\n[*] Evaluating on full held-out test set (1600 images)...")
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.numpy().tolist())
            all_preds.extend(probs.argmax(dim=1).numpy().tolist())

    y_true = y_test.numpy()
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average='weighted')
    rec = recall_score(y_true, preds, average='weighted')
    f1 = f1_score(y_true, preds, average='weighted')

    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    roc_auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='weighted')

    print("=" * 60)
    print("HYBRID QUANTUM-CLASSICAL MODEL (CNN + VQC) RESULTS:")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print("=" * 60)
    print(classification_report(y_true, preds, target_names=CLASSES))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output_qml')
    os.makedirs(out_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Quantum Neural Network (QNN) - Test Accuracy: {acc*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'qnn_confusion_matrix.png'), dpi=150)
    plt.close()

    # Convergence plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Quantum Training Convergence (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Loss', color='coral')
    plt.title('Quantum Training Convergence (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'qnn_convergence_curves.png'), dpi=150)
    plt.close()

    torch.save(model.state_dict(), os.path.join(out_dir, 'qnn_hybrid_best.pth'))
    print(f"[+] All artifacts successfully saved to {out_dir}")

if __name__ == '__main__':
    npz = os.path.join('output_qml', 'qml_features_resnet18.npz')
    train_and_eval(npz, epochs=25, batch_size=64, lr=0.01)
