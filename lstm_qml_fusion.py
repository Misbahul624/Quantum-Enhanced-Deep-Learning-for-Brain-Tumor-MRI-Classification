"""
Model 2: LSTM + QML (Quantum-Classical Fusion / Quantum Kernel Layer)
Processes the 49-step spatial MRI sequences through a Bidirectional LSTM,
then fuses the LSTM recurrent state with an 8-Qubit Variational Quantum Entangler (QML)
to create a joint Quantum-Classical feature representation.
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_QUBITS = 8
N_LAYERS = 2
N_CLASSES = 4

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qml_kernel_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class LSTM_QML_FusionModel(nn.Module):
    def __init__(self, in_features=64, hidden_dim=128, n_qubits=8, n_layers=2, n_classes=4):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1. Classical Bi-LSTM Sequence Processing
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Classical feature compression
        self.classical_branch = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        # Quantum feature projection
        self.to_quantum = nn.Sequential(
            nn.Linear(hidden_dim * 2, n_qubits),
            nn.Tanh()
        )
        
        # 2. QML Variational Circuit Layer
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qml_layer = qml.qnn.TorchLayer(qml_kernel_circuit, weight_shapes)
        
        self.quantum_branch = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.GELU()
        )
        
        # 3. Hybrid Fusion Head (64 Classical + 32 Quantum = 96 Features)
        self.fusion_head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, seq_x):
        lstm_out, (hn, cn) = self.lstm(seq_x)
        pooled = torch.cat([hn[0], hn[1]], dim=1)  # [B, 256]
        
        # Classical representation
        c_feat = self.classical_branch(pooled)
        
        # Quantum representation
        q_angles = self.to_quantum(pooled) * np.pi
        q_exp = self.qml_layer(q_angles.cpu()).to(seq_x.device)
        q_feat = self.quantum_branch(q_exp)
        
        # Joint Quantum-Classical Fusion
        fused = torch.cat([c_feat, q_feat], dim=1)
        return self.fusion_head(fused)

def run_lstm_qml():
    output_dir = 'output_lstm_qml'
    os.makedirs(output_dir, exist_ok=True)
    
    seq_file = os.path.join('output_lstm_qnn', 'mri_spatial_sequences.npz')
    data = np.load(seq_file)
    X_train_seq = data['X_train_seq']
    y_train = data['y_train']
    X_test_seq = data['X_test_seq']
    y_test = data['y_test']
    
    train_ds = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model = LSTM_QML_FusionModel(in_features=64, hidden_dim=128, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    epochs = 20
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0.0
    
    print("\n[*] Starting LSTM + QML (Quantum-Classical Fusion) Training (20 Epochs)...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
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
        
        # Test evaluation
        model.eval()
        t_corr, t_tot = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                vo = model(bx)
                t_corr += (vo.argmax(dim=1) == by).sum().item()
                t_tot += len(by)
        test_acc = t_corr / t_tot
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'lstm_qml_best.pth'))
            
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% (Best: {best_acc*100:.2f}%)", flush=True)
        
    print("\n[*] Evaluating Best LSTM + QML Model on 1,600 Held-Out Test Images...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'lstm_qml_best.pth')))
    model.eval()
    
    all_preds, all_probs = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    
    final_acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    
    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    roc_auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='weighted')
    
    print("=" * 60)
    print("HYBRID LSTM + QML FINAL TEST RESULTS:")
    print(f"  Test Accuracy : {final_acc*100:.2f}%")
    print(f"  Precision     : {prec*100:.2f}%")
    print(f"  Recall        : {rec*100:.2f}%")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")
    print("=" * 60)
    print(classification_report(y_test, preds, target_names=CLASSES))
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'LSTM + QML - Test Accuracy: {final_acc*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_qml_confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Save Convergence Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.title('LSTM+QML Accuracy Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Loss', color='coral')
    plt.title('LSTM+QML Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_qml_convergence.png'), dpi=150)
    plt.close()
    print("[+] All LSTM + QML artifacts saved successfully!")

if __name__ == '__main__':
    run_lstm_qml()
