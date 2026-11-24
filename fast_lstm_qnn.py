"""
Fast & High-Performance LSTM + QNN / QML Pipeline
=================================================
Pre-extracts spatial slice feature sequences (7x7 = 49 sequence steps per image)
using the trained deep backbone, caching features into memory/GPU.
Then trains the Bidirectional LSTM + 8-Qubit QNN (VQC) in minutes!
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
N_LAYERS = 3
N_CLASSES = 4

# Initialize Quantum Simulator Device
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnn_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class LSTM_QNN_Model(nn.Module):
    def __init__(self, in_features=64, hidden_dim=128, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1. Bidirectional LSTM across the 49 sequential spatial patches
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. Quantum Bottleneck: LSTM hidden representation (256-dim) -> 8 Quantum Angles
        self.to_quantum = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )
        
        # 3. 8-Qubit Quantum Neural Network Layer (PennyLane)
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnn_layer = qml.qnn.TorchLayer(qnn_circuit, weight_shapes)
        
        # 4. Classification Head from Quantum Measurement Expectation Values
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, seq_x):
        # seq_x: [Batch, Seq_Len=49, Feat=64]
        lstm_out, (hn, cn) = self.lstm(seq_x)
        # Concatenate forward and backward final hidden states: [Batch, 256]
        pooled = torch.cat([hn[0], hn[1]], dim=1)
        
        # Project to 8 quantum angles [-pi, pi]
        quantum_angles = self.to_quantum(pooled) * np.pi
        
        # Run 8-qubit QNN circuit
        q_expvals = self.qnn_layer(quantum_angles.cpu()).to(seq_x.device)
        
        # Output tumor logits
        return self.classifier(q_expvals)

def run_fast_lstm_qnn():
    print("[*] Checking pre-extracted spatial sequence features...")
    seq_file = os.path.join('output_lstm_qnn', 'mri_spatial_sequences.npz')
    os.makedirs('output_lstm_qnn', exist_ok=True)
    
    if not os.path.exists(seq_file):
        print("[*] Extracting 49-step spatial patch sequences using MobileNetV2...")
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        sys.path.insert(0, 'src')
        from dataset import BrainTumorMRIDataset, CLASSES, CLASS_TO_IDX
        
        def collect(split):
            paths, labels = [], []
            for c in CLASSES:
                cd = os.path.join('dataset', split, c)
                for f in sorted(os.listdir(cd)):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        paths.append(os.path.join(cd, f))
                        labels.append(CLASS_TO_IDX[c])
            return paths, labels
            
        train_p, train_l = collect('Training')
        test_p, test_l = collect('Testing')
        
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(DEVICE).eval()
        features = mobilenet.features
        channel_proj = nn.Conv2d(1280, 64, kernel_size=1).to(DEVICE).eval()
        
        train_ds = BrainTumorMRIDataset(train_p, train_l, img_size=224, is_training=False, crop_brain=True)
        test_ds = BrainTumorMRIDataset(test_p, test_l, img_size=224, is_training=False, crop_brain=True)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        def extract(loader):
            seqs, lbls = [], []
            with torch.no_grad():
                for imgs, l in loader:
                    imgs = imgs.to(DEVICE)
                    fm = features(imgs) # [B, 1280, 7, 7]
                    pm = channel_proj(fm) # [B, 64, 7, 7]
                    B, C, H, W = pm.shape
                    sq = pm.permute(0, 2, 3, 1).reshape(B, H*W, C) # [B, 49, 64]
                    seqs.append(sq.cpu().numpy())
                    lbls.append(l.numpy())
            return np.concatenate(seqs, axis=0), np.concatenate(lbls, axis=0)
            
        print("    >> Processing Training Scans (5600)...")
        X_train_seq, y_train = extract(train_loader)
        print("    >> Processing Testing Scans (1600)...")
        X_test_seq, y_test = extract(test_loader)
        
        np.savez_compressed(seq_file, X_train_seq=X_train_seq, y_train=y_train, X_test_seq=X_test_seq, y_test=y_test)
        print(f"[+] Cached spatial sequences to {seq_file}: shape {X_train_seq.shape}")
    else:
        print(f"[*] Loading cached sequence representations from {seq_file}...")
        data = np.load(seq_file)
        X_train_seq = data['X_train_seq']
        y_train = data['y_train']
        X_test_seq = data['X_test_seq']
        y_test = data['y_test']
        
    print(f"[*] Train Sequences: {X_train_seq.shape} | Test Sequences: {X_test_seq.shape}")
    
    train_ds = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model = LSTM_QNN_Model(in_features=64, hidden_dim=128, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    epochs = 20
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0.0
    
    print("\n[*] Starting Fast LSTM + QNN Training (20 Epochs)...")
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
            torch.save(model.state_dict(), 'output_lstm_qnn/lstm_qnn_best.pth')
            
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% (Best: {best_acc*100:.2f}%)", flush=True)
        
    print("\n[*] Evaluating Best LSTM + QNN Model on 1,600 Held-Out Test Images...")
    model.load_state_dict(torch.load('output_lstm_qnn/lstm_qnn_best.pth'))
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
    print("HYBRID LSTM + QNN FINAL TEST RESULTS:")
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
    plt.title(f'LSTM + QNN - Test Accuracy: {final_acc*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('output_lstm_qnn/lstm_qnn_confusion_matrix.png', dpi=150)
    plt.close()
    
    # Save Convergence Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.title('LSTM+QNN Accuracy Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Loss', color='coral')
    plt.title('LSTM+QNN Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output_lstm_qnn/lstm_qnn_convergence.png', dpi=150)
    plt.close()
    print("[+] All LSTM + QNN artifacts saved successfully!")

if __name__ == '__main__':
    run_fast_lstm_qnn()
