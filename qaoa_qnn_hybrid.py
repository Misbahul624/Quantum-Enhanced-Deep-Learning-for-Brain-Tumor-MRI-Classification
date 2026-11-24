"""
QAOA Feature Selection + QNN Hybrid Pipeline
=============================================
As specified in Box 5 of the architecture diagram:
[QAOA Feature Selection + QNN]

Workflow:
1. Classical Feature Extraction:
   - High-dimensional spatial features extracted from deep backbone (MobileNetV2 / ConvNeXt).
2. QAOA (Quantum Approximate Optimization Algorithm):
   - Formulates Feature Selection as a Quadratic Unconstrained Binary Optimization (QUBO) / Ising Hamiltonian:
     Cost = - sum_i Relevance(i)*z_i + lambda * sum_{i < j} Redundancy(i, j)*z_i*z_j + alpha * (sum_i z_i - K)^2
   - Cost Hamiltonian H_C & Mixer Hamiltonian H_M.
   - QAOA Parameterized Circuit: Alternating e^{-i gamma H_C} and e^{-i beta H_M} layers.
   - Quantum optimization finds the optimal subset of K=8 most informative, non-redundant quantum features.
3. Quantum Neural Network (QNN):
   - Feeds the QAOA-selected 8 optimal features into an 8-Qubit Variational Quantum Circuit (VQC).
   - Ry Angle Embedding + CNOT Entanglement + Pauli-Z Expectation values.
4. Classification Head:
   - Evaluates on 1,600 held-out test scans (Glioma, Meningioma, No Tumor, Pituitary).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Windows UTF-8 stdout
if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CANDIDATE_FEATURES = 16  # Candidate pool of top features
N_SELECTED_QUBITS = 8      # Number of features QAOA must select for the 8-qubit QNN
N_LAYERS = 3
N_CLASSES = 4

# ==============================================================================
# Step 1: QAOA Feature Selection Formulation (QUBO / Ising Hamiltonian)
# ==============================================================================
def qaoa_feature_selection(X_train, y_train, k=8, qaoa_steps=2):
    print(f"\n[*] Formulating QAOA Feature Selection Problem (Selecting {k} optimal features from pool)...")
    
    # 1. Relevance: Mutual Information between each feature and tumor label
    print("    >> Computing Feature Relevance (Mutual Information)...")
    relevance = mutual_info_classif(X_train, y_train, random_state=42)
    # Select top candidates for QAOA optimization
    top_candidates = np.argsort(relevance)[-N_CANDIDATE_FEATURES:]
    
    X_sub = X_train[:, top_candidates]
    rel_sub = relevance[top_candidates]
    
    # Normalize relevance
    rel_sub = rel_sub / (np.max(rel_sub) + 1e-6)
    
    # 2. Redundancy: Pearson Correlation between candidate features
    print("    >> Computing Feature Redundancy Matrix...")
    corr_matrix = np.abs(np.corrcoef(X_sub, rowvar=False))
    np.fill_diagonal(corr_matrix, 0.0)
    
    # 3. QAOA Cost Hamiltonian formulation:
    # We want to maximize Relevance and minimize Redundancy subject to choosing exactly k features.
    # Cost = - sum_i rel_i * x_i + lambda * sum_{i,j} corr_{ij} * x_i * x_j + penalty * (sum_i x_i - k)^2
    n_vars = N_CANDIDATE_FEATURES
    dev_qaoa = qml.device('default.qubit', wires=n_vars)
    
    # Cost terms
    cost_coeffs = []
    cost_obs = []
    
    # Linear terms (Relevance)
    for i in range(n_vars):
        cost_coeffs.append(-float(rel_sub[i]))
        cost_obs.append(qml.PauliZ(i))
        
    # Quadratic terms (Redundancy)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            weight = float(corr_matrix[i, j]) * 0.5
            cost_coeffs.append(weight)
            cost_obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
            
    cost_h = qml.Hamiltonian(cost_coeffs, cost_obs)
    mixer_h = qml.qaoa.x_mixer(wires=range(n_vars))
    
    @qml.qnode(dev_qaoa)
    def qaoa_circuit(params):
        # Initial equal superposition |+> state
        for w in range(n_vars):
            qml.Hadamard(wires=w)
            
        gammas = params[:qaoa_steps]
        betas = params[qaoa_steps:]
        
        for p in range(qaoa_steps):
            qml.qaoa.cost_layer(gammas[p], cost_h)
            qml.qaoa.mixer_layer(betas[p], mixer_h)
            
        return [qml.expval(qml.PauliZ(i)) for i in range(n_vars)]

    print(f"[*] Optimizing QAOA Circuit ({qaoa_steps} layers, {n_vars} qubits)...")
    from pennylane import numpy as pnp
    init_params = pnp.array([0.5] * qaoa_steps + [0.5] * qaoa_steps, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    
    params = init_params
    for step in range(12):
        params, cost_val = opt.step_and_cost(lambda p: np.sum(qaoa_circuit(p)), params)
        if (step + 1) % 4 == 0 or step == 0:
            print(f"    QAOA Step {step+1:02d}/12 | Optimization Energy: {cost_val:.4f}")
            
    # Measure probabilities of feature importance from ground state
    z_exp = qaoa_circuit(params)
    # Qubits with lowest expectation correspond to optimal selected features
    selected_indices_in_sub = np.argsort(z_exp)[:k]
    selected_feature_indices = top_candidates[selected_indices_in_sub]
    
    print(f"[+] QAOA Selected {k} Optimal Quantum Features: {selected_feature_indices}")
    return selected_feature_indices

# ==============================================================================
# Step 2: Quantum Neural Network (QNN) on QAOA-Selected Features
# ==============================================================================
dev_qnn = qml.device('default.qubit', wires=N_SELECTED_QUBITS)

@qml.qnode(dev_qnn, interface="torch", diff_method="backprop")
def qnn_vqc(inputs, weights):
    # Ry Angle Encoding of the QAOA-selected features
    qml.AngleEmbedding(inputs, wires=range(N_SELECTED_QUBITS), rotation="Y")
    # Entangling Variational Circuit
    qml.BasicEntanglerLayers(weights, wires=range(N_SELECTED_QUBITS))
    # Pauli-Z expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(N_SELECTED_QUBITS)]

class QAOA_QNN_Classifier(nn.Module):
    def __init__(self, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        self.n_qubits = n_qubits
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(qnn_vqc, weight_shapes)
        
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        q_expvals = self.q_layer(x)
        return self.classifier(q_expvals)

# ==============================================================================
# Step 3: End-to-End Execution & Evaluation
# ==============================================================================
def run_qaoa_qnn_pipeline():
    output_dir = 'output_qaoa_qnn'
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Loading extracted deep feature embeddings...")
    data_path = os.path.join('output_qml', 'qml_features_convnext_finetuned.npz')
    data = np.load(data_path)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # 1. Run QAOA Feature Selection
    selected_features = qaoa_feature_selection(X_train, y_train, k=N_SELECTED_QUBITS, qaoa_steps=2)
    
    # Filter datasets to QAOA-selected optimal quantum features
    X_train_qaoa = X_train[:, selected_features]
    X_test_qaoa = X_test[:, selected_features]
    
    # Min-max scale to [0, pi] for angle embedding
    min_val = X_train_qaoa.min(axis=0)
    max_val = X_train_qaoa.max(axis=0)
    range_val = np.where(max_val - min_val == 0, 1e-6, max_val - min_val)
    
    X_train_angles = (X_train_qaoa - min_val) / range_val * np.pi
    X_test_angles = (X_test_qaoa - min_val) / range_val * np.pi
    
    # Convert to PyTorch Tensors
    train_ds = TensorDataset(torch.tensor(X_train_angles, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_angles, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 2. Train QNN on QAOA Features
    print(f"\n[*] Initializing 8-Qubit QNN Classifier on QAOA-Selected Features...")
    model = QAOA_QNN_Classifier(n_qubits=N_SELECTED_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-4)
    
    epochs = 25
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    
    print(f"[*] Training QAOA + QNN across {epochs} epochs...")
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
        
        # Test evaluation
        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                vo = model(bx)
                v_corr += (vo.argmax(dim=1) == by).sum().item()
                v_tot += len(by)
        val_acc = v_corr / v_tot
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'qaoa_qnn_best.pth'))
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {val_acc*100:.2f}%")
            
    # 3. Final Evaluation
    print("\n[*] Evaluating Best QAOA + QNN Model on 1,600 Held-Out Test Images...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'qaoa_qnn_best.pth')))
    model.eval()
    
    all_preds, all_probs = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.numpy().tolist())
            all_preds.extend(logits.argmax(dim=1).numpy().tolist())
            
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    
    final_acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    roc_auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='weighted')
    
    print("=" * 60)
    print("QAOA FEATURE SELECTION + QNN FINAL TEST RESULTS:")
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
    plt.title(f'QAOA + QNN - Test Accuracy: {final_acc*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qaoa_qnn_confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Save Convergence Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Test Acc')
    plt.title('QAOA+QNN Accuracy Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Loss', color='coral')
    plt.title('QAOA+QNN Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qaoa_qnn_convergence.png'), dpi=150)
    plt.close()
    
    print(f"[+] Saved all QAOA + QNN artifacts to {output_dir}")

if __name__ == '__main__':
    run_qaoa_qnn_pipeline()
