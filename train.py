"""
Main training script — Brain Tumor MRI Classification (Ensemble)
================================================================
Trains three models independently on the same data, then evaluates each
independently and as an ensemble.

Architectures:
  1. ConvNeXt-Small  (IN-22k → IN-1k → fine-tune)
  2. EfficientNetV2-M (IN-21k → IN-1k → fine-tune)
  3. Swin-Base (IN-22k → IN-1k → fine-tune)

Usage:
  python src/train.py

Output (in ./output/):
  - {arch}_best.pth          : best checkpoint per model
  - {arch}_confusion_matrix.png
  - {arch}_history.png
  - ensemble_confusion_matrix.png
  - results_summary.txt
"""

import os, sys, random, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import BrainTumorMRIDataset, CLASSES, CLASS_TO_IDX
from model import build_model
from trainer import train_model, evaluate_model, plot_training_history, DEVICE

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE    = 288
BATCH_SIZE  = 32
NUM_WORKERS = 0        # Windows: use 0 to avoid multiprocessing spawn issues
VAL_SPLIT   = 0.15

ARCHS = [
    'convnext_small',
    'efficientnetv2_m',
    'swin_base',
]


def collect_files(split_root):
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


if __name__ == '__main__':

    print("Collecting file paths...")
    train_all_paths, train_all_labels = collect_files(os.path.join(DATA_ROOT, 'Training'))
    test_paths,      test_labels      = collect_files(os.path.join(DATA_ROOT, 'Testing'))

    # Stratified split: Training → train + validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
    train_idx, val_idx = next(sss.split(train_all_paths, train_all_labels))

    train_paths  = [train_all_paths[i] for i in train_idx]
    train_labels = [train_all_labels[i] for i in train_idx]
    val_paths    = [train_all_paths[i] for i in val_idx]
    val_labels   = [train_all_labels[i] for i in val_idx]

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")

    # ─── DataLoaders ──────────────────────────────────────────────────────────
    train_ds = BrainTumorMRIDataset(train_paths, train_labels, img_size=IMG_SIZE, is_training=True,  crop_brain=True)
    val_ds   = BrainTumorMRIDataset(val_paths,   val_labels,   img_size=IMG_SIZE, is_training=False, crop_brain=True)
    test_ds  = BrainTumorMRIDataset(test_paths,  test_labels,  img_size=IMG_SIZE, is_training=False, crop_brain=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ─── Train each model ─────────────────────────────────────────────────────
    trained_models = []
    all_results = {}

    for arch in ARCHS:
        print(f"\n{'#'*65}")
        print(f"# Training: {arch}")
        print(f"{'#'*65}")

        model = build_model(arch, pretrained=True, dropout=0.3)

        model, history = train_model(
            model, train_loader, val_loader,
            arch_name=arch,
            output_dir=OUTPUT_DIR,
            warmup_epochs=5,
            finetune_epochs=50,
            warmup_lr=1e-3,
            finetune_lr=4e-5,
            patience=12,
            label_smoothing=0.08,
        )

        plot_training_history(history, arch, OUTPUT_DIR)
        results = evaluate_model(model, test_loader, CLASSES, arch, OUTPUT_DIR)
        all_results[arch] = results
        trained_models.append(model)

        torch.cuda.empty_cache()

    # ─── Ensemble evaluation ──────────────────────────────────────────────────
    print(f"\n{'#'*65}")
    print("# Ensemble Evaluation (avg probabilities)")
    print(f"{'#'*65}")

    all_probs_stack = np.array([all_results[arch]['probs'] for arch in ARCHS])  # (3, N, 4)
    avg_probs = all_probs_stack.mean(axis=0)
    ens_preds = np.argmax(avg_probs, axis=1)
    true_labels = all_results[ARCHS[0]]['labels']

    ens_acc = accuracy_score(true_labels, ens_preds)
    ens_f1  = f1_score(true_labels, ens_preds, average='macro')

    print(f"\nEnsemble Test Accuracy : {ens_acc*100:.3f}%")
    print(f"Ensemble Macro F1 Score: {ens_f1:.4f}")
    print(classification_report(true_labels, ens_preds, target_names=CLASSES))

    cm = confusion_matrix(true_labels, ens_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f'Ensemble (3 models) - Accuracy {ens_acc*100:.2f}%  |  F1 {ens_f1:.4f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_confusion_matrix.png'), dpi=150)
    plt.close()

    # ─── Summary ──────────────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Brain Tumor MRI Classification - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        for arch, res in all_results.items():
            f.write(f"{arch}\n")
            f.write(f"  Accuracy : {res['accuracy']*100:.3f}%\n")
            f.write(f"  Macro F1 : {res['f1']:.4f}\n\n")
        f.write("Ensemble (avg probs)\n")
        f.write(f"  Accuracy : {ens_acc*100:.3f}%\n")
        f.write(f"  Macro F1 : {ens_f1:.4f}\n")

    print(f"\n✓ Results saved to {OUTPUT_DIR}")
    print(f"✓ Summary: {summary_path}")
