"""
Training engine for Brain Tumor MRI classification.

Strategy:
  Phase 1 (Warmup, frozen backbone)  - 5 epochs, LR 1e-3, only head trained
  Phase 2 (Full fine-tune, cosine LR) - up to 50 epochs with early stopping (patience=12)

Key techniques:
  - Label smoothing (0.08) reduces overconfidence on a 4-class MRI problem
  - CosineAnnealingLR with warm restarts for stable convergence
  - Mixed precision (AMP) for speed on RTX 5060
  - Gradient clipping (max_norm=1.0) for stability
  - Best checkpoint saved on validation accuracy
"""

import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Training loop helpers ---------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, is_training, scheduler=None):
    model.train(is_training)
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            with autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = len(loader.dataset)
    return total_loss / n, accuracy_score(all_labels, all_preds)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    arch_name: str,
    output_dir: str,
    warmup_epochs: int = 5,
    finetune_epochs: int = 50,
    warmup_lr: float = 1e-3,
    finetune_lr: float = 4e-5,
    patience: int = 12,
    label_smoothing: float = 0.08,
):
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(DEVICE)
    scaler = GradScaler('cuda')

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Phase 1: Warmup (frozen backbone, head only)
    print(f"\n[{arch_name}] Phase 1 - Warmup ({warmup_epochs} epochs, backbone frozen)")
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'head_drop' not in name:
            param.requires_grad = False

    opt_warm = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=warmup_lr, weight_decay=1e-4
    )
    warm_scheduler = LinearLR(opt_warm, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    for epoch in range(warmup_epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt_warm, scaler, is_training=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, opt_warm, scaler, is_training=False)
        warm_scheduler.step()
        elapsed = time.time() - t0
        print(f"  Warmup E{epoch+1:02d} | Loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"Acc {tr_acc*100:.2f}%/{va_acc*100:.2f}% | {elapsed:.1f}s", flush=True)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

    # Phase 2: Full fine-tune
    print(f"\n[{arch_name}] Phase 2 - Full fine-tune (up to {finetune_epochs} epochs)")
    for param in model.parameters():
        param.requires_grad = True

    # Layer-wise LR decay: backbone at 0.1x head LR for stability
    backbone_params = [p for n, p in model.named_parameters()
                       if 'classifier' not in n and 'head_drop' not in n]
    head_params = [p for n, p in model.named_parameters()
                   if 'classifier' in n or 'head_drop' in n]

    opt_ft = AdamW([
        {'params': backbone_params, 'lr': finetune_lr * 0.1},
        {'params': head_params,     'lr': finetune_lr}
    ], weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    cosine_scheduler = CosineAnnealingWarmRestarts(
        opt_ft, T_0=10 * steps_per_epoch, T_mult=1, eta_min=finetune_lr * 0.01
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(finetune_epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt_ft, scaler,
                                    is_training=True, scheduler=cosine_scheduler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, opt_ft, scaler,
                                    is_training=False)
        elapsed = time.time() - t0
        print(f"  FT E{epoch+1:02d}/{finetune_epochs} | Loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"Acc {tr_acc*100:.2f}%/{va_acc*100:.2f}% | {elapsed:.1f}s", flush=True)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            ckpt_path = os.path.join(output_dir, f'{arch_name}_best.pth')
            torch.save(best_state, ckpt_path)
            print(f"    >> New best val acc: {best_val_acc*100:.3f}% -- checkpoint saved", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {patience} non-improving epochs.")
                break

    # Reload best weights
    model.load_state_dict(best_state)
    return model, history


# --- Evaluation --------------------------------------------------------------

def evaluate_model(model, test_loader, class_names, arch_name, output_dir):
    model.eval().to(DEVICE)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            with autocast('cuda'):
                logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')

    print(f"\n{'='*60}")
    print(f"[{arch_name}] Test Accuracy : {acc*100:.3f}%")
    print(f"[{arch_name}] Macro F1 Score: {f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f'{arch_name} - Test Accuracy {acc*100:.2f}%  |  F1 {f1:.4f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{arch_name}_confusion_matrix.png'), dpi=150)
    plt.close()

    return {
        'accuracy': acc, 'f1': f1,
        'preds': all_preds, 'labels': all_labels, 'probs': all_probs
    }


def plot_training_history(history, arch_name, output_dir):
    epochs = range(1, len(history['train_acc']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history['train_acc'], label='Train Acc')
    axes[0].plot(epochs, history['val_acc'], label='Val Acc')
    axes[0].set_title(f'{arch_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_loss'], label='Train Loss')
    axes[1].plot(epochs, history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{arch_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{arch_name}_history.png'), dpi=150)
    plt.close()
