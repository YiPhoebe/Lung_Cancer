import csv
import os
import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef
)
from ResNet18_CBAM_MGA.core.config import CFG
from ResNet18_CBAM_MGA.core.transforms import tta_transforms


# ✅ TTA 적용 평가 함수

def evaluate_with_tta(model, dataset_cls, file_paths, labels, bbox_dict):
    model.eval()
    all_probs = []
    all_preds = []

    for i, tta_transform in enumerate(tta_transforms):
        print(f"🔁 TTA #{i+1} transform 적용 중...")

        tta_dataset = dataset_cls(file_paths, labels, bbox_dict, transform=tta_transform)
        loader = DataLoader(tta_dataset, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)

        probs, preds = [], []

        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(CFG.device)
                outputs = model(images)
                prob = F.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1)

                probs.append(prob.cpu())
                preds.append(pred.cpu())

        all_probs.append(torch.cat(probs, dim=0))
        all_preds.append(torch.cat(preds, dim=0))

    avg_probs = torch.stack(all_probs).mean(dim=0)
    avg_preds = avg_probs.argmax(dim=1)

    return labels, avg_probs[:, 1].numpy(), avg_preds.numpy()


# ✅ 일반 평가 함수

def evaluate_without_tta(model, dataset, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(CFG.device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


# ✅ 평가지표 계산 함수

def compute_metrics(y_true, y_probs, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp + 1e-6)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics_dict = {
        "timestamp": now,
        "model": "ResNet18_CBAM_MGA",
        "phase": "test",
        "accuracy (정확도)": round(acc, 4),
        "AUC (곡선 아래 면적)": round(auc, 4),
        "precision (정밀도)": round(precision, 4),
        "recall (재현율)": round(recall, 4),
        "specificity (특이도)": round(specificity, 4),
        "F1 (F1 점수)": round(f1, 4),
        "balanced_acc (균형 정확도)": round(balanced_acc, 4),
        "mcc (매튜 상관계수)": round(mcc, 4),
        "tn (진음성)": tn,
        "fp (위양성)": fp,
        "fn (위음성)": fn,
        "tp (진양성)": tp,
        "pth 파일 이름": CFG.model_save_name
    }

    return metrics_dict


# ✅ 결과 저장 함수

def save_metrics_to_csv(metrics_dict, save_path="logs/test_metrics.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.isfile(save_path)

    if "timestamp" not in metrics_dict:
        metrics_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

    print(f"📁 평가 결과 저장 완료: {save_path}")