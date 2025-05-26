import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split
from ResNet18_CBAM_MGA.core.dataset import extract_label

# ✅ 시드 고정
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ✅ Confusion Matrix 시각화
def plot_confusion_matrix(cm, class_names=["Negative", "Positive"], save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Confusion matrix 저장됨: {save_path}")
    else:
        plt.show()

# ✅ 텐서 → 넘파이 변환 (1채널용)
def tensor_to_numpy(tensor):
    return tensor.detach().cpu().squeeze().numpy()

# 📁 CFG.model_save_name 자동으로 저장되게 관리하는 유틸
def generate_run_name(
    model="r18_cbam",
    mga=True,
    rotate_mask=True,
    loss="ce",
    weight="sc",
    note=None
):
    parts = [model]

    if mga:
        parts.append("mga")
    if not rotate_mask:
        parts.append("norotate")
    if weight:
        parts.append(weight)
    if loss and loss != "ce":
        parts.append(loss)

    if note:
        parts.append(note)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    parts.append(timestamp)

    name = "_".join(parts)
    return name + ".pth", f"train_log_{name}.csv"


def split_by_patient_3way(all_files, val_size=0.1, test_size=0.2, random_state=42):
    """환자 단위로 train/val/test 분할"""
    patient_dict = defaultdict(list)

    for path in all_files:
        fname = os.path.basename(path)
        pid = os.path.basename(os.path.dirname(path))
        label = extract_label(fname)
        if label is not None:
            patient_dict[pid].append((path, label))

    pids = list(patient_dict.keys())
    print(f"✅ 전체 환자 수: {len(pids)}명")
    print(f"🧬 환자 ID 목록 예시: {pids[:5]}")

    if len(pids) < 3:
        raise ValueError(f"❗ 환자 수가 너무 적습니다: {len(pids)}명. 최소 3명 이상 필요합니다.")

    # test 나누고 나머지 → train+val
    trainval_pids, test_pids = train_test_split(pids, test_size=test_size, random_state=random_state)
    train_pids, val_pids = train_test_split(trainval_pids, test_size=val_size / (1 - test_size), random_state=random_state)

    def collect_files(pids):
        files, labels = [], []
        for pid in pids:
            for f, l in patient_dict[pid]:
                files.append(f)
                labels.append(l)
        return files, labels

    train_files, train_labels = collect_files(train_pids)
    val_files, val_labels = collect_files(val_pids)
    test_files, test_labels = collect_files(test_pids)

    print(f"📊 분할 완료: train={len(train_files)}개, val={len(val_files)}개, test={len(test_files)}개")

    return train_files, val_files, test_files, train_labels, val_labels, test_labels