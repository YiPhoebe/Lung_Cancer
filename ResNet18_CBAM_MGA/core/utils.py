import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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