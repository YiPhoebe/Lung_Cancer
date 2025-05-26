import os
import torch
import pandas as pd
from glob import glob
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ResNet18_CBAM_MGA.core.config import CFG
from ResNet18_CBAM_MGA.core.model import ResNet18_CBAM
from ResNet18_CBAM_MGA.core.dataset import CTDataset, load_bbox_dict, extract_label
from ResNet18_CBAM_MGA.core.transforms import val_transform
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ✅ 설정
CFG.use_mga = False
CFG.use_rotate_mask = False
CFG.batch_size = 1
model_name = "/home/iujeong/lung_cancer/logs/r18_cbam_norotate_sc_no_mask_no_rotate_0522_2045.pth"  # ← 너가 저장한 모델 이름 정확히 넣어!
model_path = os.path.join(CFG.save_dir, model_name)

# ✅ 모델 로드
model = ResNet18_CBAM(num_classes=CFG.num_classes).to(CFG.device)
model.load_state_dict(torch.load(model_path, map_location=CFG.device))
model.eval()
print(f"✅ 모델 로드 완료: {model_path}")

# ✅ 데이터 준비
all_files = glob(os.path.join(CFG.data_root, "**/*.npy"), recursive=True)
file_label_pairs = [(f, extract_label(f)) for f in all_files if extract_label(f) is not None]
files, labels = zip(*file_label_pairs)

bbox_dict = load_bbox_dict(CFG.bbox_csv)
dataset = CTDataset(files, labels, bbox_dict, transform=val_transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# ✅ 저장 리스트
records = []

# ✅ 평가 시작
with torch.no_grad():
    for images, labels, _, fnames, paths in tqdm(loader):
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)

        output = model(images)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)

        loss = F.cross_entropy(output, labels).item()
        acc = (pred == labels).float().item()
        file = paths[0]

        records.append({
            "file": file,
            "label": labels.item(),
            "pred": pred.item(),
            "prob": probs[0, 1].item(),  # 클래스 1의 확률
            "loss": loss,
            "acc": acc,
        })

# ✅ CSV 저장
df = pd.DataFrame(records)
df.to_csv("samplewise_metrics.csv", index=False)
print("✅ samplewise_metrics.csv 저장 완료!")