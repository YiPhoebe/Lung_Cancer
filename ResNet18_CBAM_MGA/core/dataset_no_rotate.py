import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ResNet18_CBAM_MGA.core.config import CFG

# ====== Bounding Box → Binary Mask ======
def create_mask(bboxes, image_size=(224, 224)):
    mask = np.zeros(image_size, dtype=np.float32)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        mask[y_min:y_max, x_min:x_max] = 1.0
    return torch.tensor(mask, dtype=torch.float32)

# ====== Bounding Box CSV 로딩 ======
def load_bbox_dict(csv_path):
    df = pd.read_csv(csv_path)
    bbox_dict = {}
    for _, row in df.iterrows():
        pid = row['pid']
        slice_str = str(row['slice'])
        slice_idx = int(''.join(filter(str.isdigit, slice_str)))
        fname = f"{pid}_slice{slice_idx:04d}.npy"
        bbox = eval(row['bb'])  # ⚠️ 내부 신뢰 데이터일 때만 사용
        bbox_dict.setdefault(fname, []).append(bbox)
    return bbox_dict

# ====== 파일명에서 라벨 추출 ======
def extract_label(fname):
    try:
        score = int(fname.split("_")[-1].replace(".npy", ""))
        if score == 3:
            return None
        return int(score >= 4)  # score ≥ 4: 1 / ≤ 2: 0
    except:
        return None

# ====== CT Dataset 클래스 ======
class CTDataset_NoRotate(Dataset):
    def __init__(self, file_paths, labels, bbox_dict, transform=None):
        self.paths = file_paths
        self.labels = labels
        self.bbox_dict = bbox_dict
        self.transform = transform

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        fname = os.path.basename(path)

        # ---- CT 이미지 로드 및 정규화 ----
        img = np.load(path)
        img = np.clip(img, -1000, 400)
        img = (img + 1000) / 1400.
        img = img.astype(np.float32)  # (H, W)

        # ---- 마스크 로드 ----
        if fname in self.bbox_dict:
            mask = create_mask(self.bbox_dict[fname], image_size=img.shape)
        else:
            mask = np.zeros(img.shape, dtype=np.float32)

        # ---- transform ----
        if self.transform:
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)  # (H, W, 1)

            print(f"Before transform: {img.shape}")  # 🔍 로그
            aug = self.transform(image=img)
            img = aug['image']
            print(f"After transform: {img.shape}")  # 🔍 로그
        else:
            img = torch.tensor(img[None, ...], dtype=torch.float32)

        # ✅ 마스크는 항상 원본 기준
        # mask = torch.tensor(mask[None, ...], dtype=torch.float32)  ← ❌ 이건 H, W가 다름

        # 💡 마스크도 img와 같은 크기로 resize
        mask = torch.nn.functional.interpolate(
            torch.tensor(mask[None, None, ...]),  # [1, 1, H, W]
            size=CFG.input_size,
            mode='nearest'  # 이진 마스크이므로 bilinear ❌
        )[0]  # -> [1, H, W]

        return img, torch.tensor(label).long(), mask

    def __len__(self):
        return len(self.paths)