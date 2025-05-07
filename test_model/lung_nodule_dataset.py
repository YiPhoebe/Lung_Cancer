# LungNoduleSliceDataset - NPY 기반 버전

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A

class LungNoduleSliceDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

        self.samples = []
        for _, row in self.df.iterrows():
            image_path = os.path.join(self.image_root, row['image_path'])
            if os.path.exists(image_path):
                # ✅ 회귀용 score 사용
                self.samples.append((image_path, row['score']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        img_np = np.load(image_path).astype(np.float32)

        # Resize 강제 적용
        img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LINEAR)

        if self.transform:
            img_np = self.transform(image=(img_np * 255).astype(np.uint8))['image'] / 255.0

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=0)

        return img_np, label
