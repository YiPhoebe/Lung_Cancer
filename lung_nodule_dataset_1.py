# lung_nodule_dataset.py (df 직접 입력 지원 + HU Clipping 포함)

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import cv2

class LungNoduleSliceDataset(Dataset):
    def __init__(self, df=None, csv_path=None, image_root=None, transform=None, label_threshold=3.0, hu_clip=(-1000, 400), resize=(224, 224)):
        if df is not None:
            self.df = df.reset_index(drop=True)
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either df or csv_path must be provided.")

        self.image_root = image_root
        self.transform = transform
        self.label_threshold = label_threshold
        self.hu_clip = hu_clip
        self.resize = resize

        if 'binary_label' not in self.df.columns:
            self.df['binary_label'] = (self.df['malignancy'] >= label_threshold).astype(int)

        self.samples = []
        for _, row in self.df.iterrows():
            pid = row['patient_id']
            nid = row['nodule_index']
            image_path = os.path.join(image_root, pid, f"nodule-{nid}", "images", "slice-1.png")
            if os.path.exists(image_path):
                self.samples.append((image_path, row['binary_label']))

    def __len__(self):
        return len(self.samples)

    def preprocess_slice(self, img_np):
        img_np = np.clip(img_np, self.hu_clip[0], self.hu_clip[1])
        img_np = (img_np - self.hu_clip[0]) / (self.hu_clip[1] - self.hu_clip[0])
        img_np = np.clip(img_np, 0.0, 1.0)
        img_np = cv2.resize(img_np, self.resize, interpolation=cv2.INTER_LINEAR)
        return img_np

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        img = Image.open(image_path).convert("L")
        img_np = np.array(img).astype(np.float32)
        img_np = self.preprocess_slice(img_np)

        if self.transform:
            img_np = self.transform(image=(img_np * 255).astype(np.uint8))['image'] / 255.0

        img_tensor = np.expand_dims(img_np, axis=0)  # [1, H, W]
        return img_tensor, label


if __name__ == "__main__":
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=15, p=0.5),
    ])

    df = pd.read_csv("/data2/lijin/lidc-prep/nodule_matched.csv")
    df['binary_label'] = df['malignancy'].astype(int)

    dataset = LungNoduleSliceDataset(
        df=df,
        image_root="/data2/lijin/lidc-prep/LIDC-IDRI-slices",
        transform=transform
    )

    print(f"Total samples: {len(dataset)}")
    img, label = dataset[0]
    print("Sample shape:", img.shape, "Label:", label)
