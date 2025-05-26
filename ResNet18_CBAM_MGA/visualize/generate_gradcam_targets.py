import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ResNet18_CBAM_MGA.core.model import ResNet18_CBAM
from ResNet18_CBAM_MGA.core.config import CFG
from ResNet18_CBAM_MGA.core.transforms import val_transform
from ResNet18_CBAM_MGA.visualize import grad_cam  # 너가 가진 Grad-CAM 함수로 대체할 것

# Load CSV
df = pd.read_csv("/home/iujeong/lung_cancer/csv/allbb_noPoly.csv")
print(f"총 샘플 수: {len(df)}")

# 조건 선택: loss 상위 10개
top_k = 10
selected = df.sort_values(by="loss", ascending=False).head(top_k)

# 모델 로드
model_path = os.path.join(CFG.save_dir, CFG.model_save_name)
model = ResNet18_CBAM(num_classes=CFG.num_classes).to(CFG.device)
model.load_state_dict(torch.load(model_path, map_location=CFG.device))
model.eval()
print(f"✅ 모델 로드 완료: {model_path}")

# transform 정의
transform = transforms.Compose([
    transforms.ToTensor()
])

# 시각화 폴더 생성
os.makedirs("gradcam_outputs", exist_ok=True)

# Grad-CAM 시각화
for i, row in selected.iterrows():
    npy_path = row["file"]
    label = row["label"]
    pred = row["pred"]
    prob = row["prob"]
    loss = row["loss"]

    # 원본 이미지 로딩
    img = np.load(npy_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=0)
    elif img.shape[0] != 3:
        raise ValueError(f"❌ 3채널 아님: {npy_path}")
    img_tensor = torch.tensor(img).float().unsqueeze(0).to(CFG.device)

    # Grad-CAM 얻기
    cam = grad_cam(model, img_tensor)  # (H, W)

    # 시각화
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (img.shape[2], img.shape[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    ori_img = img[0]  # 채널 하나
    ori_img = (ori_img - ori_img.min()) / (ori_img.max() - ori_img.min() + 1e-8)
    ori_img = np.uint8(255 * ori_img)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(ori_img, 0.6, cam_heatmap, 0.4, 0)

    out_name = os.path.basename(npy_path).replace(".npy", f"_loss{loss:.3f}_pred{pred}_label{label}.png")
    cv2.imwrite(os.path.join("gradcam_outputs", out_name), overlay)
    print(f"📸 저장 완료: {out_name}")