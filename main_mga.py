import os
import pandas as pd
from datetime import datetime
from core.config import CFG
from lung_cancer.ResNet18_CBAM_MGA.core.model import ResNet18_CBAM
from core.dataset import CTDataset, load_bbox_dict, extract_label
from lung_cancer.ResNet18_CBAM_MGA.core.transforms import val_transform, tta_transforms
from train.eval import evaluate_with_tta, compute_metrics
from glob import glob

import torch
from sklearn.model_selection import train_test_split

def main():
    # 시드 고정 및 경로 설정
    os.makedirs("logs", exist_ok=True)

    # 데이터 불러오기
    all_files = glob(os.path.join(CFG.data_root, "LIDC-IDRI-*", "*.npy"))
    file_label_pairs = [(f, extract_label(os.path.basename(f))) for f in all_files]
    file_label_pairs = [(f, l) for f, l in file_label_pairs if l is not None]
    files, labels = zip(*file_label_pairs)

    # 데이터 split
    train_files, temp_files, train_labels, temp_labels = train_test_split(files, labels, test_size=0.3, random_state=42)
    val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)

    # bbox 정보 로드
    bbox_dict = load_bbox_dict(CFG.bbox_csv)

    # 모델 로드
    model = ResNet18_CBAM().to(CFG.device)
    model_path = os.path.join(CFG.save_dir, CFG.model_save_name)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    print(f"✅ 모델 로드 완료: {model_path}")

    # TTA 평가
    probs, preds = evaluate_with_tta(model, CTDataset, test_files, test_labels, bbox_dict)
    metrics = compute_metrics(test_labels, probs, preds)

    # 출력
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    # CSV 저장
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"}
    }])
    df.to_csv("logs/test_metrics_tta.csv", index=False)
    print("📁 logs/test_metrics_tta.csv 저장 완료")

if __name__ == "__main__":
    main()
