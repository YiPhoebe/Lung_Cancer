### ✅ main_test.py (CBAM ❌ / MGA ❌ / 마스크 회전 ❌)

import os
import torch
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ResNet18_CBAM_MGA.core.config import CFG
from ResNet18_CBAM_MGA.core.model import ResNet18_CBAM
from ResNet18_CBAM_MGA.core.dataset_no_rotate import CTDataset_NoRotate as DatasetClass, extract_label, load_bbox_dict
from ResNet18_CBAM_MGA.core.transform_no_rotate import val_transform
from ResNet18_CBAM_MGA.train.eval import (
    evaluate_with_tta, evaluate_without_tta,
    compute_metrics, save_metrics_to_csv
)

# 설정
CFG.use_mga = False
CFG.use_rotate_mask = False
CFG.use_tta = False  # TTA 사용 여부 (True면 evaluate_with_tta 사용)

# 모델 로드
CFG.model_save_name = "r18_cbam_norotate_sc_no_mask_no_rotate_0520_1222.pth"
model_path = os.path.join(CFG.save_dir, CFG.model_save_name)
model = ResNet18_CBAM(num_classes=CFG.num_classes).to(CFG.device)
model.load_state_dict(torch.load(model_path, map_location=CFG.device))
print(f"✅ 모델 로드 완료: {model_path}")

# 데이터 준비
all_files = sorted(glob(os.path.join(CFG.data_root, "**/*.npy"), recursive=True))
file_label_pairs = [(f, extract_label(f)) for f in all_files if extract_label(f) is not None]
files, labels = zip(*file_label_pairs)
_, test_files, _, test_labels = train_test_split(files, labels, test_size=0.2, random_state=CFG.seed)

bbox_dict = load_bbox_dict(CFG.bbox_csv)
test_dataset = DatasetClass(test_files, test_labels, bbox_dict, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)

# 평가
if CFG.use_tta:
    print("🔁 TTA 기반 테스트 수행 중...")
    y_true, y_probs, y_pred = evaluate_without_tta(model, DatasetClass, test_files, test_labels, bbox_dict)
else:
    print("✅ 일반 테스트 수행 중...")
    y_true, y_probs, y_pred = evaluate_without_tta(model, test_dataset, test_loader)

# 메트릭 계산 및 저장
metrics = compute_metrics(y_true, y_probs, y_pred)

print("📊 최종 평가 지표:")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f" - {k}: {v:.4f}")
    else:
        print(f" - {k}: {v}")

save_metrics_to_csv(metrics)