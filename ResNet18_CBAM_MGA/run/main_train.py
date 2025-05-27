### ✅ main_train.py (CBAM ❌ / MGA ❌ / 마스크 회전 ❌)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pandas as pd
from datetime import datetime
from glob import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ResNet18_CBAM_MGA.core.model import ResNet18
from ResNet18_CBAM_MGA.core.dataset import CTDataset, load_bbox_dict, extract_label
from ResNet18_CBAM_MGA.core.transforms import train_transform, val_transform
from ResNet18_CBAM_MGA.train.train_mga import train_one_epoch, validate
from ResNet18_CBAM_MGA.core.utils import generate_run_name

from ResNet18_CBAM_MGA.core.config import CFG
CFG.use_mga = False
# CFG.model_save_name = "r18_cbam_mga_sc_weight_rotate.pth"

from ResNet18_CBAM_MGA.core.utils import generate_run_name

# 예: MGA O, 회전 O, CE loss, soft class weight
model_name, log_name = generate_run_name(
    mga=False,
    rotate_mask=False,
    loss="ce",
    weight="sc",
    note="mask_rotate"
)

CFG.model_save_name = model_name  # 📦 자동 저장 경로
log_path = os.path.join("logs", log_name)

def main():
    # 🔹 시드 고정, 저장 폴더 확보
    CFG.ensure_dirs()
    os.makedirs("logs", exist_ok=True)

    # 🔹 데이터 로딩 및 분할
    all_files = sorted(glob(os.path.join(CFG.data_root, "**/*.npy"), recursive=True))
    file_label_pairs = [(f, extract_label(f)) for f in all_files if extract_label(f) is not None]
    files, labels = zip(*file_label_pairs)

    # 🔹 train_*.csv 저장
    # 전체 파일 리스트 저장 (train_malA.csv)
    csv_rows = []
    for f, y in zip(files, labels):
        rel_path = os.path.relpath(f, CFG.data_root)
        csv_rows.append({"filename": rel_path, "label": y})

    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv("train_malA.csv", index=False)
    print("✅ train_malA.csv 저장 완료")

    # 🔹 train_test_split 먼저
    train_files, val_files, train_labels, val_labels = train_test_split(
        files, labels, test_size=0.2, random_state=CFG.seed
    )

    # 🔹 그 다음에 train/val split csv 저장
    pd.DataFrame({
        "filename": [os.path.relpath(f, CFG.data_root) for f in train_files],
        "label": train_labels
    }).to_csv("train_splitA.csv", index=False)

    pd.DataFrame({
        "filename": [os.path.relpath(f, CFG.data_root) for f in val_files],
        "label": val_labels
    }).to_csv("val_splitA.csv", index=False)

    print("✅ train_splitA.csv / val_splitA.csv 저장 완료")
    
    bbox_dict = load_bbox_dict(CFG.bbox_csv)

    train_dataset = CTDataset(train_files, train_labels, bbox_dict, transform=train_transform)
    val_dataset   = CTDataset(val_files,   val_labels,   bbox_dict, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)

    # 🔹 모델 및 학습 세팅
    model = ResNet18(num_classes=CFG.num_classes).to(CFG.device)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.6653, 0.3347], device=CFG.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)

    # 🔹 스케줄러 설정
    if CFG.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    elif CFG.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    # 🔹 학습 루프
    best_acc = 0.0
    history = []

    for epoch in range(CFG.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_acc = validate(model, val_loader)

        print(f"📌 Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # 🔹 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(CFG.save_dir, CFG.model_save_name)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to: {save_path}")

        # 🔹 스케줄러 업데이트 (에폭 끝날 때)
        if scheduler is not None:
            scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_loss": train_loss,
        })

    # 🔹 로그 저장
    df = pd.DataFrame(history)
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(os.path.join("logs", f"train_log_{CFG.model_save_name.replace('.pth', '')}.csv"), index=False)
    print("📁 logs/train_log_.csv 저장 완료")
    print("🎉 학습 종료")

    
if __name__ == "__main__":
    main()
