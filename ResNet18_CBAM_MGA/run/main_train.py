import os
import pandas as pd
from datetime import datetime
from glob import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ResNet18_CBAM_MGA.core.model import ResNet18_CBAM
from ResNet18_CBAM_MGA.core.dataset import CTDataset, load_bbox_dict, extract_label
from ResNet18_CBAM_MGA.core.transforms import train_transform, val_transform
from ResNet18_CBAM_MGA.train.train_mga import train_one_epoch, validate

from ResNet18_CBAM_MGA.core.config import CFG
CFG.use_mga = False
# CFG.model_save_name = "r18_cbam_sc_weight.pth"
# 설정 파일 (config.py > CFG)에서 MGA 사용 비활성화

from ResNet18_CBAM_MGA.core.utils import generate_run_name

# 예: MGA O, 회전 X, focal loss, soft class weight
model_name, log_name = generate_run_name(
    mga=False,
    rotate_mask=False,
    loss="ce",
    weight="sc",
    note="no_mask_no_rotate"
) # 실험 설정 기반으로 고유 모델 이름과 로그 파일 이름 생성

CFG.model_save_name = model_name  # 📦 자동 저장 경로
log_path = os.path.join("logs", log_name)

def main(): # 메인 학습 함수 정의
    # 🔹 시드 고정, 저장 폴더 확보
    CFG.ensure_dirs()
    os.makedirs("logs", exist_ok=True)
    # 필요한 디렉토리 생성 (없으면 만들어줌)

    # 🔹 데이터 로딩 및 분할
    all_files = glob(os.path.join(CFG.data_root, "**/*.npy"), recursive=True)
    file_label_pairs = [(f, extract_label(f)) for f in all_files if extract_label(f) is not None]
    # 전체 CT 슬라이스 파일 불러오고, 라벨(score 3 제외) 붙이기
    files, labels = zip(*file_label_pairs)
    # 파일 경로와 라벨 분리

    train_files, val_files, train_labels, val_labels = train_test_split(
        files, labels, test_size=0.2, random_state=CFG.seed
    )   # 학습/검증 데이터 분할 (8:2)
    bbox_dict = load_bbox_dict(CFG.bbox_csv)
    # BBox 좌표 csv를 dict로 로딩

    # 데이터 불러오기
    train_dataset = CTDataset(train_files, train_labels, bbox_dict, transform=train_transform)
    val_dataset   = CTDataset(val_files,   val_labels,   bbox_dict, transform=val_transform)

    # 데이터 로더 (학습은 shuffle on, 검증은 고정)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)

    # 🔹 모델 및 학습 세팅
    model = ResNet18_CBAM(num_classes=CFG.num_classes).to(CFG.device)   # ResNet18 + CBAM 모델 정의 및 GPU로 보내기
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.6653, 0.3347], device=CFG.device))
    # Soft class weight 적용한 CrossEntropyLoss 정의
    # 왜???? 데이터가 불균형해서 
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    # Adam 옵티마이저 설정

    # 🔹 스케줄러 설정
    if CFG.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    elif CFG.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None
    # 스케줄러 종류에 따라 학습률 조절 방식 설정

    # 🔹 학습 루프
    best_acc = 0.0
    history = []
    # 최고 정확도 저장용 변수와 로그 저장 리스트 초기화

    for epoch in range(CFG.epochs): # 전체 epoch 만큼 반복
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_acc = validate(model, val_loader)
        # 1 epoch 학습하고 검증 정확도 계산

        print(f"📌 Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        # epoch 당 결과 출력

        # 🔹 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(CFG.save_dir, CFG.model_save_name)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to: {save_path}")
            # 이전보다 성능 좋으면 모델 저장

        # 🔹 스케줄러 업데이트 (에폭 끝날 때)
        if scheduler is not None:
            scheduler.step()
            # 스케줄러 적용

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_loss": train_loss,
        })  # 로그 기록 저장

    # 🔹 로그 저장
    df = pd.DataFrame(history)
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv("logs/train_log1.csv", index=False)
    # 결과를 CSV로 저장

    print("📁 logs/train_log1.csv 저장 완료")
    print("🎉 학습 종료")
    # 완료 메시지 출력

    
if __name__ == "__main__":
    main()
# 스크립트 직접 실행 시 main() 호출