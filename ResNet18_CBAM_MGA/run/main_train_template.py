# ============ 🔧 Imports & Setup ============
import os, sys, torch, pandas as pd
from datetime import datetime
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ResNet18_CBAM_MGA.core.config import CFG
from ResNet18_CBAM_MGA.core.model import ResNet18_CBAM
from ResNet18_CBAM_MGA.core.dataset import CTDataset, load_bbox_dict, extract_label
from ResNet18_CBAM_MGA.core.transforms import train_transform, val_transform
from ResNet18_CBAM_MGA.train.train_mga import train_one_epoch, validate
from ResNet18_CBAM_MGA.core.utils import generate_run_name

# ============ 🔧 Config ============
CFG.use_mga = False
model_name, log_name = generate_run_name(mga=False, rotate_mask=False, loss="ce", weight="sc", note="no_mask_no_rotate")
CFG.model_save_name = model_name
log_path = os.path.join("logs", log_name)
CFG.ensure_dirs()
os.makedirs("logs", exist_ok=True)

# ============ 📦 Data Load ============
all_files = glob(os.path.join(CFG.data_root, "**/*.npy"), recursive=True)
file_label_pairs = [(f, extract_label(f)) for f in all_files if extract_label(f) is not None]
files, labels = zip(*file_label_pairs)

train_files, val_files, train_labels, val_labels = train_test_split(files, labels, test_size=0.2, random_state=CFG.seed)
bbox_dict = load_bbox_dict(CFG.bbox_csv)

train_dataset = CTDataset(train_files, train_labels, bbox_dict, transform=train_transform)
val_dataset   = CTDataset(val_files,   val_labels,   bbox_dict, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ============ 🧠 Model Setup ============
model = ResNet18_CBAM(num_classes=CFG.num_classes).to(CFG.device)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.6653, 0.3347], device=CFG.device))
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)

scheduler = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs),
    "step":   torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
}.get(CFG.scheduler, None)

# ============ 🔁 Train Loop ============
best_acc, history = 0.0, []
for epoch in range(CFG.epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
    val_acc = validate(model, val_loader)
    print(f"📌 Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        save_path = os.path.join(CFG.save_dir, CFG.model_save_name)
        torch.save(model.state_dict(), save_path)
        print(f"✅ Best model saved: {save_path}")
    
    if scheduler: scheduler.step()

    history.append({
        "epoch": epoch + 1,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
    })

# ============ 📊 Save Logs ============
df = pd.DataFrame(history)
df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df.to_csv("logs/train_log4.csv", index=False)
print("📁 logs/train_log4.csv 저장 완료")
print("🎉 학습 종료")