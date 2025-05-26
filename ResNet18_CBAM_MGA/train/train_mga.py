import torch
from tqdm import tqdm
import torch.nn.functional as F
import gc

from ResNet18_CBAM_MGA.core.config import CFG

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, bbox_dict=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    lambda_mga = CFG.lambda_mga_initial + (CFG.lambda_mga_final - CFG.lambda_mga_initial) * (epoch / CFG.epochs)

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch+1}]")
    for images, labels, masks, _, _ in pbar:
        images, labels, masks = images.to(CFG.device), labels.to(CFG.device), masks.to(CFG.device)

        outputs = model(images)
        ce_loss = criterion(outputs, labels)

        # MGA Loss 적용 (layer3[1]에 CBAM이 있다고 가정)
        attn_map = model.layer3[1].cbam.last_attention if CFG.use_mga else None
        if attn_map is not None:
            attn_map = F.interpolate(attn_map, size=CFG.input_size, mode='bilinear', align_corners=False).squeeze(1)
            masks = masks.squeeze(1) if masks.ndim == 4 and masks.shape[1] == 1 else masks  # ✅ 안전 처리
            masks = masks.squeeze(-1) if masks.ndim == 4 and masks.shape[-1] == 1 else masks  # ✅ 여기도
            attn_loss = F.mse_loss(attn_map, masks)
            loss = ce_loss + lambda_mga * attn_loss
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    torch.cuda.empty_cache(); gc.collect()
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _, _, _ in dataloader:
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total
