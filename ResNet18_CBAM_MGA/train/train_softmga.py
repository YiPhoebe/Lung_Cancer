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
    for images, labels, masks in pbar:
        images, labels, masks = images.to(CFG.device), labels.to(CFG.device), masks.to(CFG.device)

        outputs = model(images)
        ce_loss = criterion(outputs, labels)

        # MGA Loss 적용 (layer3[1]에 CBAM이 있다고 가정)
        attn_map = model.layer3[1].cbam.last_attention if CFG.use_mga else None
        if attn_map is not None:
            # [B, 1, H, W] → [B, H*W]
            # [B, 1, H, W] → [B, H*W]
            attn_map = F.interpolate(attn_map, size=CFG.input_size, mode='bilinear', align_corners=False)
            attn_map = attn_map.view(attn_map.size(0), -1)
            masks = masks.view(masks.size(0), -1)

            # 마스크를 확률 분포처럼 정규화 (0이 아닌 부분만 집중)
            mask_probs = masks / (masks.sum(dim=1, keepdim=True) + 1e-6)
            attn_probs = F.log_softmax(attn_map, dim=1)

            # KL Divergence Loss (soft constraint)
            attn_loss = F.kl_div(attn_probs, mask_probs, reduction='batchmean')
            loss = ce_loss + lambda_mga * attn_loss   # ← 이 줄 꼭 필요해!!
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
        for images, labels, _ in dataloader:
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total
