import os
import torch
import numpy as np
import argparse
from ResNet18_CBAM_MGA.core.model import ResNet18_CBAM
from ResNet18_CBAM_MGA.core.config import CFG
from ResNet18_CBAM_MGA.core.transforms import val_transform
from ResNet18_CBAM_MGA.core.dataset import create_mask
import cv2

def load_image(path):
    img = np.load(path)
    img = np.clip(img, -1000, 400)
    img = (img + 1000) / 1400.
    img = img.astype(np.float32)[..., None]
    img = img.transpose(2, 0, 1)  # (1, H, W)
    img = val_transform(image=img.squeeze(0))['image']  # ToTensorV2 사용
    return img.unsqueeze(0)  # (1, 1, H, W)

def inference(model_path, input_path):
    model = ResNet18_CBAM(num_classes=CFG.num_classes).to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    img_tensor = load_image(input_path).to(CFG.device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
        pred = output.argmax(1).item()

    print(f"📦 예측 결과: {pred} (양성 확률: {prob:.4f})")
    return pred, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input .npy file path')
    parser.add_argument('--model', type=str, default=os.path.join(CFG.save_dir, CFG.model_save_name))
    args = parser.parse_args()

    inference(args.model, args.input)
