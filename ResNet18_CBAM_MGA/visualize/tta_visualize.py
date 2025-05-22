import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from ResNet18_CBAM_MGA.core.model import ResNet18_CBAM
from ResNet18_CBAM_MGA.core.config import CFG
from ResNet18_CBAM_MGA.core.transforms import tta_transforms

def load_base_image(path):
    img = np.load(path)
    img = np.clip(img, -1000, 400)
    img = (img + 1000) / 1400.
    img = img.astype(np.float32)[..., None]
    img = img.transpose(2, 0, 1)  # (1, H, W)
    return img.squeeze(0)  # (H, W)

def visualize_tta(model_path, input_path):
    model = ResNet18_CBAM().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    base_img = load_base_image(input_path)
    fig, axes = plt.subplots(1, len(tta_transforms), figsize=(5 * len(tta_transforms), 5))

    for i, tform in enumerate(tta_transforms):
        aug = tform(image=base_img)
        img = aug['image'].unsqueeze(0).to(CFG.device)

        with torch.no_grad():
            output = model(img)
            prob = torch.softmax(output, dim=1)[0, 1].item()

        show_img = aug['image'].squeeze(0).cpu().numpy()
        axes[i].imshow(show_img, cmap='gray')
        axes[i].set_title(f"TTA {i+1}\n양성 확률: {prob:.2f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("tta_result.png")
    print("🖼️ TTA 예측 시각화 저장 완료: tta_result.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input .npy file path')
    parser.add_argument('--model', type=str, default=os.path.join(CFG.save_dir, CFG.model_save_name))
    args = parser.parse_args()

    visualize_tta(args.model, args.input)
