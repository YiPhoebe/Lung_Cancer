import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

def visualize_gradcam(image, cam, save_path=None):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image = image.permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())
    overlay = heatmap + image
    overlay = overlay / np.max(overlay)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("CAM")
    plt.imshow(cam, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



def generate_cam(model, image_tensor):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # 💡 hook 걸 layer 정확히 지정
    target_layer = model.layer4[1].conv2
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    model.eval()
    model.zero_grad()
    output = model(image_tensor)

    class_idx = torch.argmax(output)
    output[0, class_idx].backward(retain_graph=True)

    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)

    print(f"[DEBUG] act shape: {act.shape}, grad shape: {grad.shape}")  # 👈 여기!

    weights = grad.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * act, dim=0)

    print(f"[DEBUG] CAM max: {cam.max():.4f}, min: {cam.min():.4f}")  # 👈 그리고 여기!

    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    forward_handle.remove()
    backward_handle.remove()
    return cam.detach()

