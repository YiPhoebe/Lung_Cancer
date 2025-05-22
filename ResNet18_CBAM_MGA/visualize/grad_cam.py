import matplotlib.pyplot as plt
import cv2
import numpy as np

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