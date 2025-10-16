"""
utils.py
--------
Funções auxiliares para pipeline do TransUNet.
"""

import cv2
import numpy as np
import os

def preprocess_image(img, target_size=(224, 224)):
    """Pré-processa imagem (resize + normalização)"""
    img_resized = cv2.resize(img, target_size).astype(np.float32) / 255.0
    return np.expand_dims(img_resized, axis=-1)

def save_mask(mask, out_path):
    """Salva máscara segmentada em disco"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mask.astype(np.uint8))

def visualize_pair(img, mask):
    """Exibe imagem + máscara lado a lado (para debug)"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title("Imagem")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="viridis", alpha=0.7)
    plt.title("Máscara")
    plt.axis("off")
    plt.show()
