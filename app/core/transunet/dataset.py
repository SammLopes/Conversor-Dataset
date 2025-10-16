"""
dataset.py
----------
Carregamento de imagens e máscaras para o TransUNet.
"""

import os
import cv2
import numpy as np
from core.transunet.augmentation import get_augmentation

def load_image_mask_pairs(img_dir, mask_dir, target_size=(224, 224), augment=False):
    """
    Carrega imagens e máscaras em arrays numpy.

    Args:
        img_dir (str): diretório das imagens
        mask_dir (str): diretório das máscaras
        target_size (tuple): tamanho final (H, W)
        augment (bool): aplicar data augmentation
    """
    images, masks = [], []
    aug = get_augmentation() if augment else None

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"⚠️ Erro ao carregar {img_file}, pulando...")
            continue

        # Resize
        img_resized = cv2.resize(img, target_size).astype(np.float32) / 255.0
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Data augmentation
        if augment:
            augmented = aug(image=img_resized, mask=mask_resized)
            img_resized, mask_resized = augmented["image"], augmented["mask"]

        # Ajusta dimensões
        img_resized = np.expand_dims(img_resized, axis=-1)  # (H, W, 1)

        images.append(img_resized)
        masks.append(mask_resized)

    X = np.array(images)
    Y = np.array(masks)

    # One-hot encoding
    num_classes = len(np.unique(Y))
    Y_onehot = np.eye(num_classes)[Y]

    return X, Y_onehot, num_classes
