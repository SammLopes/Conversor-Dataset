"""
augmentation.py
---------------
Rotinas de Data Augmentation consistentes entre imagem e máscara.
Usa Albumentations para gerar variações.

Uso:
    from core.transunet.augmentation import get_augmentation
    transform = get_augmentation()
    augmented = transform(image=img, mask=mask)
    img_aug, mask_aug = augmented["image"], augmented["mask"]
"""

import albumentations as A

def get_augmentation():
    """Define pipeline de augmentation para imagens médicas."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ElasticTransform(p=0.2),
    ])
