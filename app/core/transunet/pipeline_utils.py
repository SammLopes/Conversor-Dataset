"""
pipeline_utils.py
-----------------
Funções auxiliares para o pipeline do TransUNet (PyTorch).
Contém a lógica de Data Augmentation (Step 3).
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Importa o augmentation.py (deve estar na mesma pasta)
try:
    from augmentation import get_augmentation
except ImportError:
    print("❌ Erro: 'augmentation.py' não encontrado.")
    sys.exit(1)

def create_augmented_dataset(
    img_dir, 
    mask_dir, 
    output_dir, 
    num_augmentations=10, 
    target_size=(224, 224)
):
    """
    Carrega imagens e pseudo-máscaras, aplica augmentation N vezes
    e salva o novo dataset.
    """
    
    aug_img_dir = os.path.join(output_dir, "images")
    aug_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)
    
    aug = get_augmentation()
    
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    if not img_files:
        print(f"❌ Erro: Nenhuma imagem encontrada em '{img_dir}'")
        return

    print(f"Gerando {len(img_files) * (num_augmentations + 1)} amostras aumentadas...")
    
    for img_file in tqdm(img_files, desc="Augmenting dataset"):
        img_path = os.path.join(img_dir, img_file)
        # Assume que a máscara tem o mesmo nome, mas extensão .png
        mask_file_name = os.path.splitext(img_file)[0] + ".png" 
        mask_path = os.path.join(mask_dir, mask_file_name) 

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Erro ao carregar imagem {img_file}, pulando...")
            continue
        if mask is None:
            print(f"⚠️ Erro ao carregar máscara {mask_file_name}, pulando...")
            continue
            
        # Redimensiona ANTES da augmentation
        img_resized = cv2.resize(img, target_size)
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Salva a versão original (sempre bom ter)
        cv2.imwrite(os.path.join(aug_img_dir, f"{base_name(img_file)}_orig.png"), img_resized)
        cv2.imwrite(os.path.join(aug_mask_dir, f"{base_name(img_file)}_orig.png"), mask_resized)

        # Gera N versões aumentadas
        for i in range(num_augmentations):
            augmented = aug(image=img_resized, mask=mask_resized)
            img_aug, mask_aug = augmented["image"], augmented["mask"]
            
            # Gera nome do arquivo (ex: img_01_aug_0.png)
            aug_name = f"{base_name(img_file)}_aug_{i}.png"
            
            cv2.imwrite(os.path.join(aug_img_dir, aug_name), img_aug.astype(np.uint8))
            cv2.imwrite(os.path.join(aug_mask_dir, aug_name), mask_aug.astype(np.uint8))

    print("✅ Dataset aumentado criado com sucesso.")

def base_name(file_name):
    """Retorna o nome do arquivo sem extensão."""
    return os.path.splitext(file_name)[0]

