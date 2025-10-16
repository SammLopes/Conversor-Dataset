"""
pseudo_masks.py
----------------
Gera pseudo-máscaras a partir de um modelo TransUNet pré-treinado.

Uso:
    from core.transunet.pseudo_masks import generate_pseudo_masks
    
    model = ... # vindo do pretrain_model ou carregado de disco
    generate_pseudo_masks(model, input_dir="datasets/BrainTC/raw", output_dir="datasets/BrainTC/pseudo_masks")

"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(img, target_size=(224, 224)):
    """Pré-processa a imagem para o modelo TransUNet"""
    img_resized = cv2.resize(img, target_size)
    img_norm = img_resized.astype(np.float32) / 255.0
    # Adiciona dimensões: (batch, H, W, C)
    return np.expand_dims(np.expand_dims(img_norm, axis=-1), axis=0)


def generate_pseudo_masks(model, input_dir: str, output_dir: str, target_size=(224, 224)):
    """
    Gera pseudo-máscaras para imagens sem rótulos.

    Args:
        model: modelo TransUNet pré-treinado
        input_dir (str): diretório com as imagens originais
        output_dir (str): diretório onde salvar as pseudo-máscaras
        target_size (tuple): tamanho esperado pelo modelo
    """
    os.makedirs(output_dir, exist_ok=True)

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_file in tqdm(img_files, desc="Gerando pseudo-máscaras"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Erro ao carregar {img_file}, pulando...")
            continue

        # Pré-processa e faz inferência
        img_prep = preprocess_image(img, target_size)
        pred = model.predict(img_prep, verbose=0)

        # Obtém a classe de cada pixel (argmax do mapa de probabilidade)
        mask_pred = np.argmax(pred[0], axis=-1).astype(np.uint8)

        # Redimensiona de volta ao tamanho original
        mask_resized = cv2.resize(mask_pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Salva a máscara
        out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(out_path, mask_resized)

        print(f"✅ Pseudo-máscara salva: {out_path}")
