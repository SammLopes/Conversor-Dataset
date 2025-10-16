"""
finetune.py
-----------
Realiza fine-tuning do modelo TransUNet em imagens de TC com pseudo-máscaras.

Uso:
    from core.transunet.finetune import finetune_model
    
    model = ... # vindo do pretrain_model ou carregado de disco
    finetune_model(
        dataset="BrainTC",
        pretrained=model,
        img_dir="datasets/BrainTC/raw",
        mask_dir="datasets/BrainTC/pseudo_masks",
        save_dir="models/transunet/finetuned"
    )
"""

import os
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_dataset(img_dir, mask_dir, target_size=(224, 224)):
    """Carrega imagens e máscaras em arrays numpy."""
    X, Y = [], []

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"⚠️ Problema com {img_file}, pulando...")
            continue

        # Redimensiona
        img_resized = cv2.resize(img, target_size).astype(np.float32) / 255.0
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Ajusta dimensões
        img_resized = np.expand_dims(img_resized, axis=-1)  # (H, W, 1)
        X.append(img_resized)
        Y.append(mask_resized)

    X = np.array(X)
    Y = np.array(Y)

    # Converte máscara para one-hot
    num_classes = len(np.unique(Y))
    Y_onehot = np.eye(num_classes)[Y]

    return X, Y_onehot, num_classes


def finetune_model(dataset: str, pretrained, img_dir: str, mask_dir: str,
                   save_dir: str = "models/transunet/finetuned",
                   batch_size: int = 4, epochs: int = 20, lr: float = 1e-4,
                   target_size=(224, 224)):
    """
    Fine-tuning do TransUNet com pseudo-máscaras.

    Args:
        dataset (str): nome do dataset (ex: BrainTC)
        pretrained: modelo pré-treinado carregado
        img_dir (str): diretório com imagens de entrada
        mask_dir (str): diretório com pseudo-máscaras
        save_dir (str): onde salvar o modelo ajustado
        batch_size (int): tamanho do batch
        epochs (int): número de épocas
        lr (float): learning rate
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"📂 Carregando dataset {dataset}...")
    X, Y, num_classes = load_dataset(img_dir, mask_dir, target_size)

    # Ajusta saída do modelo se necessário
    if pretrained.output_shape[-1] != num_classes:
        print(f"⚠️ Ajustando modelo: saída {pretrained.output_shape[-1]} → {num_classes}")
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.models import Model

        x = pretrained.layers[-2].output
        output = Conv2D(num_classes, (1, 1), activation="softmax")(x)
        pretrained = Model(inputs=pretrained.input, outputs=output)

    pretrained.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])

    checkpoint = ModelCheckpoint(
        os.path.join(save_dir, f"{dataset}_finetuned.keras"),
        save_best_only=True, monitor="val_loss", mode="min"
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    print("🚀 Iniciando fine-tuning...")
    history = pretrained.fit(
        X, Y,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print(f"✅ Fine-tuning concluído! Modelo salvo em {save_dir}")
    return pretrained, history
