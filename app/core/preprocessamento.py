# app/core/preprocessamento.py

import os
import cv2
import numpy as np
from skimage.exposure import equalize_hist
from skimage.morphology import remove_small_objects

def window_image(img, window_center=40, window_width=80):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(img, img_min, img_max)
    return ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)

def preprocess_image(img_path, output_size=(224, 224)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = window_image(img, window_center=40, window_width=80)
    img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 3)

    _, mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    mask = remove_small_objects(mask.astype(bool), min_size=500)
    img = img * mask.astype(np.uint8)

    img = cv2.resize(img, output_size)
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_dataset(input_root, output_root, output_size=(224, 224)):
    os.makedirs(output_root, exist_ok=True)
    for split in ["train", "valid"]:
        split_in = os.path.join(input_root, split)
        split_out = os.path.join(output_root, split)
        if not os.path.isdir(split_in):
            continue

        for cls in os.listdir(split_in):
            in_dir = os.path.join(split_in, cls)
            out_dir = os.path.join(split_out, cls)
            os.makedirs(out_dir, exist_ok=True)

            for fname in os.listdir(in_dir):
                fpath = os.path.join(in_dir, fname)
                img = preprocess_image(fpath, output_size)
                if img is not None:
                    out_path = os.path.join(out_dir, fname)
                    cv2.imwrite(out_path, (img * 255).astype(np.uint8))
    print(f"✅ Pré-processamento concluído: {output_root}")

def carregar_dataset_preprocessado(root_dir):
    print(f"📦 Carregando dataset pré-processado de: {root_dir}")
    imagens = []
    rotulos = []
    classes = {}
    class_id = 0

    for split in ["train", "valid"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue


    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue


    if class_name not in classes:
        classes[class_name] = class_id
        class_id += 1


    for fname in tqdm(os.listdir(class_path), desc=f"{split}/{class_name}"):
        img_path = os.path.join(class_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1) # canal único
        imagens.append(img)
        rotulos.append(classes[class_name])


    X = np.array(imagens)
    y = np.array(rotulos)
    print(f"✅ Dataset carregado: {X.shape[0]} amostras, {len(classes)} classes")
    return X, y