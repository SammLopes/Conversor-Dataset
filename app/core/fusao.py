# app/core/fusao.py

# Código movido diretamente de app.py, sem alterações

import os
import shutil
import yaml

def merge_datasets(classe_desejada, classe_destino, dataset_P, dataset_D, split_D = 'train', split_P='train'):
    print(f" De {classe_desejada} para {classe_destino}")
    print(f" De {dataset_D} para {dataset_P}\n")
    print(f" De {split_D} para {split_P}\n")

    p_images_dir = os.path.join(dataset_P, f'{split_P}/images/')
    p_labels_dir = os.path.join(dataset_P, f'{split_P}/labels/')

    d_images_dir = os.path.join(dataset_D, f'{split_D}/images/')
    d_labels_dir = os.path.join(dataset_D, f'{split_D}/labels/')
    d_yaml = os.path.join(dataset_D, "data.yaml")

    classes_P = ['Hemorrhagic Stroke', 'Ischemic Stroke', 'Normal']
    map_P = {name.lower(): idx for idx, name in enumerate(classes_P)}

    with open(d_yaml, "r") as f:
        d_data = yaml.safe_load(f)
    classes_D = d_data.get("names", [])

    print("Classes Dataset D:", classes_D)

    if classe_desejada not in classes_D:
        print(f"⚠️ Classe {classe_desejada} não encontrada no Dataset D")
        return

    id_D = classes_D.index(classe_desejada)
    id_P = map_P[classe_destino.lower()]

    for label_file in os.listdir(d_labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(d_labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id, bbox = int(parts[0]), parts[1:]

            if class_id == id_D:
                new_lines.append(" ".join([str(id_P)] + bbox))

        if not new_lines:
            continue

        base_name = os.path.splitext(label_file)[0]
        new_image_name = f"{classe_destino}-{base_name}.jpg"
        new_label_name = f"{classe_destino}-{base_name}.txt"

        src_img = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(d_images_dir, base_name + ext)
            if os.path.exists(candidate):
                src_img = candidate
                break

        if src_img:
            dst_img = os.path.join(p_images_dir, new_image_name)
            dst_label = os.path.join(p_labels_dir, new_label_name)
            shutil.copy(src_img, dst_img)
            with open(dst_label, "w") as f:
                f.write("\n".join(new_lines))

    print(f"✅ Fusão concluída: {classe_desejada} → {classe_destino}")