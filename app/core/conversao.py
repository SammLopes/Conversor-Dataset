# app/core/conversao.py

# Código movido diretamente de app.py, sem alterações

import os
import shutil

def gerar_labels_multiclasse(dataset_root, output_dir, classes_P, split="train", gerar_yaml=False):
    print(f'Dataset {dataset_root} - split {split} ')
    map_P = {name.lower(): idx for idx, name in enumerate(classes_P)}
    images_out = os.path.join(output_dir, split, "images")
    labels_out = os.path.join(output_dir, split, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for class_folder in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, class_folder)
        if not os.path.isdir(class_path):
            continue

        class_name = class_folder.lower().strip()
        if class_name not in map_P:
            print(f"⚠️ Classe {class_folder} ignorada (não existe no padrão)")
            continue

        class_id = map_P[class_name]

        for file in os.listdir(class_path):
            if not (file.endswith(".jpg") or file.endswith(".png")):
                continue

            src_path = os.path.join(class_path, file)
            dst_img_path = os.path.join(images_out, file)
            dst_lbl_path = os.path.join(labels_out, file.rsplit(".", 1)[0] + ".txt")
            shutil.copy(src_path, dst_img_path)
            with open(dst_lbl_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

        print(f"✅ Classe {class_folder} convertida para YOLOv8 com ID {class_id}")

    if gerar_yaml:
        data_yaml = (
            f"train: ./{os.path.join(output_dir, 'train/images')}\n"
            f"val: ./{os.path.join(output_dir, 'val/images')}\n"
            f"test: ./{os.path.join(output_dir, 'test/images')}\n\n"
            f"nc: {len(classes_P)}\n"
            f"names: {classes_P}"
        )
        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            f.write(data_yaml)
        print("📄 data.yaml gerado no formato array único!")

    print(f"🎯 Conversão finalizada no split {split}!\n")

def yolo_to_custom(yolo_root, output_root, classes):
    for split in ["train", "valid"]:
        images_dir = os.path.join(yolo_root, split, "images")
        labels_dir = os.path.join(yolo_root, split, "labels")

        for cls in classes:
            os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue

            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, "r") as f:
                lines = f.readlines()

            if not lines:
                continue

            class_id = int(lines[0].split()[0])
            class_name = classes[class_id]

            base_name = os.path.splitext(label_file)[0]
            img_file = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = os.path.join(images_dir, base_name + ext)
                if os.path.exists(candidate):
                    img_file = candidate
                    break

            if img_file:
                dst_path = os.path.join(output_root, split, class_name, os.path.basename(img_file))
                shutil.copy(img_file, dst_path)

        print(f"✅ Split '{split}' convertido para formato customizado em {output_root}/{split}")
