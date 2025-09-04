import os
import random
import shutil
from collections import Counter, defaultdict

def check_split_proportion(labels_dir, classes):
    counts = Counter()
    total = 0
    if not os.path.exists(labels_dir):
        return counts, 0
    for lbl_file in os.listdir(labels_dir):
        if not lbl_file.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, lbl_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                counts[cls_id] += 1
                total += 1
    return counts, total

def check_proporcao_dataset(dataset_root, classes):
    totals = Counter()
    total_all = 0

    for split in ["train", "valid"]:
        counts, total = check_split_proportion(os.path.join(dataset_root, split, "labels"), classes)
        total_all += total
        for k,v in counts.items():
            totals[k] += v

        print(f"\n📂 Diretório {split} (Total {total})")
        for idx, name in enumerate(classes):
            qtd = counts[idx]
            prop = (qtd / total * 100) if total > 0 else 0
            print(f"  Classe: {name}, Quantidade: {qtd}, Proporção: {prop:.2f}%")

    print(f"\n📊 Dataset completo (Total {total_all})")
    for idx, name in enumerate(classes):
        qtd = totals[idx]
        prop = (qtd / total_all * 100) if total_all > 0 else 0
        print(f"  Classe: {name}, Quantidade: {qtd}, Proporção: {prop:.2f}%")

def balancear_dataset(dataset_root, output_root, extra_root, seed=42, gerar_yaml=True):
    random.seed(seed)

    classes_P = ["Hemorrhagic Stroke", "Ischemic Stroke", "Normal"]
    target_total = 20000
    target_counts = {
        "Ischemic Stroke": int(target_total * 0.40),   # 8000
        "Hemorrhagic Stroke": int(target_total * 0.30),# 6000
        "Normal": int(target_total * 0.30),            # 6000
    }

    split_ratio = {"train": 0.8, "valid": 0.2}

    images_dir = os.path.join(dataset_root, "train", "images")
    labels_dir = os.path.join(dataset_root, "train", "labels")

    for split in ["train", "valid"]:
        os.makedirs(os.path.join(output_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(extra_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(extra_root, "labels"), exist_ok=True)

    class_to_files = defaultdict(list)
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        path = os.path.join(labels_dir, label_file)
        with open(path, "r") as f:
            lines = f.readlines()

        if not lines:
            continue

        class_id = int(lines[0].split()[0])
        class_name = classes_P[class_id]
        image_file = label_file.replace(".txt", ".jpg")

        if os.path.exists(os.path.join(images_dir, image_file)):
            class_to_files[class_name].append((image_file, label_file))

    for class_name, target in target_counts.items():
        files = class_to_files[class_name]
        random.shuffle(files)

        selected = files[:target]
        extra = files[target:]

        n_train = int(target * split_ratio["train"])
        n_valid = target - n_train

        train_files = selected[:n_train]
        valid_files = selected[n_train:n_train+n_valid]

        for img_file, lbl_file in train_files:
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(output_root, "train/images", img_file))
            shutil.copy(os.path.join(labels_dir, lbl_file), os.path.join(output_root, "train/labels", lbl_file))

        for img_file, lbl_file in valid_files:
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(output_root, "valid/images", img_file))
            shutil.copy(os.path.join(labels_dir, lbl_file), os.path.join(output_root, "valid/labels", lbl_file))

        for img_file, lbl_file in extra:
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(extra_root, "images", img_file))
            shutil.copy(os.path.join(labels_dir, lbl_file), os.path.join(extra_root, "labels", lbl_file))

        print(f"✅ Classe {class_name}: {len(train_files)} treino, {len(valid_files)} validação, {len(extra)} extras.")

    if gerar_yaml:
        data_yaml = (
            f"train: {os.path.join(output_root, 'train/images')}\n"
            f"val: {os.path.join(output_root, 'val/images')}\n"
            f"nc: {len(classes_P)}\n"
            f"names: {classes_P}"
        )
        
        yaml_path = os.path.join(output_root, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(data_yaml)
        print("📄 data.yaml gerado no formato array único!")

    print("\n🎯 Dataset balanceado gerado em:", output_root)
    print("📦 Extras salvos em:", extra_root)
