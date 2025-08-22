import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import os
from collections import Counter;
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO
import yaml
import shutil

train_path = "./yolov8/train/images"

def smart_fit():
    model = YOLO("yolov8s.pt")

    model.info()

    model.train(
        name="peso_volov8m_50ep",
        data='yolov8/data.yaml',
        epochs=300,
        imgsz=640,
        batch=4,
        patience=50,
        plots=True,
        cos_lr=True,
        device=torch.device('cuda:0'),
        half = True
    )

def validate_model(model, isOnlyPredict=False):

    print(f"\n📊 Validando modelo: {model}")
    model = YOLO(model)

    print("\nClasses do modelo ");
    for class_id, class_name in model.names.items():
        print(f"\n{class_id}: {class_name}")

    if not isOnlyPredict:
        print(f"Validate model \n")
        model.val(
            data="yolov8/data.yaml",
            plots=True,
            conf=0.25
        )

    # image_src = ["./caba-dormindo.jpg", "./image1.jpg", "./1.jpg" ]
    # print(f"\n🔍 Fazendo predição da imagem: {image_src}")
    # for src in image_src:
    #     print(f"\nCaminho: {src}")
    #     model.predict(
    #         src,
    #         save=True,
    #         iou=0.45,
    #         augment=True
    #     )

def checkPropracaoDataset():
    counts = Counter()

    for file in os.listdir(train_path):
        fname = file.lower()
        if "hemorrhagic" in fname:
            counts["AVCh"] += 1
        elif "ischemic" in fname:
            counts["AVCi"] += 1
        elif "normal" in fname:
            counts["Normal"] += 1
    
    total = sum(counts.values())

    target_total = 20000
    target_counts = {
        "AVCi": int(target_total * 0.4),   
        "AVCh": int(target_total * 0.3),   
        "Normal": int(target_total * 0.3)  
    }

    print("📊 Distribuição atual do dataset de treino:")
    for classe, qtd in counts.items():
        perc = (qtd / total) * 100 if total > 0 else 0
        print(f"Classe: {classe}, Quantidade: {qtd}, Proporção: {perc:.2f}%")

    print(f"\nTotal de imagens: {total}")

    print("\n🎯 Alvo para 20.000 imagens:")
    for classe, qtd in target_counts.items():
        print(f"Classe: {classe}, Quantidade Alvo: {qtd}")

    print("\n📌 Imagens que ainda faltam por classe:")
    for classe, qtd in target_counts.items():
        falta = qtd - counts.get(classe, 0)
        print(f" - {classe}: {max(falta, 0)} faltando")

def merge_datasets(classe_desejada, classe_destino):
    
    # === CONFIGURAÇÃO ===
    p_images_dir = "dataset_P/images/train" # dataset Principal
    p_labels_dir = "dataset_P/labels/train" # dataset Principal
   
    d_images_dir = "dataset_D/images/train" # dataset D
    d_labels_dir = "dataset_D/labels/train" # dataset D
    d_yaml = "dataset_D/data.yaml"
    
    classe_desejada = classe_desejada
    classe_destino = classe_destino

    classes_P = ['Hemorrhagic Stroke', 'Ischemic Stroke', 'Normal']
    
    map_P = {name.lower(): idx for idx, name in enumerate(classes_P)}

    # === Lê classes do Dataset D ===
    with open(d_yaml, "r") as f:
        d_data = yaml.safe_load(f)
    classes_D = d_data.get("names", [])

    print("Classes Dataset D:", classes_D)

    if classe_desejada not in classes_D:
        raise ValueError(f"Classe {classe_desejada} não encontrada no Dataset D")

    id_D = classes_D.index(classe_desejada)
    id_P = map_P[classe_destino.lower()]

    # === Processa labels do Dataset D ===
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

        # Troca o nome do arquivo adicionando o prefixo da classe destino
        base_name = label_file.replace(".txt", "")
        new_image_name = f"{classe_destino}-{base_name}.jpg"
        new_label_name = f"{classe_destino}-{base_name}.txt"

        src_img = os.path.join(d_images_dir, label_file.replace(".txt", ".jpg"))
        dst_img = os.path.join(p_images_dir, new_image_name)
        dst_label = os.path.join(p_labels_dir, new_label_name)

        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            with open(dst_label, "w") as f:
                f.write("\n".join(new_lines))


    print(f"✅ Fusão concluída! Classe {classe_desejada} do Dataset D adicionada ao Dataset P.")

def gerar_labels_multiclasse(dataset_root, output_dir, classes_P, split="train"):
    """
    Converte dataset estruturado em subpastas (uma por classe) para YOLOv8 format.
    Cada imagem recebe 1 bounding box cobrindo 100% da imagem.

    dataset_root: raiz com subpastas por classe (ex: Normal/, Ischemic/, Hemorrhagic/)
    output_dir: saída no formato YOLOv8 (train/images/, train/labels/)
    classes_P: lista padrão de classes do projeto
    split: "train", "val" ou "test"
    """
    # Mapeamento das classes
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
            
            # Copia imagem
            shutil.copy(src_path, dst_img_path)
            
            # Gera label cobrindo toda a imagem
            with open(dst_lbl_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        print(f"✅ Classe {class_folder} convertida para YOLOv8 com ID {class_id}")

    print(f"🎯 Conversão finalizada no split {split}!")
if __name__ == "__main__":

    # Converter datase fora do formato YOLOv8 para o formato YOLOv8
    convert_format();
    # check proporcao do dataset antes do merge 
    checkPropracaoDataset()
    print("\n")
    merge_datasets('Normal', 'Normal')
    print("\n")
    # proporção do dataset depois do merge
    checkPropracaoDataset()
    print("\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    