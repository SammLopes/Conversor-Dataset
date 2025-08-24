import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import os
from collections import Counter;
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO
import yaml
import shutil

train_path = "./yolov8-copy/train/images"

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

def check_proporcao_dataset(dataset_root):
    """
    Checa a proporção de classes em todo o dataset YOLOv8 (train, test, valid).
    Assume que as labels estão em formato YOLO (labels/<split>/*.txt).
    """

    # Mapeamento das classes do projeto
    classes_P = ['Normal', 'AVCh', 'AVCi']
    total_counts = Counter()
    split_counts = {}

    for split in ["train", "test", "valid"]:
        labels_dir = os.path.join(dataset_root, "labels", split)
        if not os.path.exists(labels_dir):
            continue

        counts = Counter()
        for file in os.listdir(labels_dir):
            if not file.endswith(".txt"):
                continue

            with open(os.path.join(labels_dir, file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        counts[class_id] += 1
                        total_counts[class_id] += 1

        split_counts[split] = counts

    # 📊 Exibe proporção por split
    for split, counts in split_counts.items():
        total = sum(counts.values())
        print(f"\n📂 Split: {split} (Total {total})")
        for i, cls in enumerate(classes_P):
            qtd = counts[i]
            prop = (qtd / total * 100) if total > 0 else 0
            print(f"  Classe: {cls}, Quantidade: {qtd}, Proporção: {prop:.2f}%")

    # 📊 Exibe proporção geral
    total_all = sum(total_counts.values())
    print(f"\n📊 Dataset completo (Total {total_all})")
    for i, cls in enumerate(classes_P):
        qtd = total_counts[i]
        prop = (qtd / total_all * 100) if total_all > 0 else 0
        print(f"  Classe: {cls}, Quantidade: {qtd}, Proporção: {prop:.2f}%")


def merge_datasets(classe_desejada, classe_destino, dataset_P, dataset_D, split_D = 'train',split_P='train'):
    
    """
    Yolo To Yolo 
    
    Faz o merge de UMA classe específica do dataset_D para dataset_P.
    
    classe_desejada: nome da classe no dataset_D
    classe_destino: nome da classe no dataset_P
    dataset_P: caminho raiz do dataset principal (com images/ e labels/)
    dataset_D: caminho raiz do dataset a ser fundido (com images/, labels/, data.yaml)
    """

    print(f" De {classe_desejada} para {classe_destino}")
    print(f" De {dataset_D} para {dataset_P}\n")
    print(f" De {split_D} para {split_P}\n")

   # pastas do dataset principal
    p_images_dir = os.path.join(dataset_P, f'{split_P}/images/')
    p_labels_dir = os.path.join(dataset_P, f'{split_P}/labels/')

    # pastas do dataset auxiliar
    d_images_dir = os.path.join(dataset_D, f'{split_D}/images/')
    d_labels_dir = os.path.join(dataset_D, f'{split_D}/labels/')
    d_yaml = os.path.join(dataset_D, "data.yaml")

    # classes do dataset principal
    classes_P = ['Hemorrhagic Stroke', 'Ischemic Stroke', 'Normal']
    map_P = {name.lower(): idx for idx, name in enumerate(classes_P)}

    # lê classes do dataset D
    with open(d_yaml, "r") as f:
        d_data = yaml.safe_load(f)
    classes_D = d_data.get("names", [])

    print("Classes Dataset D:", classes_D)

    if classe_desejada not in classes_D:
        print(f"⚠️ Classe {classe_desejada} não encontrada no Dataset D")
        return

    id_D = classes_D.index(classe_desejada)
    id_P = map_P[classe_destino.lower()]

    # percorre todos os labels
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

        # gera nomes únicos para evitar sobrescrever
        base_name = os.path.splitext(label_file)[0]
        new_image_name = f"{classe_destino}-{base_name}.jpg"
        new_label_name = f"{classe_destino}-{base_name}.txt"

        # procura imagem correspondente
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

def gerar_labels_multiclasse(dataset_root, output_dir, classes_P, split="train", gerar_yaml=False):
    """
    Converte dataset estruturado em subpastas (uma por classe) para YOLOv8 format.
    Cada imagem recebe 1 bounding box cobrindo 100% da imagem.

    dataset_root: raiz com subpastas por classe (ex: Normal/, Ischemic/, Hemorrhagic/)
    output_dir: saída no formato YOLOv8 (train/images/, train/labels/)
    classes_P: lista padrão de classes do projeto
    split: "train", "val" ou "test"
    """

    print(f'Dataset {dataset_root} - split {split} ')

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
    # Gera o data.yaml se pedido
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

if __name__ == "__main__":
    dataset_root = './yolov8-copy'
    # Converter datase fora do formato YOLOv8 para o formato YOLOv8
    # ======================= Conversão de datasets formato direptorio para Yolo format ================================================== 
    # classes_P = ['Hemorrhagic Stroke', 'Ischemic Stroke', 'Normal']
    # gerar_labels_multiclasse('./datasets/dataset_test', 'output_dataset_test', classes_P, split='train');
    
    #dataset 04
    classes_P_04 = ['Bleeding', 'Ischemia', 'Normal'];
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_04/train', 'output_dataset_04', classes_P_04, split='train', gerar_yaml=True);
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_04/test', 'output_dataset_04', classes_P_04, split='test', gerar_yaml=True);
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_04/valid', 'output_dataset_04', classes_P_04, split='valid', gerar_yaml=True);

    #dataset 03 
    classes_P_03 = ['Normal', 'Stroke']
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_03/train', 'output_dataset_03', classes_P_03, split='train', gerar_yaml=True);
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_03/test', 'output_dataset_03', classes_P_03, split='test', gerar_yaml=True);
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_03/valid', 'output_dataset_03', classes_P_03, split='valid', gerar_yaml=True);

    #dataset 02
    classes_P_02 = ['Hemorrhagic', 'Ischemic', 'Normal']
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_02/train', 'output_dataset_02', classes_P_02, split='train', gerar_yaml=True);
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_02/test', 'output_dataset_02', classes_P_02, split='test', gerar_yaml=True);
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_02/valid', 'output_dataset_02', classes_P_02, split='valid', gerar_yaml=True);

    #dataset 01
    classes_P_01 = ['Normal', 'Stroke']
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_01/train', 'output_dataset_01', classes_P_01, split='train', gerar_yaml=True);
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_01/test', 'output_dataset_01', classes_P_01, split='test', gerar_yaml=True);
    gerar_labels_multiclasse('./datasets/datasets_unformat/dataset_01/valid', 'output_dataset_01', classes_P_01, split='valid', gerar_yaml=True);

    print("\n")
    # check proporcao do dataset antes do merge 
    check_proporcao_dataset(dataset_root)
    print("\n")
    # ======================= Conversão de datasets Yolo para o nosso formato Yolo ==================================================

    #dataset yolo datset yaml 01 classe Hemorrágico - Isquemico
    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_01', 'train', 'train')
    merge_datasets('Iskemik'  , 'Ischemic Stroke'   , './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_01', 'train', 'train')

    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_01', 'test', 'test')
    merge_datasets('Iskemik'  , 'Ischemic Stroke'   , './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_01', 'test', 'test')
    
    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_01', 'valid', 'valid')
    merge_datasets('Iskemik'  , 'Ischemic Stroke'   , './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_01', 'valid', 'valid')
    
    #dataset yolo datset yaml 02 classe Hemorrágico - Isquemico
    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_02', 'train', 'train')
    merge_datasets('Iskemik'  , 'Ischemic Stroke'   , './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_02', 'train', 'train')

    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_02', 'test', 'test')
    merge_datasets('Iskemik'  , 'Ischemic Stroke'   , './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_02', 'test', 'test')
    
    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_02', 'valid', 'valid')
    merge_datasets('Iskemik'  , 'Ischemic Stroke'   , './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_02', 'valid', 'valid')

    #dataset yolo dataset yaml 03 classes - Ischemia
    merge_datasets('Ischemia', 'Ischemic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_03', 'train', 'train')

    merge_datasets('Ischemia', 'Ischemic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_03', 'test', 'test')

    merge_datasets('Ischemia', 'Ischemic Stroke', './yolov8-copy/', './datasets/dataset_yolo/dataset_yaml_03', 'valid', 'valid')

    #output dataset 01 - classe Normal
    merge_datasets('Normal', 'Normal', './yolov8-copy/', './output_dataset_01', 'train', 'train')

    merge_datasets('Normal', 'Normal', './yolov8-copy/', './output_dataset_01', 'test', 'test')
    
    merge_datasets('Normal', 'Normal', './yolov8-copy/', './output_dataset_01', 'valid', 'valid')

    #output dataset 02 - classe 'Hemorrhagic', 'Ischemic', 'Normal'
    merge_datasets('Hemorrhagic', 'Hemorrhagic Stroke', './yolov8-copy/', './output_dataset_02/', 'train', 'train')
    merge_datasets('Ischemic'   , 'Ischemic Stroke'   , './yolov8-copy/', './output_dataset_02/', 'train', 'train')
    merge_datasets('Normal'     , 'Normal'            , './yolov8-copy/', './output_dataset_02/', 'train', 'train')

    merge_datasets('Hemorrhagic', 'Hemorrhagic Stroke', './yolov8-copy/', './output_dataset_02/', 'test', 'test')
    merge_datasets('Ischemic'   , 'Ischemic Stroke'   , './yolov8-copy/', './output_dataset_02/', 'test', 'test')
    merge_datasets('Normal'     , 'Normal'            , './yolov8-copy/', './output_dataset_02/', 'test', 'test')

    merge_datasets('Hemorrhagic', 'Hemorrhagic Stroke', './yolov8-copy/', './output_dataset_02/', 'valid', 'valid')
    merge_datasets('Ischemic'   , 'Ischemic Stroke'   , './yolov8-copy/', './output_dataset_02/', 'valid', 'valid')
    merge_datasets('Normal'     , 'Normal'            , './yolov8-copy/', './output_dataset_02/', 'valid', 'valid')

    #output dataset 03 - classe 'Normal', 'Stroke' somente classe normal. 
    merge_datasets('Normal', 'Normal', './yolov8-copy/', './output_dataset_03', 'train', 'train')

    merge_datasets('Normal', 'Normal', './yolov8-copy/', './output_dataset_03', 'test', 'test')
    
    merge_datasets('Normal', 'Normal', './yolov8-copy/', './output_dataset_03', 'valid', 'valid')

    #out put dataset 04 - classe 'Bleeding', 'Ischemia', 'Normal'
    merge_datasets('Bleeding', 'Hemorrhagic Stroke', './yolov8-copy/', './output_dataset_04/', 'train', 'train')
    merge_datasets('Ischemia', 'Ischemic Stroke'   , './yolov8-copy/', './output_dataset_04/', 'train', 'train')
    merge_datasets('Normal'  , 'Normal'            , './yolov8-copy/', './output_dataset_04/', 'train', 'train')

    merge_datasets('Bleeding', 'Hemorrhagic Stroke', './yolov8-copy/', './output_dataset_04/', 'test', 'test')
    merge_datasets('Ischemia', 'Ischemic Stroke'   , './yolov8-copy/', './output_dataset_04/', 'test', 'test')
    merge_datasets('Normal'  , 'Normal'            , './yolov8-copy/', './output_dataset_04/', 'test', 'test')

    merge_datasets('Bleeding', 'Hemorrhagic Stroke', './yolov8-copy/', './output_dataset_04/', 'valid', 'valid')
    merge_datasets('Ischemia', 'Ischemic Stroke'   , './yolov8-copy/', './output_dataset_04/', 'valid', 'valid')
    merge_datasets('Normal'  , 'Normal'            , './yolov8-copy/', './output_dataset_04/', 'valid', 'valid')

    print("\n")
    # proporção do dataset depois do merge
    check_proporcao_dataset(dataset_root)
    #print("\n")
