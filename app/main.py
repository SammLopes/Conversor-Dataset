from core.conversao import gerar_labels_multiclasse
from core.fusao import merge_datasets
from core.balanceamento import balancear_dataset, check_proporcao_dataset
from core.conversao import yolo_to_custom
from core.treino import smart_fit, validate_model
from utils.paths import (
    RAW_UNFORMATTED,
    YOLO_COPY,
    YOLO_BALANCED,
    YOLO_EXTRA,
    CUSTOM_DATASET,
    CLASSES_PADRAO,
    OUTPUT_DATASETS,
    YOLO_YAML_DATASETS,
)

from core.preprocessamento import preprocess_dataset

def main(): 

    train_path = "./yolov8-copy/train/images"
    path_train_images = './yolov8-copy/train/images'
    path_valid_images = './yolov8-copy/valid/images'

    classes = ['Hemorrhagic Stroke', 'Ischemic Stroke', 'Normal']
    # Converter datase fora do formato YOLOv8 para o formato YOLOv8
    # ======================= Conversão de datasets formato direptorio para Yolo format ================================================== 

    #dataset 04
    classes_P_04 = ['Bleeding', 'Ischemia', 'Normal'];
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_04/train", OUTPUT_DATASETS["dataset_04"], classes_P_04, split='train', gerar_yaml=True)
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_04/test", OUTPUT_DATASETS["dataset_04"], classes_P_04, split='test', gerar_yaml=True)
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_04/valid", OUTPUT_DATASETS["dataset_04"], classes_P_04, split='valid', gerar_yaml=True)

    #dataset 03 
    classes_P_03 = ['Normal', 'Stroke']
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_03/train", OUTPUT_DATASETS["dataset_03"], classes_P_03, split='train', gerar_yaml=True)
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_03/test", OUTPUT_DATASETS["dataset_03"], classes_P_03, split='test', gerar_yaml=True)
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_03/valid", OUTPUT_DATASETS["dataset_03"], classes_P_03, split='valid', gerar_yaml=True)

    #dataset 02
    classes_P_02 = ['Hemorrhagic', 'Ischemic', 'Normal']
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_02/train", OUTPUT_DATASETS["dataset_02"], classes_P_02, split='train', gerar_yaml=True)
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_02/test", OUTPUT_DATASETS["dataset_02"], classes_P_02, split='test', gerar_yaml=True)
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_02/valid", OUTPUT_DATASETS["dataset_02"], classes_P_02, split='valid', gerar_yaml=True)


    #dataset 01
    classes_P_01 = ['Normal', 'Stroke']
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_01/train", OUTPUT_DATASETS["dataset_01"], classes_P_01, split='train', gerar_yaml=True)
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_01/test", OUTPUT_DATASETS["dataset_01"], classes_P_01, split='test', gerar_yaml=True)
    gerar_labels_multiclasse(f"{RAW_UNFORMATTED}/dataset_01/valid", OUTPUT_DATASETS["dataset_01"], classes_P_01, split='valid', gerar_yaml=True)


    print("\n")
    # check proporcao do dataset antes do merge 
    check_proporcao_dataset(YOLO_COPY, classes)
    print("\n")
    # ======================= Conversão de datasets Yolo para o nosso formato Yolo ==================================================
    # dataset yolo dataset yaml 01 - classes Hemorrágico / Isquêmico
    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_01"], 'train', 'train')
    merge_datasets('Iskemik', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_01"], 'train', 'train')

    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_01"], 'test', 'test')
    merge_datasets('Iskemik', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_01"], 'test', 'test')

    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_01"], 'valid', 'valid')
    merge_datasets('Iskemik', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_01"], 'valid', 'valid')

    # dataset yolo dataset yaml 02 - classes Hemorrágico / Isquêmico
    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_02"], 'train', 'train')
    merge_datasets('Iskemik', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_02"], 'train', 'train')

    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_02"], 'test', 'test')
    merge_datasets('Iskemik', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_02"], 'test', 'test')

    merge_datasets('Hemoragik', 'Hemorrhagic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_02"], 'valid', 'valid')
    merge_datasets('Iskemik', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_02"], 'valid', 'valid')

    # dataset yolo dataset yaml 03 - classe Isquemia
    merge_datasets('Ischemia', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_03"], 'train', 'train')
    merge_datasets('Ischemia', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_03"], 'test', 'test')
    merge_datasets('Ischemia', 'Ischemic Stroke', YOLO_COPY, YOLO_YAML_DATASETS["dataset_yaml_03"], 'valid', 'valid')

    # output dataset 01 - classe Normal
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_01"], 'train', 'train')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_01"], 'test', 'test')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_01"], 'valid', 'valid')

    # output dataset 02 - classes Hemorrhagic / Ischemic / Normal
    merge_datasets('Hemorrhagic', 'Hemorrhagic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'train', 'train')
    merge_datasets('Ischemic', 'Ischemic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'train', 'train')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'train', 'train')

    merge_datasets('Hemorrhagic', 'Hemorrhagic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'test', 'test')
    merge_datasets('Ischemic', 'Ischemic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'test', 'test')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'test', 'test')

    merge_datasets('Hemorrhagic', 'Hemorrhagic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'valid', 'valid')
    merge_datasets('Ischemic', 'Ischemic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'valid', 'valid')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_02"], 'valid', 'valid')

    # output dataset 03 - classe Normal (dataset binário original Normal vs Stroke, mas aqui só Normal)
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_03"], 'train', 'train')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_03"], 'test', 'test')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_03"], 'valid', 'valid')

    # output dataset 04 - classes Bleeding / Ischemia / Normal
    merge_datasets('Bleeding', 'Hemorrhagic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'train', 'train')
    merge_datasets('Ischemia', 'Ischemic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'train', 'train')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'train', 'train')

    merge_datasets('Bleeding', 'Hemorrhagic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'test', 'test')
    merge_datasets('Ischemia', 'Ischemic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'test', 'test')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'test', 'test')

    merge_datasets('Bleeding', 'Hemorrhagic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'valid', 'valid')
    merge_datasets('Ischemia', 'Ischemic Stroke', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'valid', 'valid')
    merge_datasets('Normal', 'Normal', YOLO_COPY, OUTPUT_DATASETS["dataset_04"], 'valid', 'valid')

    print("\n")
    # proporção do dataset depois do merge
    check_proporcao_dataset(YOLO_COPY, classes)
    print("\n")
    balancear_dataset(
        dataset_root=YOLO_COPY,
        output_root=YOLO_BALANCED,
        extra_root=YOLO_EXTRA
    )

    print("\n")
    check_proporcao_dataset(YOLO_BALANCED, classes)
    classes = ["Hemorrhagic Stroke", "Ischemic Stroke", "Normal"]

    yolo_to_custom(
        yolo_root=YOLO_BALANCED,
        output_root=CUSTOM_DATASET,
        classes=classes
    )

    # ======================= Pré-processamento ============================
    print("\n")
    preprocess_dataset(
        input_root=CUSTOM_DATASET,
        output_root="./data/pre-processed/dataset_custom_preprocessed" 
    )
    # ======================= Treinamento e Validação ============================
    # smart_fit()
    # validate_model("./runs/detect/peso_volov8m_50ep/weights/best.pt", isOnlyPredict=False)

if __name__ == "__main__": 
    main()