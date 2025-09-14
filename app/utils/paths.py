# app/utils/paths.py

# Caminhos principais organizados para uso centralizado no projeto

# Diretórios brutos (sem formatação YOLO)
RAW_UNFORMATTED = "./data/raw/datasets_unformat"

# Diretórios intermediários e processados
YOLO_COPY = "./data/processed/yolov8-copy"
YOLO_BALANCED = "./data/balanced/yolov8-balanced"
YOLO_EXTRA = "./data/balanced/yolov8-extra"
CUSTOM_DATASET = "./data/final/dataset_custom"

# Saída dos datasets convertidos
OUTPUT_DATASETS = {
    "dataset_01": "./data/processed/output_dataset_01",
    "dataset_02": "./data/processed/output_dataset_02",
    "dataset_03": "./data/processed/output_dataset_03",
    "dataset_04": "./data/processed/output_dataset_04",
}

# Datasets no formato YOLO YAML
YOLO_YAML_DATASETS = {
    "dataset_yaml_01": "./data/raw/dataset_yolo/dataset_yaml_01",
    "dataset_yaml_02": "./data/raw/dataset_yolo/dataset_yaml_02",
    "dataset_yaml_03": "./data/raw/dataset_yolo/dataset_yaml_03",
}

# Classes padrão
CLASSES_PADRAO = ["Hemorrhagic Stroke", "Ischemic Stroke", "Normal"]
