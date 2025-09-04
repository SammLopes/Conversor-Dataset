from core.conversao import gerar_labels_multiclasse
from core.fusao import merge_datasets
from core.balanceamento import balancear_dataset, check_proporcao_dataset
from core.conversao import yolo_to_custom
from core.treino import smart_fit, validate_model
from utils.paths import RAW_UNFORMATTED, YOLO_COPY, YOLO_BALANCED, YOLO_EXTRA, CUSTOM_DATASET, CLASSES_PADRAO, OUTPUT_DATASETS, YOLO_YAML_DATASETS

def main(): 

    train_path = "./yolov8-copy/train/images"
    path_train_images = './yolov8-copy/train/images'
    path_valid_images = './yolov8-copy/valid/images'

    classes = ['Hemorrhagic Stroke', 'Ischemic Stroke', 'Normal']
    # Converter datase fora do formato YOLOv8 para o formato YOLOv8
    # ======================= Conversão de datasets formato direptorio para Yolo format ================================================== 

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
    check_proporcao_dataset("./yolov8-copy", classes)
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
    check_proporcao_dataset("./yolov8-copy", classes)
    print("\n")
    balancear_dataset(
        dataset_root="./yolov8-copy",         # seu dataset atual
        output_root="./yolov8-balanced",      # dataset balanceado final
        extra_root="./yolov8-extra"           # imagens que sobraram
    )
    print('\n')
    check_proporcao_dataset("./yolov8-balanced", classes)
    classes = ["Hemorrhagic Stroke", "Ischemic Stroke", "Normal"]

    yolo_to_custom(
        yolo_root="./yolov8-balanced",
        output_root="./dataset_custom",
        classes=classes
    )

    # ======================= Treinamento e Validação ============================
    # smart_fit()
    # validate_model("./runs/detect/peso_volov8m_50ep/weights/best.pt", isOnlyPredict=False)


if __name__ == "__main__": 
    main()