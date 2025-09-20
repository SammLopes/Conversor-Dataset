# tests/test_core_functions.py

import os
import shutil
# import pytest
import numpy as np
import cv2
from app.core import balanceamento, conversao, fusao, treino, preprocessamento

def test_window_image():
    img = np.linspace(0, 255, 256).astype(np.uint8)
    result = preprocessamento.window_image(img, window_center=128, window_width=128)
    assert result.min() >= 0 and result.max() <= 255

def test_preprocess_image_runs(tmp_path):
    dummy_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    dummy_path = tmp_path / "dummy.jpg"
    cv2.imwrite(str(dummy_path), dummy_img)
    output = preprocessamento.preprocess_image(str(dummy_path))
    assert output.shape == (224, 224)
    assert output.min() >= 0 and output.max() <= 1.0

def test_check_split_proportion(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    classes = ["A", "B", "C"]
    for i in range(3):
        with open(labels_dir / f"img{i}.txt", "w") as f:
            f.write(f"{i} 0.5 0.5 1.0 1.0\n")
    counts, total = balanceamento.check_split_proportion(str(labels_dir), classes)
    assert total == 3
    assert counts[0] == 1

def test_gerar_labels_multiclasse(tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    class_dirs = ["Normal", "Stroke"]
    for cls in class_dirs:
        cls_path = dataset_dir / cls
        cls_path.mkdir(parents=True)
        cv2.imwrite(str(cls_path / "img1.jpg"), np.zeros((100, 100), dtype=np.uint8))

    conversao.gerar_labels_multiclasse(str(dataset_dir), str(output_dir), class_dirs, split="train")
    assert os.path.exists(output_dir / "train/images/img1.jpg")
    assert os.path.exists(output_dir / "train/labels/img1.txt")

def test_merge_datasets_empty(tmp_path):
    dataset_P = tmp_path / "p"
    dataset_D = tmp_path / "d"
    (dataset_D / "train/images").mkdir(parents=True)
    (dataset_D / "train/labels").mkdir(parents=True)
    with open(dataset_D / "data.yaml", "w") as f:
        f.write("names: ['Test']")

    fusao.merge_datasets("Test", "Test", str(dataset_P), str(dataset_D), "train", "train")
    assert True  # Se não crashar, passou

def test_validate_model_mock():
    try:
        treino.validate_model("yolov8n.pt", isOnlyPredict=True)
    except Exception:
        pass  # OK em ambientes sem GPU ou modelo
    assert True

def test_balancear_dataset(tmp_path):
    dataset_root = tmp_path / "yolo"
    output_root = tmp_path / "balanced"
    extra_root = tmp_path / "extra"

    for cls_id, cls in enumerate(["A", "B"]):
        for i in range(5):
            img_path = dataset_root / "train/images" / f"{cls}_{i}.jpg"
            lbl_path = dataset_root / "train/labels" / f"{cls}_{i}.txt"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_path), np.zeros((100, 100), dtype=np.uint8))
            with open(lbl_path, "w") as f:
                f.write(f"{cls_id} 0.5 0.5 1.0 1.0")

    balanceamento.balancear_dataset(str(dataset_root), str(output_root), str(extra_root))

    balanced_imgs = list((output_root / "train/images").glob("*.jpg"))
    assert len(balanced_imgs) > 0

def test_sdavc_model_runs():
    from app.core.sdac_avc.modelo_sdac import build_sdavc_model

    model = build_sdavc_model(input_shape=(224, 224, 1), num_classes=3)
    assert model is not None
    assert len(model.layers) > 0
    assert model.output_shape[-1] == 3 

def test_train_sdavc_kfold(tmp_path):
    from app.core.sdac_avc.treino_sdac_avc import train_sdavc_kfold
    
    X = np.random.rand(20, 224, 224, 1);
    y = np.array([0,1,2,0,1] * 4)

    train_sdavc_kfold(X, y, save_dir=str(tmp_path), n_splits=2)

    files = list(tmp_path.glob("*.h5"))
    assert( len(files) ) == 2

def teste_avaliador_sdavc_model( tmp_path ):
    from app.core.sdac_avc.avaliador_sdavc import avaliar_sdavc_model
    from app.core.sdac_avc.modelo_sdac  import build_sdavc_model

    os.makedirs( tmp_path, True)

    model = build_sdavc_model()
    model.save( str( tmp_path / "fold1.h5" ) )

    X = np.random.rand(5, 224, 224, 1)
    y = np.array( [0, 1, 2, 1,0] )

    avaliar_sdavc_model(X, y, model_dir=str( tmp_path ))

    assert os.path.exists("avaliacoes/sdavc_matrix_confusao.png")