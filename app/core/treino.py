# app/core/treino.py

# Código movido diretamente de app.py, sem alterações

import torch
from ultralytics import YOLO

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
        half=True
    )

def validate_model(model, isOnlyPredict=False):
    print(f"\n📊 Validando modelo: {model}")
    model = YOLO(model)

    print("\nClasses do modelo ")
    for class_id, class_name in model.names.items():
        print(f"\n{class_id}: {class_name}")

    if not isOnlyPredict:
        print(f"Validate model \n")
        model.val(
            data="yolov8/data.yaml",
            plots=True,
            conf=0.25
        )

    # Exemplo comentado de predições:
    # image_src = ["./caba-dormindo.jpg", "./image1.jpg", "./1.jpg"]
    # for src in image_src:
    #     model.predict(
    #         src,
    #         save=True,
    #         iou=0.45,
    #         augment=True
    #     )
