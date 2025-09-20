# app/core/treino.py
import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import numpy as np
from ultralytics import YOLO

def train():
    model = YOLO("yolov8m.pt")
    model.info()
    model.train(
        name="yolo_avc_v8m_multiclasse",
        data='yolov8/data.yaml',
        epochs=100,
        imgsz=224,
        batch=32,                 # 🔧 otimizado para datasets médicos pequenos
        lr0=0.0005,               # 🔧 taxa de aprendizado refinada
        lrf=0.0001,               # 🔧 taxa mínima (cosine lr decay)
        patience=50,             # 🔧 early stopping precoce
        dropout=0.3,             # 🔧 regularização
        cos_lr=True,             # 🔁 agendamento de LR (Cosine decay)
        device='cuda:0',
        half=True,               # 🔧 FP16 para performance
        plots=True,
        val=True,
        save=True
    )

def validate_model(model, isOnlyPredict=False):

    model = YOLO(model)
    results = model.val(data="yolov8/data.yaml", conf=0.25)

    y_true = np.array(results.boxes.cls)
    y_pred = np.array(results.results[0].probs.top1)  # ou argmax pro multi

    print("\n📋 Relatório de Classificação")
    print(classification_report(y_true, y_pred, target_names=model.names.values()))

    try:
        auc = roc_auc_score(y_true, results.results[0].probs.data.cpu().numpy(), multi_class='ovr')
        print(f"🏅 AUC-ROC: {auc:.4f}")
    except Exception as e:
        print("⚠️ AUC não pôde ser calculado:", e)
