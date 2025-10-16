import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import numpy as np
from ultralytics import YOLO
import os

def train():
    """
    Função para treinar o modelo YOLOv8.
    Carrega o modelo pré-treinado 'yolov8m.pt' e inicia o treinamento
    com os parâmetros especificados.
    """
    # Garante que os diretórios para os resultados existam
    if not os.path.exists('runs/classify'):
        os.makedirs('runs/classify')

    # Carrega um modelo pré-treinado
    model = YOLO("yolov8m.pt")
    
    # Exibe informações do modelo
    model.info()
    
    # Treina o modelo com as configurações definidas
    model.train(
        name="yolo_avc_v8m_multiclasse",
        data='yolov8/data.yaml',
        epochs=100,
        imgsz=224,
        batch=32,                 # Otimizado para datasets médicos pequenos
        lr0=0.0005,               # Taxa de aprendizado inicial refinada
        lrf=0.0001,               # Taxa de aprendizado final (cosine lr decay)
        patience=50,              # Paciência para early stopping
        dropout=0.3,              # Regularização para evitar overfitting
        cos_lr=True,              # Usa o agendador de taxa de aprendizado Cosine
        device='cuda:0',          # Utiliza a GPU
        half=True,                # Usa precisão de 16 bits (FP16) para acelerar
        plots=True,               # Gera gráficos do treinamento
        val=True,                 # Realiza validação durante o treinamento
        save=True                 # Salva o modelo treinado
    )
    print("\n✅ Treinamento concluído!")
    print("O melhor modelo foi salvo em: runs/classify/yolo_avc_v8m_multiclasse/weights/best.pt")

def validate_model(model_path):
    """
    Função para validar um modelo YOLOv8 treinado.

    Args:
        model_path (str): O caminho para o arquivo do modelo treinado (ex: 'best.pt').
    """
    if not os.path.exists(model_path):
        print(f"❌ Erro: O arquivo do modelo não foi encontrado em '{model_path}'")
        return

    # Carrega o modelo treinado
    model = YOLO(model_path)
    
    # Executa a validação no conjunto de dados especificado
    results = model.val(data="yolov8/data.yaml", conf=0.25)

    # Extrai os rótulos verdadeiros e as predições
    # Nota: A extração pode variar dependendo da versão da Ultralytics
    # Este é um exemplo baseado na sua estrutura
    y_true = []
    y_pred = []
    y_scores = []
    
    # Acessando as predições de forma correta para tarefas de classificação
    for result in results.pred:
        if result is not None and len(result) > 0:
            # result é uma lista de tensores, um para cada imagem no batch
            # Para classificação, usamos `results.probs`
            pass # A lógica principal será baseada nos `results` do `val`

    y_true = np.array(results.boxes.cls.cpu())
    y_pred_probs = results.probs.data.cpu().numpy()
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n📋 Relatório de Classificação")
    print("="*30)
    # Garante que os nomes das classes estejam disponíveis
    class_names = model.names if hasattr(model, 'names') and model.names else [str(i) for i in range(len(np.unique(y_true)))]
    print(classification_report(y_true, y_pred, target_names=class_names.values()))
    print("="*30)

    try:
        # Para multi-classe, usamos as probabilidades para o cálculo da AUC
        auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
        print(f"🏅 AUC-ROC (One-vs-Rest): {auc:.4f}")
    except Exception as e:
        print(f"⚠️ AUC não pôde ser calculado: {e}")
