import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import numpy as np
from ultralytics import YOLO
import os
import datetime 

def train():
    """
    Função para treinar o modelo YOLOv8.
    Carrega o modelo pré-treinado 'yolov8m.pt' e inicia o treinamento
    com os parâmetros especificados.
    """
    # Garante que os diretórios para os resultados existam
    if not os.path.exists('runs/classify'):
        os.makedirs('runs/classify',  exist_ok=True)

    # Carrega um modelo pré-treinado
    #model = YOLO("yolov8m-cls.pt")
    model= YOLO("yolov8m.pt") # Treinamento de detecção
    # data='./data/training/dataset_custom'
    # Exibe informações do modelo
    model.info()
    # workspace/Conversor-Dataset/data/training/yolov8-balanced
    # Treina o modelo com as configurações definidas
    model.train(
        name="yolo_avc_v8m_detect",
        data= "./data/training/datasets_training/yolov8-balanced/data.yaml",
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

def validate_model(model_path, dataset_path):
    """
    Função para validar um modelo YOLOv8 (Classificação ou Detecção).

    Args:
        model_path (str): O caminho para o arquivo do modelo treinado (ex: 'best.pt').
        dataset_path (str): O caminho para os dados de validação.
                            - Para Detecção: 'caminho/para/data.yaml'
                            - Para Classificação: 'caminho/para/diretorio_raiz_do_dataset/'
    """

    # --- 1. Verificação de Arquivos e Caminhos ---
    if not os.path.exists(model_path):
        print(f"❌ Erro: O arquivo do modelo não foi encontrado em '{model_path}'")
        return
    
    # Verificação genérica de existência (funciona para arquivos e diretórios)
    if not os.path.exists(dataset_path):
        print(f"❌ Erro: O caminho do dataset não foi encontrado em '{dataset_path}'")
        print("   (Lembre-se: .yaml para detecção, diretório para classificação)")
        return

    # --- 2. Carregar Modelo e Detectar Tarefa ---
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo: {e}")
        return
    
    task = model.task
    print(f"ℹ️ Modelo detectado: {model_path}")
    print(f"ℹ️ Tarefa do modelo: '{task}'")
    print(f"ℹ️ Dataset: {dataset_path}")

    try:
        # Salva os resultados no diretório principal do modelo
        # Ex: model_path = '.../runs/classify/exp1/weights/best.pt'
        #     model_dir = '.../runs/classify/exp1/'
        model_dir = os.path.dirname(os.path.dirname(model_path))
        output_filename = os.path.join(model_dir, "validation_results.txt")
        
        # Garante que o diretório existe
        os.makedirs(model_dir, exist_ok=True) 
        
        print(f"ℹ️ Resultados serão salvos em: {output_filename}")
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível determinar o diretório para salvar os resultados: {e}")
        output_filename = None # Continua sem salvar

    # --- 3. Executar Validação Baseada na Tarefa ---
    
    # ======================================================
    #   CASO 1: Modelo de CLASSIFICAÇÃO
    # ======================================================
    if task == 'classify':
        # Verifica se o caminho é um DIRETÓRIO (esperado para classificação)
        if not os.path.isdir(dataset_path):
            print(f"❌ Erro de Classificação: O caminho '{dataset_path}' não é um diretório.")
            print("   Para classificação, forneça o caminho para o diretório de validação (ex: .../dataset/val/).")
            return

        print("\nIniciando validação de CLASSIFICAÇÃO...")
        print(f"Executando predições em '{dataset_path}' para calcular métricas...")
        
        # 1. USAR model.predict(), NÃO model.val()
        # 'stream=True' economiza memória ao processar muitas imagens
        # A biblioteca sabe o que fazer com o diretório
        try:

            # Usar um glob pattern (**) para buscar imagens recursivamente
            # A função predict() não busca em subpastas por padrão.
            glob_pattern = os.path.join(dataset_path, '**', '*')
            print(f"Buscando imagens recursivamente com o padrão: {glob_pattern}")
            
            results_pred = model.predict(source=glob_pattern, stream=True, verbose=False)
            # --- FIM DA CORREÇÃO ---
        except Exception as e:
            print(f"❌ Erro durante a predição: {e}")
            print("   Verifique se o caminho do dataset está correto e contém imagens.")
            return

        y_true = []
        y_pred = []
        y_pred_probs = []
        
        # 2. Precisamos mapear o NOME DA PASTA para o ÍNDICE DA CLASSE
        # model.names é {0: 'classe_A', 1: 'classe_B'}
        # Precisamos do inverso: {'classe_A': 0, 'classe_B': 1}
        class_name_to_index = {name: index for index, name in model.names.items()}
        print(f"Mapeamento de classes (Pasta -> Índice): {class_name_to_index}")

        try:
            for r in results_pred:
                # 3. Extrair probabilidades
                probs = r.probs.data.cpu().numpy()
                y_pred_probs.append(probs)
                y_pred.append(np.argmax(probs))
                
                # 4. Extrair rótulo verdadeiro do caminho do arquivo
                # ex: r.path = '/.../dataset/val/classe_A/img123.jpg'
                #     true_class_name = 'classe_A'
                true_class_name = os.path.basename(os.path.dirname(r.path))
                
                if true_class_name in class_name_to_index:
                    true_class_index = class_name_to_index[true_class_name]
                    y_true.append(true_class_index)
                else:
                    print(f"⚠️ Aviso: A pasta '{true_class_name}' (do arquivo {r.path}) não corresponde a nenhuma classe em model.names.")
                    # Removemos os itens que acabamos de adicionar para manter a sincronia
                    y_pred_probs.pop()
                    y_pred.pop()

        except Exception as e:
            print(f"❌ Erro ao processar predições: {e}")
            return

        if not y_true:
            print("❌ Erro: Nenhum resultado de validação encontrado.")
            print("   Verifique se o diretório de validação contém subpastas com os nomes das classes.")
            return

        print(f"\nResultados processados para {len(y_true)} imagens.")
        print("\n📋 Relatório de Classificação")
        print("="*30)
        
        # Nomes das classes na ordem correta
        class_names = [model.names[i] for i in sorted(model.names.keys())]
        #print(classification_report(y_true, y_pred, target_names=class_names))
        report_str = classification_report(y_true, y_pred, target_names=class_names)
        print(report_str) # Imprime no console
        print("="*30)

        try:
            auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
            print(f"🏅 AUC-ROC (One-vs-Rest): {auc:.4f}")
        except Exception as e:
            print(f"⚠️ AUC-ROC não pôde ser calculado: {e}")

            # --- Salvar resultados de Classificação ---
        if output_filename:
            try:
                with open(output_filename, "a", encoding="utf-8") as f:
                    # MUDANÇA AQUI
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n--- Validation Results (Classify) @ {timestamp} ---\n")
                    f.write(f"Model: {model_path}\n")
                    f.write(f"Dataset: {dataset_path}\n\n")
                    f.write("📋 Relatório de Classificação\n")
                    f.write("="*30 + "\n")
                    f.write(report_str + "\n")
                    f.write("="*30 + "\n")
                    f.write(f"🏅 AUC-ROC (One-vs-Rest): {auc:.4f}\n")
                    f.write("-"*50 + "\n")
                print(f"✅ Resultados de classificação salvos em {output_filename}")
            except Exception as e:
                print(f"❌ Erro ao salvar o arquivo de resultados: {e}")

    # ======================================================
    #   CASO 2: Modelo de DETECÇÃO
    # ======================================================
    elif task == 'detect':
        # Verifica se o caminho é um ARQUIVO .yaml (esperado para detecção)
        if not os.path.isfile(dataset_path) or not dataset_path.endswith('.yaml'):
            print(f"❌ Erro de Detecção: O caminho '{dataset_path}' não é um arquivo .yaml.")
            print("   Modelos de detecção exigem um arquivo 'data.yaml'.")
            return

        print("\nIniciando validação de DETECÇÃO...")
        # A biblioteca sabe o que fazer com o .yaml
        results_val = model.val(data=dataset_path, conf=0.70, iou=0.5)
        
        print("\n📊 Resultados da Validação de Detecção (mAP)")
        print("="*40)
        print(f"mAP50-95: {results_val.box.map:.4f}")
        print(f"mAP50:    {results_val.box.map50:.4f}")
        print(f"mAP75:    {results_val.box.map75:.4f}")
        print(f"Precision (P): {results_val.box.p.mean():.4f}")
        print(f"Recall (R):    {results_val.box.r.mean():.4f}")
        print("\n(A tabela completa de resultados por classe foi impressa acima.)")

    # ======================================================
    #   CASO 3: Tarefas Não Suportadas
    # ======================================================
    else:
        print(f"\n❌ ERRO: Tarefa não suportada.")
        print(f"Este script foi feito apenas para 'classify' e 'detect', mas o modelo é do tipo '{task}'.")