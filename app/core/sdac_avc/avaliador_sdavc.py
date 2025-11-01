import os
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from keras.models import load_model
from keras.utils import to_categorical

# Corrigido: compile=False é mais rápido para predição
def avaliar_sdavc_model(X, y, model_dir="modelos/sdavc", output_dir='avaliacoes', is_compile=False):

    os.makedirs(output_dir, exist_ok=True)

    y_cat = to_categorical(y)
    
    # --- MUDANÇA: Vamos acumular probabilidades em vez de listas ---
    # Inicializa uma matriz de zeros para somar as probabilidades de todos os folds
    # Ex: se X tem 4000 amostras e y_cat tem 3 classes -> shape (4000, 3)
    all_probs = np.zeros((len(X), y_cat.shape[1]))

    # Listas para guardar métricas de performance de cada fold
    tempos_predicao = []
    throughputs = []
    mems_usadas = []
    
    process = psutil.Process(os.getpid()) 
    
    # Pega a lista de modelos .keras no diretório
    model_files = [f for f in sorted(os.listdir(model_dir)) if f.endswith(".keras")]
    
    if not model_files:
        print(f"❌ Erro: Nenhum arquivo .keras encontrado em '{model_dir}'")
        return

    print(f"ℹ️ Encontrados {len(model_files)} folds para avaliação (Ensemble).")

    for fold_file in model_files:
        print(f"\n--- Processando fold: {fold_file} ---")
        
        model = load_model(os.path.join(model_dir, fold_file), compile=is_compile)
        
        mem_before = process.memory_info().rss / (1024 * 1024)  
        start = time.time()
        
        # y_pred é uma matriz de probabilidades (ex: 4000, 3)
        y_pred_probs = model.predict(X)
        
        end = time.time()
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        # --- Acumula métricas de performance ---
        mem_used = mem_after - mem_before
        tempo_predicao = (end - start) / len(X) * 1000
        throughput = len(X) / (end - start )

        tempos_predicao.append(tempo_predicao)
        throughputs.append(throughput)
        mems_usadas.append(mem_used)

        print(f"Tempo de predição: {tempo_predicao:.2f} ms/imagem")
        print(f"Throughput: {throughput:.2f} imagens/segundo") 
        print(f"Uso de memória adicional: {mem_used:.2f} MB")

        # --- MUDANÇA: Soma as probabilidades ---
        all_probs += y_pred_probs

    # --- FIM DO LOOP ---
    
    print("\n--- Calculando Métricas Finais (Ensemble Média) ---")

    # --- MUDANÇA: Calcula a média das probabilidades ---
    avg_probs = all_probs / len(model_files)
    
    # Converte as probabilidades médias em classes (0, 1, 2)
    final_predicoes = np.argmax(avg_probs, axis=1)
    
    # Pega os rótulos reais (0, 1, 2)
    reais = y # Usa o 'y' original

    print("\n📊 Relatório de Classificação (Ensemble Média dos Folds)")
    report = classification_report(reais, final_predicoes, output_dict=True,  zero_division=0)
    print(classification_report(reais, final_predicoes, zero_division=0))

    recall_macro = report['macro avg']['recall']
    print(f"🌟 Sensibilidade (Recall - macro): {recall_macro:.4f}")

    cm = confusion_matrix(reais, final_predicoes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    n_classes = cm.shape[0]

    especificidades = []
    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        especificidade_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Evita divisão por zero
        especificidades.append(especificidade_i)
        print(f"Especificidade (classe {i}): {especificidade_i:.4f}")

    especificidade_macro = np.mean(especificidades)
    print(f"Especificidade (macro): {especificidade_macro:.4f}")

    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão SDAVC (Ensemble Média)")
    path_avaliable = os.path.join(output_dir, "sdavc_matriz_confusao.png");
    plt.savefig(path_avaliable)
    plt.close()

    auc = None
    try:
        # --- MUDANÇA: Usa y_cat (N, 3) e avg_probs (N, 3) ---
        # Agora os shapes batem e o AUC vai funcionar.
        auc = roc_auc_score(y_cat, avg_probs, multi_class='ovr')
        print(f"🏅 AUC-ROC (Ensemble): {auc:.4f}")
    except Exception as e:
        print(f"⚠️ AUC-ROC não pôde ser calculado: {e}")

    # Calcula a MÉDIA das métricas de performance
    avg_tempo_ms_img = np.mean(tempos_predicao)
    avg_throughput_img_s = np.mean(throughputs)
    avg_memoria_mb = np.mean(mems_usadas)

    resultados = {
        "acuracia": report["accuracy"],
        "precisao_macro": report["macro avg"]["precision"],
        "recall_macro": recall_macro,
        "f1_macro": report["macro avg"]["f1-score"],
        "especificidade_macro": especificidade_macro,
        "auc_roc": auc if auc is not None else np.nan,
        "tempo_ms_img_medio": avg_tempo_ms_img,
        "throughput_img_s_medio": avg_throughput_img_s,
        "memoria_mb_media": avg_memoria_mb
    }

    for i, spec in enumerate(especificidades):
        resultados[f"especificidade_classe_{i}"] = spec

    df = pd.DataFrame([resultados])
    path_csv = os.path.join(output_dir, "resultados_sdavc.csv")
    if os.path.exists(path_csv):
        df.to_csv(path_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(path_csv, index=False)

    path_txt = os.path.join(output_dir, "resultados_sdavc.txt")
    with open(path_txt, "a") as f:
        f.write(f"=== Avaliação SDAVC (Ensemble Média de {len(model_files)} folds) ===\n")
        f.write(f"Modelos em: {model_dir}\n")
        f.write(f"Dados em: {output_dir}\n") # Tentativa de registrar o dir de dados
        f.write("-----------------------------\n")
        f.write(f"Acurácia: {report['accuracy']:.4f}\n")
        f.write(f"Precisão (macro): {report['macro avg']['precision']:.4f}\n")
        f.write(f"Recall (macro): {recall_macro:.4f}\n")
        f.write(f"F1-score (macro): {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"AUC-ROC (Ensemble): {auc:.4f}\n" if auc is not None else "AUC-ROC: N/A\n")
        f.write(f"Tempo médio por imagem (ms): {avg_tempo_ms_img:.2f}\n")
        f.write(f"Throughput médio (imagens/s): {avg_throughput_img_s:.2f}\n")
        f.write(f"Memória adicional média (MB): {avg_memoria_mb:.2f}\n")
        f.write(f"Especificidade (macro): {especificidade_macro:.4f}\n")
        f.write("Especificidade por classe (Ensemble):\n")
        for i, spec in enumerate(especificidades):
            f.write(f"  Classe {i}: {spec:.4f}\n")
        f.write("\n")  

    print(f"\n✅ Métricas salvas em: {path_csv} e {path_txt}")

