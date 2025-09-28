import os
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from keras.models import load_model
from keras.utils import to_categorical

def avaliar_sdavc_model(X, y, model_dir="modelos/sdavc", output_dir='avaliacoes', is_compile=True):

    os.makedirs(output_dir, exist_ok=True)

    y_cat = to_categorical(y)
    predicoes = []
    reais = []
    probs = []

    process = psutil.Process(os.getpid()) 

    for fold_file in sorted(os.listdir(model_dir)):
        if not fold_file.endswith(".keras"):
            continue

        model = load_model(os.path.join(model_dir, fold_file), compile=is_compile)
        mem_before = process.memory_info().rss / (1024 * 1024)  
        
        start = time.time()
        y_pred = model.predict(X)
        end = time.time()

        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before

        tempo_predicao = (end - start) / len(X) * 1000
        throughput = len(X) / (end - start )

        print(f"Tempo de predição: {tempo_predicao:.2f} ms/imagem")
        print(f"Throughput: {throughput:.2f} imagens/segundo") 
        print(f"Uso de memória adicional: {mem_used:.2f} MB")

        predicoes.extend(np.argmax(y_pred, axis=1))
        reais.extend(np.argmax(y_cat, axis=1))
        probs.extend(y_pred)

    print("\n📊 Relatório de Classificação (média de todos os folds)")
    report = classification_report(reais, predicoes, output_dict=True,  zero_division=0)
    print(classification_report(reais, predicoes, zero_division=0))

    recall_macro =report['macro avg']['recall']
    print(f"🌟 Sensibilidade (Recall - macro): {recall_macro:.4f}")

    cm = confusion_matrix(reais, predicoes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    n_classes = cm.shape[0]

    especificidades = []
    for i in range(n_classes):

        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        especificidade_i = tn / (tn + fp)
        especificidades.append(especificidade_i)
        print(f"Especificidade (classe {i}): {especificidade_i:.4f}")

    especificidade_macro = np.mean(especificidades)
    print(f"Especificidade (macro): {especificidade_macro:.4f}")

    disp.plot(cmap=plt.cm.Blues)
    
    plt.title("Matriz de Confusão SDAVC")
    path_avaliable = os.path.join(output_dir, "sdavc_matriz_confusao.png");
    plt.savefig(path_avaliable)
    plt.close()

    try:
        auc = roc_auc_score(y_cat, np.array(probs), multi_class='ovr')
        print(f"🏅 AUC-ROC médio: {auc:.4f}")
    except Exception as e:
        print("⚠️ AUC-ROC não pôde ser calculado:", e)


    resultados = {
        "acuracia": report["accuracy"],
        "precisao_macro": report["macro avg"]["precision"],
        "recall_macro": recall_macro,
        "f1_macro": report["macro avg"]["f1-score"],
        "especificidade_macro": especificidade_macro,
        "auc_roc": auc if auc is not None else np.nan,
        "tempo_ms_img": tempo_predicao,
        "throughput_img_s": throughput,
        "memoria_mb": mem_used
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
        f.write("=== Avaliação SDAVC ===\n")
        f.write(f"Modelo / Fold: {fold_file}\n")
        f.write("-----------------------------\n")
        f.write(f"Acurácia: {report['accuracy']:.4f}\n")
        f.write(f"Precisão (macro): {report['macro avg']['precision']:.4f}\n")
        f.write(f"Recall (macro): {recall_macro:.4f}\n")
        f.write(f"F1-score (macro): {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"AUC-ROC: {auc:.4f}\n" if auc is not None else "AUC-ROC: N/A\n")
        f.write(f"Tempo médio por imagem (ms): {tempo_predicao:.2f}\n")
        f.write(f"Throughput (imagens/s): {throughput:.2f}\n")
        f.write(f"Memória adicional (MB): {mem_used:.2f}\n")
        f.write(f"Especificidade (macro): {especificidade_macro:.4f}\n")
        f.write("Especificidade por classe:\n")
        for i, spec in enumerate(especificidades):
            f.write(f"  Classe {i}: {spec:.4f}\n")
        f.write("\n")  

    print(f"\n✅ Métricas salvas em: {path_csv} e {path_txt}")

