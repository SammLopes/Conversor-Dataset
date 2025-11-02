import os
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from keras.models import load_model
from keras.utils import to_categorical

def _calcular_especificidade(cm):
    """Função auxiliar para calcular especificidade macro e por classe a partir da matriz de confusão."""
    n_classes = cm.shape[0]
    especificidades = []
    for i in range(n_classes):
        # Verdadeiros Negativos (TN) para a classe 'i'
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        # Falsos Positivos (FP) para a classe 'i'
        fp = cm[:, i].sum() - cm[i, i]
        
        especificidade_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        especificidades.append(especificidade_i)
        
    especificidade_macro = np.mean(especificidades)
    return especificidade_macro, especificidades

def avaliar_sdavc_model(X, y, model_dir="modelos/sdavc", output_dir='avaliacoes', is_compile=False):

    os.makedirs(output_dir, exist_ok=True)

    # Prepara os rótulos verdadeiros (em 1D e em One-Hot)
    y_reais_1d = y # y original (ex: [0, 1, 2, 0])
    y_reais_2d_cat = to_categorical(y) # y em one-hot (ex: [[1,0,0], [0,1,0], ...])
    
    # --- Acumuladores para o Ensemble Final ---
    all_probs = np.zeros((len(X), y_reais_2d_cat.shape[1]))
    fold_results_list = [] # Lista para salvar os dicts de cada fold
    
    # Listas para guardar métricas de performance de cada fold
    tempos_predicao = []
    throughputs = []
    mems_usadas = []
    
    process = psutil.Process(os.getpid()) 
    
    model_files = [f for f in sorted(os.listdir(model_dir)) if f.endswith(".keras")]
    
    if not model_files:
        print(f"❌ Erro: Nenhum arquivo .keras encontrado em '{model_dir}'")
        return

    print(f"ℹ️ Encontrados {len(model_files)} folds para avaliação.")

    # Abre o arquivo .txt para salvar resultados
    path_txt = os.path.join(output_dir, "resultados_sdavc_completo.txt")
    with open(path_txt, "w", encoding="utf-8") as f_txt: # 'w' para apagar o arquivo antigo

        for fold_file in model_files:
            print(f"\n--- Processando Fold Individual: {fold_file} ---")
            f_txt.write(f"--- Avaliação Individual: {fold_file} ---\n")
            
            model = load_model(os.path.join(model_dir, fold_file), compile=is_compile)
            
            mem_before = process.memory_info().rss / (1024 * 1024)  
            start = time.time()
            
            # 1. Faz a predição do fold
            fold_pred_probs = model.predict(X)
            
            end = time.time()
            mem_after = process.memory_info().rss / (1024 * 1024)
            
            # 2. Calcula performance do fold
            mem_used = mem_after - mem_before
            tempo_predicao = (end - start) / len(X) * 1000
            throughput = len(X) / (end - start )

            tempos_predicao.append(tempo_predicao)
            throughputs.append(throughput)
            mems_usadas.append(mem_used)
            
            # 3. Calcula métricas de acurácia do fold
            fold_pred_1d = np.argmax(fold_pred_probs, axis=1)
            
            fold_report_dict = classification_report(y_reais_1d, fold_pred_1d, output_dict=True, zero_division=0)
            fold_cm = confusion_matrix(y_reais_1d, fold_pred_1d)
            fold_spec_macro, fold_specs_list = _calcular_especificidade(fold_cm)
            
            fold_auc = None
            try:
                fold_auc = roc_auc_score(y_reais_2d_cat, fold_pred_probs, multi_class='ovr')
            except Exception as e:
                print(f"⚠️ AUC-ROC (fold) não pôde ser calculado: {e}")

            # 4. Imprime o relatório individual
            print(f"Relatório do Fold: {fold_file}")
            print(classification_report(y_reais_1d, fold_pred_1d, zero_division=0))
            print(f"AUC-ROC (Fold): {fold_auc:.4f}" if fold_auc is not None else "AUC-ROC (Fold): N/A")
            print(f"Especificidade (Fold Macro): {fold_spec_macro:.4f}")

            # 5. Salva o resultado individual
            fold_result_data = {
                "fold": fold_file,
                "acuracia": fold_report_dict["accuracy"],
                "precisao_macro": fold_report_dict["macro avg"]["precision"],
                "recall_macro": fold_report_dict["macro avg"]["recall"],
                "f1_macro": fold_report_dict["macro avg"]["f1-score"],
                "especificidade_macro": fold_spec_macro,
                "auc_roc": fold_auc if fold_auc is not None else np.nan,
                "tempo_ms_img": tempo_predicao,
                "throughput_img_s": throughput,
                "memoria_mb": mem_used
            }
            fold_results_list.append(fold_result_data)
            
            # Salva no arquivo de texto
            f_txt.write(classification_report(y_reais_1d, fold_pred_1d, zero_division=0) + "\n")
            f_txt.write(f"AUC-ROC: {fold_auc:.4f}\n" if fold_auc is not None else "AUC-ROC: N/A\n")
            f_txt.write(f"Especificidade (Macro): {fold_spec_macro:.4f}\n\n")

            # 6. Acumula para o Ensemble
            all_probs += fold_pred_probs

        # --- FIM DO LOOP ---
        
        print("\n" + "="*50)
        print("--- Calculando Métricas Finais (Ensemble Média) ---")
        print("="*50)

        # --- Calcula o Ensemble (Média das Probabilidades) ---
        avg_probs = all_probs / len(model_files)
        
        # Converte as probabilidades médias em classes
        ensemble_pred_1d = np.argmax(avg_probs, axis=1)
        
        # Calcula métricas do Ensemble
        ensemble_report_dict = classification_report(y_reais_1d, ensemble_pred_1d, output_dict=True, zero_division=0)
        ensemble_cm = confusion_matrix(y_reais_1d, ensemble_pred_1d)
        ensemble_spec_macro, ensemble_specs_list = _calcular_especificidade(ensemble_cm)
        
        ensemble_auc = None
        try:
            ensemble_auc = roc_auc_score(y_reais_2d_cat, avg_probs, multi_class='ovr')
        except Exception as e:
            print(f"⚠️ AUC-ROC (Ensemble) não pôde ser calculado: {e}")

        # Calcula a MÉDIA das métricas de performance
        avg_tempo_ms_img = np.mean(tempos_predicao)
        avg_throughput_img_s = np.mean(throughputs)
        avg_memoria_mb = np.mean(mems_usadas)
        
        # --- Imprime Relatório do Ensemble ---
        print("\n📊 Relatório de Classificação (Ensemble Média dos Folds)")
        print(classification_report(y_reais_1d, ensemble_pred_1d, zero_division=0))
        print(f"🌟 Sensibilidade (Recall - macro): {ensemble_report_dict['macro avg']['recall']:.4f}")
        for i, spec in enumerate(ensemble_specs_list):
            print(f"Especificidade (classe {i}): {spec:.4f}")
        print(f"Especificidade (macro): {ensemble_spec_macro:.4f}")
        print(f"🏅 AUC-ROC (Ensemble): {ensemble_auc:.4f}" if ensemble_auc is not None else "AUC-ROC (Ensemble): N/A")

        # --- Salva Matriz de Confusão do Ensemble ---
        disp = ConfusionMatrixDisplay(confusion_matrix=ensemble_cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusão SDAVC (Ensemble Média)")
        path_matriz = os.path.join(output_dir, "sdavc_matriz_confusao_ensemble.png");
        plt.savefig(path_matriz)
        plt.close()

        # --- Salva CSVs ---
        
        # 1. Salva CSV dos folds individuais
        df_folds = pd.DataFrame(fold_results_list)
        path_csv_folds = os.path.join(output_dir, "resultados_sdavc_por_fold.csv")
        df_folds.to_csv(path_csv_folds, index=False)
        
        # 2. Salva CSV do resultado Ensemble
        ensemble_result_data = {
            "metrica": "Ensemble",
            "acuracia": ensemble_report_dict["accuracy"],
            "precisao_macro": ensemble_report_dict["macro avg"]["precision"],
            "recall_macro": ensemble_report_dict["macro avg"]["recall"],
            "f1_macro": ensemble_report_dict["macro avg"]["f1-score"],
            "especificidade_macro": ensemble_spec_macro,
            "auc_roc": ensemble_auc if ensemble_auc is not None else np.nan,
            "tempo_ms_img_medio": avg_tempo_ms_img,
            "throughput_img_s_medio": avg_throughput_img_s,
            "memoria_mb_media": avg_memoria_mb
        }
        df_ensemble = pd.DataFrame([ensemble_result_data])
        path_csv_ensemble = os.path.join(output_dir, "resultados_sdavc_ensemble.csv")
        df_ensemble.to_csv(path_csv_ensemble, index=False)
        
        # --- Salva Resumo Ensemble no TXT ---
        f_txt.write("\n" + "="*50 + "\n")
        f_txt.write(f"=== Avaliação SDAVC (Ensemble Média de {len(model_files)} folds) ===\n")
        f_txt.write("="*50 + "\n")
        f_txt.write(f"Acurácia: {ensemble_report_dict['accuracy']:.4f}\n")
        f_txt.write(f"Precisão (macro): {ensemble_report_dict['macro avg']['precision']:.4f}\n")
        f_txt.write(f"Recall (macro): {ensemble_report_dict['macro avg']['recall']:.4f}\n")
        f_txt.write(f"F1-score (macro): {ensemble_report_dict['macro avg']['f1-score']:.4f}\n")
        f_txt.write(f"AUC-ROC (Ensemble): {ensemble_auc:.4f}\n" if ensemble_auc is not None else "AUC-ROC: N/A\n")
        f_txt.write(f"Tempo médio por imagem (ms): {avg_tempo_ms_img:.2f}\n")
        f_txt.write(f"Throughput médio (imagens/s): {avg_throughput_img_s:.2f}\n")
        f_txt.write(f"Memória adicional média (MB): {avg_memoria_mb:.2f}\n")
        f_txt.write(f"Especificidade (macro): {ensemble_spec_macro:.4f}\n")
        f_txt.write("Especificidade por classe (Ensemble):\n")
        for i, spec in enumerate(ensemble_specs_list):
            f_txt.write(f"  Classe {i}: {spec:.4f}\n")
        f_txt.write("\n")  

    print(f"\n✅ Métricas salvas em: {output_dir}")
    print(f"   -> CSV Individual: {path_csv_folds}")
    print(f"   -> CSV Ensemble: {path_csv_ensemble}")
    print(f"   -> TXT Completo: {path_txt}")
