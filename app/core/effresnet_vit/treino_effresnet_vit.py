import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# --- Configurações ---
# (Certifique-se que estes valores batem com os do seu treino)
IMG_SIZE = 224
BATCH_SIZE = 32 # O batch_size para avaliação pode ser diferente, mas 32 é eficiente
NUM_CLASSES = 3 # Número de classes no seu problema
VALID_DIR = "dataset_custom_preprocessed/valid" # O diretório que você especificou
MODEL_PATH = "effresnet_vit_best_model.h5"     # O modelo salvo pelo script de treino

def calculate_specificity(y_true, y_pred_labels, num_classes):
    """ Calcula a especificidade para cada classe em um cenário multiclasse. """
    cm = confusion_matrix(y_true, y_pred_labels)
    specificities = []
    
    for i in range(num_classes):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificities.append(specificity)
        
    return specificities

def evaluate_model(model_path, validation_dir, num_classes):
    """ Executa a avaliação completa do modelo no conjunto de validação. """
    
    print(f"--- Iniciando Avaliação em '{validation_dir}' ---")
    
    # --- 1. Carregar Modelo Treinado ---
    print(f"Carregando modelo de '{model_path}'...")
    # Precisamos registrar as camadas customizadas para que o Keras possa carregá-las
    custom_objects = {"PatchEncoder": PatchEncoder}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # --- 2. Carregar Dados de Validação ---
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int', 
        shuffle=False  # Crucial para alinhamento de predições/labels
    )
    
    num_images = len(val_dataset.file_paths)
    if num_images == 0:
        print(f"Erro: Nenhum imagem encontrada em {validation_dir}")
        return
    print(f"Encontradas {num_images} imagens de validação.")

    # --- 3. Obter Labels Verdadeiros (y_true) ---
    y_true = []
    for images, labels in val_dataset:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    class_names = val_dataset.class_names

    # --- 4. Calcular Métricas de Desempenho (Tempo e Throughput) ---
    print("\n--- Métricas de Desempenho (Inferência) ---")
    start_time = time.perf_counter()
    
    y_pred_probs = model.predict(val_dataset, verbose=1)
    
    end_time = time.perf_counter()

    total_time_sec = end_time - start_time
    tempo_medio_ms = (total_time_sec / num_images) * 1000
    throughput_imgs_s = num_images / total_time_sec

    print(f"Tempo total de predição: {total_time_sec:.2f} segundos")
    print(f"Tempo médio por imagem: {tempo_medio_ms:.4f} ms")
    print(f"Throughput médio: {throughput_imgs_s:.2f} imagens/s")

    # --- 5. Calcular Métricas de Classificação ---
    print("\n--- Métricas de Classificação ---")
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    # Acurácia (Accuracy)
    accuracy = np.sum(y_true == y_pred_labels) / num_images
    print(f"Acurácia Global: {accuracy:.4f}")

    # Precisão, Recall, F1-score
    print("\nRelatório de Classificação (Precisão, Recall, F1-score):")
    print(classification_report(y_true, y_pred_labels, target_names=class_names, digits=4))

    # Especificidade (Specificity)
    specificities = calculate_specificity(y_true, y_pred_labels, num_classes)
    print("\nEspecificidade (por classe):")
    for i, name in enumerate(class_names):
        print(f"  - {name}: {specificities[i]:.4f}")

    # AUC-ROC
    y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        auc_macro = roc_auc_score(y_true_binarized, y_pred_probs, average='macro', multi_class='ovr')
        auc_weighted = roc_auc_score(y_true_binarized, y_pred_probs, average='weighted', multi_class='ovr')
        print("\nAUC-ROC (Area Under the Curve):")
        print(f"  - AUC (Macro Avg): {auc_macro:.4f}")
        print(f"  - AUC (Weighted Avg): {auc_weighted:.4f}")
    except ValueError as e:
        print(f"\nNão foi possível calcular o AUC-ROC: {e}")

    print("\n--- Avaliação Concluída ---")

# --- Para executar este script ---
if __name__ == "__main__":
    # Certifique-se que o diretório 'valid' existe e o modelo .h5 está no mesmo local
    try:
        evaluate_model(MODEL_PATH, VALID_DIR, NUM_CLASSES)
    except FileNotFoundError:
        print(f"Erro: Arquivo do modelo '{MODEL_PATH}' ou diretório '{VALID_DIR}' não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")