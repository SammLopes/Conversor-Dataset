import os
import argparse
import numpy as np
import cv2
from keras.models import load_model

# --- CONFIGURAÇÃO ---
# Defina os nomes das suas classes NA MESMA ORDEM que o Keras usou 
# (ordem alfabética das pastas 'train')
CLASSES = {
    0: "Hemorrhagic Stroke",
    1: "Ischemic Stroke",
    2: "Normal"
}
IMG_SIZE = (224, 224) # O tamanho que seu modelo espera
# --------------------

def load_and_preprocess_image(image_path):
    """Carrega uma única imagem, converte para escala de cinza, redimensiona e normaliza."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Erro: Não foi possível ler a imagem em '{image_path}'")
            return None
        
        # Redimensiona para o tamanho que o modelo espera
        img_resized = cv2.resize(img, IMG_SIZE)
        
        # Normaliza (0-255 -> 0-1)
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # Adiciona a dimensão do canal (224, 224) -> (224, 224, 1)
        img_expanded = np.expand_dims(img_norm, axis=-1)
        
        # Adiciona a dimensão do lote (224, 224, 1) -> (1, 224, 224, 1)
        img_batch = np.expand_dims(img_expanded, axis=0)
        
        return img_batch
    except Exception as e:
        print(f"❌ Erro ao pré-processar a imagem: {e}")
        return None

def predict_single_model(model_path, image_array):
    """Carrega um único modelo e faz uma predição."""
    print(f"--- Modo de Predição: Modelo Único ---")
    print(f"Carregando modelo: {model_path}")
    try:
        model = load_model(model_path, compile=False) # compile=False é mais rápido
        
        print("Fazendo predição...")
        probabilities = model.predict(image_array)[0] # [0] para pegar o primeiro (e único) item do lote
        
        return probabilities
    except Exception as e:
        print(f"❌ Erro ao carregar ou prever com modelo único: {e}")
        return None

def predict_ensemble(model_dir, image_array):
    """Carrega todos os modelos de um diretório e prevê com a média (Ensemble)."""
    print(f"--- Modo de Predição: Ensemble (K-Fold) ---")
    
    model_files = [f for f in sorted(os.listdir(model_dir)) if f.endswith(".keras")]
    if not model_files:
        print(f"❌ Erro: Nenhum arquivo .keras encontrado em '{model_dir}'")
        return None
        
    print(f"Encontrados {len(model_files)} modelos para o ensemble.")
    
    all_probs = []
    
    try:
        for i, fold_file in enumerate(model_files):
            print(f"Carregando e prevendo com fold {i+1}/{len(model_files)}: {fold_file}")
            model_path = os.path.join(model_dir, fold_file)
            model = load_model(model_path, compile=False)
            
            fold_pred_probs = model.predict(image_array)[0]
            all_probs.append(fold_pred_probs)
            
        # Calcula a média das probabilidades
        # 'all_probs' é uma lista de arrays, ex: [ [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], ... ]
        avg_probabilities = np.mean(np.array(all_probs), axis=0)
        
        return avg_probabilities
        
    except Exception as e:
        print(f"❌ Erro durante o processo de ensemble: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Script para prever uma única imagem com o modelo SDAVC.")
    parser.add_argument("--image", type=str, required=True, help="Caminho para a imagem a ser classificada.")
    
    # Grupo para escolher OU um modelo único OU um diretório de ensemble
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Caminho para um ÚNICO arquivo .keras (Ex: o 'melhor' fold).")
    group.add_argument("--model-dir", type=str, help="Caminho para o DIRETÓRIO que contém todos os 5 folds (.keras).")
    
    args = parser.parse_args()

    # 1. Carrega e prepara a imagem
    print(f"Carregando imagem: {args.image}")
    image_array = load_and_preprocess_image(args.image)
    
    if image_array is None:
        return # Erro já foi impresso

    final_probabilities = None
    
    # 2. Decide qual modo de predição usar
    if args.model:
        # Modo "Bom": Modelo Único
        final_probabilities = predict_single_model(args.model, image_array)
        
    elif args.model_dir:
        # Modo "Excelente": Ensemble
        final_probabilities = predict_ensemble(args.model_dir, image_array)

    # 3. Mostra o resultado final
    if final_probabilities is not None:
        print("\n--- Resultado Final ---")
        
        # Pega a classe com maior probabilidade
        predicted_class_index = np.argmax(final_probabilities)
        confidence = final_probabilities[predicted_class_index]
        predicted_class_name = CLASSES.get(predicted_class_index, "Classe Desconhecida")
        
        print(f"Classe Prevista: {predicted_class_name}")
        print(f"Confiança: {confidence:.2%}")
        
        print("\nProbabilidades por Classe:")
        for i, prob in enumerate(final_probabilities):
            class_name = CLASSES.get(i, f"Classe {i}")
            print(f"  {class_name}: {prob:.2%}")

if __name__ == "__main__":
    main()
