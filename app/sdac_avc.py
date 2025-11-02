import argparse
import numpy as np
import os
import sys
import cv2
from tqdm import tqdm

# Adiciona o diretório raiz do projeto ao path para permitir a importação do pacote 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.core.sdac_avc.treino_sdac_avc import train_sdavc_kfold
from app.core.sdac_avc.avaliador_sdavc import avaliar_sdavc_model

from app.core.preprocessamento import carregar_dataset_preprocessado, carregar_split_preprocessado

from app.core.sdac_avc.predicao_sdac_avc import (
        load_and_preprocess_image, 
        predict_single_model, 
        predict_ensemble, 
        CLASSES
    )

def main():
    parser = argparse.ArgumentParser(description="Script para treinar, avaliar e prever com o modelo SDAVC (Keras).")
    # --- MUDANÇA: Adiciona 'predict' à ajuda ---
    subparsers = parser.add_subparsers(dest="command", required=True, help="Ação a ser executada: 'train', 'evaluate' ou 'predict'")

    # --- Sub-comando para Treinamento ---
    parser_train = subparsers.add_parser("train", help="Executa o treinamento k-fold do modelo.")
    parser_train.add_argument("--data-dir", type=str, required=True, help="Caminho para o diretório raiz do dataset pré-processado (com subpastas train/valid).")
    parser_train.add_argument("--save-dir", type=str, default="modelos/sdavc", help="Diretório para salvar os modelos treinados.")
    
    # --- Sub-comando para Avaliação ---
    parser_evaluate = subparsers.add_parser("evaluate", help="Avalia os modelos treinados em um dataset.")
    parser_evaluate.add_argument("--data-dir", type=str, required=True, help="Caminho para o diretório de dados de avaliação (ex: /path/to/valid).")
    parser_evaluate.add_argument("--model-dir", type=str, default="modelos/sdavc", help="Diretório onde os 5 modelos (.keras) estão salvos.")
    parser_evaluate.add_argument("--output-dir", type=str, default="avaliacoes", help="Diretório para salvar os resultados da avaliação.")

    # --- NOVO Sub-comando para Predição ---
    parser_predict = subparsers.add_parser("predict", help="Classifica uma única imagem.")
    parser_predict.add_argument("--image", type=str, required=True, help="Caminho para a imagem a ser classificada.")
    
    # Grupo de argumentos mutuamente exclusivos para o modelo
    group = parser_predict.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Caminho para um ÚNICO arquivo .keras (Ex: o 'melhor' fold).")
    group.add_argument("--model-dir", type=str, help="Caminho para o DIRETÓRIO que contém todos os 5 folds (Modo Ensemble).")
    # ------------------------------------

    args = parser.parse_args()

    # --- LÓGICA DE COMANDO ATUALIZADA ---
    # A lógica de carregamento de dados foi movida para DENTRO de cada 'if'
    
    if args.command == "train":
        print("\nIniciando o Treinamento K-Fold...")
        # Carrega o dataset de TREINO (que contém subpastas train/valid)
        
        print(" Teste train ")
        exit()
        X, y = carregar_dataset_preprocessado(args.data_dir)
        if X.size == 0:
             print("❌ Erro fatal: O dataset de treino está vazio. Verifique o caminho e a estrutura.")
             return
        train_sdavc_kfold(X, y, save_dir=args.save_dir)
        
    elif args.command == "evaluate":
        print("\nIniciando a Avaliação do Modelo...")
        # Carrega o dataset de AVALIAÇÃO (ex: a pasta 'valid' ou 'test' direto)
        
        print(" Teste evaluate ")
        exit()
        X, y = carregar_split_preprocessado(args.data_dir)
        if X.size == 0:
             print("❌ Erro fatal: O dataset de avaliação está vazio. Verifique o caminho.")
             return
        avaliar_sdavc_model(X, y, model_dir=args.model_dir, output_dir=args.output_dir)
    
    elif args.command == "predict":

        print(" Teste Predict ")
        exit()
        
        print(f"\nIniciando Predição da Imagem: {args.image}")
        
        # 1. Carrega e prepara a imagem
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
            
            predicted_class_index = np.argmax(final_probabilities)
            confidence = final_probabilities[predicted_class_index]
            predicted_class_name = CLASSES.get(predicted_class_index, "Classe Desconhecida")
            
            print(f"Classe Prevista: {predicted_class_name}")
            print(f"Confiança: {confidence:.2%}")
            
            print("\nProbabilidades por Classe:")
            for i, prob in enumerate(final_probabilities):
                class_name = CLASSES.get(i, f"Classe {i}")
                print(f"  {class_name}: {prob:.2%}")
        else:
            print("❌ Erro: A predição falhou.")

if __name__ == '__main__':
    main()

