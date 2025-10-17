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
from app.core.preprocessamento import carregar_dataset_preprocessado

def main():
    parser = argparse.ArgumentParser(description="Script para treinar e avaliar o modelo SDAVC (Keras).")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Ação a ser executada: 'train' ou 'evaluate'")

    # --- Sub-comando para Treinamento ---
    parser_train = subparsers.add_parser("train", help="Executa o treinamento k-fold do modelo.")
    parser_train.add_argument("--data-dir", type=str, required=True, help="Caminho para o diretório raiz do dataset pré-processado.")
    parser_train.add_argument("--save-dir", type=str, default="modelos/sdavc", help="Diretório para salvar os modelos treinados.")
    
    # --- Sub-comando para Avaliação ---
    parser_evaluate = subparsers.add_parser("evaluate", help="Avalia os modelos treinados.")
    parser_evaluate.add_argument("--data-dir", type=str, required=True, help="Caminho para o diretório raiz do dataset de avaliação.")
    parser_evaluate.add_argument("--model-dir", type=str, default="modelos/sdavc", help="Diretório onde os modelos (.keras) estão salvos.")
    parser_evaluate.add_argument("--output-dir", type=str, default="avaliacoes", help="Diretório para salvar os resultados da avaliação.")

    args = parser.parse_args()

    # Carrega os dados para a ação escolhida
    X, y = carregar_dataset_preprocessado(args.data_dir)

    if args.command == "train":
        print("\nIniciando o Treinamento K-Fold...")
        train_sdavc_kfold(X, y, save_dir=args.save_dir)
    elif args.command == "evaluate":
        print("\nIniciando a Avaliação do Modelo...")
        avaliar_sdavc_model(X, y, model_dir=args.model_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main()