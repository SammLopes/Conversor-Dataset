import argparse
import numpy as np
import os
import sys
import cv2
from tqdm import tqdm

# Adiciona o diretório raiz do projeto ao path para permitir a importação do pacote 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.core.sdac_avc.treino_sdac_avc import train_sdavc_kfold, train_sdavc_simple
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
    subparsers = parser.add_subparsers(dest="command", required=True, help="Ação a ser executada: 'train', 'evaluate' ou 'predict'")

    # Sub-comando para Treinamento
    parser_train = subparsers.add_parser("train", help="Executa o treinamento k-fold do modelo.")
    parser_train.add_argument("--data-dir", type=str, required=True)
    parser_train.add_argument("--save-dir", type=str, default="modelos/sdavc")
    parser_train.add_argument("--mode", type=str, choices=["kfold", "simple"], default="simple")
    parser_train.add_argument("--val-split", type=float, default=0.2)
    parser_train.add_argument("--epochs", type=int, default=100)
    parser_train.add_argument("--batch-size", type=int, default=32)

    # Sub-comando para Avaliação
    parser_evaluate = subparsers.add_parser("evaluate", help="Avalia os modelos treinados em um dataset.")
    parser_evaluate.add_argument("--data-dir", type=str, required=True)
    parser_evaluate.add_argument("--model", type=str, help="Caminho para um modelo único (.keras)")
    parser_evaluate.add_argument("--model-dir", type=str, help="Diretório com os modelos (.keras) para ensemble k-fold")
    parser_evaluate.add_argument("--output-dir", type=str, default="avaliacoes")

    # Sub-comando para Predição
    parser_predict = subparsers.add_parser("predict", help="Classifica uma única imagem.")
    parser_predict.add_argument("--image", type=str, required=True)
    group = parser_predict.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--model-dir", type=str)

    args = parser.parse_args()

    if args.command == "train":
        print("\nIniciando o Treinamento do Modelo SDAVC...")
        X, y = carregar_dataset_preprocessado(args.data-dir)
        if X.size == 0:
            print("❌ Erro fatal: O dataset de treino está vazio.")
            return
        if args.mode == "kfold":
            print("Modo de treinamento selecionado: K-Fold")
            train_sdavc_kfold(X, y, save_dir=args.save_dir, n_splits=5, epochs=args.epochs, batch_size=args.batch_size, is_include=True)
        else:
            print("Modo de treinamento selecionado: Simples (train/valid)")
            train_sdavc_simple(X, y, diretorio_de_saida=args.save_dir, quantidade_de_epocas=args.epochs, tamanho_do_lote=args.batch_size, fracao_de_validacao=args.val_split, incluir_otimizador=True)

    elif args.command == "evaluate":
        print("\nIniciando a Avaliação do Modelo...")
        X, y = carregar_split_preprocessado(args.data_dir)
        if X.size == 0:
            print("❌ Erro fatal: O dataset de avaliação está vazio.")
            return
        if args.model:
            from app.core.sdac_avc.avaliador_sdavc import avaliar_modelo_simples
            avaliar_modelo_simples(X, y, model_path=args.model, output_dir=args.output_dir)
        else:
            avaliar_sdavc_model(X, y, model_dir=args.model_dir, output_dir=args.output_dir)

    elif args.command == "predict":
        print(f"\nIniciando Predição da Imagem: {args.image}")
        image_array = load_and_preprocess_image(args.image)
        if image_array is None:
            return
        if args.model:
            probs = predict_single_model(args.model, image_array)
        else:
            probs = predict_ensemble(args.model_dir, image_array)
        if probs is not None:
            idx = np.argmax(probs)
            print(f"Classe Prevista: {CLASSES.get(idx, 'Desconhecida')} - Confiança: {probs[idx]:.2%}")
        else:
            print("❌ Erro: A predição falhou.")

if __name__ == '__main__':
    main()
