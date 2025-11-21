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
    # --- MUDANÇA: Adiciona 'predict' à ajuda ---
    subparsers = parser.add_subparsers(dest="command", required=True, help="Ação a ser executada: 'train', 'evaluate' ou 'predict'")

    # --- Sub-comando para Treinamento ---
    parser_train = subparsers.add_parser("train", help="Executa o treinamento k-fold do modelo.")
    parser_train.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Caminho para o diretório raiz do dataset pré-processado."
    )
    parser_train.add_argument(
        "--save-dir",
        type=str,
        default="modelos/sdavc",
        help="Diretório para salvar os modelos treinados."
    )
    parser_train.add_argument(
        "--mode",
        type=str,
        choices=["kfold", "simple"],
        default="kfold",
        help="Modo de treinamento: 'kfold' (padrão) ou 'simple' (treino/validacao único)."
    )
    parser_train.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraçao de validacao usada no modo 'simple' (padrão: 0.2 = 20%%)."
    )
    parser_train.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Quantidade máxima de épocas para o treinamento."
    )
    parser_train.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote (batch size)."
    )
    
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
        print("\nIniciando o Treinamento do Modelo SDAVC...")

        matriz_de_imagens, vetor_de_rotulos = carregar_dataset_preprocessado(args.data_dir)
        if matriz_de_imagens.size == 0:
            print("❌ Erro fatal: O dataset de treino está vazio. Verifique o caminho e a estrutura.")
            return

        if args.mode == "kfold":
            print("Modo de treinamento selecionado: K-Fold")
            train_sdavc_kfold(
                matriz_de_imagens,
                vetor_de_rotulos,
                save_dir=args.save_dir,
                n_splits=5,
                epochs=args.epochs,
                batch_size=args.batch_size,
                is_include=True,
            )
        else:
            print("Modo de treinamento selecionado: Simples (train/valid)")
            train_sdavc_simple(
                matriz_de_imagens,
                vetor_de_rotulos,
                diretorio_de_saida=args.save_dir,
                quantidade_de_epocas=args.epochs,
                tamanho_do_lote=args.batch_size,
                fracao_de_validacao=args.val_split,
                incluir_otimizador=True,
            )
            
    elif args.command == "evaluate":
        print("\nIniciando a Avaliação do Modelo...")

        matriz_de_imagens_de_avaliacao, vetor_de_rotulos_de_avaliacao = carregar_split_preprocessado(args.data_dir)
        if matriz_de_imagens_de_avaliacao.size == 0:
            print("❌ Erro fatal: O dataset de avaliação está vazio. Verifique o caminho.")
            return

        avaliar_sdavc_model(
            matriz_de_imagens_de_avaliacao,
            vetor_de_rotulos_de_avaliacao,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
        )
    
    elif args.command == "predict":

        print(f"\nIniciando Predição da Imagem: {args.image}")

        matriz_da_imagem = load_and_preprocess_image(args.image)
        if matriz_da_imagem is None:
            # A função de carregamento já imprime a mensagem de erro
            return

        probabilidades_finais = None

        if args.model:
            # Usa um único modelo (.keras)
            probabilidades_finais = predict_single_model(args.model, matriz_da_imagem)

        elif args.model_dir:
            # Usa ensemble com todos os folds dentro do diretório
            probabilidades_finais = predict_ensemble(args.model_dir, matriz_da_imagem)

        if probabilidades_finais is not None:
            print("\n--- Resultado Final ---")

            indice_da_classe_prevista = np.argmax(probabilidades_finais)
            confianca = probabilidades_finais[indice_da_classe_prevista]
            nome_da_classe_prevista = CLASSES.get(indice_da_classe_prevista, "Classe Desconhecida")

            print(f"Classe Prevista: {nome_da_classe_prevista}")
            print(f"Confiança: {confianca:.2%}")

            print("\nProbabilidades por Classe:")
            for indice_da_classe, probabilidade in enumerate(probabilidades_finais):
                nome_da_classe = CLASSES.get(indice_da_classe, f"Classe {indice_da_classe}")
                print(f"  {nome_da_classe}: {probabilidade:.2%}")
        else:
            print("❌ Erro: A predição falhou.")

if __name__ == '__main__':
    main()

