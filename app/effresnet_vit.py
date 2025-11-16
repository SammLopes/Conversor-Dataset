import argparse
import sys
import os

# Garante que a pasta raiz (onde está "app/") esteja no sys.path
CAMINHO_DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
if CAMINHO_DIRETORIO_ATUAL not in sys.path:
    sys.path.append(CAMINHO_DIRETORIO_ATUAL)

from app.core.effresnet_vit import treinar_modelo_effresnet_vit, avaliar_modelo_effresnet_vit


def criar_argument_parser_principal():
    argument_parser = argparse.ArgumentParser(
        prog="effresnet_vit",
        description="Orquestrador para treinamento e avaliação do modelo EFFResNet-ViT."
    )

    subparsers = argument_parser.add_subparsers(
        title="comandos",
        dest="nome_do_comando",
        required=True
    )

    # Subcomando: treinar
    subparser_treinar = subparsers.add_parser(
        "treinar",
        help="Realiza o treinamento do modelo EFFResNet-ViT a partir de um arquivo de configuração JSON."
    )
    subparser_treinar.add_argument(
        "--config",
        dest="caminho_arquivo_configuracao",
        type=str,
        required=True,
        help="Caminho para o arquivo JSON de configuração de treinamento."
    )

    # Subcomando: avaliar
    subparser_avaliar = subparsers.add_parser(
        "avaliar",
        help="Avalia um modelo EFFResNet-ViT salvo em disco e calcula métricas."
    )
    subparser_avaliar.add_argument(
        "--modelo",
        dest="caminho_arquivo_modelo",
        type=str,
        required=True,
        help="Caminho para o arquivo .h5 contendo o modelo treinado."
    )
    subparser_avaliar.add_argument(
        "--validacao",
        dest="caminho_diretorio_validacao",
        type=str,
        required=True,
        help="Diretório raiz contendo as imagens de validação (subpastas por classe)."
    )
    subparser_avaliar.add_argument(
        "--tamanho_imagem",
        dest="tamanho_da_imagem",
        type=int,
        default=224,
        help="Tamanho do lado das imagens (padrão: 224)."
    )
    subparser_avaliar.add_argument(
        "--tamanho_lote",
        dest="tamanho_do_lote",
        type=int,
        default=32,
        help="Tamanho do lote usado na avaliação (padrão: 32)."
    )

    return argument_parser


def executar_comando_treinar(caminho_arquivo_configuracao):
    print(f"Iniciando treinamento do modelo EFFResNet-ViT com configuração em: {caminho_arquivo_configuracao}")
    modelo_treinado, historico_de_treinamento = treinar_modelo_effresnet_vit(
        caminho_arquivo_configuracao=caminho_arquivo_configuracao
    )
    print("Treinamento concluído.")
    return modelo_treinado, historico_de_treinamento


def executar_comando_avaliar(
    caminho_arquivo_modelo,
    caminho_diretorio_validacao,
    tamanho_da_imagem,
    tamanho_do_lote
):
    print(f"Avaliando modelo EFFResNet-ViT em: {caminho_arquivo_modelo}")
    print(f"Diretório de validação: {caminho_diretorio_validacao}")

    dicionario_de_metricas = avaliar_modelo_effresnet_vit(
        caminho_arquivo_modelo=caminho_arquivo_modelo,
        caminho_diretorio_validacao=caminho_diretorio_validacao,
        tamanho_da_imagem=tamanho_da_imagem,
        tamanho_do_lote=tamanho_do_lote
    )

    print("\nResumo das principais métricas numéricas retornadas:")
    for nome_da_metrica, valor_da_metrica in dicionario_de_metricas.items():
        print(f"{nome_da_metrica}: {valor_da_metrica}")

    return dicionario_de_metricas


def main():
    argument_parser_principal = criar_argument_parser_principal()
    argumentos = argument_parser_principal.parse_args()

    if argumentos.nome_do_comando == "treinar":
        executar_comando_treinar(
            caminho_arquivo_configuracao=argumentos.caminho_arquivo_configuracao
        )

    elif argumentos.nome_do_comando == "avaliar":
        executar_comando_avaliar(
            caminho_arquivo_modelo=argumentos.caminho_arquivo_modelo,
            caminho_diretorio_validacao=argumentos.caminho_diretorio_validacao,
            tamanho_da_imagem=argumentos.tamanho_da_imagem,
            tamanho_do_lote=argumentos.tamanho_do_lote
        )


if __name__ == "__main__":
    main()
