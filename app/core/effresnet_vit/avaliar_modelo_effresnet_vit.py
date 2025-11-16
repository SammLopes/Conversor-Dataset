import time
import numpy
import tensorflow as tensorflow
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from .modelo_effresnet_vit import CodificadorDePatches, BlocoCodificadorTransformer


def avaliar_modelo_effresnet_vit(
    caminho_arquivo_modelo,
    caminho_diretorio_validacao,
    tamanho_da_imagem=224,
    tamanho_do_lote=32
):
    """
    Carrega o modelo EFFResNet-ViT salvo, avalia em um diretório de validação
    e imprime métricas de classificação e desempenho.
    """

    # Carregar modelo com objetos personalizados
    dicionario_de_objetos_personalizados = {
        "CodificadorDePatches": CodificadorDePatches,
        "BlocoCodificadorTransformer": BlocoCodificadorTransformer,
    }

    modelo = keras.models.load_model(
        filepath=caminho_arquivo_modelo,
        custom_objects=dicionario_de_objetos_personalizados
    )

    # Criar conjunto de validação (rótulo inteiro para facilitar métricas)
    conjunto_de_validacao = keras.utils.image_dataset_from_directory(
        directory=caminho_diretorio_validacao,
        labels="inferred",
        label_mode="int",
        batch_size=tamanho_do_lote,
        image_size=(tamanho_da_imagem, tamanho_da_imagem),
        shuffle=False
    )
    conjunto_de_validacao = conjunto_de_validacao.prefetch(buffer_size=tensorflow.data.AUTOTUNE)

    lista_de_rotulos_reais = []
    for lote_de_imagens, lote_de_rotulos in conjunto_de_validacao:
        lista_de_rotulos_reais.extend(lote_de_rotulos.numpy())
    vetor_de_rotulos_reais = numpy.array(lista_de_rotulos_reais)
    quantidade_de_imagens = vetor_de_rotulos_reais.shape[0]

    # ----------------- Métricas de desempenho (tempo, throughput, memória) -----------------
    print("\n--- Métricas de Desempenho de Inferência ---")

    informacoes_iniciais_de_memoria = None
    if tensorflow.config.list_physical_devices("GPU"):
        try:
            informacoes_iniciais_de_memoria = tensorflow.config.experimental.get_memory_info("GPU:0")
        except Exception as erro_de_memoria_inicial:
            print(f"Não foi possível obter memória inicial da GPU: {erro_de_memoria_inicial}")

    instante_inicial = time.perf_counter()
    matriz_de_probabilidades_preditas = modelo.predict(conjunto_de_validacao, verbose=1)
    instante_final = time.perf_counter()

    tempo_total_de_predicao_segundos = instante_final - instante_inicial
    tempo_medio_por_imagem_milisegundos = (tempo_total_de_predicao_segundos / quantidade_de_imagens) * 1000.0
    vazao_media_imagens_por_segundo = quantidade_de_imagens / tempo_total_de_predicao_segundos

    print(f"Tempo total de predição: {tempo_total_de_predicao_segundos:.2f} segundos")
    print(f"Tempo médio por imagem: {tempo_medio_por_imagem_milisegundos:.4f} ms")
    print(f"Throughput médio: {vazao_media_imagens_por_segundo:.2f} imagens/s")

    memoria_adicional_maxima_megabytes = None
    if informacoes_iniciais_de_memoria is not None:
        try:
            informacoes_finais_de_memoria = tensorflow.config.experimental.get_memory_info("GPU:0")
            pico_inicial_bytes = informacoes_iniciais_de_memoria.get("peak", 0)
            pico_final_bytes = informacoes_finais_de_memoria.get("peak", 0)
            memoria_adicional_bytes = max(0, pico_final_bytes - pico_inicial_bytes)
            memoria_adicional_maxima_megabytes = memoria_adicional_bytes / (1024.0 * 1024.0)
            print(f"Memória adicional máxima aproximada: {memoria_adicional_maxima_megabytes:.2f} MB")
        except Exception as erro_de_memoria_final:
            print(f"Não foi possível calcular memória adicional de GPU: {erro_de_memoria_final}")

    # ----------------- Métricas de classificação -----------------
    vetor_de_rotulos_preditos = numpy.argmax(matriz_de_probabilidades_preditas, axis=1)
    valor_de_acuracia = numpy.mean(vetor_de_rotulos_preditos == vetor_de_rotulos_reais)

    print("\n--- Métricas de Classificação ---")
    print(f"Acurácia global: {valor_de_acuracia:.4f}")

    relatorio_de_classificacao = classification_report(
        vetor_de_rotulos_reais,
        vetor_de_rotulos_preditos,
        digits=4
    )
    print("\nRelatório de classificação (precisão, revocação, F1 por classe):")
    print(relatorio_de_classificacao)

    # Matriz de confusão e especificidade por classe
    matriz_de_confusao = confusion_matrix(vetor_de_rotulos_reais, vetor_de_rotulos_preditos)
    quantidade_de_classes = matriz_de_confusao.shape[0]

    print("\nEspecificidade por classe:")
    for indice_da_classe in range(quantidade_de_classes):
        verdadeiros_positivos = matriz_de_confusao[indice_da_classe, indice_da_classe]
        falsos_positivos = matriz_de_confusao[:, indice_da_classe].sum() - verdadeiros_positivos
        falsos_negativos = matriz_de_confusao[indice_da_classe, :].sum() - verdadeiros_positivos
        verdadeiros_negativos = matriz_de_confusao.sum() - (
            verdadeiros_positivos + falsos_positivos + falsos_negativos
        )
        valor_de_especificidade = verdadeiros_negativos / (verdadeiros_negativos + falsos_positivos + 1e-8)
        print(f"Classe {indice_da_classe}: especificidade = {valor_de_especificidade:.4f}")

    # AUC-ROC multiclasse
    vetor_de_rotulos_binarizados = label_binarize(
        vetor_de_rotulos_reais,
        classes=list(range(quantidade_de_classes))
    )

    try:
        valor_de_auc_roc_macro = roc_auc_score(
            vetor_de_rotulos_binarizados,
            matriz_de_probabilidades_preditas,
            multi_class="ovr",
            average="macro"
        )
        valor_de_auc_roc_ponderado = roc_auc_score(
            vetor_de_rotulos_binarizados,
            matriz_de_probabilidades_preditas,
            multi_class="ovr",
            average="weighted"
        )
        print(f"\nAUC-ROC macro: {valor_de_auc_roc_macro:.4f}")
        print(f"AUC-ROC ponderado: {valor_de_auc_roc_ponderado:.4f}")
    except ValueError as erro_de_auc:
        print(f"\nNão foi possível calcular AUC-ROC: {erro_de_auc}")

    dicionario_de_metricas = {
        "acuracia": float(valor_de_acuracia),
        "tempo_total_de_predicao_segundos": float(tempo_total_de_predicao_segundos),
        "tempo_medio_por_imagem_milisegundos": float(tempo_medio_por_imagem_milisegundos),
        "vazao_media_imagens_por_segundo": float(vazao_media_imagens_por_segundo),
        "memoria_adicional_maxima_megabytes": float(memoria_adicional_maxima_megabytes)
        if memoria_adicional_maxima_megabytes is not None
        else None,
    }

    return dicionario_de_metricas
