import time
import os  # Necessário para criar pastas e caminhos
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
    Carrega o modelo EFFResNet-ViT salvo, avalia em um diretório de validação,
    imprime métricas no terminal e salva um relatório em 'avaliacoes/'.
    """

    # --- PREPARAÇÃO PARA SALVAR RELATÓRIO ---
    linhas_do_relatorio = []

    def registrar(texto):
        """Função auxiliar para imprimir no terminal e guardar na memória."""
        print(texto)
        linhas_do_relatorio.append(texto)

    registrar(f"Iniciando avaliação do modelo: {caminho_arquivo_modelo}")
    registrar(f"Diretório de validação: {caminho_diretorio_validacao}")
    # ----------------------------------------

    # Carregar modelo com objetos personalizados
    dicionario_de_objetos_personalizados = {
        "CodificadorDePatches": CodificadorDePatches,
        "BlocoCodificadorTransformer": BlocoCodificadorTransformer,
    }

    try:
        modelo = keras.models.load_model(
            filepath=caminho_arquivo_modelo,
            custom_objects=dicionario_de_objetos_personalizados
        )
    except Exception as e:
        registrar(f"ERRO CRÍTICO ao carregar o modelo: {e}")
        return {}

    # Criar conjunto de validação
    conjunto_de_validacao = keras.utils.image_dataset_from_directory(
        directory=caminho_diretorio_validacao,
        labels="inferred",
        label_mode="int",
        batch_size=tamanho_do_lote,
        image_size=(tamanho_da_imagem, tamanho_da_imagem),
        shuffle=False,
        color_mode="grayscale"
    )
    conjunto_de_validacao = conjunto_de_validacao.prefetch(buffer_size=tensorflow.data.AUTOTUNE)

    lista_de_rotulos_reais = []
    for lote_de_imagens, lote_de_rotulos in conjunto_de_validacao:
        lista_de_rotulos_reais.extend(lote_de_rotulos.numpy())
    vetor_de_rotulos_reais = numpy.array(lista_de_rotulos_reais)
    quantidade_de_imagens = vetor_de_rotulos_reais.shape[0]

    # ----------------- Métricas de desempenho -----------------
    registrar("\n--- Métricas de Desempenho de Inferência ---")

    informacoes_iniciais_de_memoria = None
    if tensorflow.config.list_physical_devices("GPU"):
        try:
            informacoes_iniciais_de_memoria = tensorflow.config.experimental.get_memory_info("GPU:0")
        except Exception:
            pass

    instante_inicial = time.perf_counter()
    matriz_de_probabilidades_preditas = modelo.predict(conjunto_de_validacao, verbose=1)
    instante_final = time.perf_counter()

    tempo_total_de_predicao_segundos = instante_final - instante_inicial
    tempo_medio_por_imagem_milisegundos = (tempo_total_de_predicao_segundos / quantidade_de_imagens) * 1000.0
    vazao_media_imagens_por_segundo = quantidade_de_imagens / tempo_total_de_predicao_segundos

    registrar(f"Tempo total de predição: {tempo_total_de_predicao_segundos:.2f} segundos")
    registrar(f"Tempo médio por imagem: {tempo_medio_por_imagem_milisegundos:.4f} ms")
    registrar(f"Throughput médio: {vazao_media_imagens_por_segundo:.2f} imagens/s")

    memoria_adicional_maxima_megabytes = None
    if informacoes_iniciais_de_memoria is not None:
        try:
            informacoes_finais_de_memoria = tensorflow.config.experimental.get_memory_info("GPU:0")
            pico_inicial_bytes = informacoes_iniciais_de_memoria.get("peak", 0)
            pico_final_bytes = informacoes_finais_de_memoria.get("peak", 0)
            memoria_adicional_bytes = max(0, pico_final_bytes - pico_inicial_bytes)
            memoria_adicional_maxima_megabytes = memoria_adicional_bytes / (1024.0 * 1024.0)
            registrar(f"Memória adicional máxima aproximada: {memoria_adicional_maxima_megabytes:.2f} MB")
        except Exception:
            pass

    # ----------------- Métricas de classificação -----------------
    vetor_de_rotulos_preditos = numpy.argmax(matriz_de_probabilidades_preditas, axis=1)
    valor_de_acuracia = numpy.mean(vetor_de_rotulos_preditos == vetor_de_rotulos_reais)

    registrar("\n--- Métricas de Classificação ---")
    registrar(f"Acurácia global: {valor_de_acuracia:.4f}")

    # Relatório do Sklearn (Precision, Recall, F1)
    relatorio_de_classificacao = classification_report(
        vetor_de_rotulos_reais,
        vetor_de_rotulos_preditos,
        digits=4
    )
    registrar("\nRelatório de classificação (precisão, revocação, F1 por classe):")
    registrar(relatorio_de_classificacao)

    # Especificidade
    matriz_de_confusao = confusion_matrix(vetor_de_rotulos_reais, vetor_de_rotulos_preditos)
    quantidade_de_classes = matriz_de_confusao.shape[0]

    registrar("\nEspecificidade por classe:")
    for indice_da_classe in range(quantidade_de_classes):
        verdadeiros_positivos = matriz_de_confusao[indice_da_classe, indice_da_classe]
        falsos_positivos = matriz_de_confusao[:, indice_da_classe].sum() - verdadeiros_positivos
        falsos_negativos = matriz_de_confusao[indice_da_classe, :].sum() - verdadeiros_positivos
        verdadeiros_negativos = matriz_de_confusao.sum() - (
            verdadeiros_positivos + falsos_positivos + falsos_negativos
        )
        # Evitar divisão por zero
        denominador = verdadeiros_negativos + falsos_positivos
        if denominador == 0:
            valor_de_especificidade = 0.0
        else:
            valor_de_especificidade = verdadeiros_negativos / denominador
            
        registrar(f"Classe {indice_da_classe}: especificidade = {valor_de_especificidade:.4f}")

    # AUC-ROC
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
        registrar(f"\nAUC-ROC macro: {valor_de_auc_roc_macro:.4f}")
        registrar(f"AUC-ROC ponderado: {valor_de_auc_roc_ponderado:.4f}")
    except ValueError as erro_de_auc:
        registrar(f"\nNão foi possível calcular AUC-ROC: {erro_de_auc}")

    nome_diretorio_saida = "avaliacoes/effresnet_vit"
    os.makedirs(nome_diretorio_saida, exist_ok=True)
    
    caminho_arquivo_saida = os.path.join(nome_diretorio_saida, "resultado_avaliacao.txt")
    
    try:
        with open(caminho_arquivo_saida, "w", encoding="utf-8") as arquivo:
            arquivo.write("\n".join(linhas_do_relatorio))
        print(f"\n[SUCESSO] Relatório completo salvo em: {caminho_arquivo_saida}")
    except Exception as e:
        print(f"\n[ERRO] Falha ao salvar o arquivo de relatório: {e}")

    # Retornar dicionário (para uso programático se necessário)
    dicionario_de_metricas = {
        "acuracia": float(valor_de_acuracia),
        "tempo_total": float(tempo_total_de_predicao_segundos),
    }

    return dicionario_de_metricas