import json
import tensorflow as tensorflow
from tensorflow import keras

from .modelo_effresnet_vit import criar_modelo_effresnet_vit


def carregar_conjunto_de_dados_de_imagens(
    caminho_diretorio_imagens,
    tamanho_da_imagem,
    tamanho_do_lote,
    modo_de_rotulo,
    embaralhar
):
    conjunto_de_dados = keras.utils.image_dataset_from_directory(
        directory=caminho_diretorio_imagens,
        labels="inferred",
        label_mode=modo_de_rotulo,
        batch_size=tamanho_do_lote,
        image_size=(tamanho_da_imagem, tamanho_da_imagem),
        shuffle=embaralhar
    )
    conjunto_de_dados = conjunto_de_dados.prefetch(buffer_size=tensorflow.data.AUTOTUNE)
    return conjunto_de_dados


def treinar_modelo_effresnet_vit(caminho_arquivo_configuracao):
    """
    Lê um arquivo JSON de configuração, constrói o modelo EFFResNet-ViT,
    compila e treina usando os parâmetros especificados.
    """

    with open(caminho_arquivo_configuracao, "r") as arquivo_de_configuracao:
        dicionario_de_configuracao = json.load(arquivo_de_configuracao)

    # Diretórios de dados
    caminho_diretorio_treinamento = dicionario_de_configuracao["training_directory"]
    caminho_diretorio_validacao = dicionario_de_configuracao["validation_directory"]

    # Hiperparâmetros de entrada
    tamanho_da_imagem = dicionario_de_configuracao.get("image_size", 224)
    quantidade_de_classes = dicionario_de_configuracao["num_classes"]

    # Otimizador e perda
    taxa_de_aprendizado = dicionario_de_configuracao["optimizer"]["learning_rate"]
    tipo_de_otimizador = dicionario_de_configuracao["optimizer"]["type"]
    funcao_de_perda = dicionario_de_configuracao["loss_function"]

    # Tamanho do lote e épocas
    tamanho_do_lote_maximo = dicionario_de_configuracao["batch_size"]["max"]
    quantidade_de_epocas = dicionario_de_configuracao["epochs"]

    # Parada antecipada
    parametro_de_monitoramento = dicionario_de_configuracao["callbacks"]["early_stopping"]["monitor"]
    paciencia_para_parada_antecipada = dicionario_de_configuracao["callbacks"]["early_stopping"]["patience"]

    # Regularização
    taxa_de_dropout = dicionario_de_configuracao["regularization"]["dropout"]

    # Fine-tuning
    quantidade_de_camadas_para_finetuning_efficientnet = dicionario_de_configuracao.get(
        "efficientnet_finetune_layers",
        15
    )
    quantidade_de_camadas_para_finetuning_resnet = dicionario_de_configuracao.get(
        "resnet_finetune_layers",
        20
    )

    # Caminho do melhor modelo
    caminho_para_salvar_melhor_modelo = dicionario_de_configuracao.get(
        "best_model_path",
        "melhor_modelo_effresnet_vit.h5"
    )

    # Conjuntos de dados
    conjunto_de_treinamento = carregar_conjunto_de_dados_de_imagens(
        caminho_diretorio_imagens=caminho_diretorio_treinamento,
        tamanho_da_imagem=tamanho_da_imagem,
        tamanho_do_lote=tamanho_do_lote_maximo,
        modo_de_rotulo="categorical",
        embaralhar=True
    )

    conjunto_de_validacao = carregar_conjunto_de_dados_de_imagens(
        caminho_diretorio_imagens=caminho_diretorio_validacao,
        tamanho_da_imagem=tamanho_da_imagem,
        tamanho_do_lote=tamanho_do_lote_maximo,
        modo_de_rotulo="categorical",
        embaralhar=False
    )

    # Construir modelo
    modelo_effresnet_vit = criar_modelo_effresnet_vit(
        formato_da_entrada=(tamanho_da_imagem, tamanho_da_imagem, 3),
        quantidade_de_classes=quantidade_de_classes,
        quantidade_de_blocos_transformer=dicionario_de_configuracao.get("num_transformer_blocks", 4),
        dimensao_de_incorporacao=dicionario_de_configuracao.get("hidden_dim", 128),
        quantidade_de_cabecas_de_atencao=dicionario_de_configuracao.get("num_heads", 4),
        dimensao_da_camada_alimentada_adiante=dicionario_de_configuracao.get("feedforward_dim", 256),
        dimensao_da_mlp_final=dicionario_de_configuracao.get("mlp_dim", 64),
        taxa_de_dropout=taxa_de_dropout,
        quantidade_de_camadas_para_finetuning_efficientnet=quantidade_de_camadas_para_finetuning_efficientnet,
        quantidade_de_camadas_para_finetuning_resnet=quantidade_de_camadas_para_finetuning_resnet,
    )

    # Otimizador
    if tipo_de_otimizador.lower() == "adam":
        otimizador = keras.optimizers.Adam(learning_rate=taxa_de_aprendizado)
    else:
        raise ValueError(f"Tipo de otimizador não suportado: {tipo_de_otimizador}")

    modelo_effresnet_vit.compile(
        optimizer=otimizador,
        loss=funcao_de_perda,
        metrics=["accuracy"]
    )

    nome_do_arquivo_txt = "arquitetura_modelo.txt"
        
    with open(nome_do_arquivo_txt, "w") as arquivo:
        
        arquivo.write("--- Resumo da Arquitetura do Modelo ---\n")
        modelo_effresnet_vit.summary(print_fn=lambda x: arquivo.write(x + '\n'))
        
        arquivo.write("---------------------------------------\n")

    print(f"Resumo da arquitetura salvo com sucesso em: {nome_do_arquivo_txt}")
    print("Iniciando treinamento")

    # Callbacks
    callback_de_parada_antecipada = keras.callbacks.EarlyStopping(
        monitor=parametro_de_monitoramento,
        patience=paciencia_para_parada_antecipada,
        restore_best_weights=True
    )

    callback_de_salvamento_do_melhor_modelo = keras.callbacks.ModelCheckpoint(
        filepath=caminho_para_salvar_melhor_modelo,
        monitor=parametro_de_monitoramento,
        save_best_only=True,
        save_weights_only=False
    )

    print("--- INICIANDO TREINAMENTO ---")
    
    historico = modelo_effresnet_vit.fit(
        conjunto_de_treinamento,
        validation_data=conjunto_de_validacao,
        epochs=quantidade_de_epocas,
        callbacks=[callback_de_parada_antecipada, callback_de_salvamento_do_melhor_modelo]
    )

    # print("--- INICIANDO MODO DE TESTE RÁPIDO (Remova isso depois!) ---")

    # historico = modelo_effresnet_vit.fit(
    #     conjunto_de_treinamento,
    #     validation_data=conjunto_de_validacao,
        
    #     # --- CONFIGURAÇÃO DE TESTE ---
    #     epochs=1,              # <--- Garanta que esta é a ÚNICA linha 'epochs'
    #     steps_per_epoch=2,     
    #     validation_steps=2,    
    #     # -----------------------------
        
    #     callbacks=[callback_de_parada_antecipada, callback_de_salvamento_do_melhor_modelo]
    # )

    return modelo_effresnet_vit, historico
