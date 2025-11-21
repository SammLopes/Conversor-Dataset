import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from app.core.sdac_avc.modelo_sdac import build_sdavc_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from tqdm.keras import TqdmCallback

def train_sdavc_kfold(X, y, save_dir="modelos/sdavc", n_splits=5, epochs=100, batch_size=32, is_include=True):
    os.makedirs(save_dir, exist_ok=True)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_cat = to_categorical(y)
    fold = 1
    for train_idx, val_idx in kfold.split(X, y):
        print(f"\n🚀 Treinando fold {fold}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_cat[train_idx], y_cat[val_idx]

        model = build_sdavc_model(input_shape=X.shape[1:], num_classes=y_cat.shape[1])

        callbacks = [
            EarlyStopping(patience=50, monitor='val_loss', restore_best_weights=True),
            ReduceLROnPlateau(patience=25, factor=0.3, verbose=1),
            TqdmCallback(verbose=1)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        model.save(os.path.join(save_dir, f"sdavc_fold{fold}.keras"), include_optimizer=is_include)
        fold += 1
    
    print(" Fim do treino em K-fold ")

def train_sdavc_simple(
    matriz_de_imagens,
    vetor_de_rotulos,
    diretorio_de_saida="modelos/sdavc",
    quantidade_de_epocas=100,
    tamanho_do_lote=32,
    fracao_de_validacao=0.2,
    incluir_otimizador=True
):
    """
    Treinamento "simples" (sem K-Fold), usando uma divisão treino/validação.

    - matriz_de_imagens: numpy array com shape (quantidade_de_amostras, altura, largura, canais)
    - vetor_de_rotulos: numpy array com os rótulos em formato inteiro (0, 1, 2, ...)
    - diretorio_de_saida: onde salvar o modelo final (.keras)
    - quantidade_de_epocas: número máximo de épocas
    - tamanho_do_lote: tamanho do batch
    - fracao_de_validacao: porcentagem do conjunto usada para validação (ex: 0.2 = 20%)
    - incluir_otimizador: se o otimizador deve ser salvo dentro do arquivo .keras
    """
    os.makedirs(diretorio_de_saida, exist_ok=True)

    # Converte rótulos para one-hot
    rotulos_categoricos = to_categorical(vetor_de_rotulos)

    quantidade_total_de_amostras = matriz_de_imagens.shape[0]
    quantidade_de_amostras_para_treino = int(quantidade_total_de_amostras * (1.0 - fracao_de_validacao))

    imagens_de_treino = matriz_de_imagens[:quantidade_de_amostras_para_treino]
    imagens_de_validacao = matriz_de_imagens[quantidade_de_amostras_para_treino:]

    rotulos_de_treino = rotulos_categoricos[:quantidade_de_amostras_para_treino]
    rotulos_de_validacao = rotulos_categoricos[quantidade_de_amostras_para_treino:]

    print(f"\n🚀 Treinando modelo SDAVC em modo simples")
    print(f"   Amostras de treino: {imagens_de_treino.shape[0]}")
    print(f"   Amostras de validacao: {imagens_de_validacao.shape[0]}")

    modelo_sdavc = build_sdavc_model(
        input_shape=matriz_de_imagens.shape[1:],
        num_classes=rotulos_categoricos.shape[1]
    )

    lista_de_callbacks = [
        EarlyStopping(patience=50, monitor="val_loss", restore_best_weights=True),
        ReduceLROnPlateau(patience=25, factor=0.3, verbose=1),
        TqdmCallback(verbose=1),
    ]

    historico_de_treinamento = modelo_sdavc.fit(
        imagens_de_treino,
        rotulos_de_treino,
        validation_data=(imagens_de_validacao, rotulos_de_validacao),
        epochs=quantidade_de_epocas,
        batch_size=tamanho_do_lote,
        callbacks=lista_de_callbacks,
        verbose=1,
    )

    caminho_modelo = os.path.join(diretorio_de_saida, "sdavc_unico.keras")
    modelo_sdavc.save(caminho_modelo, include_optimizer=incluir_otimizador)

    print(f"\n✅ Treinamento simples concluído. Modelo salvo em: {caminho_modelo}")

    return modelo_sdavc, historico_de_treinamento
