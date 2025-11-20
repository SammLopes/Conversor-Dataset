import tensorflow as tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.regularizers import l2


class CodificadorDePatches(layers.Layer):
    """
    Camada que projeta patches e soma a posição.
    Usa layers.Add() para evitar erros de grafo no Keras 3.
    """

    def __init__(self, quantidade_de_patches, dimensao_de_incorporacao, **argumentos_adicionais):
        super().__init__(**argumentos_adicionais)
        self.quantidade_de_patches = quantidade_de_patches
        self.camada_de_projecao = layers.Dense(units=dimensao_de_incorporacao)
        self.camada_de_incorporacao_posicional = layers.Embedding(
            input_dim=quantidade_de_patches,
            output_dim=dimensao_de_incorporacao
        )
        # FIX: Camada de soma explícita
        self.camada_soma = layers.Add()

    def call(self, sequencia_de_patches, **kwargs):
        indices_de_posicao = tensorflow.range(
            start=0,
            limit=self.quantidade_de_patches,
            delta=1
        )
        incorporacoes_posicionais = self.camada_de_incorporacao_posicional(indices_de_posicao)
        sequencia_projetada = self.camada_de_projecao(sequencia_de_patches)
        
        # FIX: Substituído o '+' por self.camada_soma([])
        return self.camada_soma([sequencia_projetada, incorporacoes_posicionais])

    def get_config(self):
        configuracao_base = super().get_config()
        configuracao_base.update(
            {
                "quantidade_de_patches": self.quantidade_de_patches,
                "dimensao_de_incorporacao": self.camada_de_projecao.units,
            }
        )
        return configuracao_base


class BlocoCodificadorTransformer(layers.Layer):
    """
    Bloco transformer.
    Usa layers.Add() para as conexões residuais.
    """

    def __init__(
        self,
        dimensao_de_incorporacao,
        quantidade_de_cabecas_de_atencao,
        dimensao_da_camada_alimentada_adiante,
        taxa_de_dropout,
        **argumentos_adicionais
    ):
        super().__init__(**argumentos_adicionais)

        self.camada_de_normalizacao_antes_da_atencao = layers.LayerNormalization(epsilon=1e-6)
        
        self.camada_de_atencao_multi_cabecas = layers.MultiHeadAttention(
            num_heads=quantidade_de_cabecas_de_atencao,
            key_dim=dimensao_de_incorporacao,
            dropout=taxa_de_dropout
        )
        
        # FIX: Camadas de soma explícitas para os resíduos
        self.camada_soma_atencao = layers.Add()
        self.camada_soma_mlp = layers.Add()

        self.camada_de_normalizacao_antes_da_mlp = layers.LayerNormalization(epsilon=1e-6)
        
        self.rede_alimentada_adiante = keras.Sequential(
            [
                layers.Dense(units=dimensao_da_camada_alimentada_adiante, activation="gelu"),
                layers.Dropout(rate=taxa_de_dropout),
                layers.Dense(units=dimensao_de_incorporacao),
                layers.Dropout(rate=taxa_de_dropout),
            ]
        )

    def call(self, sequencia_de_entrada, training=False, **kwargs):
        
        # --- Parte 1: Atenção ---
        sequencia_normalizada_para_atencao = self.camada_de_normalizacao_antes_da_atencao(sequencia_de_entrada)
        
        sequencia_apos_atencao = self.camada_de_atencao_multi_cabecas(
            sequencia_normalizada_para_atencao,
            sequencia_normalizada_para_atencao,
            training=training 
        )
        
        # FIX: Substituído o '+' por self.camada_soma_atencao([])
        sequencia_apos_residual_de_atencao = self.camada_soma_atencao(
            [sequencia_de_entrada, sequencia_apos_atencao]
        )

        # --- Parte 2: MLP ---
        sequencia_normalizada_para_mlp = self.camada_de_normalizacao_antes_da_mlp(
            sequencia_apos_residual_de_atencao
        )
        
        sequencia_apos_mlp = self.rede_alimentada_adiante(
            sequencia_normalizada_para_mlp,
            training=training
        )
        
        # FIX: Substituído o '+' por self.camada_soma_mlp([])
        sequencia_apos_residual_de_mlp = self.camada_soma_mlp(
            [sequencia_apos_residual_de_atencao, sequencia_apos_mlp]
        )

        return sequencia_apos_residual_de_mlp
    
    def get_config(self):
        configuracao_base = super().get_config()
        return configuracao_base


def criar_backbones_cnn(
    formato_da_entrada=(224, 224, 3),
    quantidade_de_camadas_para_finetuning_efficientnet=15,
    quantidade_de_camadas_para_finetuning_resnet=20
):
    camada_de_entrada = layers.Input(shape=formato_da_entrada, name="entrada_imagem")

    # Backbone EfficientNet-B0
    modelo_base_efficientnet = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=camada_de_entrada
    )
    modelo_base_efficientnet.trainable = True
    if quantidade_de_camadas_para_finetuning_efficientnet > 0:
        for camada in modelo_base_efficientnet.layers[:-quantidade_de_camadas_para_finetuning_efficientnet]:
            camada.trainable = False
    saida_efficientnet = modelo_base_efficientnet.get_layer("block7a_project_conv").output

    # Backbone ResNet-50
    modelo_base_resnet = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=formato_da_entrada
    )
    modelo_base_resnet.trainable = True
    if quantidade_de_camadas_para_finetuning_resnet > 0:
        for camada in modelo_base_resnet.layers[:-quantidade_de_camadas_para_finetuning_resnet]:
            camada.trainable = False
    saida_resnet = modelo_base_resnet.get_layer("conv5_block3_out").output

    return camada_de_entrada, saida_efficientnet, saida_resnet


def criar_modelo_effresnet_vit(
    formato_da_entrada=(224, 224, 3),
    quantidade_de_classes=3,
    quantidade_de_blocos_transformer=4,
    dimensao_de_incorporacao=128,
    quantidade_de_cabecas_de_atencao=4,
    dimensao_da_camada_alimentada_adiante=256,
    dimensao_da_mlp_final=64,
    taxa_de_dropout=0.3,
    quantidade_de_camadas_para_finetuning_efficientnet=15,
    quantidade_de_camadas_para_finetuning_resnet=20,
):
    # Passo 1
    camada_de_entrada, mapa_de_caracteristicas_efficientnet, mapa_de_caracteristicas_resnet = \
        criar_backbones_cnn(
            formato_da_entrada=formato_da_entrada,
            quantidade_de_camadas_para_finetuning_efficientnet=quantidade_de_camadas_para_finetuning_efficientnet,
            quantidade_de_camadas_para_finetuning_resnet=quantidade_de_camadas_para_finetuning_resnet,
        )

    # Passo 2
    mapa_de_caracteristicas_concatenado = layers.Concatenate(axis=-1, name="fusao_de_caracteristicas")(
        [mapa_de_caracteristicas_efficientnet, mapa_de_caracteristicas_resnet]
    )

    mapa_de_caracteristicas_fundido = layers.Conv2D(
        filters=256,
        kernel_size=1,
        padding="same",
        activation="relu",
        name="convolucao_de_fusao_de_caracteristicas"
    )(mapa_de_caracteristicas_concatenado)

    mapa_de_incorporacao_de_patches = layers.Conv2D(
        filters=dimensao_de_incorporacao,
        kernel_size=1,
        padding="same",
        activation=None,
        name="convolucao_de_incorporacao_de_patches"
    )(mapa_de_caracteristicas_fundido)

    # Passo 3
    altura_estatica, largura_estatica = keras_backend.int_shape(mapa_de_incorporacao_de_patches)[1:3]
    quantidade_de_patches = altura_estatica * largura_estatica

    sequencia_de_patches = layers.Reshape(
        target_shape=(quantidade_de_patches, dimensao_de_incorporacao),
        name="remoldagem_para_sequencia_de_patches"
    )(mapa_de_incorporacao_de_patches)

    # Passo 4
    sequencia_codificada = CodificadorDePatches(
        quantidade_de_patches=quantidade_de_patches,
        dimensao_de_incorporacao=dimensao_de_incorporacao,
        name="camada_codificadora_de_patches"
    )(sequencia_de_patches)

    # Passo 5
    sequencia_transformada = sequencia_codificada
    for indice_do_bloco in range(quantidade_de_blocos_transformer):
        sequencia_transformada = BlocoCodificadorTransformer(
            dimensao_de_incorporacao=dimensao_de_incorporacao,
            quantidade_de_cabecas_de_atencao=quantidade_de_cabecas_de_atencao,
            dimensao_da_camada_alimentada_adiante=dimensao_da_camada_alimentada_adiante,
            taxa_de_dropout=taxa_de_dropout,
            name=f"bloco_transformer_{indice_do_bloco + 1}"
        )(sequencia_transformada)

    # Passo 6
    mapa_pos_transformer = layers.Reshape(
        target_shape=(altura_estatica, largura_estatica, dimensao_de_incorporacao),
        name="remoldagem_para_mapa_2d_pos_transformer"
    )(sequencia_transformada)

    mapa_pos_transformer = layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="convolucao_pos_transformer"
    )(mapa_pos_transformer)
    mapa_pos_transformer = layers.BatchNormalization(name="normalizacao_pos_transformer")(mapa_pos_transformer)
    mapa_pos_transformer = layers.ReLU(name="ativacao_pos_transformer")(mapa_pos_transformer)

    # Passo 7
    vetor_pooling = layers.GlobalAveragePooling2D(name="pooling_global_medio")(mapa_pos_transformer)

    vetor_denso = layers.Dense(
        units=dimensao_da_mlp_final,
        activation="relu",
        kernel_regularizer=l2(0.01),
        name="camada_densa_final"
    )(vetor_pooling)
    vetor_denso = layers.Dropout(rate=taxa_de_dropout, name="dropout_final")(vetor_denso)

    saida_de_classe = layers.Dense(
        units=quantidade_de_classes,
        activation="softmax",
        name="saida_de_classificacao"
    )(vetor_denso)

    modelo = keras.Model(
        inputs=camada_de_entrada,
        outputs=saida_de_classe,
        name="modelo_effresnet_vit"
    )

    return modelo