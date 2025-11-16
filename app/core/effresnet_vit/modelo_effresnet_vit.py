import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50

# --- 1. Hiperparâmetros Globais e do Artigo (Tabela 3) ---
IMG_SIZE = 224
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_TRANSFORMER_BLOCKS = 4
MLP_DIM = 64

# --- 2. [ATUALIZADO] Parâmetros do seu JSON ---
# (Estes são os valores que você forneceu)
NUM_CLASSES = 3       # Atualize para o seu dataset (ex: Hemorrhagic, Ischemic, Normal)
NEW_LEARNING_RATE = 0.0005
NEW_LOSS_FUNCTION = "categorical_crossentropy"
NEW_BATCH_SIZE = 32   # Usando o 'max' do seu JSON
NEW_EPOCHS = 100
NEW_PATIENCE = 50
NEW_DROPOUT = 0.3

# --- 3. Funções de Construção do Modelo (Blocos ViT) ---

class PatchEncoder(layers.Layer):
    """ Camada customizada para adicionar embeddings de posição aos patches. """
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


def create_transformer_encoder_block(inputs, num_heads, projection_dim, mlp_dim):
    """ Cria um único bloco Transformer Encoder (conforme Figura 4) """
    x1 = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1
    )(x1, x1)
    x2 = layers.Add()([attention_output, inputs])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x3)
    x3 = layers.Dropout(0.1)(x3)
    x3 = layers.Dense(projection_dim)(x3)
    outputs = layers.Add()([x3, x2])
    return outputs

# --- 4. Funções de Construção do Modelo (Backbones e Modelo Principal) ---

def create_cnn_backbones(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """ Cria os backbones EfficientNetB0 e ResNet50 para extração de features. """
    input_layer = layers.Input(shape=input_shape, name="input_image")

    # Ramificação 1: EfficientNet-B0
    base_effnet = EfficientNetB0(
        include_top=False, weights='imagenet', input_tensor=input_layer
    )
    base_effnet.trainable = False 
    effnet_output = base_effnet.get_layer('block7a_project_conv').output
    
    # Ramificação 2: ResNet-50
    base_resnet = ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base_resnet.trainable = False
    resnet_output_tensor = base_resnet(input_layer) 
    # Precisamos obter a saída pelo nome da camada no *objeto* base_resnet
    resnet_output = base_resnet.get_layer('conv5_block3_out').output

    return input_layer, effnet_output, resnet_output


def build_effresnet_vit(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    dropout_rate=0.3  # <-- [ATUALIZADO]
):
    """ Constrói e retorna o modelo EFFResNet-ViT completo (Figura 4). """
    
    # PASSO 1: Backbones CNN
    input_layer, effnet_features, resnet_features = create_cnn_backbones(input_shape)
    
    # PASSO 2: Concatenação
    fused_features = layers.Concatenate(axis=-1, name="concatenation")(
        [effnet_features, resnet_features]
    )

    # PASSO 3: "Get Feature Map" (Conv2D pós-concatenação)
    feature_map = layers.Conv2D(
        filters=256, kernel_size=(1, 1), activation='relu', padding='same', name="feature_fusion_conv"
    )(fused_features)

    # PASSO 4: Geração de Patches (do mapa de features 7x7)
    patch_embeddings = layers.Conv2D(
        filters=hidden_dim, kernel_size=(1, 1), strides=(1, 1),
        padding='valid', name="patch_embedding_conv"
    )(feature_map)
    
    h, w = patch_embeddings.shape[1], patch_embeddings.shape[2]
    num_patches = h * w
    
    patch_embeddings_flat = layers.Reshape(
        (num_patches, hidden_dim), name="flatten_patches"
    )(patch_embeddings)

    # PASSO 5: Módulo Transformer
    encoded_patches = PatchEncoder(num_patches, hidden_dim)(patch_embeddings_flat)
    transformer_output = encoded_patches
    for _ in range(num_transformer_blocks):
        transformer_output = create_transformer_encoder_block(
            transformer_output, num_heads, hidden_dim, mlp_dim
        )

    # PASSO 6: Bloco Pós-Transformer
    features_2d = layers.Reshape((h, w, hidden_dim), name="reshape_post_transformer")(transformer_output)
    post_transformer_features = layers.Conv2D(
        filters=64, kernel_size=(3, 3), padding='same', name="post_transformer_conv"
    )(features_2d)
    post_transformer_features = layers.BatchNormalization(name="post_transformer_bn")(post_transformer_features)
    post_transformer_features = layers.Activation('relu', name="post_transformer_relu")(post_transformer_features)
    pooled_features = layers.GlobalAveragePooling2D(name="gap_layer")(post_transformer_features)

    # PASSO 7: Cabeça de Classificação (MLP Head)
    mlp_output = layers.Dense(
        mlp_dim, activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(0.01), 
        name="mlp_dense"
    )(pooled_features)
    
    # [ATUALIZADO] Usando o dropout_rate do seu JSON
    mlp_output = layers.Dropout(dropout_rate, name="mlp_dropout")(mlp_output)

    classifier_output = layers.Dense(
        num_classes, activation='softmax', name="classifier_output"
    )(mlp_output)

    # PASSO 8: Criar modelo final
    model = Model(
        inputs=input_layer, 
        outputs=classifier_output, 
        name="EFFResNet_ViT_Custom"
    )
    
    return model

# --- 5. Script Principal: Montagem e Treinamento ---

# Diretórios de dados (ajuste os nomes conforme sua estrutura)
TRAIN_DIR = "dataset_custom_preprocessed/train"
VALID_DIR = "dataset_custom_preprocessed/validation" 

# Construir o modelo com o dropout customizado
print("Construindo o modelo...")
model = build_effresnet_vit(
    num_classes=NUM_CLASSES,
    dropout_rate=NEW_DROPOUT
)

# Compilar o modelo com o optimizer e loss customizados
print("Compilando o modelo...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=NEW_LEARNING_RATE), # [ATUALIZADO]
    loss=NEW_LOSS_FUNCTION,                                 # [ATUALIZADO]
    metrics=["accuracy"]
)

# Visualizar a arquitetura
model.summary()

# Preparar os dados com o batch_size customizado
print("Carregando datasets...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=NEW_BATCH_SIZE, # [ATUALIZADO]
    label_mode='categorical'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=NEW_BATCH_SIZE, # [ATUALIZADO]
    label_mode='categorical'
)

# Definir Callbacks com a paciência customizada
print("Configurando callbacks...")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=NEW_PATIENCE, # [ATUALIZADO]
    restore_best_weights=True
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'effresnet_vit_best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

# Treinar o modelo com as épocas customizadas
print("Iniciando o treinamento...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=NEW_EPOCHS, # [ATUALIZADO]
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

print("Treinamento concluído. O melhor modelo foi salvo em 'effresnet_vit_best_model.h5'")