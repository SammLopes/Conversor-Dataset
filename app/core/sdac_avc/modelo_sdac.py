from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from keras.optimizers import Adam

def build_sdavc_model(input_shape=(224, 224, 1), num_classes=3, dropout_rate=0.3, lr=0.0005):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for filters in [32, 64, 128, 256]:
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    # --- CORREÇÃO AQUI ---
    # Substitua Flatten() pelo GlobalAveragePooling2D()
    # model.add(Flatten()) # <-- Causa do erro de memória
    model.add(GlobalAveragePooling2D()) # <-- Solução leve
    # ---------------------

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
