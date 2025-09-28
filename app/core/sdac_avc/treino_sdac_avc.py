import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from app.core.sdac_avc.modelo_sdac import build_sdavc_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from tqdm.keras import TqdmCallback

def train_sdavc_kfold(X, y, save_dir="modelos/sdavc", n_splits=5, epochs=100, batch_size=32):
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

        model.save(os.path.join(save_dir, f"sdavc_fold{fold}.h5"))
        fold += 1
    
    print(" Fim do treino em K-fold ")
