import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

def avaliar_sdavc_model(X, y, model_dir="modelos/sdavc"):
    y_cat = to_categorical(y)
    predicoes = []
    reais = []
    probs = []

    for fold_file in sorted(os.listdir(model_dir)):
        if not fold_file.endswith(".h5"):
            continue

        model = load_model(os.path.join(model_dir, fold_file))
        y_pred = model.predict(X)

        predicoes.extend(np.argmax(y_pred, axis=1))
        reais.extend(np.argmax(y_cat, axis=1))
        probs.extend(y_pred)

    print("\n📊 Relatório de Classificação (média de todos os folds)")
    print(classification_report(reais, predicoes))

    cm = confusion_matrix(reais, predicoes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão SDAVC")
    plt.savefig("avaliacoes/sdavc_matriz_confusao.png")
    plt.close()

    try:
        auc = roc_auc_score(y_cat, np.array(probs), multi_class='ovr')
        print(f"🏅 AUC-ROC médio: {auc:.4f}")
    except Exception as e:
        print("⚠️ AUC-ROC não pôde ser calculado:", e)