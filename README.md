# Pipeline de Processamento e Treinamento YOLOv8, TransUnet, SDAVC.

Este projeto organiza o processo completo de preparação, fusão, balanceamento, conversão e pré-processamento de datasets médicos para uso com o modelo YOLOv8 e outras arquiteturas de redes neurais, dentro da arquitetura do sistema Meu Plano.

---

## 📁 Estrutura de Pastas

```
app/
├── core/               # Módulos funcionais (balanceamento, fusão, conversão, treino, pré-processamento)
├── utils/              # Constantes e paths reutilizáveis
├── main.py             # Pipeline principal com chamadas diretas

datasets/
├── datasets_unformat/  # Dados brutos não formatados (por classe)
├── dataset_yolo/       # Dados já no formato YOLO + YAML

output_dataset_*/        # Saídas intermediárias convertidas para YOLO
yolov8-copy/             # Pasta de fusão principal (dataset acumulado)
yolov8-balanced/         # Dataset final balanceado
yolov8-extra/            # Imagens excedentes após balanceamento
dataset_custom/          # Dataset final convertido para o formato customizado
dataset_custom_preprocessed/  # Dataset custom com imagens pré-processadas
```

---

## ▶️ Como Executar

### 1. Preparar ambiente (recomendação)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Executar pipeline completo

```bash
python app/main.py
```

> As etapas serão executadas na ordem:
> 1. Conversão de datasets (multiclasse para YOLO)
> 2. Fusão de datasets
> 3. Balanceamento por classe
> 4. Conversão final para formato customizado
> 5. Pré-processamento das imagens (para uso em CNNs ou análise supervisionada)

---

## 📦 Módulos

- `core/conversao.py` → Gera labels em formato YOLOv8 a partir de pastas separadas por classe
- `core/fusao.py` → Junta imagens de diferentes fontes sob um padrão de classes
- `core/balanceamento.py` → Garante proporções fixas entre as classes
- `core/treino.py` → Treina e valida modelos YOLOv8 (Ultralytics)
- `core/preprocessamento.py` → Aplica janelamento, equalização, filtragem e normalização para uso em classificadores

---

## ⚙️ Dependências principais

- Python 3.9+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Torch / CUDA (para treinamento)
- OpenCV, PyYAML, scikit-image, scikit-learn

---
---

## 📌 Observações


- Nenhum dataset original é versionado por Git.
- Diretórios `output_*/`, `yolov8-copy/`, `balanced/`, `preprocessed/` e `modelos/` são gerados dinamicamente.
- Todos os caminhos podem ser ajustados via `utils/paths.py`


---


## 🧠 Sobre o SDAVC Adaptado


O modelo SDAVC-AVC é uma adaptação expandida da proposta de Reis (2021), originalmente binária (AVCi vs AVCh), agora estendida para multiclasse (incluindo Normal). A arquitetura implementada mantém os blocos convolucionais originais com filtros `[32, 64, 128, 256]`, ativação ReLU, pooling, batch normalization e regularização por dropout.


Ele é treinado com validação cruzada estratificada (5 folds), utilizando `Adam`, `categorical_crossentropy`, e avaliado com métricas clínicas como sensibilidade, precisão, F1-score e AUC-ROC.


---


## 🧪 Comparativo de Parâmetros de Treinamento

| Parâmetro / Recurso                    | SDAC-AVC Adaptado ✅ | TransUNet ✅ | YOLOv8 ⚠️ |
|----------------------------------------|----------------------|--------------|-----------|
| Modificação da arquitetura             | ✅ Total              | ✅ Total     | ❌ Limitada |
| Otimizador Adam com LR custom          | ✅ Sim                | ✅ Sim       | ✅ Sim (`lr0`) |
| Scheduler (ReduceLROnPlateau)          | ✅ Sim                | ✅ Sim       | ⚠️ Limitado (`lrf`) |
| Early stopping                         | ✅ Sim                | ✅ Sim       | ✅ Sim (`patience`) |
| Batch size ajustável (16–32)           | ✅ Sim                | ✅ Sim       | ✅ Sim |
| Função de perda customizada            | ✅ Qualquer           | ✅ Qualquer  | ❌ Fixa para detecção |
| Dropout customizado                    | ✅ Sim                | ✅ Sim       | ❌ Não aplicável diretamente |
| Data augmentation                      | ✅ Sim (`ImageDataGenerator`) | ✅ Sim  | ✅ Sim (embutido) |
| K-Fold Cross-validation                | ✅ Sim (`StratifiedKFold`) | ✅ Sim  | ❌ Manual |
| Callbacks personalizados (Keras)       | ✅ Total              | ✅ Total     | ⚠️ Limitado |
| Transfer learning custom               | ✅ Sim                | ✅ Sim       | ⚠️ Apenas via backbones |

---

## 📊 Comparativo de Métricas Suportadas

> ❗ Nota: YOLOv8 gera a **F1-Confidence Curve**, que mostra o F1 em diferentes thresholds de confiança, **mas não é o F1-score tradicional** (média harmônica entre precisão e recall fixos). Para cálculo exato do F1-score, é necessário processamento adicional com `sklearn`.

| Métrica               | SDAC-AVC Adaptado ✅ | TransUNet ✅ | YOLOv8 ⚠️ |
|------------------------|----------------------|--------------|-----------|
| Acurácia               | ✅ Sim                | ✅ Sim       | ✅ Sim |
| Precisão               | ✅ Sim                | ✅ Sim       | ✅ Sim |
| Sensibilidade (Recall) | ✅ Sim                | ✅ Sim       | ✅ Sim |
| Especificidade         | ✅ Sim                | ✅ Sim       | ⚠️ Manual |
| F1-score tradicional   | ✅ Sim                | ✅ Sim       | ⚠️ Requer extração manual |
| AUC-ROC                | ✅ Sim                | ✅ Sim       | ⚠️ Requer workaround |
| Tempo de predição      | ✅ Sim (`time`)       | ✅ Sim       | ✅ Nativo |
| Throughput (img/s)     | ✅ Sim                | ✅ Sim       | ✅ Sim |
| Uso de memória         | ✅ Sim (`psutil`)     | ✅ Sim       | ⚠️ Estimado |

---

## 👨‍💻 Autor

Samuel Paviotti

> Foco em simplicidade, clareza e segurança de execução.
