# Meu Plano - Pipeline de Processamento e Treinamento YOLOv8

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

## 📌 Observações

- Nenhum dataset original é versionado por Git.
- Diretórios `output_*/`, `yolov8-copy/`, `balanced/` e `preprocessed/` devem ser gerados dinamicamente.
- Todos os caminhos podem ser ajustados via `utils/paths.py`

---

## 👨‍💻 Autor

Samuel Paviotti

> Foco em simplicidade, clareza e segurança de execução.
