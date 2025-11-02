# Pipeline de Processamento e Treinamento YOLOv8, TransUnet, SDAVC.

Este projeto organiza o processo completo de preparação, fusão, balanceamento, conversão e pré-processamento de datasets médicos para uso com o modelo YOLOv8 e outras arquiteturas de redes neurais, dentro da arquitetura do sistema Meu Plano.

---

## 📁 Estrutura de Pastas

```
app/
├── core/               # Módulos funcionais (balanceamento, fusão, conversão, treino, pré-processamento)
├── core/
    ├── sdac_avc        # modulos do modeloes SDAC_AVC(avaliador_sdavc, modelo_sdacv, treino_sdac_avc)
    ├── transunet       # modulos de interação com o TransUNet (augmentation, dataset, finetune, pretrain, pseudo_masks, utils)
├── utils/              # Constantes e paths reutilizáveis
├── main.py             # Pipeline principal com chamadas diretas
├── yolo.py             # Scritp que executa imediatamente o treinamento e validação
data/
├── balanced 
    ├── yolov8-balanced/     # Dataset final balanceado
    ├── yolov8-extra/        # Imagens excedentes após balanceamento 
├── final
    ├──dataset_custom/       # Dataset final convertido para o formato customizado
├── pre-processed 
    ├── dataset_custom_preprocessed/  # Dataset custom com imagens pré-processadas
├── processed 
    ├── output_dataset_*/    # Saídas intermediárias convertidas para YOLO
    ├── yolov8-copy/         # Pasta de fusão principal (dataset acumulado)    
├──raw
    ├── datasets_unformat/  # Dados brutos não formatados (por classe)
    ├── dataset_yolo/       # Dados já no formato YOLO + YAML
├── results                 # Armazena os resultados dos treinamentos
tests                   # Diretório de tests 
    ├── test_core_functions # Testes unitários do modulo core 
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
- `core/yolo.py` →  Orquestrador da execução do treinamento do yolo e do validador do modelo.
- `core/conversao.py` → Gera labels em formato YOLOv8 a partir de pastas separadas por classe
- `core/fusao.py` → Junta imagens de diferentes fontes sob um padrão de classes
- `core/balanceamento.py` → Garante proporções fixas entre as classes
- `core/treino_yolo.py` → Treina e valida modelos YOLOv8 (Ultralytics)
- `core/preprocessamento.py` → Aplica janelamento, equalização, filtragem e normalização para uso em classificadores
- `core/sdacv ` → Modulo que engloba a criação treinamento e validação do modelo sdacv_avc.
- `code/transunet` → Modulo que trata do pre-treinamento, fine-tuning, criação de pseudo-masks, carregamento do datasets e augmentation.
---

## ⚙️ Dependências principais

- Python 3.9+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Torch / CUDA (para treinamento)
- OpenCV, PyYAML, scikit-image, scikit-learn
- Requirements do projeto [TransUnet](https://github.com/Beckschen/TransUNet)

---
---

## 📌 Observações


- Nenhum dataset original é versionado por Git.
- Diretórios `output_*/`, `yolov8-copy/`, `balanced/`, `preprocessed/` e `modelos/` são gerados dinamicamente.
- Todos os caminhos podem ser ajustados via `utils/paths.py`


---


## 🧠 Sobre o SDAVC Adaptado>

Ele é treinado com validação cruzada estratificada (5 folds), utilizando `Adam`, `categorical_crossentropy`, e avaliado com métricas clínicas como sensibilidade, precisão, F1-score e AUC-ROC.


---

## Sobre a adaptação do TransUNet no treino do modelo

O **TransUNet** é um modelo voltado para tarefas de **segmentação de imagens médicas**, e portanto requer que o dataset contenha **máscaras de segmentação** (labels pixel a pixel).  

Como o dataset utilizado neste projeto não possui anotações manuais (máscaras), adotamos a seguinte estratégia:  

1. **Pré-treinamento:** primeiro treinamos o TransUNet em datasets públicos já segmentados (ex.: **BTCV** ou **ACDC**) para que o modelo aprenda padrões gerais de segmentação em imagens médicas.  
2. **Geração de pseudo-máscaras:** com o modelo pré-treinado, aplicamos inferência sobre as imagens do nosso dataset, gerando **pseudo-rótulos** (máscaras automáticas).  
3. **Fine-tuning:** em seguida, refinamos o modelo utilizando as imagens do nosso dataset junto das pseudo-máscaras, aplicando **data augmentation** para aumentar a robustez e generalização.  

Esse processo possibilita adaptar o TransUNet para domínios específicos (como tomografia computadorizada do cérebro) mesmo sem a necessidade inicial de anotações manuais feitas por especialistas.  


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

## Treinamento e Validação do Modelo Yolo 
- Abra seu terminal na pasta raiz do projeto e utilize os seguintes comandos:
> 1 Para Treinar o Modelo
- Execute o comando abaixo. Ele chamará a função ```train()``` do arquivo treino.py.
```bash
python -m app.yolo train

```
> 2 Para Validar o Modelo
- Após o treinamento, você pode validar o melhor modelo salvo. Para isso, use o comando ```validate``` e forneça o caminho para o arquivo ```.pt``` usando o argumento ```--model```.
```bash
python -m app.yolo validate --model "./data/results/yolo_avc_v8m_multiclasse/weights/best.pt" --dataset "./data/datasets_training/dataset_custom"
```
- Caso seja um modelo de detecção.
```bash     
python -m app.yolo validate --model "./data/results/yolo_avc_v8m_detect3/weights/best.pt" --dataset "./data/datasets_training/yolov8-balanced"
```

## Treinamento e Validação do Modelo SDAC_AVC
>1 Estrutura do datasets
```
├── dataset_custom_preprocessed
    ├── train
        ├── Hemorrhagic Stroke
        ├── Ischemic Stroke
        ├── Normal
    ├── valid
        ├── Hemorrhagic Stroke
        ├── Ischemic Stroke
        ├── Normal
```

>2 Uso
- Use o comando ```train``` e forneça o caminho do diretório para o dataset se treino, nesse trabalho o nome do diretório é ```dataset_custom_preprocessed```, pode mudar o lugar a medida que seja necessário.
```bash
python -m app.sdac_avc.py train --data_dir /caminho/para/seu_dataset
```
Os modelos (```.keras```) serão salvos no diretório ```modelos/sdavc/```.

>3 Avaliar Modelo
- Use o comando ```evaluate``` e aponte para o diretório do dataset que deseja usar para a avaliação (pode ser o mesmo do treino ou um de teste separado com a mesma estrutura).

```bash
python -m app.sdac_avc.py evaluate --data-dir /caminho/para/seu_dataset_de_teste
```
>4 Predição do Modelo
- Este script foi projetado para carregar os modelos .keras que você treinou e usá-los para classificar uma única imagem.
Ele oferece duas estratégias de predição, que refletem os resultados da sua avaliação (o arquivo resultados_sdavc_por_fold.csv vs. resultados_sdavc_ensemble.csv):

  - Modo de Modelo Único: Você escolhe o seu "melhor" fold (ex: sdavc_fold3.keras) e o utiliza para a previsão. É rápido, mas depende do desempenho desse único modelo.

  - Modo Ensemble (Recomendado): Você fornece a pasta com todos os 5 modelos. O script passa a imagem por cada um, tira a média das probabilidades e dá um resultado final. Este método é mais lento (5x), porém muito mais robusto e confiável, pois representa o seu resultado final (ex: AUC de 0.932).

#### Usando o predictor
- O script a ser executado é o ```app/core/sdavc/predicao_sdac_avc.py```.

    - Argumento ```--image```: O caminho para a imagem que você quer classificar.

    - Você deve escolher OU ```--model``` OU ```--model-dir```.

### Modo 1: Usando Ensemble

Os resultados (gráficos, relatórios) serão salvos no diretório ```avaliacoes/```.
- Este é o método mais robusto e reflete seu resultado de AUC de 93.2%. Ele usa os 5 modelos para "votar" no resultado final.

- Use o argumento ```--model-dir``` e passe o caminho para a pasta que contém todos os 5 folds.

```bash
python -m app.sdac_avc.py \
    --image /caminho/para/minha_imagem.png \
    --model-dir ./modelos/sdavc/
```

#### Exemplo de saída, Ensemble
```bash
--- Modo de Predição: Ensemble (K-Fold) ---
Encontrados 5 modelos para o ensemble.
Carregando e prevendo com fold 1/5...
Carregando e prevendo com fold 2/5...
Carregando e prevendo com fold 3/5...
Carregando e prevendo com fold 4/5...
Carregando e prevendo com fold 5/5...

--- Resultado Final ---
Classe Prevista: Ischemic Stroke
Confiança: 82.45%

Probabilidades por Classe:
  Hemorrhagic Stroke: 10.15%
  Ischemic Stroke: 82.45%
  Normal: 7.40%
```

### Modo 2: Usando um Modelo Único (Ex: o "Melhor" Fold)

- Se você olhou seu arquivo resultados_sdavc_por_fold.csv e viu que o sdavc_fold3.keras foi o melhor individualmente, você pode usá-lo sozinho.

- Use o argumento ```--model``` e passe o caminho para o arquivo .keras específico.
```bash
python -m app.sdac_avc.py \
    --image /caminho/para/minha_imagem.png \
    --model ./modelos/sdavc/sdavc_fold3.keras
```

#### Exemplo de saída(Modelo Único)
```bash
--- Modo de Predição: Modelo Único ---
Carregando modelo: ./modelos/sdavc/sdavc_fold3.keras
Fazendo predição...

--- Resultado Final ---
Classe Prevista: Ischemic Stroke
Confiança: 80.11%

Probabilidades por Classe:
  Hemorrhagic Stroke: 12.05%
  Ischemic Stroke: 80.11%
  Normal: 7.84%

```

## Configuração do Ambiente e Instalação

-  Instale o python na maquina. 

> 1 Crie o Ambiente Virtual (```venv```)
```bash 
python3 -m venv .venv
```
> 2 Ative o Ambiente Virtual
```bash 
source .venv/bin/activate
```
> 3 Instale as dependencias
```bash
pip install -r requirements.txt
```

## 👨‍💻 Autor

Samuel Paviotti

> Foco em simplicidade, clareza e segurança de execução.
