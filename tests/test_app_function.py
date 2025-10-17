import pytest
import sys
import os
import numpy as np
import cv2

# Adiciona o diretório raiz ao path para que possamos importar os scripts
# de execução (yolo.py, sdac_avc.py) e os módulos da 'app'.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa as funções main dos scripts que queremos testar
import app.yolo as yolo_script
import app.sdac_avc as sdac_script

# --- SEÇÃO DE TESTES PARA O SCRIPT YOLO ---

def test_yolo_script_calls_train(mocker):
    """
    Verifica se `yolo.py train` chama a função `train()` corretamente.
    """
    
    # CORREÇÃO: O patch deve apontar para onde a função é USADA (no script 'yolo.py').
    # O arquivo yolo.py importa 'train', então fazemos o patch em 'yolo.train'.
    mock_train = mocker.patch('app.yolo.train')

    # Simula os argumentos da linha de comando
    mocker.patch.object(sys, 'argv', ['yolo.py', 'train'])
    
    yolo_script.main()
    
    # Verifica se a função simulada foi chamada uma vez
    mock_train.assert_called_once()

def test_yolo_script_calls_validate(mocker):
    """
    Verifica se `yolo.py validate --model path` chama a função `validate_model()` corretamente.
    """
    # CORREÇÃO: O patch deve apontar para onde a função é USADA.
    mock_validate = mocker.patch('app.yolo.validate_model')
    model_path = 'path/to/fake_model.pt'
    
    mocker.patch.object(sys, 'argv', ['yolo.py', 'validate', '--model', model_path])
    
    yolo_script.main()
    
    # Verifica se a função foi chamada com o caminho do modelo correto
    mock_validate.assert_called_once_with(model_path)

def test_yolo_script_validate_fails_without_model(mocker):
    """
    Verifica se o script para com um erro se 'validate' é chamado sem '--model'.
    """
    mocker.patch.object(sys, 'argv', ['yolo.py', 'validate'])
    
    # O teste passa se o script terminar com um SystemExit (comportamento do parser.error)
    with pytest.raises(SystemExit):
        yolo_script.main()

# --- SEÇÃO DE TESTES PARA O SCRIPT SDAVC (KERAS) ---

@pytest.fixture
def mock_keras_functions(mocker):
    """
    Fixture para simular (mock) todas as funções externas usadas pelo script do Keras.
    Isso nos permite testar a lógica do script isoladamente.
    """
    # Simula o carregador de dataset para retornar dados falsos
    dummy_x = np.random.rand(10, 32, 32, 1)
    dummy_y = np.random.randint(0, 2, 10)
    mock_loader = mocker.patch('app.core.preprocessamento.carregar_dataset_preprocessado', return_value=(dummy_x, dummy_y))
    
    # Simula as funções de treino e avaliação
    # CORREÇÃO: O patch deve apontar para onde as funções são USADAS (no script 'sdac_avc.py').
    mock_loader = mocker.patch('sdac_avc.carregar_dataset_preprocessado', return_value=(dummy_x, dummy_y))
    mock_train = mocker.patch('app.core.sdac_avc.treino_sdac_avc.train_sdavc_kfold')
    mock_evaluate = mocker.patch('app.core.sdac_avc.avaliador_sdavc.avaliar_sdavc_model')
    
    # Retorna os mocks para que possam ser usados nos testes
    return mock_loader, mock_train, mock_evaluate, dummy_x, dummy_y

def test_keras_script_calls_train(mocker, tmp_path):
    """
    Verifica se `sdac_avc.py train` chama as funções corretas.
    Usa um diretório temporário para o dataset.
    """
    # Cria uma estrutura de diretório falsa que a função de carregar espera.
    data_dir = tmp_path / "fake_dataset"
    class_a_train_dir = data_dir / "train" / "class_A"
    class_a_valid_dir = data_dir / "valid" / "class_A" # Define o caminho para o diretório de validação
    class_a_train_dir.mkdir(parents=True)
    class_a_valid_dir.mkdir(parents=True)
    
    # CORREÇÃO: Cria um arquivo de imagem falso nos diretórios de treino E validação.
    fake_image = np.zeros((10, 10), dtype=np.uint8)
    cv2.imwrite(str(class_a_train_dir / "fake_img_train.jpg"), fake_image)
    cv2.imwrite(str(class_a_valid_dir / "fake_img_valid.jpg"), fake_image) # Garante que a validação também encontre um arquivo
    
    # Simula a função de treino para que não rode de verdade.
    mock_train_kfold = mocker.patch.object(sdac_script, 'train_sdavc_kfold')
    
    # Simula os argumentos da linha de comando
    mocker.patch.object(sys, 'argv', ['sdac_avc.py', 'train', '--data-dir', str(data_dir)])

    sdac_script.main()

    # Verifica se a função de treino foi chamada
    mock_train_kfold.assert_called_once()
    # Verifica se o primeiro argumento da chamada (X) não está vazio
    assert mock_train_kfold.call_args[0][0].size > 0


def test_keras_script_calls_evaluate(mocker, tmp_path):
    """
    Verifica se `sdac_avc.py evaluate` chama as funções corretas.
    Usa um diretório temporário para o dataset.
    """
    # Cria a mesma estrutura de diretório falsa.
    data_dir = tmp_path / "fake_dataset"
    class_a_train_dir = data_dir / "train" / "class_A"
    class_a_valid_dir = data_dir / "valid" / "class_A"
    class_a_train_dir.mkdir(parents=True)
    class_a_valid_dir.mkdir(parents=True)
    
    fake_image = np.zeros((10, 10), dtype=np.uint8)
    cv2.imwrite(str(class_a_train_dir / "fake_img_train.jpg"), fake_image)
    cv2.imwrite(str(class_a_valid_dir / "fake_img_valid.jpg"), fake_image)
    
    # CORREÇÃO: Cria um diretório de modelo falso e um arquivo de modelo falso dentro dele.
    model_dir = tmp_path / "fake_models"
    model_dir.mkdir()
    (model_dir / "fake_model.keras").touch() # Cria um arquivo .keras vazio
    
    # CORREÇÃO: Usa o caminho completo e explícito para a função no patch.
    mock_evaluate_model = mocker.patch.object(sdac_script, 'avaliar_sdavc_model')
    
    # CORREÇÃO: Passa o diretório do modelo falso para o script.
    mocker.patch.object(sys, 'argv', [
        'sdac_avc.py', 'evaluate', 
        '--data-dir', str(data_dir),
        '--model-dir', str(model_dir) 
    ])

    sdac_script.main()

    # Verifica se a função de avaliação foi chamada
    mock_evaluate_model.assert_called_once()
    # Verifica se o primeiro argumento da chamada (X) não está vazio
    assert mock_evaluate_model.call_args[0][0].size > 0