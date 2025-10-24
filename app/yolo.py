import argparse
import os
from app.core.treino_yolo import train, validate_model

def main():
    """
    Função principal que gerencia os argumentos da linha de comando
    para decidir se o modelo será treinado ou validado.
    """
    # Configura o parser de argumentos
    parser = argparse.ArgumentParser(
        description="Script para treinar e validar o modelo YOLOv8.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Adiciona o argumento 'action' para escolher entre 'train' e 'validate'
    parser.add_argument(
        'action',
        type=str,
        choices=['train', 'validate'],
        help="A ação a ser executada:\n"
             "  train    - Inicia um novo treinamento do zero.\n"
             "  validate - Avalia um modelo já treinado."
    )

    # Adiciona o argumento opcional '--model' para o caminho do modelo na validação
    parser.add_argument(
        '--model',
        type=str,
        help="Caminho para o modelo treinado (obrigatório para a ação 'validate').\n"
             "Exemplo: runs/classify/yolo_avc_v8m_multiclasse/weights/best.pt"
    )

    args = parser.parse_args()

    # Verifica a ação e executa a função correspondente
    if args.action == 'train':
        print("\n🚀 Iniciando o treinamento do modelo...")
        train()
    
    elif args.action == 'validate':
        if not args.model:
            # Se a ação for 'validate', o argumento '--model' é obrigatório
            parser.error("--model é obrigatório para a ação 'validate'.")
        
        print(f"\n🔍 Iniciando a validação do modelo: {args.model}")
        validate_model(args.model)

if __name__ == '__main__':
    # Cria o diretório app/core se ele não existir, para garantir a estrutura
    if not os.path.exists('app/core'):
        os.makedirs('app/core')
        print("Diretório 'app/core' criado.")
        # É necessário criar um __init__.py para que a pasta seja um pacote
        with open('app/__init__.py', 'w') as f:
            pass
        with open('app/core/__init__.py', 'w') as f:
            pass
    
    main()
