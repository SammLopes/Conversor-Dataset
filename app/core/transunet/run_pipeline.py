"""
run_pipeline.py
----------------
Orquestrador principal para executar o pipeline de 
semi-supervised learning com TransUNet (PyTorch).

Este script CHAMA os scripts `train.py` e `test.py` do 
repositório original do TransUNet via terminal.
"""

import argparse
import os
import sys
import subprocess
from pipeline_utils import create_augmented_dataset

# --- CONFIGURAÇÃO OBRIGATÓRIA ---
# Ajuste este caminho para apontar para a pasta do TransUNet que você baixou
# (Ex: se esta pasta está em 'app/core/transunet' e o TransUNet está em 'TransUNet')
TRANSUNET_REPO_PATH = "../../TransUNet/" 
# -----------------------------------

# Converte o caminho para absoluto
TRANSUNET_REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), TRANSUNET_REPO_PATH))
TRAIN_SCRIPT_PATH = os.path.join(TRANSUNET_REPO_PATH, "train.py")
TEST_SCRIPT_PATH = os.path.join(TRANSUNET_REPO_PATH, "test.py") # ou predict.py

def run_command(command, cwd=None):
    """Executa um comando no terminal e imprime a saída."""
    print(f"\n[CMD] Executando comando: {' '.join(command)}")
    print(f"[CMD] No diretório: {cwd or os.getcwd()}")
    try:
        # Executa o comando e imprime a saída em tempo real
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='')
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ ERRO AO EXECUTAR O COMANDO: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ ERRO: Script não encontrado. '{command[1]}' existe?")
        print(f"   Verifique se TRANSUNET_REPO_PATH está correto em run_pipeline.py")
        print(f"   Caminho configurado: {TRANSUNET_REPO_PATH}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pipeline TransUNet (PyTorch) Semi-Supervisionado")
    
    # --- Argumentos do Pipeline ---
    parser.add_argument('--skip_pretrain', action='store_true', help="Pular Step 1 (usar modelo pré-treinado existente)")
    parser.add_argument('--skip_pseudo_masks', action='store_true', help="Pular Step 2 (usar pseudo-máscaras existentes)")
    parser.add_argument('--skip_augmentation', action='store_true', help="Pular Step 3 (usar dataset aumentado existente)")
    
    # --- Argumentos de Caminhos (Paths) ---
    parser.add_argument('--pretrain_data_dir', type=str, required=True, 
                        help="[Step 1] Caminho para o dataset de pré-treino (Ex: .../BTCV/train_npz/)")
    parser.add_argument('--model_save_dir', type=str, default="models/transunet_pipeline", 
                        help="Diretório raiz para salvar todos os modelos gerados")
    parser.add_argument('--user_data_images', type=str, required=True, 
                        help="[Step 2] Diretório das SUAS imagens (sem rótulo) (Ex: .../meu_dataset/images/)")
    parser.add_argument('--pseudo_mask_dir', type=str, default="datasets/pseudo_masks", 
                        help="[Step 2] Onde salvar/carregar as pseudo-máscaras geradas")
    parser.add_argument('--augmented_data_dir', type=str, default="datasets/finetune_augmented",
                        help="[Step 3] Onde salvar o novo dataset aumentado (imagens e máscaras)")
    
    # --- Argumentos de Treinamento ---
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=3, help="Número de classes (ex: 3 para AVC-H/AVC-I/Fundo)")

    args = parser.parse_args()
    
    # --- Define os caminhos dos artefatos ---
    pretrain_model_dir = os.path.join(args.model_save_dir, "pretrained")
    # O script train.py do TransUNet salva dentro de uma subpasta, ex: 'epoch_100.pth'
    pretrain_model_path = os.path.join(pretrain_model_dir, f"epoch_{args.pretrain_epochs - 1}.pth") # Ajuste se o nome for diferente

    finetune_model_dir = os.path.join(args.model_save_dir, "finetuned")
    
    os.makedirs(pretrain_model_dir, exist_ok=True)
    os.makedirs(finetune_model_dir, exist_ok=True)
    os.makedirs(args.pseudo_mask_dir, exist_ok=True)
    os.makedirs(args.augmented_data_dir, exist_ok=True)
    
    print("="*50)
    print("🚀 INICIANDO PIPELINE TRANSUNET (PyTorch)")
    print(f"Usando Repositório TransUNet em: {TRANSUNET_REPO_PATH}")
    print("="*50)

    # ========================================
    # STEP 1: PRÉ-TREINAMENTO (PyTorch)
    # ========================================
    if not args.skip_pretrain:
        print("\n--- STEP 1: Iniciando Pré-treinamento (PyTorch) ---")
        cmd_pretrain = [
            "python", TRAIN_SCRIPT_PATH,
            "--dataset", "Synapse", # Nome do dataset como esperado pelo repo
            "--root_path", args.pretrain_data_dir,
            "--output_dir", pretrain_model_dir,
            "--max_epochs", str(args.pretrain_epochs),
            "--batch_size", str(args.batch_size),
            "--base_lr", str(args.base_lr),
            "--num_classes", str(args.num_classes)
        ]
        run_command(cmd_pretrain, cwd=TRANSUNET_REPO_PATH)
    else:
        print("\n--- STEP 1: Pré-treinamento pulado (skip_pretrain=True) ---")
        
    if not os.path.exists(pretrain_model_path):
        print(f"❌ Erro: Modelo pré-treinado não encontrado em '{pretrain_model_path}'.")
        print("   Verifique o nome do arquivo .pth salvo pelo script de treino (ex: epoch_99.pth).")
        return

    # ========================================
    # STEP 2: GERAÇÃO DE PSEUDO-MÁSCARAS (PyTorch)
    # ========================================
    if not args.skip_pseudo_masks:
        print("\n--- STEP 2: Iniciando Geração de Pseudo-Máscaras (PyTorch) ---")
        cmd_pseudo = [
            "python", TEST_SCRIPT_PATH,
            "--dataset", "Synapse", # Use o mesmo nome de dataset
            "--volume_path", args.user_data_images, # Caminho para suas imagens
            "--output_dir", args.pseudo_mask_dir, # Onde salvar
            "--model_path", pretrain_model_path, # O modelo que treinamos
            "--is_savenii", # Salva as máscaras (o script test.py deve suportar isso)
            "--num_classes", str(args.num_classes)
        ]
        # NOTA: O script 'test.py' do repo pode precisar de adaptação 
        # para carregar imagens .png/.jpg em vez de .npz (Synapse)
        print("⚠️ Atenção: O script 'test.py' do TransUNet deve ser adaptado")
        print("   para ler imagens (.jpg/.png) do seu '--user_data_images'")
        print("   e salvar em formato .png (não .nii ou .npz).")
        run_command(cmd_pseudo, cwd=TRANSUNET_REPO_PATH)
    else:
        print("\n--- STEP 2: Geração de Pseudo-Máscaras pulada (skip_pseudo_masks=True) ---")

    # ========================================
    # STEP 3: DATA AUGMENTATION
    # ========================================
    if not args.skip_augmentation:
        print("\n--- STEP 3: Iniciando Data Augmentation ---")
        create_augmented_dataset(
            img_dir=args.user_data_images,
            mask_dir=args.pseudo_mask_dir,
            output_dir=args.augmented_data_dir
        )
    else:
        print("\n--- STEP 3: Data Augmentation pulada (skip_augmentation=True) ---")

    # ========================================
    # STEP 4: FINE-TUNING (PyTorch)
    # ========================================
    print("\n--- STEP 4: Iniciando Fine-Tuning (PyTorch) ---")
    cmd_finetune = [
        "python", TRAIN_SCRIPT_PATH,
        "--dataset", "Synapse", # O loader do repo (precisa ser adaptado)
        "--root_path", args.augmented_data_dir, # Aponta para os dados aumentados
        "--output_dir", finetune_model_dir,
        "--max_epochs", str(args.finetune_epochs),
        "--batch_size", str(args.batch_size),
        "--base_lr", str(args.base_lr / 10), # LR 10x menor para fine-tuning
        "--num_classes", str(args.num_classes),
        "--pretrained_model", pretrain_model_path # Carrega o modelo pré-treinado
    ]
    print("⚠️ Atenção: O 'dataset.py' do TransUNet deve ser adaptado")
    print("   para ler o novo dataset aumentado em '--root_path'.")
    run_command(cmd_finetune, cwd=TRANSUNET_REPO_PATH)

    print("="*50)
    print(f"✅ PIPELINE CONCLUÍDO!")
    print(f"Modelo final salvo na pasta: {finetune_model_dir}")
    print("="*50)

if __name__ == "__main__":
    main()

