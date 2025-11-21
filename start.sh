#!/bin/bash
set -e

python -m app.effresnet_vit treinar --config config/effresnet_vit_config.json

python -m app.effresnet_vit avaliar \
  --modelo modelos/effresnet_vit/melhor_modelo_effresnet_vit.keras \
  --validacao data/datasets_training/dataset_custom_preprocessed/valid
