#!/bin/bash
set -e

python -m app.effresnet_vit treinar --config config/effresnet_vit_config.json

python -m effresnet_vit avaliar \
  --modelo modelos/melhor_modelo_effresnet_vit.keras \
  --validacao dataset_custom_preprocessed/validation
