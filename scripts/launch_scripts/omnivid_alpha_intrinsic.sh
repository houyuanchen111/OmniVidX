accelerate launch \
    --config_file "configs/accelerate_config.yaml" \
    "scripts/train.py" \
    --config "configs/omnivid_intrinsic_train.yaml"     
