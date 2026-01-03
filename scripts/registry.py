import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATASET_REGISTRY = {

}


MODEL_REGISTRY = {
}

OPTIMIZER_REGISTRY = {
    'AdamW': AdamW,
}

SCHEDULER_REGISTRY = {
    'CosineAnnealingLR': CosineAnnealingLR,
}

CALLBACK_REGISTRY = {

}   


TRAINER_REGISTRY = {
   
}

