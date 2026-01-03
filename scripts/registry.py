import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.pipelines.omnivid_alpha import OmniVidAlpha
from src.pipelines.omnivid_intrinsic import OmniVidIntrinsic
from src.trainers.omnivid_alpha_trainer import ModelCheckpointCallback_OmnividAlpha, TensorboardLoggingCallback_OmnividAlpha, Trainer_OmnividAlpha

DATASET_REGISTRY = {

}


MODEL_REGISTRY = {
    'OmniVidAlpha': OmniVidAlpha,
    'OmniVidIntrinsic': OmniVidIntrinsic
}

OPTIMIZER_REGISTRY = {
    'AdamW': AdamW,
}

SCHEDULER_REGISTRY = {
    'CosineAnnealingLR': CosineAnnealingLR,
}

CALLBACK_REGISTRY = {
    'ModelCheckpointCallback_OmnividAlpha': ModelCheckpointCallback_OmnividAlpha,
    'TensorboardLoggingCallback_OmnividAlpha': TensorboardLoggingCallback_OmnividAlpha,
}   


TRAINER_REGISTRY = {
    'Trainer_OmnividAlpha': Trainer_OmnividAlpha,
}

