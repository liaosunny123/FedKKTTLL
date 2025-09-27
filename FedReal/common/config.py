from dataclasses import dataclass
from typing import Optional

@dataclass
class FedConfig:
    num_clients: int = 3
    total_rounds: int = 30
    local_epochs: int = 5
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    seed: int = 42
    sample_fraction: float = 1.0
    model_name: str = "resnet18"
    feature_dim: int =512
    max_message_mb: int = 128
    num_classes: int = 10
    encoder_ratio: float = 1.0
    algorithm: str = "FedEXT"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    feature_batch_size: int = 128
    feature_keep_spatial: bool = False
    feature_include_test_split: bool = True
    tail_batch_size: int = 64
    tail_epochs: int = 20
    tail_lr: float = 0.01
    tail_momentum: float = 0.9
    tail_weight_decay: float = 1e-4
    tail_device: Optional[str] = None

    def to_dict(self):
        return self.__dict__.copy()
