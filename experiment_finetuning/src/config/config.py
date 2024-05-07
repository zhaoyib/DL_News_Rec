from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    random_seed: int = 42
    textual_pretrained: str = "google-bert/bert-base-uncased"
    visual_pretrained: str = "google/vit-base-patch16-224"
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 2
    gradient_accumulation_steps: int = 16  # batch_size = 16 x 2 = 32
    epochs: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_len: int = 30


cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainConfig)