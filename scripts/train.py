# scripts/train.py
from core.builder import load_config
from training.base_trainer import BaseTrainer

if __name__ == "__main__":
    cfg = load_config("configs/train_default.yaml")
    trainer = BaseTrainer(cfg)
    trainer.train()