from pathlib import Path
from typing import Optional

import torch

from network import DQN

MODEL_SAVE_PATH = Path("checkpoints")
BEST_MODEL_NAME = "best_model.pt"
if not MODEL_SAVE_PATH.exists():
    MODEL_SAVE_PATH.mkdir()


def model_checkpoint(model: DQN, score: int, episode: int, is_best=False):
    torch.save(
        {
            "best_score": score,
            "model_state_dict": model.state_dict(),
        },
        MODEL_SAVE_PATH / (BEST_MODEL_NAME if is_best else f"model_{episode}.pt"),
    )
    return score


def model_load(model: DQN, default_score=0, episode: Optional[int] = None):
    best_score = default_score
    target = BEST_MODEL_NAME if episode is not None else f"model_{episode}.pt"
    if (MODEL_SAVE_PATH / target).exists():
        checkpoint = torch.load(MODEL_SAVE_PATH / target, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_score = checkpoint["best_score"]
    return best_score
