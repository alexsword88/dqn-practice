from itertools import count
from typing import List

import numpy as np
import torch

from envs.grid_world import GridWorldEnv
from modules.logging import logging
from network import DQN
from utils import model_load

GRID_SIZE = 25
STEP_NAME = ["Right", "Down", "Left", "Up"]
env = GridWorldEnv(size=GRID_SIZE, render_mode="human")

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

n_actions = env.action_space.n
state, info = env.reset()
n_observations = 4

policy_net = DQN(n_observations, n_actions, device=device).to(device)
# NOTE: 400 always get a good result
best_score = model_load(policy_net, episode=400)
# best_score = model_load(policy_net)
logging.info(f"Train Best Score: {best_score}")


def select_action(env: GridWorldEnv, state: torch.tensor, action_mask: List[int]):
    actions_value = policy_net(state)
    actions_value[0, action_mask] = float("-inf")
    return actions_value.max(1).indices.view(1, 1)


def generate_masking(
    x: int,
    y: int,
):
    global GRID_SIZE
    masking_index = []
    if x - 1 < 0:
        masking_index.append(2)
    if y - 1 < 0:
        masking_index.append(3)
    if x + 1 > GRID_SIZE - 1:
        masking_index.append(0)
    if y + 1 > GRID_SIZE - 1:
        masking_index.append(1)
    return masking_index


state, info = env.reset()
logging.info(f'Start Point [{state["agent"][0]}, {state["agent"][1]}]')
logging.info(f'Goal Point [{state["target"][0]}, {state["target"][1]}]')
action_mask = generate_masking(state["agent"][0], state["agent"][1])
state = torch.tensor(
    np.append(state["agent"], [GRID_SIZE, info["distance"]]),
    dtype=torch.float32,
    device=device,
).unsqueeze(0)

for t in count():
    action = select_action(state, action_mask)
    logging.debug(action_mask)
    logging.info("Agent Action: " + STEP_NAME[action.item()])
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(
            np.append(observation["agent"], [GRID_SIZE, info["distance"]]),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        action_mask = generate_masking(observation["agent"][0], observation["agent"][1])
    state = next_state

    if done:
        logging.info(f"Play Time: {t}")
        input("Enter to exit")
        break
