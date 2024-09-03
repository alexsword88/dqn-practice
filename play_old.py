from itertools import count

import gymnasium as gym
import torch

from network import DQN
from utils import model_load

env = gym.make("CartPole-v1", render_mode="human")


# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
best_score = model_load(policy_net)
print(f"Train Best Score: {best_score}")


steps_done = 0

episode_durations = []


def select_action(state):
    return policy_net(state).max(1).indices.view(1, 1)


state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
for t in count():
    action = select_action(state)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
    state = next_state

    if done:
        print(f"Play Time: {t}")
        break
