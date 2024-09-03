import math
import random
from datetime import datetime
from itertools import count
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from envs.grid_world import GridWorldEnv
from modules.logging import logging
from network import DQN
from utils import model_checkpoint

GRID_SIZE = 11
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
PATH_MEMORY_SIZE = GRID_SIZE
TRUNCATED_NUM = 200

envs = [GridWorldEnv(size=5), GridWorldEnv(size=7), GridWorldEnv(size=9)]
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Get number of actions from gym action space
n_actions: int = envs[0].action_space.n
# NOTE: [agentX, agentY, Grid Size, distance]
n_observations = 4

best_score = math.inf
policy_net = DQN(
    n_observations, n_actions, batch_size=BATCH_SIZE, device=device, gamma=GAMMA
).to(device)
target_net = DQN(
    n_observations, n_actions, batch_size=BATCH_SIZE, device=device, gamma=GAMMA
).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

steps_done = 0
episode_durations = []


def select_action(env: GridWorldEnv, state: torch.tensor, action_mask: List[int]):
    global steps_done, policy_net
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            actions_value = policy_net(state)
            actions_value[0, action_mask] = float("-inf")
            # 將已訪問狀態的動作設為負無窮
            # for a in range(self.n_actions):
            #     if self.get_next_state(env.current_state, a) in self.visited_states:
            #         actions_value[0, a] = float('-inf')
            # action = torch.max(actions_value, 1)[1].data.numpy()[0]
            # self.last_action = action
            return actions_value.max(1).indices.view(1, 1)
    else:
        random_action = env.action_space.sample()
        while random_action in action_mask:
            random_action = env.action_space.sample()
        return torch.tensor([[random_action]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def generate_masking(x: int, y: int, last_action: Optional[int] = None):
    global GRID_SIZE
    masking_index = []
    # 0:right, 1: down, 2: left, 3:up
    if x - 1 < 0:
        masking_index.append(2)
    if y - 1 < 0:
        masking_index.append(3)
    if x + 1 > GRID_SIZE - 1:
        masking_index.append(0)
    if y + 1 > GRID_SIZE - 1:
        masking_index.append(1)
    return masking_index


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50
for i_episode in range(num_episodes):
    env = random.choice(envs)
    GRID_SIZE = env.size
    # Initialize the environment and get its state
    state, info = env.reset()
    action_mask = generate_masking(state["agent"][0], state["agent"][1])
    walked = []
    state = torch.tensor(
        np.append(state["agent"], [GRID_SIZE, info["distance"]]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    rewards = []
    last_actions = []
    last_action = None
    for t in count():
        action = select_action(env, state, action_mask)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated or t > TRUNCATED_NUM
        reward = int(GRID_SIZE * 2 / info["distance"]) if info["distance"] > 0 else 1000
        if len(walked) > PATH_MEMORY_SIZE:
            walked.pop(0)
        if not terminated:
            for index, walk_point in enumerate(walked):
                if np.array_equal(walk_point, observation["agent"]):
                    reward -= index * 2
                    break
            if t > GRID_SIZE * 2:  # let agent over min step is no good
                reward -= t // GRID_SIZE

        rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            action_mask = generate_masking(
                observation["agent"][0], observation["agent"][1], last_action
            )
            walked.append(observation["agent"])
            next_state = torch.tensor(
                np.append(observation["agent"], [GRID_SIZE, info["distance"]]),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

        # Store the transition in memory
        target_net.update_memory(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        target_net.learn(policy_net=policy_net, optimizer=optimizer)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            logging.debug(f"Done time: {t + 1}s")
            episode_durations.append(t + 1)
            means = sum(rewards) / len(rewards)
            if (i_episode + 1) % 100 == 0:
                model_checkpoint(policy_net, means, i_episode + 1)
            plot_durations()
            break
logging.info("Complete")
plot_durations(show_result=True)
model_checkpoint(policy_net, means, i_episode + 1, is_best=True)
plt.ioff()
plt.savefig(datetime.now().strftime("result_%Y%m%d%H%M%S.png"))
plt.show()
