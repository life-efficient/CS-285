import torch
import gym
from utils.NN_boilerplate import NN
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


value = NN([np.prod(env.observation_space.shape), 16, 16, 1], distribution=False)

# policy iteration needs a value table to perform the argmax over actions
class ValueTable():
    def __init___(self, grid_size):
        self.table = np.zeros((grid_size, grid_size))

    def update_value_table():
        #  update value table at point in table

def train():
    env = gym.make('CartPole-v0')
    value_table = create_value_table(env)
    for policy_idx in range(num_updates): # every update creates a new policy
        # RUN EPISODES
        for episode_idx in range(episodes_per_update):
            done = False
            state = env.reset()
            while not done:
                state = torch.tensor(state)

                for a in range(env.action_space.n): # cant resample action spaces because you'd have to revisit them

                    advantage = reward
                    print(value(state))
                # action = # argmax over actions
                k


episodes_per_update = 10
num_updates = 100
train()
