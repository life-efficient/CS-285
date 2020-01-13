import torch
import gym
from utils.NN_boilerplate import NN
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Griddy import 

writer = SummaryWriter()


value = NN([np.prod(env.observation_space.shape), 16, 16, 1], distribution=False)

# policy iteration needs a value table to perform the argmax over actions
class ValueTable():
    def __init___(self, env, state_size, action_size):
        self.table = np.zeros((*state_size, action_size))

    def update(state, action, new_value):
        self.table[*state][action] = new_value # find value of idx in table

    def policy(state):
        return np.argmax(self.table[state])

def train(value_table):
    env = gym.make('CartPole-v0')
    value_table = create_value_table(env)
    for policy_idx in range(num_updates): # every update creates a new policy
        # POLICY EVALUATION = RUN EPISODES
        for episode_idx in range(episodes_per_update):
            done = False
            state = env.reset()
            while not done:
                state = torch.tensor(state)
                action = policy
                
                # action = # argmax over actions
                k

        # POLICY UPDATE
        new_value = reward + discount_factor * value(next_state) # ONLY WORKS FOR DETERMINISTIC ENVIRONMENTS CURRENTLY - otherwise we'd need to take the expectance over the next states
        value_table.update(state, action, new_value)
        policy


episodes_per_update = 10
num_updates = 100
discount_factor = 0.9
train()
