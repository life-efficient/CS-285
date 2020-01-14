import torch
from time import sleep
import gym
from utils.NN_boilerplate import NN
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import types

writer = SummaryWriter()

# policy iteration needs a value table to perform the argmax over actions
class ValueTable():
    def __init__(self, env):
        self.table = np.zeros((env.observation_space.n, env.action_space.n))

    def __getitem__(self, index):
        state, action = index
        return self.table[state, action]
        
    def update(self, state, action, new_value):
        self.table[state][action] = new_value # find value of idx in table

    def policy(self, state):
        # if np.random.rand() > epsilon:
        #     return np.random.randint(self.table.shape[1]) # take a random action
        return np.argmax(self.table[state])

def train(env, value_table):
    for policy_idx in range(num_updates): # every update creates a new policy
        # POLICY EVALUATION step 1 = SIMULATE EPISODES 
        # for state in range(env.observation_space.n):            
        #     old_value = value_table[1, 1]
        #     print(old_value)
        #     # new_value = reward + discount_factor * value_table[new_state]

            # l
        for episode_idx in range(episodes_per_update):
            done = False
            state = env.reset()
            total_reward = 0
            while not done:
                state = torch.tensor(state)
                action = value_table.policy(state)
                # print('action:', action)
                new_state, reward, done, _ = env.step(1)
                total_reward += reward
                env.render()

                print()
                row, col = new_state // 4, new_state % 4
                print(row, col)
                print(env.desc)
                sleep(0.1)
                # experiences.append((state, action, reward, new_state))
                state = new_state
                # print('STATE;',state)
                print(done)
                print()
                k
            k

            # writer.add_scalar('Reward/Train', total_reward, episode_idx + policy_idx * episodes_per_update)

        # POLICY EVALUATION step 2 = FIT VALUE FUNCTION
        experience_dataset = ValueDataset(experiences)
        experience_loader = DataLoader(experience_dataset, shuffle=True, batch_size=16)
        for epoch_idx, epoch in enumerate(epochs):
            avg_loss = 0
            for state, target_value in experience_loader:
                print(state)
                input_state = torch.tensor(state).view()
                predicted_value = value(state)
                loss = torch.nn.MSELoss(predicted_value, target_value)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
            avg_loss /= len(experiences)
            writer.add_scalar('ValueLoss/Train', avg_loss, epoch + policy_idx * len(experiences))

        # POLICY ITERATION = UPDATE POLICY
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                print(state)
                new_value = value(torch.tensor(state).view(1, -1))
                value_table.update(state, action, new_value) # update the value for this (s, a) in the value table
        



episodes_per_update = 1
num_updates = 100
discount_factor = 0.75

is_slippery=False
env = gym.make('FrozenLake-v0', is_slippery=is_slippery)

def model(self, state, action):
    row, col = state // self.ncol, state % self.ncol
    new_states = [(row, max(col-1, 0)), (min(row+1, self.nrow - 1), col), (row, min(col+1, self.ncol - 1)), (max(row-1, 0), col)] # new (row, col) if action [left, down, right, up] is taken
    if is_slippery:
        transition_probs = np.ones(self.action_space.n)
        transition_probs[(action + 2) % 4] = 0   # all possible actions can be taken, unless its opposite to the one you tried to take (if we try to go up we will never go down, only left, up or right)
        transition_probs /= sum(transition_probs)    # each remaining action which actually occurs has equal probability
    else:
        transition_probs = np.zeros(self.action_space.n)
        transition_probs[action] = 1    # deterministic - the state in the direction of the action is certain to be observed next
    rewards = [np.float(self.desc[row, col] ==b'G') for (row, col) in new_states]
    return zip(new_states, rewards, transition_probs) # return list of (new state, probability of going into that state)

env.model = types.MethodType(model, env) # attach model to env
env.model(11, 3)



# value = NN([env.observation_space.n, 16, 16, 1], distribution=False)
value_table = ValueTable(env)
epochs = 100
train(env, value_table)
