import torch
from utils.NN_boilerplate import NN
from MyDatasets import DPValueDataset
from torch.utils.data import DataLoader
import gym
import Models
import types
import torch.nn.functional as F
import numpy as np

def fit_value_function(value_function, model, episodes, discount_factor, epochs=100):
    dataset = DPValueDataset(model, value_function, episodes, discount_factor)
    loader = DataLoader(dataset, shuffle=True, batch_size=64)
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (state, _return) in enumerate(loader):
            prediction = value_function(state)
            # print('state:', state, 'target:', _return, 'prediction:', prediction)
            loss = F.mse_loss(prediction, _return)
            # print('Batch:', batch_idx, 'Loss:', loss.item())
            epoch_loss += loss.item()
            loss.backward()
            value_function.optimiser.step()
            value_function.optimiser.zero_grad()
        epoch_loss /= len(loader)
        print('avg batch loss:', epoch_loss)
    print('done fitting')
    print('current value function:')
    print({s: value_function(torch.tensor(s).view(1, -1).float()).item() for s in range(16)})
    print()

def sample_episodes(value_function, env, discount_factor, num_episodes=100, epsilon=0.1):
    episodes = []
    for episode_idx in range(num_episodes):
        episode = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        done = False
        state = env.reset()
        while not done:
            best_val = -float('inf')
            for action in range(4): # act greedily wrt value function
                possible_transitions = env.model(state, action)
                val = 0
                for possible_transition in possible_transitions:
                    new_state, reward, probability = possible_transition
                    new_state = torch.tensor(new_state).view((1, -1)).float()
                    val += probability * (reward + discount_factor * value_function(new_state))
                if val >= best_val:
                    best_val = val
                    action_to_take = action
            if np.random.rand() < epsilon:
                action_to_take = np.random.randint(4)
            episode['states'].append(state)
            episode['actions'].append(action_to_take)
            state, reward, done, info = env.step(action_to_take)
            episode['rewards'].append(reward)
        episodes.append(episode)
        # print(episode['states'])
        print('Episode ', episode_idx, 'lasted', len(episode['states']), 'timesteps')
        # kl
    return episodes

def train(env, value_function, discount_factor=0.9, num_iterations=10, epochs=10):
    for iteration in range(num_iterations):
        print('Iteration ', iteration)
        episodes = sample_episodes(value_function, env, discount_factor)
        fit_value_function(value_function, env.model, episodes, discount_factor, epochs)

discount_factor = 0.9
is_slippery=False # stochastic environment?
env = gym.make('FrozenLake-v0', is_slippery=is_slippery) # intialise environment
model = Models.FrozenLakeModel
env.model = types.MethodType(model, env) # attach model to env
env.is_slippery = is_slippery # attach is slippery attribute to env for model to use
value_function = NN([1, 32, 16, 32, 1]).float()
value_function.optimiser = torch.optim.Adam(value_function.parameters(), lr=0.005)
train(env, value_function, discount_factor)

def extract_policy(value_function, model, discount_factor):
    policy = {}
    for state in range(15):
        best_action = None
        best_val = -float('inf')
        for action in range(4):
            possible_transitions = model(state, action)
            val = 0
            for possible_transition in possible_transitions:
                new_state, reward, probability = possible_transition
                new_state = torch.tensor(new_state).view((1, -1)).float()
                val += probability * (reward + discount_factor * value_function(new_state))
            if val > best_val:
                best_val = val
                best_action = action
        policy[state] = best_action
    print(policy)
    return policy

extract_policy(value_function, env.model, discount_factor)