from torch.utils.data import Dataset
import torch

class MCValueDataset(Dataset):
    def __init__(self, episodes, discount_factor):
        super().__init__()
        self.examples = []  # initialise empty list of datapoints
        for episode in episodes:    # for each episode that just ran
            states = episode['states']
            rewards = episode['rewards']
            T = len(rewards)    # length of episode
            _return = 0    # initialise future reward for last timestep
            for t in reversed(range(T)):    # do back up
                state = states[t]    # get state
                _return = rewards[t] + discount_factor * _return     # calculate return recursively
                self.examples.append((state, _return))  # add (state, value) tuple to list of datapoints

    def __getitem__(self, idx):
        return self.examples[idx]   # return (state, value) tuple

    def __len__(self):
        return len(self.examples)   # how many examples generated  

class DPValueDataset(Dataset):
    """Create a dataset using the model"""
    def __init__(self, model, value_function, episodes, discount_factor):
        super().__init__()
        self.examples = []
        for episode in episodes:
            states = episode['states']
            action = episode['actions']
            for state in states:
                possible_transitions = model(state, action)
                _return = 0
                for possible_transition in possible_transitions:
                    new_state, reward, probability = possible_transition
                    new_state = torch.tensor(new_state).view((1, -1)).float()
                    _return += probability * (reward + discount_factor * value_function(new_state))
                state = torch.tensor(state).view((-1, 1)).float()
                _return = _return.detach() # needs to be detached because it has a history in the computation graph as a result of it beign computed using the value function
                self.examples.append((state, _return))

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


class QDataset(Dataset):
    def __init__(self, episodes, discount_factor):
        super().__init__()
        self.examples = []  # initialise empty list of datapoints
        for episode in episodes:    # for each episode that just ran
            states = episode['states']
            actions = episode['actions']
            rewards = episode['rewards']
            T = len(rewards)    # length of episode
            _return = 0    # initialise future reward for last timestep
            for t in reversed(range(T)):    # do back up
                state = states[t]    # get state
                action = actions[t]
                _return = rewards[t] + discount_factor * _return     # calculate return recursively
                self.examples.append(((state, action), _return))  # add (state, value) tuple to list of datapoints

    def __getitem__(self, idx):
        return self.examples[idx]   # return (state, value) tuple

    def __len__(self):
        return len(self.examples)   # how many examples generated  