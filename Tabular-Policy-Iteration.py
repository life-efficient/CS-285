import torch
from time import sleep
import gym
from utils.NN_boilerplate import NN
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import types

# policy iteration needs a value table to perform the argmax over actions
class ValueTable():
    def __init__(self, num_states=16):
        self.table = {state: 0 for state in range(num_states)}

    def __getitem__(self, state):
        return self.table[state]

    def __repr__(self):
        return str(self.table)
        
    def update(self, state, new_value):
        self.table[state] = new_value # find value of idx in table

class GreedyPolicy():
    def __init__(self, num_states=16):
        self.policy = {state: 0 for state in range(num_states)} # initialise action 0 for each state

    def update(self, state, value):
        self.policy[state] = value

    def __getitem__(self, state):
        return self.policy[state]

    def __repr__(self):
        return str(self.policy)

def policy_evaluation(policy, value_table, model, error_threshold=0.01):
    converged = False
    sweep_idx = 0
    while not converged:   
        print('sweep ', sweep_idx)
        sweep_idx += 1
        worst_delta = 0     
        for state in value_table.table.keys():
            old_val = value_table[state]
            action = policy[state]
            possible_transitions = model(state, action)
            new_val = 0
            for possible_transition in possible_transitions: # considering all possible transitions
                new_state, reward, probability = possible_transition # unpack that transition
                new_val += probability * (reward + discount_factor * value_table[new_state]) # cumulate expected value

            # UPDATE VALUE IN TABLE
            value_table.update(state, new_val)

            # EVALUATE DELTA
            delta = abs(new_val - old_val) # difference between state values between iterations
            worst_delta = max(worst_delta, delta) # update worst difference

        # CHECK CONVERGED
        if worst_delta < error_threshold:
            converged = True

    return value_table

def policy_improvement(value_table, model):
    new_policy = GreedyPolicy() # we don't need the previous policy, we'll just create a new one from the current value table
    for state in value_table.table.keys():
        best_value = -float('inf')
        best_action = None
        for action in range(4):
            possible_transitions = model(state, action)
            val = 0
            for possible_transition in possible_transitions:
                new_state, reward, probability = possible_transition
                val += probability * (reward + discount_factor * value_table[new_state])
            if val > best_value:
                best_value = val
                best_action = action
        new_policy.update(state, best_action)
    return new_policy
        
def train(env, value_table):

    def check_stable(new_policy, old_policy):
        stable = True
        for state in value_table.table.keys():
            if new_policy[state] != old_policy[state]:
                stable = False
        return stable

    policy_stable = False
    policy = GreedyPolicy() # initialise randomly
    policy_idx = 0
    while not policy_stable:

        print('policy ', policy_idx)

        # POLICY EVALUATION
        value_table = policy_evaluation(policy, value_table, env.model) # find value function for current policy
        print('value_table:', value_table)

        # POLICY IMPROVEMENT
        new_policy = policy_improvement(value_table, env.model) # get new policy by acting greedily with respect to the current value table

        # CHECK STABLE POLICY REACHED
        policy_stable = check_stable(new_policy, policy) 

        # ITERATE CURRENT POLICY
        policy = new_policy  

        print()

    print('Optimal policy found')
    print(policy)
    return policy

discount_factor = 0.9

def model(self, state, action):
    row, col = state // self.ncol, state % self.ncol
    if self.desc[row, col] == b'G': return [] # if current state is goal state, no transitions can occur - so return empty list of possible transitions
    new_states = [(row, max(col-1, 0)), (min(row+1, self.nrow - 1), col), (row, min(col+1, self.ncol - 1)), (max(row-1, 0), col)] # new (row, col) if action [left, down, right, up] is taken
    if self.is_slippery: # if stochastic environment
        transition_probs = np.ones(self.action_space.n)
        transition_probs[(action + 2) % 4] = 0   # all possible actions can be taken, unless its opposite to the one you tried to take (if we try to go up we will never go down, only left, up or right)
        transition_probs /= sum(transition_probs)    # each remaining action which actually occurs has equal probability
    else:
        transition_probs = np.zeros(self.action_space.n)
        transition_probs[action] = 1    # deterministic - the state in the direction of the action is certain to be observed next
    rewards = [np.float(self.desc[row, col] == b'G') for (row, col) in new_states]
    new_states = [self.nrow * s[0] + s[1] for s in new_states]
    return list(zip(new_states, rewards, transition_probs)) # return list of (new state, reward, probability of that transition)

is_slippery=False # stochastic environment?
env = gym.make('FrozenLake-v0', is_slippery=is_slippery) # intialise environment
env.model = types.MethodType(model, env) # attach model to env
env.is_slippery = is_slippery # attach is slippery attribute to env for model to use
t = env.model(11, 3) # test model
print(t)
print()

value_table = ValueTable()
epochs = 100
train(env, value_table)