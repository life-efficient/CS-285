from time import sleep
import gym
import numpy as np
import types
import Models

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

def extract_policy(model, value_table): # returns a policy from a value function
    policy = {}
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
        policy[state] = best_action

    print('Extracted policy:')
    print(policy)
    return policy
        
def train(model, accuracy_threshold=0.01):
    value_table = ValueTable()
    converged = False
    sweep_idx = 0
    while not converged:
        print('\nsweep ', sweep_idx)
        sweep_idx += 1
        worst_delta = 0
        
        for state in value_table.table.keys():
            best_value = -float('inf')
            old_value = value_table[state]
            for action in range(4): # find action with best value
                possible_transitions = model(state, action)
                val = 0
                for possible_transition in possible_transitions:
                    new_state, reward, probability = possible_transition
                    val += probability * (reward + discount_factor * value_table[new_state])
                if val > best_value:
                    best_value = val
            new_value = best_value # set value to max value over all actions
            value_table.update(state, new_value) # this implementation updates the value function in-place. It could alternatively create a new value table each sweep which would replace the value function only after the complete sweep.
            delta = abs(new_value - old_value) # compare difference values for this state using old and new value table 
            worst_delta = max(worst_delta, delta) # update worst difference between values for a state

        # CHECK CONVERGENCE
        if worst_delta < accuracy_threshold:
            converged = True

    print('Optimal value function found')
    print(value_table)
    return value_table

discount_factor = 0.9

is_slippery=False # stochastic environment?
env = gym.make('FrozenLake-v0', is_slippery=is_slippery) # intialise environment
model = Models.FrozenLakeModel
env.model = types.MethodType(model, env) # attach model to env
env.is_slippery = is_slippery # attach is slippery attribute to env for model to use
t = env.model(11, 3) # test model
print(t)
print()

epochs = 100
optimal_value_function = train(env.model)
optimal_policy = extract_policy(env.model, optimal_value_function)

def policy_evaluation(value_table, model, error_threshold=0.01):
    converged = False
    sweep_idx = 0
    while not converged:   
        print('sweep ', sweep_idx)
        sweep_idx += 1
        worst_delta = 0     
        for state in value_table.table.keys():
            old_val = value_table[state]
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