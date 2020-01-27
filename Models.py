import numpy as np

def FrozenLakeModel(self, state, action):
    row, col = state // self.ncol, state % self.ncol
    if self.desc[row, col] == b'G': return [] # if current state is goal state, no transitions can occur - so return empty list of possible transitions
    if self.desc[row, col] == b'H': return [] # if current state is a hole in the ice, no transitions can occur - so return empty list of possible transitions
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
