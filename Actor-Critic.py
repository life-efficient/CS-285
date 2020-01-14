import gym
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from time import sleep

writer = SummaryWriter()

class ValueDataset(Dataset):
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

def train(policy, value, discount_factor=0.99, epochs=100, n_episodes=30, n_steps_for_eligibility_traces=False, generalised_advantage_error=False):
    val_idx = 0
    for epoch in range(epochs):
        avg_reward = 0
        objective = 0
        episodes = []
        for episode_idx in range(n_episodes):     # run some n_episodes
            done = False
            state = env.reset()
            log_policy = []

            episode = {
                'states': [],
                'actions': [],
                'rewards': []
            }

            # RUN AN EPISODE
            while not done:     # while the episode is not terminated
                state = torch.Tensor(state)     # correct data type for passing to model
                action_distribution = policy(state)     # get a distribution over actions from the policy given the state
                action = torch.distributions.Categorical(action_distribution).sample()      # sample from that distrbution
                action = int(action)
                
                new_state, reward, done, info = env.step(action)    # take timestep

                episode['states'].append(state)
                # episode['actions'].append(action)
                episode['rewards'].append(reward)

                state = new_state
                log_policy.append(torch.log(action_distribution[action]))
                
                if done:
                    break

            episodes.append(episode)

            avg_reward += ( sum(episode['rewards']) - avg_reward ) / ( episode_idx + 1 )   # accumulate avg reward

            # ACCUMULATE POLICY OBJECTIVE
            T = len(episode['rewards'])
            next_state_value = 0    # initialise the value of the state that follows the last state to zero 

            for t in range(T-1):     # for each timestep in the episode, compute advantage and the objective for the policy network (each computation uses the next state, so only go up to T-1)
                if t == 0:
                    state = episode['states'][t]
                    current_state_value = value(state)  # use bootstrapped prediction of state value using your current value network
                else:
                    current_state_value = next_state_value # we've already predicted the current value - we predicted the next value one timestep ago

                next_state = episode['states'][t + 1]
                next_state_value = value(next_state)


                if generalised_advantage_error:
                    # this GAE implementation is drastically underoptimised because all proceeding state values are being computed in each loop
                    # we should move it out of this loop and compute the values for all states visited in the episode in advance 
                    _lambda = generalised_advantage_error # the GAE kwarg should be the value of lambda \in [0, 1)
                    advantage = 0
                    for t_prime in range(t, T - 1):
                        reward = episode['rewards'][t_prime]
                        current_state = episode['states'][t_prime]
                        next_state = episode['states'][t_prime + 1]
                        current_state_value = value(current_state)
                        next_state_value = value(next_state)
                        advantage += (_lambda * discount_factor )** (t_prime - t) * ( reward + ( discount_factor * next_state_value ) - current_state_value )   
                elif n_steps_for_eligibility_traces: # if this is not none (it should be an integer describing how far ahead to use a monte carlo prediction for the state value)
                    advantage = 0 # just to initialise (we will add to this with the Monte Carlo and critic contributions to the advantage)
                    if t + n_steps_for_eligibility_traces > T - 1: # if the MC lookahead looks beyond the end of the episode (max value of t is T-1)
                        N = T - t # limit the lookahead
                        rewards = episode['rewards'][t:] # same as [t:t+N]
                    else:
                        N = n_steps_for_eligibility_traces # otherwise we can look the full length ahead
                        rewards = episode['rewards'][t: t + N] # get list of rewards that many steps in future
                        print('len ep:', T, '\tt:', t, '\tN:', N)
                        state_n_steps_ahead = episode['states'][t + N] # get the state N steps ahead
                        advantage += discount_factor**N * value(state_n_steps_ahead) # add critic part of the advantage (long term predictions) - the discounted value of the state N steps ahead
                    advantage += sum([discount_factor**n * rewards[n] for n in range(N)]) # calculate monte carlo part of advantage
                    advantage -= current_state_value # 
                else:
                    reward = episode['rewards'][t]
                    advantage = reward + ( discount_factor * next_state_value ) - current_state_value
                # print('reward:', reward)
                # print('current val:', current_state_value)
                # print('next val:', next_state_value)
                # print('adv:', advantage)
                objective += log_policy[t] * advantage   # add the weighted log likelihood of this taking action to the objective (log_policy is negative)
                # print('log policy:', log_policy[t])
                # print('adding to obj:', log_policy[t] * advantage)
                # print()


        writer.add_scalar('Reward/Train', avg_reward, epoch*n_episodes + episode_idx)     # plot the average reward for the episodes that were run
        
        # GENERATE SUPERVISED (STATE, VALUE) DATASET TO UPDATE VALUE NETWORK
        dataset = ValueDataset(episodes, discount_factor=discount_factor)
        loader = DataLoader(dataset, shuffle=True, batch_size=16)
        # TRAIN VALUE NETWORK
        for s, v in loader:   # single run through every state encountered in all episodes
            v_hat = value(s)
            # print('predicted value:', v_hat)
            # print('true value:', v.float().view(-1, 1))
            v_loss = F.mse_loss(v_hat, v.float().view(-1, 1))
            writer.add_scalar('ValueLoss/Train', v_loss, val_idx)
            v_loss.backward()
            value.optimiser.step()
            value.optimiser.zero_grad()
            val_idx += 1

        # UPDATE POLICY NETWORK
        objective /= n_episodes   # average over n_episodes
        objective *= -1     # invert to represent cost rather than reward
        writer.add_scalar('Objective', objective, epoch)
        objective.backward()    # backprop - puts .grad values in parameters of both value and policy network
        policy.optimiser.step()    # update params given to policy optimiser ()
        policy.optimiser.zero_grad()   # reset gradients to zero
        value.optimiser.zero_grad() # the obj is a function of the advantage which is a function of the value network's output 
        
        print('EPOCH:', epoch, f'AVG REWARD: {avg_reward:.2f}')
        print()

        # VISUALISE AT END OF EPOCH AFTER UPDATING POLICY
        state = env.reset()
        done = False
        while not done:
            env.render()
            state = torch.Tensor(state)
            action_distribution = policy(state)
            action = torch.distributions.Categorical(action_distribution).sample()
            action = int(action)
            state, reward, done, info = env.step(action)
            sleep(0.01)

    env.close()


env = gym.make('CartPole-v0')

from utils.NN_boilerplate import NN
import numpy as np
policy = NN([np.prod(env.observation_space.shape), 32, 16, 2], distribution=True)
value = NN([np.prod(env.observation_space.shape), 32, 16, 1])

p_lr = 0.01
v_lr = 0.001
weight_decay = 0
policy.optimiser = torch.optim.Adam(policy.parameters(), lr=p_lr, weight_decay=weight_decay)
value.optimiser = torch.optim.Adam(value.parameters(), lr=v_lr, weight_decay=weight_decay)

train(
    policy,
    value,
    discount_factor=0.9,
    epochs=4000,
    i_episodes=10,
    n_steps_for_eligibility_traces=5,
    generalised_advantage_error=True
)
