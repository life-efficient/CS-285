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

def train(policy, value, policy_optimiser, value_optimiser, discount_factor=0.99, epochs=100, n_episodes=30):
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
            writer.add_scalar('Reward/Train', avg_reward, epoch*n_episodes + episode_idx)     # plot the latest reward
            
            # ACCUMULATE POLICY OBJECTIVE
            T = len(episode['rewards'])
            next_state_value = 0    # initialise the value of the state that follows the last state to zero 
            for t in range(T):     # for each timestep experienced in the episode
                state = episode['states'][t]
                reward = episode['rewards'][t]
                current_state_value = value(state)
                advantage = reward + discount_factor * next_state_value - current_state_value
                objective += log_policy[t] * advantage   # add the weighted log likelihood of this taking action to the objective
                next_state_value = current_state_value

        # GENERATE SUPERVISED (STATE, VALUE) DATASET TO UPDATE VALUE NETWORK
        dataset = ValueDataset(episodes, discount_factor=0.9)
        loader = DataLoader(dataset, shuffle=True, batch_size=16)
        # TRAIN VALUE NETWORK
        for idx, (s, v) in enumerate(loader):   # single run through every state encountered in all episodes
            v_hat = value(s)
            v_loss = F.mse_loss(v_hat, v.float().view(-1, 1))
            writer.add_scalar('ValueLoss/Train', v_loss, val_idx)
            v_loss.backward()
            value_optimiser.step()
            value_optimiser.zero_grad()
            val_idx += 1

        # UPDATE POLICY NETWORK
        objective /= n_episodes   # average over n_episodes
        objective *= -1     # invert to represent reward rather than cost
        writer.add_scalar('Objective', objective, epoch)
        objective.backward()    # backprop
        policy_optimiser.step()    # update params
        policy_optimiser.zero_grad()   # reset gradients to zero
        
        print('EPOCH:', epoch, f'AVG REWARD: {avg_reward:.2f}')

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
policy = NN([4, 32, 2], distribution=True)
value = NN([4, 32, 1])

lr = 0.01
weight_decay = 0
policy_optimiser = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
value_optimiser = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)

train(
    policy,
    value,
    policy_optimiser,
    value_optimiser,
    discount_factor=0.9,
    epochs=400,
    n_episodes=3
)