import torch
import gym
from time import sleep
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

def train(optimiser, epochs=100, episodes=30, use_baseline=False, use_causality=False):
    assert not (use_baseline and use_causality)   # cant implement both simply
    baseline = 0
    for epoch in range(epochs):
        avg_reward = 0
        objective = 0
        for episode in range(episodes):
            done = False
            state = env.reset()
            log_policy = []

            rewards = []

            step = 0

            # RUN AN EPISODE
            while not done:     # while the episode is not terminated
                state = torch.Tensor(state)     # correct data type for passing to model
                # print('STATE:', state)
                action_distribution = policy(state)     # get a distribution over actions from the policy given the state
                # print('ACTION DISTRIBUTION:', action_distribution)

                action = torch.distributions.Categorical(action_distribution).sample()      # sample from that distrbution
                action = int(action)
                # print('ACTION:', action)
                
                new_state, reward, done, info = env.step(action)    # take timestep

                rewards.append(reward)

                state = new_state
                log_policy.append(torch.log(action_distribution[action]))
                
                step += 1
                if done:
                    break
                if step > 10000000:
                    # break
                    pass

            avg_reward += ( sum(rewards) - avg_reward ) / ( episode + 1 )   # accumulate avg reward
            writer.add_scalar('Reward/Train', avg_reward, epoch*episodes + episode)     # plot the latest reward
            
            # update baseline
            if use_baseline:
                baseline += ( sum(rewards) - baseline ) / (epoch*episodes + episode + 1)    # accumulate average return  

            for idx in range(len(rewards)):     # for each timestep experienced in the episode
                # add causality
                if use_causality:   
                    weight = sum(rewards[idx:])     # only weight the log likelihood of this action by the future rewards, not the total
                else:
                    weight = sum(rewards) - baseline           # weight by the total reward from this episode
                objective += log_policy[idx] * weight   # add the weighted log likelihood of this taking action to 
                

        objective /= episodes   # average over episodes
        objective *= -1     # invert to represent reward rather than cost


        # UPDATE POLICY
        # print('updating policy')
        print('EPOCH:', epoch, f'AVG REWARD: {avg_reward:.2f}')
        objective.backward()    # backprop
        optimiser.step()    # update params
        optimiser.zero_grad()   # reset gradients to zero
        
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

lr = 0.001
weight_decay = 1
optimiser = torch.optim.SGD(policy.parameters(), lr=lr, weight_decay=weight_decay)

train(
    optimiser,
    use_baseline=True,
    use_causality=False,
    epochs=400,
    episodes=30
)