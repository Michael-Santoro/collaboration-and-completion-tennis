import json
import torch
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent
from unityagents import UnityEnvironment

from easydict import EasyDict as e_dict

import pdb


seed = 42

train_mode = True

env = UnityEnvironment(file_name="Tennis_Windows_x86_64\Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=train_mode)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

ddpg_config = e_dict()

ddpg_config.state_size = 24         # enviroment size
ddpg_config.action_size = 2         # action size
ddpg_config.seed = 42

ddpg_config.update_step = 10

## DDPG Params
ddpg_config.buffer_size = int(1e4)  # replay buffer size
ddpg_config.batch_size = 32        # minibatch size
ddpg_config.gamma = 0.99            # discount factor
ddpg_config.tau = 1e-2              # for soft update of target parameters
ddpg_config.lr_actor = 1e-4         # learning rate of the actor
ddpg_config.lr_critic = 1e-3        # learning rate of the critic
ddpg_config.weight_decay = 0.0     # L2 weight decay
ddpg_config.loss = 'l1_smooth'      # loss functions include 'mae' or 'mse'

## Noise Params
ddpg_config.theta = 0.15
ddpg_config.sigma = 0.02
ddpg_config.add_noise = True

with open('ddpg/ddpg_config.json', 'w') as f:
    json.dump(ddpg_config, f)

p0_agent = Agent(ddpg_config)
p1_agent = Agent(ddpg_config)
p1_agent.memory = p0_agent.memory

n_episodes=1000
max_t=1000
print_every=50

scores_deque = deque(maxlen=100)
scores_eps_deque = deque(maxlen=2)
scores = []
eps_actor_loss = []
eps_critic_loss = []
for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
    state = env_info.vector_observations                 # get the current state
    p0_agent.reset()
    p1_agent.reset()
    p0_score = 0
    p1_score = 0
    for t in range(max_t):
        
        p0_action = p0_agent.act(state[0],ddpg_config.add_noise)
        p1_action = p1_agent.act(state[1],ddpg_config.add_noise)
        action = np.array((p0_action,p1_action)).reshape(4,) 
        env_info = env.step(action)[brain_name]         # send the action to the environment
        next_state = env_info.vector_observations       # get the next state
        reward = env_info.rewards                       # get the reward
        done = env_info.local_done                      # see if episode has finished
        
        p0_a_loss, p0_c_loss = p0_agent.step(state[0], p0_action, reward[0], next_state[0], done[0], t)
        p1_a_loss, p1_c_loss = p1_agent.step(state[1], p1_action, reward[1], next_state[1], done[1], t)

        if p0_a_loss: 
            eps_actor_loss.append(p0_a_loss)
        if p0_c_loss:
            eps_critic_loss.append(p0_c_loss)
        if p1_a_loss: 
            eps_actor_loss.append(p1_a_loss)
        if p1_c_loss:
            eps_critic_loss.append(p1_c_loss)

        state = next_state
        p0_score += reward[0]
        p1_score += reward[1]
        if np.any(done):
            break
    if i_episode == 750:
        ddpg_config.add_noise = False
    
    # pdb.set_trace()
    
    scores_deque.append(max(p0_score,p1_score))

    scores_eps_deque.append(p0_score)
    scores_eps_deque.append(p1_score)
    scores.append(p0_score)
    scores.append(p1_score)

    if i_episode % print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.4f}\tCritic Loss: {:.5f}\tActor Loss: {:.5f}\tEpisode Score: {}'.format(i_episode, np.mean(scores_deque), eps_critic_loss[-1], eps_actor_loss[-1], scores_eps_deque))
        torch.save(p0_agent.actor_local.state_dict(), 'ddpg/p0_checkpoint_actor.pth')
        torch.save(p0_agent.critic_local.state_dict(), 'ddpg/p0_checkpoint_critic.pth')
        torch.save(p1_agent.actor_local.state_dict(), 'ddpg/p1_checkpoint_actor.pth')
        torch.save(p1_agent.critic_local.state_dict(), 'ddpg/p1_checkpoint_critic.pth')

    elif eps_critic_loss:
        print('\rEpisode {}\tAverage Score: {:.4f}\tCritic Loss: {:.5f}\tActor Loss: {:.5f}\tEpisode Score: {}'.format(i_episode, np.mean(scores_deque),  eps_critic_loss[-1], eps_actor_loss[-1],scores_eps_deque,),end="")

#pdb.set_trace()

logs = pd.DataFrame({'actor_loss':eps_actor_loss, 'critic_loss':eps_critic_loss})
logs.to_csv('ddpg/ddpg_loss_logs.csv')

_scores = pd.DataFrame({'scores':scores})
_scores.to_csv('ddpg/ddpg_scores_logs.csv')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(eps_actor_loss)+1), eps_actor_loss)
plt.ylabel('Loss')
plt.xlabel('Episode #')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(eps_critic_loss)+1), eps_critic_loss)
plt.ylabel('Loss')
plt.xlabel('Episode #')
plt.show()

env.close()