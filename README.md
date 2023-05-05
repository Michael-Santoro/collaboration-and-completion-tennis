# Collaboration to Win Tennis
## Multi-Agent Reinforcement Learning
**Michael Santoro - micheal.santoro@du.edu**
# Introduction
This project explores Deep Reinforcement Learning in an enviroment with multiple agents. This particular enviroment requires continious control. The plan is use the [Proximal Policy Opimization](https://arxiv.org/abs/1707.06347) (PPO). PPO is a popular algorithm used for solving continuous control problems in reinforcement learning.

PPO, is a model-free, on-policy algorithm that is designed to handle both discrete and continuous action spaces. It uses a trust region optimization approach to update the policy, which ensures that the new policy is not too far from the old policy. This helps to prevent the policy from diverging during training. PPO has been shown to be effective in solving a variety of continuous control problems, such as humanoid locomotion and robotic manipulation.

This a solution submission for the Deep Reinforcement Learning by Udacity. The problem details can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

# Enviroment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Enviroment Set-Up
1. Find the appropiate unity reacher enviroment from this [repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).
2. After cloning the repository navigate to the repo and run the following command.
```
conda env create --file drlnd.yml
```
3. Run the Following.
```
conda activate drlnd
```

