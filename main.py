import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from DDPG_class import DDPG, trajectory
import opt_env 

env =  opt_env.RORLenv(10)

trials  = 100
trial_len = 20 # defines parameter T, the length of a trajectory for training
P = -10 # reward if the state becomes infeasible

ddpg_agent = DDPG(env=env)
beta = 0.1

steps = [] # records all the states visited in a trajectory
rewards = [] # records the sum of rewards for each trajectory
traj_rew = []
max_traj = 0

weights = ddpg_agent.model_a.get_weights()

for trial in range(trials):
    cur_state = env.reset().reshape(1,ddpg_agent.env.obs_space_n)
    tr_reward = 0
    temp = []
    traj = trajectory(ddpg_agent, np.array([0,0,0,0,0,0,0,0,0,0]), 10)
    if traj > max_traj:
      max_traj = traj
      weights = ddpg_agent.model_a.get_weights()
      if max_traj >= 300:
        # ddpg_agent.model_a.save('/content/drive/My Drive/saved_model_ddpg_5d_4')
        # ddpg_agent.target_train_a(1)
        beta = 0.01
        ddpg_agent.epsilon_decay = 0.9
    # print(traj)
    for step in range(trial_len):
        action = ddpg_agent.act(cur_state)
        new_state, reward, done= env.step(action)
        
        if done: reward = P
        temp.append((new_state, reward))
        if not done: tr_reward += reward
        new_state = new_state.reshape(1,ddpg_agent.env.obs_space_n)
        ddpg_agent.remember(cur_state, action, reward, new_state, done)
            
        ddpg_agent.replay()
        
        # update the weights of the target networks
        ddpg_agent.target_train_a(beta, weights)
        ddpg_agent.target_train_c(.8)

        cur_state = new_state
        if done:
             break
    ddpg_agent.set_eps()  # update the exploration rate
    
    steps.append(temp)
    rewards.append(tr_reward)
    
    print('trial number:', trial, ', the length of the trajectory is', step, 'with the sum of rewards equal to', tr_reward)

plt.figure(figsize=(5, 4))
plt.plot(rewards, '*-', linewidth = 2)
plt.title('Progress of reward by training the agent ($T = 20$)')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.show()