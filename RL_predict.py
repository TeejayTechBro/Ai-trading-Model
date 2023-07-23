# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 00:29:48 2023

@author: Huzaifah-Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:23:06 2022

@author: Huzaifah Wasim
"""

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, DQN, PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.buffers import ReplayBuffer
import pandas as pd
from env_fx import tgym

import os
import gym
#import ray 

env=tgym("GBBPUSD",df=)
obs=env.reset()


model_path="models/DDPG/9000.zip" 

model=DDPG.load(model_path, env) 


episodes = 2000

for i in range(episodes):
    done=False
    while not done:
        env.render()
        print("\nOBSERVATION: {}".format(obs))
        action, _=model.predict(obs)
        obs,reward,done,info=env.step(action)
        print("\nTAKEN ACTION: {},\nREWARD: {}".format(action,reward))
       
      
            
