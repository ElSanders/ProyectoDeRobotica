import gym
import torch
import numpy as np
from ReinforcementLearning.Agent import Agent
from ComputerVision.object_identifier_factory import ObjectIdentifierFactory
from ComputerVision.take_picture import TakePicture
import time
from ReinforcementLearning.utils import *

agent = Agent(10000, [6])
time_stamps = []
avg_score= 0
begin = time.time()
vision = ObjectIdentifierFactory("shape","img/env.png")

while(1):
    picked = False
    score = 0
    done = False
    env = gym.make('FetchPickAndPlace-v1')
    obs = env.reset()
    env.render(mode='rgb_array')
    env.get_viewer("rgb_array").cam.distance = 0.2
    env.get_viewer("rgb_array").cam.azimuth = 0
    env.get_viewer("rgb_array").cam.elevation = -90.0
    TakePicture(env)
    vision.ProcessImage()
    env.get_viewer("human").cam.distance = 2.5
    env.get_viewer("human").cam.azimuth = 132.0
    env.get_viewer("human").cam.elevation = -14.0
    env.render()
    last_obs = obs
    steps = 0
    start = time.time()
    while not done and steps < 500:
        picked = get_picked(last_obs)
        grab = -1 if picked else 1
        action = agent.get_action(get_observation(last_obs,picked))
        movement = torch.cat((action.cpu(),torch.tensor([grab]))).detach().numpy()
        obs_,reward,done,info = env.step(movement)
        
        obs_ = {
            "observation" : obs_["observation"],
            "desired_goal": vision.GetCordinates(obs_)["desired_goal"],
            "achieved_goal": vision.GetCordinates(obs_)["achieved_goal"]
            } 

        reward = get_reward(last_obs,obs_)
        score += reward
        done = get_done(obs_)
        agent.remember(get_observation(last_obs,picked),action.cpu().detach().numpy(),get_observation(obs_,picked),reward,done)
        agent.learn(action,get_expected(last_obs,picked=picked))
        end = time.time()
        last_obs = obs_
        steps = steps + 1
        env.render()
    end = time.time()
    time_stamps.append(end-start)
    print(f"Total Time {end-begin} Avg: {np.mean(time_stamps)}")
    env.close()
