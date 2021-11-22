import gym
import torch
import numpy as np
from ReinforcementLearning.Agent import Agent
from ComputerVision.object_identifier_factory import ObjectIdentifierFactory
from ComputerVision.take_picture import TakePicture
import time
from ReinforcementLearning.utils import *
import cv2

agent = Agent(10000, [6])
env = gym.make('FetchPickAndPlace-v1')
env.seed(2042)
time_stamps = []
avg_score= 0
begin = time.time()
vision = ObjectIdentifierFactory("shape","img/env.png")

while(1):
    picked = False
    score = 0
    done = False
    obs = env.reset()
    env.get_viewer("rgb_array").cam.distance = 0.2
    env.get_viewer("rgb_array").cam.azimuth = 0
    env.get_viewer("rgb_array").cam.elevation = -90.0
    img = env.render("rgb_array")
    TakePicture(env)    
    img = vision.ProcessImage()
    cv2.imshow("Vision Model Input",img)
    cv2.waitKey(16)

    env.get_viewer("rgb_array").cam.distance = 2.5
    env.get_viewer("rgb_array").cam.azimuth = 132.0
    env.get_viewer("rgb_array").cam.elevation = -14.0
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
        image = env.render('rgb_array')
        cv2.imshow("Simulation",cv2.resize(image, (560,560), interpolation=cv2.INTER_LINEAR))
        cv2.waitKey(16)
    end = time.time()
    time_stamps.append(end-start)
    print(f"Total Time {end-begin} Avg: {np.mean(time_stamps)}")
