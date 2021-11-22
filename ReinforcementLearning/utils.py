import torch
import numpy as np

def get_expected(obs,picked):
    if(not picked):
        start = obs['observation'][0:3]
        finish = obs['achieved_goal']
    else:
        start = obs['achieved_goal']
        finish = obs['desired_goal']
    return torch.tensor(np.array([finish[0]-start[0],finish[1]-start[1],finish[2]-start[2]]),dtype=torch.float32)


def get_observation(obs,picked):
    if(not picked):
        start = obs['observation'][0:3]
        finish = obs['achieved_goal']
    else:
        start = obs['achieved_goal'][0:3]
        finish = obs['desired_goal']
    return np.concatenate((start,finish))

def get_picked(obs):
    if(np.mean((obs['observation'][0:3]-obs['achieved_goal'])**2) > 0.00015):
        return False
    else:
        return True

def get_done(obs):
    if(np.mean((obs['desired_goal'][0:3]-obs['achieved_goal'])**2) > 0.00015):
        return False
    else:
        return True

def get_reward(last_obs,obs_):
    last_dist = abs(last_obs['achieved_goal'][1] - last_obs['observation'][1])
    new__dist = abs(obs_['achieved_goal'][1] - obs_['observation'][1])
    if(new__dist < last_dist):
        return 0.0
    elif(new__dist > last_dist):
        return 1.0