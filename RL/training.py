import gym
import torch
import numpy as np
from Agent import Agent

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

def get_reward(last_obs,obs_):
    last_dist = abs(last_obs['achieved_goal'][1] - last_obs['observation'][1])
    new__dist = abs(obs_['achieved_goal'][1] - obs_['observation'][1])
    if(new__dist < last_dist):
        return 0.0
    elif(new__dist > last_dist):
        return 1.0

agent = Agent(10000, [6])
env = gym.make('FetchPickAndPlace-v1')
env.seed(12)
scores = []
avg_score= 0
i = 0
while(avg_score < 10000):
    picked = False
    score = 0
    done = False
    obs = env.reset()
    last_obs = obs
    steps = 0
    while steps < 500:
        #env.render()
        picked = get_picked(last_obs)
        grab = -1 if picked else 1
        action = agent.get_action(get_observation(last_obs,picked))
        movement = torch.cat((action.cpu(),torch.tensor([grab]))).detach().numpy()
        obs_,reward,done,info = env.step(movement)
        reward = get_reward(last_obs,obs_)
        score += reward
        agent.remember(get_observation(last_obs,picked),action.cpu().detach().numpy(),get_observation(obs_,picked),reward,done)
        agent.learn(action,get_expected(last_obs,picked=picked))
        last_obs = obs_
        steps = steps + 1
        env.render()
    scores.append(score)
    avg_score = np.mean(scores[-50:])
    i = i + 1
    print(f"Episode {i}, average score {avg_score}")