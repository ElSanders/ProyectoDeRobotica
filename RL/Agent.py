import numpy as np
from numpy import random as random
import torch
import torch.optim as optim
from typing import NamedTuple

from DQN import DQN
from ReplayMemory import ReplayMemory

class Agent():
    def __init__(self,max_memory_size,input_dims):
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9999
        self.max_memory_size = max_memory_size
        self.model = DQN(input_dims)
        self.memory = ReplayMemory(self.max_memory_size,input_dims)

    def get_action(self,observation):
        action = self.model.forward(torch.tensor(observation,dtype=torch.float32).to(self.model.device))
        return action
    
    def remember(self, state,action,new_state,reward,finished):
        self.memory.push(state=state,action=action,new_state=new_state,reward=reward,finished=finished)


    def learn(self,prediction,expected):
        if len(self.memory) < self.BATCH_SIZE:
            return 
        self.model.optimizer.zero_grad()
        max_mem = min(self.max_memory_size,len(self.memory))
        batch = np.random.choice(max_mem,self.BATCH_SIZE,replace = False)

        batch_index = np.arange(self.BATCH_SIZE,dtype=np.int32)

        state_batch = torch.tensor(self.memory.state_batch(batch)).to(self.model.device)
        new_state_batch = torch.tensor(self.memory.new_state_batch(batch)).to(self.model.device)
        reward_batch = torch.tensor(self.memory.reward_batch(batch)).to(self.model.device)

        action_batch_iters = self.memory.action_batch(batch)
        finished_batch = torch.tensor(self.memory.finished_batch(batch),dtype=torch.bool).to(self.model.device)
        #q_eval = self.model.forward(state_batch)[batch_index,action_batch_iters]
        #q_next = self.model.forward(new_state_batch)
        #q_next[finished_batch] = 0.0
        #q_target = reward_batch + self.GAMMA * torch.max(q_next,dim=1)[0]
        loss = self.model.loss(prediction,expected.to(self.model.device)).to(self.model.device)  
        self.model.compute_loss(loss)
