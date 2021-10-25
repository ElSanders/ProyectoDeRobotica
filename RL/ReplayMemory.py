import numpy as np
from numpy.core.defchararray import index
class ReplayMemory(object):
    def __init__(self,capacity,input_dims):
        self.capacity = capacity
        self.state_memory = np.zeros((capacity, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((capacity,(3)), dtype=np.float32)
        self.new_state_memory = np.zeros((capacity, *input_dims), dtype=np.float32)
        self.reward_memory = np.zeros(capacity, dtype=np.int32)
        self.finished_memory = np.zeros(capacity, dtype=np.bool)
        self.index = 0
    
    def push(self, state, action, new_state, reward,finished):
        self.state_memory[self.index] = state
        self.action_memory[self.index] = action
        self.new_state_memory[self.index] = new_state
        self.reward_memory[self.index] = reward
        self.finished_memory[self.index] = finished
        self.index = (self.index + 1) % self.capacity
    
    def state_batch(self,batch):
        return self.state_memory[batch]
    def action_batch(self,batch):
        return self.action_memory[batch]
    def new_state_batch(self,batch):
        return self.new_state_memory[batch]
    def reward_batch(self,batch):
        return self.reward_memory[batch]
    def finished_batch(self,batch):
        return self.finished_memory[batch]


    def __len__(self):
        return self.index