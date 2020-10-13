#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:01:41 2020

@author: suryanshsoni
"""

import torch 
import numpy as np
import random 
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as autograd
from torch.autograd import Variable

#creating the architecture of neural network
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(input_size, 1000)
        self.fc3 = nn.Linear(input_size, 1000)
        self.fc4 = nn.Linear(input_size, 1000)
        self.fc5 = nn.Linear(1000, nb_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(state))
        x = F.relu(self.fc3(state))
        x = F.relu(self.fc4(state))
        q_values = self.fc5(x)
        return q_values 
        
    
#implementing experice Replay 
class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if(len(self.memory) > self.capacity):
            del self.memory[0]
            
    def sample(self,batch_size):
        samples = zip(*random.sample(self.memory,batch_size))
        #pytr=orch variable consist both a tensor and a gradient 
        #torch variable
        return map(lambda x: Variable(torch.cat(x,0)), samples)
        
#implementing deep q leanring algorithm 

class deepQnetwork():
    def __init__(self,input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        #T = TEMPERATURE PARAMETER TO MAKE PROB MORE CERTERIN IN THIS CASE IT 7 
        probs = F.softmax(self.model(Variable(state,volatile=True))*7) #
        print(probs)
        #softmax([1,2,3]) = [0.04,0.11.0.85] =. softmax([1,2,3]*3) = [0,0.02,0.98]
        action = torch.multinomial(num_samples = 1)
        return action.data[0,0]
    
    def learn(self,batch_state, batch_next_state, batch_reward,batch_action):
        outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        tdl = F.smooth_l1_loss(outputs,target)
        self.optimizer.zero_grad()
        tdl.backward(retain_variables = True)
        self.optimizer.step()
    
        
#we need to updadte the transition 
    def update(self,reward,new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state,
                          new_state,
                          torch.LongTensor([int(self.last_action)]),
                          torch.tensor([self.last_reward])))
        
        action = self.select_action(self.last_action(new_state))
        if len(self.memory.memory) > 100:
            batch_state,batch_next_state,batch_rewards,batch_action = self.memory.sample(100)
            self.learn(batch_state,batch_next_state,batch_rewards,batch_action)
            
        self.last_action = action 
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward > 1000):
            del self.reward_window[0]
        return action 
    
        
        
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        torch.save({'state_dict':self.model.state_dict()
        ,'optimizer':self.optimizer.state_dict},'last_brain.pth')
        
    def load(self):
        if(os.path.isfile('last_brain.pth')):
            print(' last model saved is loaded')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('your AI is loaded')
        else:
            print('no brain found')
            
        
        
        
        
        
        
        
        
        
        


























