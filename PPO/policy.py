import sys
sys.path.append("..") 
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

HIDDEN_SIZE = 512
HIDDEN_MEMORY_SIZE = 256
NUM_FILTERS = [64, 64, 32]
KERNEL_SIZE = [8, 4, 2]

class CNN(nn.Module):
    def __init__(self, obs_size):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_size, 64, kernel_size=8),
            nn.Conv2d(64, 32, kernel_size=4),
            nn.Conv2d(32, 32, kernel_size=2)
            )
    def forward(self, obs):
        out = self.conv(obs)
        return torch.flatten(out)
        
class GRUPolicy(nn.Module):

    def __init__(self, obs_size, action_size, num_workers):
        super(GRUPolicy, self).__init__()
        self.num_workers = num_workers
        if isinstance(obs_size, list):
            self.cnn = CNN(obs_size)
            self.image = True
        else:
            self.fnn = nn.Sequential(
                nn.Linear(obs_size, HIDDEN_SIZE),
                nn.ReLU()
                )
            self.image = False
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_MEMORY_SIZE, batch_first=True)
        self.actor = nn.Linear(HIDDEN_MEMORY_SIZE, action_size)
        self.critic = nn.Linear(HIDDEN_MEMORY_SIZE, 1)
        self.hidden_memory_size = HIDDEN_MEMORY_SIZE
        self.hidden_memory = torch.zeros(1, 
            self.num_workers,
            self.hidden_memory_size
            )

    def forward(self, obs):

        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)

        _, self.hidden_memory = self.gru(feature_vector, self.hidden_memory)

        logits = self.actor(self.hidden_memory)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        value = self.critic(self.hidden_memory)

        return action, value, log_prob, self.hidden_memory

    def evaluate_action(self, obs, action, hidden_memory):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs) 

        hidden_memory, _ = self.gru(feature_vector, hidden_memory)

        log_probs = self.actor(hidden_memory)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(action)
        entropy = action_dist.entropy()

        value = self.critic(hidden_memory)

        return value, log_prob, entropy

    def evaluate_value(self, obs):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)
        _, hidden_memory = self.gru(feature_vector, self.hidden_memory)
        value = self.critic(hidden_memory)
        return value



    def _reset_hidden_memory(self, worker):
        
        self.hidden_memory[:, worker] = torch.zeros(
            1, 1, self.hidden_memory_size
            )
        

class ModifiedGRUPolicy(nn.Module):

    def __init__(self, obs_size, action_size, num_workers):
        super(GRUPolicy, self).__init__()
        self.num_workers = num_workers
        if isinstance(obs_size, list):
            self.cnn = CNN(obs_size)
            self.image = True
        else:
            self.image = False
            self.fnn = nn.Sequential(
                nn.Linear(obs_size, HIDDEN_SIZE),
                nn.ReLU()
                )
            self.fnn2 = nn.Sequential(
                nn.Linear(HIDDEN_SIZE+HIDDEN_MEMORY_SIZE, HIDDEN_SIZE),
                nn.ReLU()
                )
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_MEMORY_SIZE, batch_first=True)
        self.actor = nn.Linear(HIDDEN_MEMORY_SIZE, action_size)
        self.critic = nn.Linear(HIDDEN_MEMORY_SIZE, 1)
        self.hidden_memory_size = HIDDEN_MEMORY_SIZE
        self._reset_hidden_memory()
    
    def forward(self, obs):

        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)

        out = self.fnn2(torch.concat(feature_vector, self.hidden_memory))
        self.hidden_memory, _ = self.gru(feature_vector, self.hidden_memory)

        logits = self.actor(out)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        value = self.critic(out)

        return action, value, log_prob

    def evaluate_actions(self, obs, actions, hidden_memory):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs) 

        hidden_memory, _ = self.gru(feature_vector, hidden_memory)

        log_probs = self.actor(hidden_memory)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        value = self.critic(hidden_memory)

        return value, log_prob, entropy

    def _reset_hidden_memory(self):
        self.hidden_memory = torch.zeros(1, 
            self.num_workers, 
            self.hidden_memory_size
            )