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
        self.recurrent = True
        if isinstance(obs_size, list):
            self.cnn = CNN(3)
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

        out, self.hidden_memory = self.gru(feature_vector, self.hidden_memory)
        logits = self.actor(out.flatten(end_dim=1))
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        value = self.critic(out.flatten(end_dim=1))

        return action, value, log_prob

    def evaluate_action(self, obs, action, old_hidden_memory, masks):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs) 
        
        seq_len = feature_vector.size(1)
        hidden_memories = []
        # NOTE: We use masks to zero out hidden memory if last 
        # step belongs to previous episode. Mask_{t-1}*hidden_memory_t
        masks = torch.cat((torch.ones(masks.size(0), 1), masks), dim=1)
        hidden_memory = old_hidden_memory[:,0].unsqueeze(0)
        for t in range(seq_len):
            out, hidden_memory = self.gru(
                feature_vector[:,t].unsqueeze(1), 
                hidden_memory*masks[:,t].view(1,-1,1)
                )
            hidden_memories.append(out)
        hidden_memories = torch.cat(hidden_memories, 1).flatten(end_dim=1)
        log_probs = self.actor(hidden_memories)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(action)
        entropy = action_dist.entropy()

        value = self.critic(hidden_memories)

        return value, log_prob, entropy

    def evaluate_value(self, obs):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)
        out, _ = self.gru(feature_vector, self.hidden_memory)
        value = self.critic(out.flatten(end_dim=1))
        return value

    def reset_hidden_memory(self, worker):
        
        self.hidden_memory[:, worker] = torch.zeros(
            1, 1, self.hidden_memory_size
            )

    def get_architecture(self):
        return 'GRU'
        

class ModifiedGRUPolicy(nn.Module):

    def __init__(self, obs_size, action_size, num_workers):
        super(ModifiedGRUPolicy, self).__init__()
        self.num_workers = num_workers
        self.recurrent = True
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
                nn.Linear(HIDDEN_SIZE+HIDDEN_MEMORY_SIZE, HIDDEN_MEMORY_SIZE),
                nn.ReLU()
                )
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

        out  = self.fnn2(torch.cat((feature_vector, self.hidden_memory.transpose(0,1)), 2)).flatten(end_dim=1)
        _, self.hidden_memory = self.gru(feature_vector, self.hidden_memory)

        logits = self.actor(out)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        value = self.critic(out)

        return action, value, log_prob

    def evaluate_action(self, obs, action, old_hidden_memory, masks):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs) 

        seq_len = feature_vector.size(1)
        out = []
        # NOTE: We use masks to zero out hidden memory if last 
        # step belongs to previous episode. Mask_{t-1}*hidden_memory_t
        masks = torch.cat((torch.ones(masks.size(0), 1), masks), dim=1)
        hidden_memory = old_hidden_memory[:,0].unsqueeze(0)
        for t in range(seq_len):
            hidden_memory = hidden_memory*masks[:,t].view(1,-1,1)
            out.append(self.fnn2(torch.cat(
                (feature_vector[:,t].unsqueeze(1), 
                 hidden_memory.transpose(0,1))
                ,2)))
            _, hidden_memory = self.gru(
                feature_vector[:,t].unsqueeze(1), 
                hidden_memory
                )
        out = torch.cat(out, 1).flatten(end_dim=1)
        log_probs = self.actor(out)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(action)
        entropy = action_dist.entropy()

        value = self.critic(out)

        return value, log_prob, entropy

    def evaluate_value(self, obs):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)
        out  = self.fnn2(torch.cat((feature_vector, self.hidden_memory.transpose(0,1)), 2)).flatten(end_dim=1)
        value = self.critic(out)
        return value

    def reset_hidden_memory(self, worker):
        
        self.hidden_memory[:, worker] = torch.zeros(
            1, 1, self.hidden_memory_size
            )
    
    def get_architecture(self):
        return 'ModifiedGRU'

class FNNPolicy(nn.Module):

    def __init__(self, obs_size, action_size, num_workers):
        super(FNNPolicy, self).__init__()
        self.num_workers = num_workers
        self.recurrent = False
        if isinstance(obs_size, list):
            self.cnn = CNN(obs_size)
            self.image = True
        else:
            self.fnn = nn.Sequential(
                nn.Linear(obs_size, HIDDEN_SIZE),
                nn.ReLU()
                )
            self.image = False
        self.fnn2 = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_MEMORY_SIZE),
            nn.ReLU()
            )
        self.actor = nn.Linear(HIDDEN_MEMORY_SIZE, action_size)
        self.critic = nn.Linear(HIDDEN_MEMORY_SIZE, 1)

    
    def forward(self, obs):

        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)

        out = self.fnn2(feature_vector)
        
        logits = self.actor(out)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        value = self.critic(out)

        return action, value, log_prob

    
    def evaluate_action(self, obs, action, hidden_memory, masks):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs) 

        out = self.fnn2(feature_vector).flatten(end_dim=1)
        log_probs = self.actor(out)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(action)
        entropy = action_dist.entropy()

        value = self.critic(out)
        return value, log_prob, entropy

    
    def evaluate_value(self, obs):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)
        out = self.fnn2(feature_vector)
        value = self.critic(out)
        
        return value

    def get_architecture(self):
        return 'FNN'



class IAMPolicy(nn.Module):

    def __init__(self, obs_size, action_size, num_workers):
        super(IAMPolicy, self).__init__()
        self.num_workers = num_workers
        self.recurrent = True
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
                nn.Linear(HIDDEN_SIZE, HIDDEN_MEMORY_SIZE//2),
                nn.ReLU()
                )
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_MEMORY_SIZE//2, batch_first=True)
        self.actor = nn.Linear(HIDDEN_MEMORY_SIZE, action_size)
        self.critic = nn.Linear(HIDDEN_MEMORY_SIZE, 1)
        self.hidden_memory_size = HIDDEN_MEMORY_SIZE//2
        self.hidden_memory = torch.zeros(1, 
            self.num_workers,
            self.hidden_memory_size
            )

    
    def forward(self, obs):
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)
        fnn_out  = self.fnn2(feature_vector)
        gru_out, self.hidden_memory = self.gru(feature_vector, self.hidden_memory)
        out = torch.cat((fnn_out, gru_out), 2).flatten(end_dim=1)

        logits = self.actor(out)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        value = self.critic(out)

        return action, value, log_prob

    
    def evaluate_action(self, obs, action, old_hidden_memory, masks):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs) 

        seq_len = feature_vector.size(1)
        out = []
        # NOTE: We use masks to zero out hidden memory if last 
        # step belongs to previous episode. Mask_{t-1}*hidden_memory_t
        masks = torch.cat((torch.ones(masks.size(0), 1), masks), dim=1)
        hidden_memory = old_hidden_memory[:,0].unsqueeze(0)
        for t in range(seq_len):
            fnn_out  = self.fnn2(feature_vector[:,t].unsqueeze(1))
            hidden_memory = hidden_memory*masks[:,t].view(1,-1,1)
            gru_out, hidden_memory = self.gru(
                feature_vector[:,t].unsqueeze(1), 
                hidden_memory
                )
            out.append(torch.cat((fnn_out, gru_out), 2))
        out = torch.cat(out, 1).flatten(end_dim=1)
        log_probs = self.actor(out)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(action)
        entropy = action_dist.entropy()

        value = self.critic(out)

        return value, log_prob, entropy

    
    def evaluate_value(self, obs):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)

        fnn_out  = self.fnn2(feature_vector)
        gru_out, _ = self.gru(feature_vector, self.hidden_memory)
        out = torch.cat((fnn_out, gru_out), 2).flatten(end_dim=1)
        value = self.critic(out)

        return value

    
    def reset_hidden_memory(self, worker):
        
        self.hidden_memory[:, worker] = torch.zeros(
            1, 1, self.hidden_memory_size
            )
    
    def get_architecture(self):
        return 'IAM'