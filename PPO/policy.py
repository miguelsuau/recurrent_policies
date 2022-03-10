import sys
sys.path.append("..") 
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

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

    def __init__(self, obs_size, action_size, hidden_size, hidden_size_2, num_workers):
        super(GRUPolicy, self).__init__()
        self.num_workers = num_workers
        self.recurrent = True
        self.gru = nn.GRU(obs_size, hidden_size, batch_first=True)
        self.fnn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size_2),
                nn.ReLU()
                )
        self.actor = nn.Linear(hidden_size_2, action_size)
        self.critic = nn.Linear(hidden_size_2, 1)
        self.hidden_memory_size = hidden_size
        self.hidden_memory = torch.zeros(1, 
            self.num_workers,
            self.hidden_memory_size
            )

    def forward(self, obs):

        out, self.hidden_memory = self.gru(obs, self.hidden_memory)
        out = self.fnn(out)
        logits = self.actor(out.flatten(end_dim=1))
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        value = self.critic(out.flatten(end_dim=1))

        return action, value, log_prob

    def evaluate_action(self, obs, action, old_hidden_memory, masks):
        

        seq_len = obs.size(1)
        hidden_memories = []
        # NOTE: We use masks to zero out hidden memory if last 
        # step belongs to previous episode. Mask_{t-1}*hidden_memory_t
        masks = torch.cat((torch.ones(masks.size(0), 1), masks), dim=1)
        hidden_memory = old_hidden_memory[:,0].unsqueeze(0)
        for t in range(seq_len):
            out, hidden_memory = self.gru(
                obs[:,t].unsqueeze(1), 
                hidden_memory*masks[:,t].view(1,-1,1)
                )
            hidden_memories.append(out)
        hidden_memories = torch.cat(hidden_memories, 1).flatten(end_dim=1)
        hidden_memories = self.fnn(hidden_memories)
        log_probs = self.actor(hidden_memories)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(action)
        entropy = action_dist.entropy()

        value = self.critic(hidden_memories)

        return value, log_prob, entropy

    def evaluate_value(self, obs):
        out, _ = self.gru(obs, self.hidden_memory)
        out = self.fnn(out)
        value = self.critic(out.flatten(end_dim=1))
        return value

    def reset_hidden_memory(self, worker):
        self.hidden_memory[:, worker] = torch.zeros(
            1, 1, self.hidden_memory_size
            )

    def get_architecture(self):
        return 'GRU'
        
class FNNPolicy(nn.Module):

    def __init__(self, obs_size, action_size, hidden_size, hidden_size_2, num_workers):
        super(FNNPolicy, self).__init__()
        self.num_workers = num_workers
        self.recurrent = False
        if isinstance(obs_size, list):
            self.cnn = CNN(obs_size)
            self.image = True
        else:
            self.fnn = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU()
                )
            self.image = False
        self.fnn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size_2),
            nn.ReLU()
            )
        self.actor = nn.Linear(hidden_size_2, action_size)
        self.critic = nn.Linear(hidden_size_2, 1)

    
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

class IAMGRUPolicy(nn.Module):

    def __init__(self, obs_size, action_size, hidden_size, hidden_size_2, hidden_memory_size, num_workers, dset=None, dset_size=0):
        super(IAMGRUPolicy, self).__init__()
        self.num_workers = num_workers
        self.recurrent = True
        if dset is not None:
            if isinstance(obs_size, list):
                self.cnn = CNN(obs_size)
                self.image = True
            else:
                self.image = False
                self.fnn = nn.Sequential(
                    nn.Linear(obs_size, hidden_size),
                    nn.ReLU()
                    )
            self.gru = nn.GRU(len(dset), hidden_memory_size, batch_first=True)
        else:
            if isinstance(obs_size, list):
                self.cnn = CNN(obs_size)
                self.image = True
            else:
                self.image = False
                self.fnn = nn.Sequential(
                    nn.Linear(obs_size, hidden_size),
                    nn.ReLU()
                    )
                self.dhat = nn.Linear(obs_size, dset_size)
            self.gru = nn.GRU(dset_size, hidden_memory_size, batch_first=True)

        self.fnn2 = nn.Sequential(
                nn.Linear(hidden_size + hidden_memory_size, hidden_size_2),
                nn.ReLU()
                )

        self.actor = nn.Linear(hidden_size_2, action_size)
        self.critic = nn.Linear(hidden_size_2, 1)
        self.hidden_memory_size = hidden_memory_size
        self.hidden_memory = torch.zeros(1, 
            self.num_workers,
            self.hidden_memory_size
            )
        self.dset = dset

    
    def forward(self, obs):
        if self.dset is not None:
            # nondset_mask = np.ones(obs.shape[2], np.bool)
            # nondset_mask[self.dset] = 0
            if self.image:
                feature_vector = self.cnn(obs)
            else:
                feature_vector = self.fnn(obs)
            dset = obs[:, :, self.dset]
        else:
            if self.image:
                feature_vector = self.cnn(obs)
            else:
                feature_vector = self.fnn(obs)
            # dset = self.dhat(obs)
            dset = obs
        
        gru_out, self.hidden_memory = self.gru(dset, self.hidden_memory)
        out = torch.cat((feature_vector, gru_out), 2).flatten(end_dim=1)
        out  = self.fnn2(out)

        logits = self.actor(out)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        value = self.critic(out)

        return action, value, log_prob

    
    def evaluate_action(self, obs, action, old_hidden_memory, masks):
        
        if self.dset is not None:
            # nondset_mask = np.ones(obs.shape[2], np.bool)
            # nondset_mask[self.dset] = 0
            if self.image:
                feature_vector = self.cnn(obs)
            else:
                feature_vector = self.fnn(obs)
            dset = obs[:, :, self.dset] 
        else:
            if self.image:
                feature_vector = self.cnn(obs)
            else:
                feature_vector = self.fnn(obs)
            # dset = self.dhat(obs)
            dset = obs

        seq_len = feature_vector.size(1)
        out = []
        # NOTE: We use masks to zero out hidden memory if last 
        # step belongs to previous episode. Mask_{t-1}*hidden_memory_t
        masks = torch.cat((torch.ones(masks.size(0), 1), masks), dim=1)
        hidden_memory = old_hidden_memory[:,0].unsqueeze(0)
        for t in range(seq_len):
            hidden_memory = hidden_memory*masks[:,t].view(1,-1,1)
            gru_out, hidden_memory = self.gru(
                dset[:,t].unsqueeze(1), 
                hidden_memory
                )
            out.append(torch.cat((feature_vector[:,t].unsqueeze(1), gru_out), 2))
        out = torch.cat(out, 1).flatten(end_dim=1)
        out = self.fnn2(out)
        log_probs = self.actor(out)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(action)
        entropy = action_dist.entropy()

        value = self.critic(out)

        return value, log_prob, entropy

    
    def evaluate_value(self, obs):
        
        if self.dset is not None:
            # nondset_mask = np.ones(obs.shape[2], np.bool)
            # nondset_mask[self.dset] = 0
            if self.image:
                feature_vector = self.cnn(obs)
            else:
                feature_vector = self.fnn(obs)
            dset = obs[:, :, self.dset]
        else:
            if self.image:
                feature_vector = self.cnn(obs)
            else:
                feature_vector = self.fnn(obs)
            # dset = self.dhat(obs)
            dset = obs
            
        gru_out, _ = self.gru(dset, self.hidden_memory)
        out = torch.cat((feature_vector, gru_out), 2).flatten(end_dim=1)
        out  = self.fnn2(out)
        value = self.critic(out)

        return value

    
    def reset_hidden_memory(self, worker):
        
        self.hidden_memory[:, worker] = torch.zeros(
            1, 1, self.hidden_memory_size
            )
    
    def get_architecture(self):
        return 'IAMGRU'

class IAMGRUPolicy_dynamic(nn.Module):

    def __init__(self, obs_size, action_size, hidden_size, hidden_size_2, hidden_memory_size, context_size, num_workers, dset=None, dset_size=0):
        super(IAMGRUPolicy_dynamic, self).__init__()
        self.num_workers = num_workers
        self.recurrent = True
    
        if isinstance(obs_size, list):
            self.cnn = CNN(obs_size)
            self.image = True
        else:
            self.image = False
            self.fnn = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU()
                )

        self.context = nn.Sequential(
            nn.Linear(hidden_memory_size + 1, context_size),
            nn.Tanh()            
            )
        self.attention = nn.Sequential(
            nn.Linear(context_size, 1),
            nn.Softmax(dim=-2)
        )

        self.gru = nn.GRU(1, hidden_memory_size, batch_first=True)

        self.fnn2 = nn.Sequential(
                nn.Linear(hidden_size + hidden_memory_size, hidden_size_2),
                nn.ReLU()
                )

        self.actor = nn.Linear(hidden_size_2, action_size)
        self.critic = nn.Linear(hidden_size_2, 1)
        self.hidden_memory_size = hidden_memory_size
        self.hidden_memory = torch.zeros(1, 
            self.num_workers,
            self.hidden_memory_size
            )
        self.dset = dset

    
    def forward(self, obs):
        
        if self.image:
            feature_vector = self.cnn(obs)
        else:
            feature_vector = self.fnn(obs)
        
        # attention
        obs_size = obs.shape[-1]
        context_input = torch.cat((obs.unsqueeze(-1), torch.cat([torch.swapaxes(self.hidden_memory, 0, 1).unsqueeze(-2)]*obs_size, -2)), -1)
        context_out = self.context(context_input)
        attention_weights = self.attention(context_out).squeeze(-1)
        dset = torch.sum(attention_weights*obs, dim=-1, keepdim=True)
        
        gru_out, self.hidden_memory = self.gru(dset, self.hidden_memory)
        out = torch.cat((feature_vector, gru_out), 2).flatten(end_dim=1)
        out  = self.fnn2(out)

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
            # hidden_memory2 = hidden_memory*masks[:,t].view(1, -1, 1)

            # attention
            obs_size = obs.shape[-1]
            context_input = torch.cat((obs[:,t].unsqueeze(1).unsqueeze(-1), torch.cat([torch.swapaxes(hidden_memory, 0, 1).unsqueeze(-2)]*obs_size, -2)), -1)
            context_out = self.context(context_input)
            attention_weights = self.attention(context_out).squeeze(-1)
            dset = torch.sum(attention_weights*obs[:,t].unsqueeze(1), dim=-1, keepdim=True)
            gru_out, hidden_memory = self.gru(dset, hidden_memory)
            out.append(torch.cat((feature_vector[:,t].unsqueeze(1), gru_out), 2))

        out = torch.cat(out, 1).flatten(end_dim=1)
        out = self.fnn2(out)
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
        
        # attention
        obs_size = obs.shape[-1]
        context_input = torch.cat((obs.unsqueeze(-1), torch.cat([torch.swapaxes(self.hidden_memory, 0, 1).unsqueeze(-2)]*obs_size, -2)), -1)
        context_out = self.context(context_input)
        attention_weights = self.attention(context_out).squeeze(-1)
        dset = torch.sum(attention_weights*obs, dim=-1, keepdim=True)
            
        gru_out, _ = self.gru(dset, self.hidden_memory)
        out = torch.cat((feature_vector, gru_out), 2).flatten(end_dim=1)
        out  = self.fnn2(out)
        value = self.critic(out)

        return value

    
    def reset_hidden_memory(self, worker):
        
        self.hidden_memory[:, worker] = torch.zeros(
            1, 1, self.hidden_memory_size
            )
    
    def get_architecture(self):
        return 'IAMGRU'


class LSTMPolicy(nn.Module):

    def __init__(self, obs_size, action_size, hidden_size, hidden_size_2, num_workers):
        super(LSTMPolicy, self).__init__()
        self.num_workers = num_workers
        self.recurrent = True
        self.lstm = nn.LSTM(hidden_size, hidden_size_2, batch_first=True)
        self.fnn = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU()
                )
        self.actor = nn.Linear(hidden_size_2, action_size)
        self.critic = nn.Linear(hidden_size_2, 1)
        self.hidden_memory_size = hidden_size_2
        h = torch.zeros(1, self.num_workers, self.hidden_memory_size)
        c = torch.zeros(1, self.num_workers, self.hidden_memory_size)
        self.hidden_memory = (h, c)

    def forward(self, obs):
        out = self.fnn(obs)
        out, self.hidden_memory = self.lstm(out, self.hidden_memory)
        
        logits = self.actor(out.flatten(end_dim=1))
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        value = self.critic(out.flatten(end_dim=1))

        return action, value, log_prob

    def evaluate_action(self, obs, action, old_hidden_memory, masks):
        

        seq_len = obs.size(1)
        hidden_memories = []
        # NOTE: We use masks to zero out hidden memory if last 
        # step belongs to previous episode. Mask_{t-1}*hidden_memory_t
        masks = torch.cat((torch.ones(masks.size(0), 1), masks), dim=1)
        hidden_memory = (old_hidden_memory[0][:,0].unsqueeze(0),
            old_hidden_memory[1][:,0].unsqueeze(0))
        fnn_out = self.fnn(obs)
        for t in range(seq_len):
            hidden_memory = (hidden_memory[0]*masks[:,t].view(1,-1,1), 
                hidden_memory[1]*masks[:,t].view(1,-1,1))
            out, hidden_memory = self.lstm(
                fnn_out[:,t].unsqueeze(1), 
                hidden_memory
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
        out = self.fnn(obs)
        out, _ = self.lstm(out, self.hidden_memory)
        value = self.critic(out.flatten(end_dim=1))
        return value

    def reset_hidden_memory(self, worker):
        self.hidden_memory[1][:, worker] = torch.zeros(1, 1, self.hidden_memory_size)
        self.hidden_memory[0][:, worker] = torch.zeros(1, 1, self.hidden_memory_size)

    def get_architecture(self):
        return 'LSTM'



class IAMLSTMPolicy(nn.Module):
    
    def __init__(self, obs_size, action_size, hidden_size, hidden_size_2, hidden_memory_size, num_workers, dset=None, dset_size=0):
        super(IAMLSTMPolicy, self).__init__()
        self.num_workers = num_workers
        self.recurrent = True
        
        if dset is not None:        
            self.fnn = nn.Sequential(
                nn.Linear(obs_size, hidden_size, bias=True),
                nn.ReLU()
                )
            self.lstm = nn.LSTM(len(dset), hidden_memory_size, batch_first=True)
        else:
            self.fnn = nn.Sequential(
                nn.Linear(obs_size, hidden_size, bias=True),
                nn.ReLU()
                )
            self.lstm = nn.LSTM(obs_size, hidden_memory_size, batch_first=True)
        self.fnn2 = nn.Sequential(
                nn.Linear(hidden_memory_size + hidden_size, hidden_size_2, bias=True),
                nn.ReLU()
                )
        self.actor = nn.Linear(hidden_size_2, action_size, bias=True)
        self.critic = nn.Linear(hidden_size_2, 1)
        self.hidden_memory_size = hidden_memory_size
        h = torch.zeros(1, self.num_workers, self.hidden_memory_size)
        c = torch.zeros(1, self.num_workers, self.hidden_memory_size)
        self.hidden_memory = (h, c)
        self.dset = dset

    
    def forward(self, obs):
        if self.dset is not None:
            # nondset_mask = np.ones(obs.shape[2], np.bool)
            # nondset_mask[self.dset] = 0
            feature_vector = self.fnn(obs)#[:, :, nondset_mask])
            # feature_vector = self.fnn3(feature_vector)
            dset = obs[:, :, self.dset]
        else:
            feature_vector = self.fnn(obs)
            dset = obs
        lstm_out, self.hidden_memory = self.lstm(dset, self.hidden_memory)
        out = torch.cat((feature_vector, lstm_out), 2).flatten(end_dim=1)
        out = self.fnn2(out)

        logits = self.actor(out)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        value = self.critic(out)

        return action, value, log_prob

    
    def evaluate_action(self, obs, action, old_hidden_memory, masks):
        
        if self.dset is not None:
            # nondset_mask = np.ones(obs.shape[2], np.bool)
            # nondset_mask[self.dset] = 0
            feature_vector = self.fnn(obs)#[:, :, nondset_mask])
            # feature_vector = self.fnn3(feature_vector)
            dset = obs[:, :, self.dset] 
        else:
            feature_vector = self.fnn(obs)
            dset = obs
        seq_len = feature_vector.size(1)
        hidden_memories = []
        # NOTE: We use masks to zero out hidden memory if last 
        # step belongs to previous episode. Mask_{t-1}*hidden_memory_t
        masks = torch.cat((torch.ones(masks.size(0), 1), masks), dim=1)
        hidden_memory = (old_hidden_memory[0][:,0].unsqueeze(0),
            old_hidden_memory[1][:,0].unsqueeze(0))
        for t in range(seq_len):
            hidden_memory = (hidden_memory[0]*masks[:,t].view(1,-1,1), 
                hidden_memory[1]*masks[:,t].view(1,-1,1))
            lstm_out, hidden_memory = self.lstm(
                dset[:,t].unsqueeze(1), 
                hidden_memory
                )
            hidden_memories.append(lstm_out)
        out = torch.cat((feature_vector, torch.cat(hidden_memories, 1)), 2).flatten(end_dim=1)
        out = self.fnn2(out)
        log_probs = self.actor(out)
        action_dist = Categorical(logits=log_probs)
        log_prob =  action_dist.log_prob(action)
        entropy = action_dist.entropy()
        value = self.critic(out)

        return value, log_prob, entropy

    
    def evaluate_value(self, obs):
        
        if self.dset is not None:
            # nondset_mask = np.ones(obs.shape[2], np.bool)
            # nondset_mask[self.dset] = 0
            feature_vector = self.fnn(obs)#[:, :, nondset_mask])
            # feature_vector = self.fnn3(feature_vector)
            dset = obs[:, :, self.dset]
        else:
            feature_vector = self.fnn(obs)
            dset = obs
            
        lstm_out, _ = self.lstm(dset, self.hidden_memory)
        out = torch.cat((feature_vector, lstm_out), 2).flatten(end_dim=1)
        out = self.fnn2(out)
        value = self.critic(out)
        return value

    
    def reset_hidden_memory(self, worker):
        self.hidden_memory[1][:, worker] = torch.zeros(1, 1, self.hidden_memory_size)
        self.hidden_memory[0][:, worker] = torch.zeros(1, 1, self.hidden_memory_size)

    def get_architecture(self):
        return 'IAM'

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)