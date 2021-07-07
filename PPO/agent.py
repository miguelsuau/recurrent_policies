import sys
sys.path.append("..") 
from recurrent_policies.PPO.buffer import Buffer
import numpy as np
import torch
from torch.nn import functional as F
from recurrent_policies.PPO.utils import LinearSchedule, LRLinearSchedule

class Agent(object):
    """
    Agent
    """
    def __init__(
        self,
        policy, 
        memory_size = 128,
        batch_size = 32,
        seq_len = 8,
        num_epoch = 4,
        learning_rate = 2.5e-4,
        total_steps = 2.0e6,
        clip_range = 0.2,
        entropy_coef = 1.0e-3
        ):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_epoch = num_epoch
        self.recurrent_policy = policy.recurrent
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.lr_schedule = LRLinearSchedule(self.optimizer, total_steps, learning_rate)
        self.clip_schedule = LinearSchedule(total_steps, clip_range, 1.0e-5)
        self.entropy_schedule = LinearSchedule(total_steps, entropy_coef, entropy_coef)
        self.buffer = Buffer(memory_size)
        self.step = 0
        

    def choose_action(self, obs):

        self.step += 1
        with torch.no_grad():
            obs = torch.FloatTensor(obs).unsqueeze(1)
            action, value, log_prob, hidden_memory = self.policy(obs)
            return action.flatten(), value, log_prob, hidden_memory

    def add_to_memory(
        self, prev_obs, action, reward, done, value, 
        log_probs, hidden_memory=None, prev_action=None
        ):

        self.buffer['obs'].append(prev_obs)
        self.buffer['actions'].append(action.flatten().detach().numpy())
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['masks'].append([1-d for d in done])
        self.buffer['values'].append(value.flatten().detach().numpy())
        self.buffer['log_probs'].append(log_probs.flatten().detach().numpy())
        if self.recurrent_policy:
            self.buffer['hidden_memories'].append(hidden_memory.squeeze(0).detach().numpy())
            # self.buffer['prev_action'].append(prev_action)
    
    def bootstrap(self, obs, time_horizon, gamma=0.99, lambd=0.95):
        """
        Computes GAE and returns for a given time horizon
        """
        obs = torch.FloatTensor(obs).unsqueeze(1)
        last_value = self.policy.evaluate_value(obs)
        batch = self.buffer.get_last_entries(
            time_horizon, 
            ['rewards', 'values','dones']
            )
        advantages = self._compute_advantages(
            np.array(batch['rewards']), 
            np.array(batch['values']), 
            np.array(batch['dones']),
            last_value.flatten().detach().numpy(), 
            gamma, lambd
            )
        self.buffer['advantages'].extend(advantages)
        returns = advantages + batch['values']
        self.buffer['returns'].extend(returns)

    def reset_hidden_memory(self, done):
        if self.recurrent_policy:
            for worker, done in enumerate(done):
                if done:
                    self.policy.reset_hidden_memory(worker)

    def update(self):
        """
        Runs multiple epoch of mini-batch gradient descent to
        update the model using experiences stored in buffer.
        """
        self.lr_schedule.update_learning_rate(self.step)
        clip_range = self.clip_schedule.update(self.step)
        entropy_coef = self.entropy_schedule.update(self.step)
        policy_loss = 0
        value_loss = 0
        n_batches = self.memory_size // self.batch_size
        for _ in range(self.num_epoch):
            self.buffer.shuffle(self.seq_len)
            for b in range(n_batches):
                batch = self.buffer.sample(b, self.batch_size, self.seq_len)
                mb_policy_loss, mb_value_loss = self._update_policy(batch, clip_range, entropy_coef)
                policy_loss += mb_policy_loss
                value_loss += mb_value_loss
        self.buffer.empty()

    ######################### Private Functions ###########################

    def _compute_advantages(
            self, reward, value, done, last_value, gamma, lambd
            ):
        """
        Calculates advantages using genralized advantage estimation (GAE)
        """
        last_advantage = 0
        shape = np.shape(reward)
        advantages = np.zeros((shape[0], shape[1]), dtype=np.float32)
        for t in reversed(range(shape[0])):
            mask = 1.0 - done[t, :]
            last_value = last_value*mask
            last_advantage = last_advantage*mask
            delta = reward[t, :] + gamma*last_value - value[t, :]
            last_advantage = delta + gamma*lambd*last_advantage
            advantages[t, :] = last_advantage
            last_value = value[t, :]
        return advantages

    def _update_policy(
            self, batch, clip_range=0.2, entropy_coef=1e-3, 
            value_coef=1.0, max_grad_norm=0.5
            ):
        
        obs = torch.FloatTensor(batch['obs']).flatten(end_dim=1)
        actions = torch.FloatTensor(batch['actions']).flatten(end_dim=1)
        if self.recurrent_policy:
            old_hidden_memories = torch.FloatTensor(batch['hidden_memories']).flatten(end_dim=1)[:, 0].unsqueeze(0)
        else:
            old_hidden_memories = None
        values, log_prob, entropy = self.policy.evaluate_action(
            obs, actions, old_hidden_memories
            )
        # Normalize advantage
        advantages = torch.FloatTensor(batch['advantages']).flatten(end_dim=1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # importance sampling ratio
        old_log_probs = torch.FloatTensor(batch['log_probs']).flatten(end_dim=1)
        ratio = torch.exp(log_prob - old_log_probs)

        # policy loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # value loss
        returns = torch.FloatTensor(batch['returns']).flatten(end_dim=1)
        old_values = torch.FloatTensor(batch['values']).flatten(end_dim=1)
        clipped_values = old_values + torch.clamp(
                values.squeeze() - old_values, -clip_range, clip_range
            )
        value_loss1 = F.mse_loss(returns, values.squeeze())
        value_loss2 = F.mse_loss(returns, clipped_values)
        value_loss = torch.max(value_loss1, value_loss2)

        # Entropy bonus
        entropy_bonus = -torch.mean(entropy)

        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_bonus

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.optimizer.step()
        return policy_loss, value_loss