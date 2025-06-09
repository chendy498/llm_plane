import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, args):
        self.capacity = int(capacity)
        self.device = args['device']

        self.obs_buf = np.zeros((self.capacity, args['obs_dim']), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, args['obs_dim']), dtype=np.float32)
        self.acts_buf = np.zeros((self.capacity, args['act_dim']), dtype=np.float32)
        self.rews_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = torch.tensor(self.obs_buf[idxs], dtype=torch.float32).to(self.device)
        acts = torch.tensor(self.acts_buf[idxs], dtype=torch.float32).to(self.device)
        rews = torch.tensor(self.rews_buf[idxs], dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32).to(self.device)
        done = torch.tensor(self.done_buf[idxs], dtype=torch.float32).to(self.device)

        return obs, acts, rews, next_obs, done
