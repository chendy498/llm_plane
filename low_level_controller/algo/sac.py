import torch
import torch.nn.functional as F
from low_level_controller.algo.networks import GaussianPolicy, MLP
import os
from datetime import datetime

class SAC:
    def __init__(self, obs_dim, act_dim, args):
        self.actor = GaussianPolicy(obs_dim, act_dim).to(args["device"])
        self.critic1 = MLP(obs_dim + act_dim, 1).to(args["device"])
        self.critic2 = MLP(obs_dim + act_dim, 1).to(args["device"])
        self.critic1_target = MLP(obs_dim + act_dim, 1).to(args["device"])
        self.critic2_target = MLP(obs_dim + act_dim, 1).to(args["device"])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args["lr"])
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=args["lr"])
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=args["lr"])

        self.gamma = args["gamma"]
        self.tau = args["tau"]
        self.alpha = args["alpha"]

    def update(self, replay_buffer, batch_size):
        obs, act, rew, next_obs, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_act, next_log_prob = self.actor.sample(next_obs)
            q1_t = self.critic1_target(torch.cat([next_obs, next_act], dim=1))
            q2_t = self.critic2_target(torch.cat([next_obs, next_act], dim=1))
            min_q = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
            q_target = rew + (1 - done) * self.gamma * min_q

        q1 = self.critic1(torch.cat([obs, act], dim=1))
        q2 = self.critic2(torch.cat([obs, act], dim=1))
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        act_sampled, log_prob = self.actor.sample(obs)
        q1_pi = self.critic1(torch.cat([obs, act_sampled], dim=1))
        q2_pi = self.critic2(torch.cat([obs, act_sampled], dim=1))
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, filepath, episode=None):
        """
        保存 SAC 模型的参数到指定文件路径。

        参数：
            filepath (str): 保存模型的文件路径（包含文件名，如 'model.pth'）。
            episode (int, optional): 当前训练的回合数，用于生成带回合数的文件名。
        """
        # 如果提供了 episode，则在文件名中添加回合数
        if episode is not None:
            base, ext = os.path.splitext(filepath)
            filepath = f"{base}_episode_{episode}{ext}"

        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic1_optim': self.critic1_optim.state_dict(),
            'critic2_optim': self.critic2_optim.state_dict(),
        }, filepath)
        print(f"[INFO] Model saved to {filepath}")

    def load_model(self, filepath):
        """
        从指定文件路径加载 SAC 模型的参数。

        参数：
            filepath (str): 保存模型的文件路径（如 'model.pth'）。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[ERROR] 模型文件未找到: {filepath}")

        checkpoint = torch.load(filepath, map_location='cpu')  # 加载 checkpoint，兼容不同设备

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic1_optim.load_state_dict(checkpoint['critic1_optim'])
        self.critic2_optim.load_state_dict(checkpoint['critic2_optim'])

        print(f"[INFO] Model loaded from {filepath}")
