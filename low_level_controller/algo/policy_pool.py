import copy
import random
import torch

class PolicyPool:
    def __init__(self, max_size=5):
        self.policies = []
        self.elos = []
        self.max_size = max_size

    def add(self, policy, elo=1000):
        """添加新策略到池中，如果池满则替换最低 ELO 的策略"""
        if len(self.policies) >= self.max_size:
            idx = self.elos.index(min(self.elos))
            del self.policies[idx]
            del self.elos[idx]
        self.policies.append(copy.deepcopy(policy))
        self.elos.append(elo)

    def sample(self):
        """随机采样一个策略"""
        if not self.policies:
            raise ValueError("Policy pool is empty")
        return random.choice(self.policies)

    def update_elo(self, idx, new_elo):
        """更新指定索引的 ELO 分数"""
        if 0 <= idx < len(self.elos):
            self.elos[idx] = new_elo

    def get_best(self):
        """获取 ELO 最高的策略"""
        if not self.policies:
            return None
        return self.policies[self.elos.index(max(self.elos))]

    def size(self):
        """返回池中策略数量"""
        return len(self.policies)

    def save(self, path):
        """保存策略池到文件"""
        torch.save({
            'policies': [policy.state_dict() for policy in self.policies],
            'elos': self.elos
        }, path)

    def load(self, path, policy_class, args):
        """从文件加载策略池"""
        checkpoint = torch.load(path)
        self.policies = []
        self.elos = checkpoint['elos']
        for state_dict in checkpoint['policies']:
            policy = policy_class(obs_dim=args['obs_dim'], act_dim=args['act_dim'], args=args)
            policy.load_state_dict(state_dict)
            self.policies.append(policy)