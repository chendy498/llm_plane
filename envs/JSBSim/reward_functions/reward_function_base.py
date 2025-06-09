import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    奖励函数的基类。具体的奖励逻辑由子类实现。
    """

    def __init__(self, config):
        self.config = config  # 奖励配置参数
        self.reward_scale = getattr(config, f'{self.__class__.__name__}_scale', 1.0)  # 奖励缩放因子
        self.is_potential = getattr(config, f'{self.__class__.__name__}_potential',
                                    False)  # 是否使用势函数（Potential-based reward）
        self.pre_rewards = defaultdict(float)  # 用于势函数差分
        self.reward_trajectory = defaultdict(list)  # 存储奖励轨迹
        self.reward_item_names = [self.__class__.__name__]  # 奖励条目名称（用于记录多个子奖励）

    def reset(self, task, env):
        """Perform reward function-specific reset after episode reset.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance
        """
        if self.is_potential:
            self.pre_rewards.clear()
            for agent_id in env.agents.keys():
                self.pre_rewards[agent_id] = self.get_reward(task, env, agent_id)
        self.reward_trajectory.clear()

    @abstractmethod
    def get_reward(self, task, env, agent_id):
        """Compute the reward at the current timestep.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        raise NotImplementedError

    def _process(self, new_reward, agent_id, render_items=()):
        """
        对计算出的奖励进行后处理：
        - 缩放
        - 势函数差分（可选）
        - 添加到奖励轨迹中

        参数：
            new_reward: 原始奖励
            agent_id: 智能体ID
            render_items: 可视化项（如果包含多个子奖励项时使用）

        返回：
            最终奖励值
        """
        reward = new_reward * self.reward_scale

        if self.is_potential:
            reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward

        self.reward_trajectory[agent_id].append([reward, *render_items])
        return reward

    def get_reward_trajectory(self):
        """
        获取当前episode中每个奖励项的历史轨迹。

        返回：
            一个字典：{奖励名称: numpy数组[time, agent_id, value]}
        """
        return dict(zip(self.reward_item_names, np.array(self.reward_trajectory.values()).transpose(2, 0, 1)))
