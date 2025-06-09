import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class PostureReward(BaseRewardFunction):
    """
    PostureReward（姿态奖励函数） = Orientation * Range
    - Orientation：鼓励飞机朝向敌人，惩罚被敌人瞄准
    - Range：鼓励接近敌人，远离则惩罚
    注意：
    - 仅适用于一对一空战场景
    """

    def __init__(self, config):
        super().__init__(config)
        # 设置版本控制，可在 config 中传参选择函数版本
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)  # 理想距离（单位 km）

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)

        # 日志中保存的奖励项名称
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    def get_reward(self, task, env, agent_id):
        """
        计算智能体的姿态奖励。

        输入：
        - task: 当前任务实例
        - env: 仿真环境
        - agent_id: 智能体编号

        输出：
        - 最终奖励值：姿态奖励 × 距离奖励
        """
        new_reward = 0

        # 获取当前智能体的位置 + 速度（拼接成一个特征向量）
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        # 遍历敌方目标（默认只有一个敌人）
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])

            # 获取角度信息（AO、TA）与距离R
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            # 计算朝向奖励
            orientation_reward = self.orientation_fn(AO, TA)

            # 计算距离奖励（单位转为 km）
            range_reward = self.range_fn(R / 1000)

            # 最终奖励 = 朝向 × 距离
            new_reward += orientation_reward * range_reward

        return self._process(new_reward, agent_id, (orientation_reward, range_reward))

    def get_orientation_function(self, version):
        if version == 'v0':
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")

    def get_range_funtion(self, version):
        if version == 'v0':
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")
