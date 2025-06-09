import numpy as np
from .reward_function_base import BaseRewardFunction

class AltitudeReward(BaseRewardFunction):
    """
    AltitudeReward（高度奖励函数）
    如果当前战机不满足一些高度相关的安全约束，则给予惩罚（通常是负值）。
    - 当飞行高度低于安全高度时，根据垂直速度给予惩罚（范围：[-1, 0]）
    - 当飞行高度低于危险高度时，根据高度给予惩罚（范围：[-1, 0]）
    """
    def __init__(self, config):
        super().__init__(config)
        # 安全高度（单位：千米），低于此高度会根据垂直速度惩罚
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)
        # 危险高度（单位：千米），低于此高度会给予严重惩罚
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)
        # 垂直速度惩罚系数（单位：马赫）
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)

        # 奖励项的名称，用于日志记录等
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        """
        奖励为所有惩罚项的总和。

        参数:
            task: 当前任务实例
            env: 当前环境实例
            agent_id: 当前智能体ID

        返回:
            float: 总奖励值
        """
        # 获取自身的飞行高度（单位：千米）
        ego_z = env.agents[agent_id].get_position()[-1] / 1000
        # 获取自身的垂直速度（单位：马赫）
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340

        # 垂直速度惩罚项
        Pv = 0.
        if ego_z <= self.safe_altitude:
            Pv = -np.clip(
                ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude,
                0., 1.
            )

        # 高度惩罚项
        PH = 0.
        if ego_z <= self.danger_altitude:
            PH = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.

        # 奖励为两个惩罚项的总和
        new_reward = (Pv + PH)*5

        # 调用基类中的处理函数，可能包括记录日志等操作
        return self._process(new_reward, agent_id, (Pv, PH))
