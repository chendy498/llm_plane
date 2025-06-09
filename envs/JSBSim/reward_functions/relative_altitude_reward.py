import numpy as np
from .reward_function_base import BaseRewardFunction

class RelativeAltitudeReward(BaseRewardFunction):
    """
    RelativeAltitudeReward（相对高度惩罚）
    当当前飞机与敌机的高度差大于设定阈值（默认 1000 米）时，给予惩罚。
    奖励值范围：[-1, 0]，一般为负值。

    注意：
    - 仅支持一对一环境（即一个敌人）。
    """
    def __init__(self, config):
        super().__init__(config)
        self.KH = getattr(self.config, f'{self.__class__.__name__}_KH', 1.0)  # 高度容差上限，单位：km

    def get_reward(self, task, env, agent_id):
        """
        根据我方与敌方的高度差计算惩罚。

        参数：
            task: 当前任务
            env: 当前环境实例
            agent_id: 智能体ID

        返回：
            new_reward (float): 惩罚值，范围为 [-∞, 0]
        """
        # 获取我方飞机的高度（单位：km，注意是负的，因为 z 向下）
        ego_z = env.agents[agent_id].get_position()[-1] / 1000

        # 获取敌方飞机的高度（单位：km）
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000

        # 计算高度差，并根据 KH 计算惩罚
        # 若 |高度差| < KH，奖励为 0；否则为负值，惩罚程度随超出幅度加重
        new_reward = min(self.KH - np.abs(ego_z - enm_z), 0)

        # 调用父类的 reward 处理方法（记录、缩放等）
        return self._process(new_reward, agent_id)

