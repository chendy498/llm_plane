import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_thetaT_thetaA_R, S  # 假设 S 是 sigmoid

class StableApproachReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__]
        self.pre_d = 5577  # 初始距离\
        self.count = 0  # 用于记录调用次数

    def reset(self, task, env):
        """
        重置函数，在每次仿真重启时调用，清空上一次记录的距离。
        """
        self.pre_d = 5577
        self.count = 0  # 重置调用次数
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        self.count+= 1  # 增加调用次数
        ego_feature = np.hstack([
            env.agents[agent_id].get_position(),
            env.agents[agent_id].get_velocity(),
            env.agents[agent_id].get_rpy()
        ])

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([
                enm.get_position(),
                enm.get_velocity(),
                enm.get_rpy()
            ])

            thetaT, thetaA, d = get_thetaT_thetaA_R(ego_feature, enm_feature)

            thetaT_bar = thetaT / np.pi
            thetaA_bar = thetaA / np.pi
            vc = -(d - self.pre_d) / 0.2  # 速度为负值代表正在接近
            if self.count % 2 == 0:
                # 每两次调用更新一次距离
                self.pre_d = d

            h = ego_feature[2]
            roll = ego_feature[6] / np.pi  # 滚转角归一化

            # 各部分奖励
            r_closure = self.r_closure(vc)
            r_stability = self.r_stability(roll, d)
            r_altitude = self.r_altitude(h)
            r_attack_angle = self.r_attack_angle(thetaT_bar, d)

            # 加权求和（避免某项主导）
            new_reward += (
                1.0 * r_closure +
                1.0 * r_stability +
                0.5 * r_altitude +
                1.0 * r_attack_angle
            )
        self.reward_trajectory[agent_id].append([new_reward])
        return new_reward

    def r_closure(self, vc):
        # 奖励接近敌人（vc > 0 时为负向接近）
        return np.clip(vc / 50.0, -2, 2)

    def r_stability(self, roll, d):
        # 平滑惩罚滚转角大，同时考虑离敌距离（越近越严格）
        penalty = (1 - S(1 - abs(roll), 1, 0.2))  # 滚转角越大，惩罚越大
        distance_factor = S(d, 1 / 100, 3000)  # 越yuan越大
        return 2.0 * (1 - penalty * distance_factor)

    def r_altitude(self, h):
        # 奖励保持在目标高度上方（目标高度约 3000m）
        return 2.0 * S(h, 1 / 100, 3000)

    def r_attack_angle(self, thetaT_bar, d):
        # 奖励保持较好的攻击角度，靠近敌人时更重要
        angle_quality = 1-S(thetaT_bar, 20, 0.5)  # 理想值在 0.5 左右
        proximity = 1 - S(d, 1 / 50, 3000)  # 越近，值越大
        return 2.0 * angle_quality * proximity
