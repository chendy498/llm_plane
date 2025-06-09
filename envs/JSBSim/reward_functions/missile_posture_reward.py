import numpy as np
from .reward_function_base import BaseRewardFunction

class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward（导弹姿态奖励）
    根据导弹的速度衰减（velocity attenuation）计算奖励。
    - 奖励的关键是评估导弹速度是否减弱以及导弹与飞机飞行方向的夹角。
    """
    def __init__(self, config):
        super().__init__(config)
        # 用于记录上一步的导弹速度
        self.previous_missile_v = None

    def reset(self, task, env):
        """
        重置函数，在每次仿真重启时调用，清空上一次记录的导弹速度。
        """
        self.previous_missile_v = None
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        计算当前时刻的奖励。

        参数:
            task: 当前任务实例
            env: 当前仿真环境
            agent_id: 智能体编号

        返回:
            float: 奖励值
        """

        reward = 0

        # 检查当前是否有导弹在飞向该智能体（返回导弹对象）
        missile_sim = env.agents[agent_id].check_missile_warning()

        if missile_sim is not None:
            # 获取导弹和飞机的速度向量
            missile_v = missile_sim.get_velocity()
            aircraft_v = env.agents[agent_id].get_velocity()

            # 如果是第一次检测到导弹，初始化导弹速度记录
            if self.previous_missile_v is None:
                self.previous_missile_v = missile_v

            # 计算导弹速度的衰减程度（速度差 / 340 m/s，单位归一化）
            v_decrease = (np.linalg.norm(self.previous_missile_v) - np.linalg.norm(missile_v)) / 340 * self.reward_scale

            # 计算导弹和飞机速度的夹角余弦值（方向对齐程度）
            angle = np.dot(missile_v, aircraft_v) / (np.linalg.norm(missile_v) * np.linalg.norm(aircraft_v))

            # 如果导弹朝向与飞机朝向夹角大于 90°（angle < 0），即导弹背离方向
            if angle < 0:
                reward = angle / (max(v_decrease, 0) + 1)
            else:
                # 否则，导弹追踪方向和飞机一致时，根据导弹速度减缓程度决定奖励强度
                reward = angle * max(v_decrease, 0)
        else:
            # 如果没有导弹在飞行，清空导弹速度记录
            self.previous_missile_v = None
            reward = 0

        # 将奖励加入轨迹记录（用于日志或训练追踪）
        self.reward_trajectory[agent_id].append([reward])

        return reward
