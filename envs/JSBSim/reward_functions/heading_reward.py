import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class HeadingReward(BaseRewardFunction):
    """
    HeadingReward（航向奖励函数）
    用于衡量当前飞行状态与目标飞行状态的差异，鼓励保持正确航向、速度、高度和姿态。
    """
    def __init__(self, config):
        super().__init__(config)
        # 定义包含多个奖励项的名字（用于可视化或日志记录）
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]

    def get_reward(self, task, env, agent_id):
        """
        计算奖励，方法是将各项误差转为高斯形状奖励，并取几何平均值。

        参数:
            task: 当前任务实例
            env: 当前仿真环境实例
            agent_id: 当前智能体的 ID

        返回:
            float: 奖励值，范围为 [0, 1]
        """

        # 1. 航向误差项（单位：角度）
        heading_error_scale = 5.0  # 容忍误差的标准差为 5 度
        delta_heading = env.agents[agent_id].get_property_value(c.delta_heading)
        heading_r = math.exp(-((delta_heading / heading_error_scale) ** 2))

        # 2. 高度误差项（单位：米）
        alt_error_scale = 15.24  # 约为 50 英尺
        delta_altitude = env.agents[agent_id].get_property_value(c.delta_altitude)
        alt_r = math.exp(-((delta_altitude / alt_error_scale) ** 2))

        # 3. 翻滚角误差项（单位：弧度）
        roll_error_scale = 0.35  # 约为 20 度
        roll_angle = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
        roll_r = math.exp(-((roll_angle / roll_error_scale) ** 2))

        # 4. 空速误差项（单位：米/秒）
        speed_error_scale = 24  # 约为最大速度的 10%
        delta_speed = env.agents[agent_id].get_property_value(c.delta_velocities_u)
        speed_r = math.exp(-((delta_speed / speed_error_scale) ** 2))

        # 5. 最终奖励是几何平均值，防止单项过低时影响太大
        reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)

        # 调用父类的 _process 方法进行奖励标准化/记录
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r))
