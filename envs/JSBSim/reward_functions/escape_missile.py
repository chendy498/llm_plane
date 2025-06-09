import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import  get_thetaT_thetaA_R,S
class EscapeMissileReward(BaseRewardFunction):
    """
    MissilePostureReward（导弹姿态奖励）
    鼓励智能体逃避导弹：速度减缓、方向偏离、拉大距离、生存时间。
    """
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = None
        self.previous_distance = None
        self.pre_d=5577  # 初始距离
        self.count = 0  # 用于记录调用次数
        self.survival_bonus = 0.05  # 每帧逃过导弹的奖励
        self.reward_scale = 1.0     # 奖励缩放因子

    def reset(self, task, env):
        self.previous_missile_v = None
        self.previous_distance = None
        self.pre_d = 5577  # 重置初始距离
        self.count = 0  # 重置调用次数
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        reward = 0
        missile_sim = env.agents[agent_id].check_missile_warning()
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity(),
                                 env.agents[agent_id].get_rpy()])

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity(), enm.get_rpy()])

            # 获取角度信息（AO、TA）与距离R
            thetaT, thetaA, R = get_thetaT_thetaA_R(ego_feature, enm_feature)

            thetaT_bar = thetaT / np.pi
            thetaA_bar = thetaA / np.pi
            d = R
            # vc是距离的变化率
            vc = -(d - self.pre_d) / 0.2  # 假设时间间隔为0.2秒
            if self.count / 2 == 0:
                self.pre_d = d  # 更新上一次的距离
            h = ego_feature[2]  # 高度

            Gamma_B_d = self.Gamma_B_d(d)  # 你需要定义
            Gamma_R_d = self.Gamma_R_d(d)  # 你需要定义

            # new_reward += (
            #         self.r_rel_pos(thetaT_bar, thetaA_bar,d) +
            #         self.r_closure(vc, thetaA_bar, d) +
            #         self.r_gunsnap_blue(Gamma_B_d, thetaT_bar) +
            #         self.r_gunsnap_red(Gamma_R_d, thetaA_bar) +
            #         self.r_deck(h) +
            #         self.r_too_close(thetaA_bar, d)+
            #         self.stop_roll(ego_feature,d)
            # )
            r_pos = self.r_rel_pos(thetaT_bar, thetaA_bar, d)
            r_closure = self.r_closure(vc, thetaA_bar, d)
            r_gunsnap_blue = self.r_gunsnap_blue(Gamma_B_d, thetaT_bar)
            r_gunsnap_red = self.r_gunsnap_red(Gamma_R_d, thetaA_bar)
            r_deck = self.r_deck(h)
            r_too_close = self.r_too_close(thetaA_bar, d)
            r_stop_roll = self.stop_roll(ego_feature, d)

            reward += (
                    r_pos +
                    r_closure +
                    r_gunsnap_blue +
                    r_gunsnap_red +
                    r_deck +
                    r_too_close +
                    r_stop_roll
            )
        # 检查当前是否有导弹在飞向该智能体（返回导弹对象）
        if missile_sim is not None:
            missile_v = missile_sim.get_velocity()
            missile_pos = missile_sim.get_position()
            aircraft_v = env.agents[agent_id].get_velocity()
            aircraft_pos = env.agents[agent_id].get_position()

            # 初始记录导弹速度和距离
            if self.previous_missile_v is None:
                self.previous_missile_v = missile_v
            if self.previous_distance is None:
                self.previous_distance = np.linalg.norm(missile_pos - aircraft_pos)

            # 当前距离与前一帧比较
            distance = np.linalg.norm(missile_pos - aircraft_pos)
            distance_delta = distance - self.previous_distance
            self.previous_distance = distance

            # 导弹速度减缓量
            v_decrease = (np.linalg.norm(self.previous_missile_v) - np.linalg.norm(missile_v)) / 340
            self.previous_missile_v = missile_v

            # 导弹与飞机的夹角余弦
            angle_cos = np.dot(missile_v, aircraft_v) / (np.linalg.norm(missile_v) * np.linalg.norm(aircraft_v))
            angle_cos = np.clip(angle_cos, -1.0, 1.0)

            # 奖励项构造
            r_escape_direction = (1 - angle_cos)  # 越背离越好，angle_cos 趋近于 -1 越优秀
            r_distance_gain = max(0, distance_delta) / 100  # 距离拉开越多奖励越大
            r_velocity_decay = max(v_decrease, 0)

            # 总奖励构成
            reward = (
                1.5 * r_escape_direction +
                1.0 * r_distance_gain +
                1.0 * r_velocity_decay +
                self.survival_bonus
            )
        else:
            # 若导弹消失或无导弹，清空状态，给予逃脱奖励
            if self.previous_missile_v is not None:
                reward = 3.0  # 代表成功逃脱
            self.previous_missile_v = None
            self.previous_distance = None

        self.reward_trajectory[agent_id].append([reward])
        return reward

    def r_rel_pos(self, theta_t_bar, phi_a_bar, d):
        return ((theta_t_bar - 2) * S(phi_a_bar, 18, 0.5) - theta_t_bar + 1) * (1 - S(d, 1 / 50, 3000))  # 控制3000米之内再狗斗

    def r_closure(self, vc, phi_a_bar, d):
        return (vc / 150) * (1 - S(phi_a_bar, 18, 0.5)) * S(d, 1 / 200, 1000)

    def r_gunsnap_blue(self, Gamma_B_d, theta_t_bar):
        return Gamma_B_d * (1 - S(theta_t_bar, 1e5, 1 / 180))

    def r_gunsnap_red(self, Gamma_R_d, phi_a_bar):
        return -Gamma_R_d * S(phi_a_bar, 800, 178 / 180)

    def r_deck(self, h):
        return 2 * S(h, 1 / 20, 4000)

    def r_too_close(self, phi_a_bar, d):
        return -1 * (1 - S(phi_a_bar, 18, 0.5)) * (1 - S(d, 1 / 20, 200))

    def Gamma_B_d(self, d):
        if d < 650:
            return self.beta_b(d) * S(d, 1 / 50, 300)
        else:
            return self.beta_b(d) * (1 - S(d, 1 / 50, 1000))

    def Gamma_R_d(self, d):
        if d < 650:
            return self.beta_r(d) * S(d, 1 / 50, 300)
        else:
            return self.beta_r(d) * (1 - S(d, 1 / 50, 1000))

    def beta_b(self, d):
        return 3

    def beta_r(self, d):
        return -3

    def stop_roll(self, ego_feature, d):
        """
        减少滚转角的变化
        """
        roll = ego_feature[6] / np.pi  # 将弧度转换为单位圆上的值
        if roll > 0.2:
            return 2 - 2 * abs(roll - 0.2) * S(d, 1 / 50, 4000)
        elif roll < -0.2:
            return 2 - 2 * abs(roll + 0.2) * S(d, 1 / 50, 4000)
        else:
            return 2