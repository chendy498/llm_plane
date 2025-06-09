from itertools import count

import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import  get_thetaT_thetaA_R,S


class RelativePositionReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__]
        self.pre_d= 5577 # 初始距离，假设初始距离为11154米的一半
        self.count = 0  # 用于记录调用次数

    def reset(self, task, env):
        self.pre_d = 5577  # 初始距离，假设初始距离为11154米的一半
        self.count = 0
        return super().reset(task, env)

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
        self.count+= 1  # 增加调用次数计数
        # 获取当前智能体的位置 + 速度（拼接成一个特征向量）
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity(),
                                 env.agents[agent_id].get_rpy()])

        # 遍历敌方目标（默认只有一个敌人）
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity(),enm.get_rpy()])

            # 获取角度信息（AO、TA）与距离R
            thetaT, thetaA, R = get_thetaT_thetaA_R(ego_feature, enm_feature)

            thetaT_bar = thetaT / np.pi
            thetaA_bar = thetaA / np.pi
            d = R
            #vc是距离的变化率
            vc = -(d - self.pre_d) / 0.2  # 假设时间间隔为0.2秒
            if self.count/2 == 0:
                self.pre_d= d  # 更新上一次的距离
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

            new_reward += (
                    r_pos +
                    r_closure +
                    r_gunsnap_blue +
                    r_gunsnap_red +
                    r_deck +
                    r_too_close +
                    r_stop_roll
            )
            # 将奖励加入轨迹记录（用于日志或训练追踪）
        self.reward_trajectory[agent_id].append([new_reward])
        return new_reward

    def r_rel_pos(self,theta_t_bar, phi_a_bar,d):
        return ((theta_t_bar - 2) * S(phi_a_bar, 18, 0.5) - theta_t_bar + 1)* (1-S(d, 1 / 50, 3000))#控制3000米之内再狗斗

    def r_closure(self,vc, phi_a_bar, d):
        return (vc / 150) * (1 - S(phi_a_bar, 18, 0.5)) * S(d, 1 / 200, 1000)

    def r_gunsnap_blue(self,Gamma_B_d, theta_t_bar):
        return Gamma_B_d * (1 - S(theta_t_bar, 1e5, 1 / 180))

    def r_gunsnap_red(self,Gamma_R_d, phi_a_bar):
        return -Gamma_R_d * S(phi_a_bar, 800, 178 / 180)

    def r_deck(self,h):
        return 2 * S(h, 1 / 20, 4000)

    def r_too_close(self,phi_a_bar, d):
        return -1 * (1 - S(phi_a_bar, 18, 0.5)) *(1 - S(d, 1/20, 200))

    def Gamma_B_d(self,d):
        if d < 650:
            return self.beta_b(d)* S(d, 1/50, 300)
        else:
            return self.beta_b(d) * (1 - S(d, 1/50, 1000))

    def Gamma_R_d(self,d):
        if d < 650:
            return self.beta_r(d) * S(d, 1/50, 300)
        else:
            return self.beta_r(d) * (1 - S(d, 1/50, 1000))

    def beta_b(self,d):
        return 3

    def beta_r(self,d):
        return -3

    def stop_roll(self,ego_feature,d):
        """
        减少滚转角的变化
        """
        roll = ego_feature[6]/ np.pi  # 将弧度转换为单位圆上的值
        if roll > 0.2:
            return 2-2 * abs(roll - 0.2) * S(d, 1 / 50, 3000)
        elif roll < -0.2:
            return 2-2 * abs(roll + 0.2) * S(d, 1 / 50, 3000)
        else:
            return 2








# import numpy as np
# from .reward_function_base import BaseRewardFunction
# from ..utils.utils import  get_thetaT_thetaA_R,S
#
#
# class RelativePositionReward(BaseRewardFunction):
#     def __init__(self, config):
#         super().__init__(config)
#         self.reward_item_names = [self.__class__.__name__]
#         # self.pre_d= 11154
#         self.pre_d= 11154/2  # 初始距离
#     def get_reward(self, task, env, agent_id):
#         """
#         计算智能体的姿态奖励。
#
#         输入：
#         - task: 当前任务实例
#         - env: 仿真环境
#         - agent_id: 智能体编号
#
#         输出：
#         - 最终奖励值：姿态奖励 × 距离奖励
#         """
#         new_reward = 0
#
#         # 获取当前智能体的位置 + 速度（拼接成一个特征向量）
#         ego_feature = np.hstack([env.agents[agent_id].get_position(),
#                                  env.agents[agent_id].get_velocity(),
#                                  env.agents[agent_id].get_rpy()])
#
#         # 遍历敌方目标（默认只有一个敌人）
#         for enm in env.agents[agent_id].enemies:
#             enm_feature = np.hstack([enm.get_position(), enm.get_velocity(),enm.get_rpy()])
#
#             # 获取角度信息（AO、TA）与距离R
#             thetaT, thetaA, R = get_thetaT_thetaA_R(ego_feature, enm_feature)
#
#             thetaT_bar = thetaT / np.pi
#             thetaA_bar = thetaA / np.pi
#             d = R
#             #vc是距离的变化率
#             vc = -(d - self.pre_d) / 0.2  # 假设时间间隔为0.2秒
#             self.pre_d= d  # 更新上一次的距离
#             h = ego_feature[2]  # 高度
#
#             Gamma_B_d = self.Gamma_B_d(d)  # 你需要定义
#             Gamma_R_d = self.Gamma_R_d(d)  # 你需要定义
#
#             # new_reward += (
#             #         self.r_rel_pos(thetaT_bar, thetaA_bar,d) +
#             #         self.r_closure(vc, thetaA_bar, d) +
#             #         self.r_gunsnap_blue(Gamma_B_d, thetaT_bar) +
#             #         self.r_gunsnap_red(Gamma_R_d, thetaA_bar) +
#             #         self.r_deck(h) +
#             #         self.r_too_close(thetaA_bar, d)+
#             #         self.stop_roll(ego_feature,d)
#             # )
#             r_pos = self.r_rel_pos(thetaT_bar, thetaA_bar, d)
#             r_closure = self.r_closure(vc, thetaA_bar, d)
#             r_gunsnap_blue = self.r_gunsnap_blue(Gamma_B_d, thetaT_bar)
#             r_gunsnap_red = self.r_gunsnap_red(Gamma_R_d, thetaA_bar)
#             r_deck = self.r_deck(h)
#             r_too_close = self.r_too_close(thetaA_bar, d)
#             r_too_far = self.too_far(d,thetaT_bar)  # 如果敌机距离过远，给予惩罚
#             # r_stop_roll = self.stop_roll(ego_feature, d)
#
#             new_reward += (
#                     r_pos +
#                     r_closure +
#                     r_gunsnap_blue +
#                     r_gunsnap_red +
#                     r_deck +
#                     r_too_close+
#                     r_too_far
#                     # r_stop_roll
#             )
#
#         return new_reward
#
#     def r_rel_pos(self,theta_t_bar, phi_a_bar,d):
#         return ((theta_t_bar - 2) * S(phi_a_bar, 18, 0.5) - theta_t_bar + 1)* (1-S(d, 1 / 20, 3000))#控制3000米之内再狗斗
#
#     def r_closure(self,vc, phi_a_bar, d):
#         return (vc / 150) * (1 - S(phi_a_bar, 18, 0.5)) * S(d, 1 / 50, 1000)
#
#     def r_gunsnap_blue(self,Gamma_B_d, theta_t_bar):
#         return Gamma_B_d * (1 - S(theta_t_bar, 1e5, 1 / 180))
#
#     def r_gunsnap_red(self,Gamma_R_d, phi_a_bar):
#         return -Gamma_R_d * S(phi_a_bar, 800, 178 / 180)
#
#     def r_deck(self,h):
#         return -5 * (1 - S(h, 1 / 20, 4000))-10* (1 - S(h, 1 , 2000))  # 奖励保持高于一定高度
#
#     def r_too_close(self,phi_a_bar, d):
#         return -1 * (1 - S(phi_a_bar, 18, 0.5)) *(1 - S(d, 1/20, 200))
#
#     def Gamma_B_d(self,d):
#         if d < 650:
#             return self.beta_b(d)* S(d, 1/20, 300)
#         else:
#             return self.beta_b(d) * (1 - S(d, 1/20, 1000))
#
#     def Gamma_R_d(self,d):
#         if d < 650:
#             return self.beta_r(d) * S(d, 1/20, 300)
#         else:
#             return self.beta_r(d) * (1 - S(d, 1/20, 1000))
#
#     def beta_b(self,d):
#         return 3
#
#     def beta_r(self,d):
#         return -3
#
#     def too_far(self,d,theta_t_bar):
#         """
#         如果敌机距离过远，给予惩罚
#         """
#         return -1 * S(d, 1 / 200, 3000)* S(theta_t_bar, 1 , 0.5)  # 当敌机距离大于3000米时，给予惩罚
#
#     def stop_roll(self,ego_feature,d):
#         """
#         减少滚转角的变化
#         """
#         roll = ego_feature[6]/ np.pi  # 将弧度转换为单位圆上的值
#         if roll > 0.2:
#             return -5 * abs(roll - 0.2) * S(d, 1 / 50, 4000)
#         elif roll < -0.2:
#             return -5 * abs(roll + 0.2) * S(d, 1 / 50, 4000)
#         else:
#             return 0
#
#
#
#
#
#
#
