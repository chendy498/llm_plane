from .reward_function_base import BaseRewardFunction


class ShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward（发射惩罚奖励函数）
    每当发射一枚导弹时，给予 -10 分的惩罚，
    用于限制一次性发射所有导弹的策略。
    """
    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
        self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        当智能体发射了一枚导弹，就给予固定惩罚（-10）。

        参数：
            task: 当前任务实例
            env: 当前环境
            agent_id: 智能体 ID

        返回：
            reward: 奖励值（一般为 0 或 -10）
        """
        reward = 0

        # 如果剩余导弹数减少了 1，说明刚刚发射了一枚导弹
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] - 1:
            reward -= 10  # 给负奖励（惩罚）

        # 更新当前导弹数量状态
        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]

        return self._process(reward, agent_id)

