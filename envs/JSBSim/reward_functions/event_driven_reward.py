from .reward_function_base import BaseRewardFunction

class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    基于事件的奖励函数：
    - 被导弹击落：扣分 -200
    - 意外坠毁：扣分 -200
    - 击落敌机：加分 +200
    """
    def __init__(self, config):
        super().__init__(config)  # 调用父类构造函数，初始化配置

    def get_reward(self, task, env, agent_id):
        """
        获取当前智能体的事件奖励。奖励是所有事件的累加。

        参数:
            task: 当前任务实例
            env: 当前仿真环境实例
            agent_id: 智能体的唯一标识符

        返回:
            (float): 当前时刻智能体的事件奖励
        """
        reward = 0  # 初始化奖励为 0

        # 如果智能体被击落（例如被导弹命中）
        if env.agents[agent_id].is_shotdown:
            reward -= 200  # 给予惩罚 -200

        # 如果智能体发生坠毁（可能是撞地或飞控失误）
        elif env.agents[agent_id].is_crash:
            reward -= 200  # 同样惩罚 -200

        # 遍历该智能体发射的导弹
        for missile in env.agents[agent_id].launch_missiles:
            # 如果某个导弹成功击中目标
            if missile.is_success:
                reward += 200  # 奖励 +200

        # 调用父类处理函数，可能包含缩放、记录或平滑处理
        return self._process(reward, agent_id)
