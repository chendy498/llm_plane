import numpy as np
import openai
import os
from envs.JSBSim.envs import SingleCombatEnv
import logging
from openai import OpenAI
# openai.api_key = os.getenv("OPENAI_API_KEY")  # 确保你设置了环境变量

client = OpenAI(
    api_key="xai-YauNBWEwg7LsL4qCkoCTCKlyub5wqsYRNUrY1cV5oPUb99MTxw1yFf4zvkrcQMQSQH1OrQxRSGbGp6x0",
    base_url="https://api.x.ai/v1",
)

logging.basicConfig(level=logging.INFO)

# 辅助函数：将obs向量转成简洁文本（你可以自定义这里的描述）
def obs_to_text(obs):
    return ", ".join([f"{x:.2f}" for x in obs])

# 辅助函数：将模型文本输出转成动作向量（假设输出类似："Action: [0.1, -0.2, 0.0, 1.0]"）
def parse_action(text):
    import re
    try:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        return np.array([float(n) for n in numbers], dtype=np.float32)
    except Exception as e:
        logging.error(f"动作解析失败: {e}, 返回默认动作")
        return np.zeros(4, dtype=np.float32)

# 调用GPT生成动作
# def build_prompt(agent, visible):
#     prompt = f"你是{agent.team}阵营的无人机 {agent.id}，坐标在 ({agent.x}, {agent.y})，生命值 {agent.hp}。\n"
#     prompt += "你看到的敌人如下：\n"
#     for e in visible:
#         prompt += f"- {e.id} 在 ({e.x}, {e.y})，生命值 {e.hp}\n"
#     prompt += "请选择你的动作，并说明理由（用 JSON 格式输出：{\"action\": ..., \"target\": ..., \"reasoning\": ...}）：\n"
#     prompt += "可选动作：move(dx, dy), attack(target_id), pass"
#     return prompt
def query_grok(obs_vector):
    prompt = f"""
你是一个空战智能体的控制模块。你将接收到一个包含15维信息的观测向量，每一维代表当前战斗机与敌机的飞行状态或相对关系。你的任务是根据这些观测信息生成一个长度为4的动作列表，以控制己方战斗机的行为。
观测向量结构如下（共15维）：
1. {obs_vector[0]} 己方高度 / 5000（单位：5km）
2. {obs_vector[1]}  己方滚转角的正弦值 sin(roll)
3. {obs_vector[2]}  己方滚转角的余弦值 cos(roll)
4. {obs_vector[3]}  己方俯仰角的正弦值 sin(pitch)
5. {obs_vector[4]}  己方俯仰角的余弦值 cos(pitch)
6. {obs_vector[5]}  己方 x 轴机体速度 / 340（单位：马赫数）
7. {obs_vector[6]}  己方 y 轴机体速度 / 340
8. {obs_vector[7]}  己方 z 轴机体速度 / 340
9. {obs_vector[8]}  己方速度模 vc / 340
10. {obs_vector[9]}  敌我 x 轴速度差（敌 - 己）/ 340
11. {obs_vector[10]}  敌我高度差（敌 - 己）/ 1000（单位：km）
12. {obs_vector[11]}  己方视角下的敌机角度 Angle-Off（单位：弧度，范围 [0, π]）
13. {obs_vector[12]}  敌方视角下的己机角度 Target-Aspect（单位：弧度，范围 [0, π]）
14. {obs_vector[13]}  敌我相对距离 / 10000（单位：10 km）
15. {obs_vector[14]}  己方在敌方左右位置（-1：左，0：正前，1：右）
输出目标：
根据上述观测向量，输出一个包含4个浮点数的动作列表，格式如下：
输出一个长度为4的动作向量，对应控制：
1. aileron（副翼）: 控制滚转，范围 [0,40]
2. elevator（升降舵）: 控制俯仰，范围 [0,40]
3. rudder（方向舵）: 控制偏航，范围 [0,40]
4. throttle（油门）: 控制推力，范围 [0,58]
输出格式要求：
只输出动作列表，例如：
[0.9, 0.05, 0.1, 0.0]

请不要输出其他解释文字。
    """
    try:
        response = client.chat.completions.create(
            model="grok-3-beta",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        action_text = response.choices[0].message.content.strip()
        # print(action_text)
        return parse_action(action_text)
    except Exception as e:
        logging.error(f"Grok API 请求失败: {e}")
        return np.zeros(4, dtype=np.float32)

# 设置环境
env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
env.seed(0)

obs = env.reset()
render = True
experiment_name = "Grok_Decision"

num_agents = 2
episode_rewards = 0

if render:
    env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

ego_obs = obs[:num_agents // 2, :]
enm_obs = obs[num_agents // 2:, :]

while True:
    # 使用 Grok 大模型为每个agent生成动作
    ego_action = query_grok(ego_obs[0])  # 只支持一个agent
    enm_action = query_grok(enm_obs[0])  # 同理


    actions = np.stack([ego_action, enm_action], axis=0)#这里有bug，到时候注意就好
    print(actions)
    # 环境一步
    obs, rewards, dones, infos = env.step(actions)
    episode_rewards += rewards[:num_agents // 2].sum()
    print("obs:", obs)
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

    if dones.all():
        print("战斗结束信息:", infos)
        break

    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step: {env.current_step}, bloods: {bloods}")

    # 更新观察
    ego_obs = obs[:num_agents // 2, :]
    enm_obs = obs[num_agents // 2:, :]

print("总奖励:", episode_rewards)#表示战斗机的总奖励
