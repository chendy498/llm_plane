import os
import torch
import numpy as np
from datetime import datetime
from low_level_controller.algo.sac import SAC
from envs.JSBSim.envs import SingleCombatEnv
import matplotlib.pyplot as plt

def evaluate(model_path_red, model_path_blue, num_episodes=5, save_dir='eval_acmi'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建环境
    env = SingleCombatEnv("1v1/ShootMissile/Selfplay")
    # env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
    obs_dim = 29
    act_dim = 4

    # 初始化红蓝双方智能体
    red_agent = SAC(obs_dim, act_dim, args= {
    'gamma': 0.99,
    'tau': 0.005,
    'lr': 1e-4,
    'alpha': 0.2,         # 熵系数，可自动调整
    'device': device,
    'batch_size': 256,
    'obs_dim': 29,
    'act_dim': 4,
})
    blue_agent = SAC(obs_dim, act_dim, args = {
    'gamma': 0.99,
    'tau': 0.005,
    'lr': 3e-4,
    'alpha': 0.2,         # 熵系数，可自动调整
    'device': device,
    'batch_size': 256,
    'obs_dim': 29,
    'act_dim': 4,
})

    # 加载模型
    red_agent.load_model(model_path_red)
    blue_agent.load_model(model_path_blue)

    os.makedirs(save_dir, exist_ok=True)

    for ep in range(num_episodes):
        obs = env.reset()
        red_obs = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0).to(device)
        blue_obs = torch.tensor(obs[1], dtype=torch.float32).unsqueeze(0).to(device)

        done = False
        ep_reward_red = 0
        ep_reward_blue = 0
        red_reward_history = []
        blue_reward_history = []

        # 每局生成一个 ACMI 文件名
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        acmi_path = os.path.join(save_dir, f'{now}_episode.txt.acmi')
        env.render(mode='txt', filepath=acmi_path)
        for step in range(10000):
            red_action, _ = red_agent.actor.sample(red_obs)
            blue_action, _ = blue_agent.actor.sample(blue_obs)
            actions = torch.stack([red_action, blue_action], dim=0).detach().cpu().numpy().squeeze(1)
            # no_shoot = np.zeros((2, 1))  # 2 agents, each with one "no shoot" flag

            # actions = np.hstack([actions, no_shoot])  # 添加不射击的动作
            thetaT = obs[1][15]  # obs[1][15]是蓝方的thetaT,航迹角
            d = obs[1][13]  # obs[1][14]是蓝方的d,距离
            blue_shoot = 1 if (abs(thetaT) < 0.3 and d < 0.4) else 0
            shoot_flags = np.array([[0], [blue_shoot]])

            actions = np.hstack([actions, shoot_flags])
            # actions = np.array([
            #     [20, 20, 20, 58, 0],
            #     [20, 20, 20, 58, 0]
            # ])
            next_obs, rewards, done, info = env.step(actions)
            ep_reward_red += rewards[0].item() if hasattr(rewards[0], 'item') else float(rewards[0])
            ep_reward_blue += rewards[1].item() if hasattr(rewards[1], 'item') else float(rewards[1])
            red_reward_history.append(rewards[0])
            blue_reward_history.append(rewards[1])

            red_obs = torch.tensor(next_obs[0], dtype=torch.float32).unsqueeze(0).to(device)
            blue_obs = torch.tensor(next_obs[1], dtype=torch.float32).unsqueeze(0).to(device)
            env.render(mode='txt', filepath=acmi_path)
            if done.all():
                break

        # 渲染并保存

        # ✅ 绘图展示每一步的奖励变化
        # 每步耗时（秒）
        step_time_seconds = 0.2

        # 构造横轴时间标签
        time_axis = np.arange(len(red_reward_history)) * step_time_seconds

        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, red_reward_history, label='Red Agent')
        plt.plot(time_axis, blue_reward_history, label='Blue Agent')
        plt.xlabel('Time (s)')
        plt.ylabel('Reward')
        plt.title('Step-wise Reward per Agent')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{now}_reward_plot.png"))

        print(f"Episode {ep}: Red reward = {ep_reward_red:.2f}, Blue reward = {ep_reward_blue:.2f}, ACMI saved to {acmi_path}")


if __name__ == "__main__":
    red_model_path = "checkpoints/episode_15000_20250609_161618.pth"
    # blue_model_path = "checkpoints/episode_29999_20250604_012649.pth"
    blue_model_path = "checkpoints/episode_15000_20250609_161618.pth"
    evaluate(red_model_path, blue_model_path, num_episodes=1)

