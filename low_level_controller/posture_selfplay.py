import sys
import os
import argparse
import logging
import time
from datetime import datetime
import copy
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from low_level_controller.algo.sac import SAC
from low_level_controller.algo.replay_buffer import ReplayBuffer
from low_level_controller.algo.policy_pool import PolicyPool
from envs.JSBSim.envs import SingleCombatEnv

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 解析超参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC with Self-Play for Fighter Tracking in JSBSim Environment")

    # 环境和智能体设置
    parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments (must be positive)')
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents (red and blue, must be 2 for 1v1)')
    parser.add_argument('--max_episodes', type=int, default=50000,
                        help='Maximum number of training episodes (must be positive)')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode (must be positive)')

    # 强化学习参数
    parser.add_argument('--replay_size', type=int, default=int(1e6), help='Replay buffer size (must be positive)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (must be positive)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (0 < gamma <= 1)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update parameter for target networks (0 < tau <= 1)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizers (must be positive)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy coefficient for SAC (must be non-negative)')

    # 自我对弈参数
    parser.add_argument('--policy_pool_size', type=int, default=5,
                        help='Size of policy pool for self-play (must be positive)')
    parser.add_argument('--play_against_latest_ratio', type=float, default=0.7,
                        help='Probability of playing against latest policy (0 <= ratio <= 1)')
    parser.add_argument('--k_factor', type=int, default=16, help='ELO update factor (must be positive)')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Interval for evaluating against policy pool (must be positive)')

    # 模型和环境维度
    parser.add_argument('--obs_dim', type=int, default=29, help='Observation dimension (must match environment)')
    parser.add_argument('--act_dim', type=int, default=4, help='Action dimension (must match environment)')

    # 其他设置
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (non-negative)')
    parser.add_argument('--save_interval', type=int, default=5000, help='Interval for saving models (must be positive)')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='Device for training (cuda:0 or cpu)')

    # 解析参数
    args = parser.parse_args()

    # 参数验证
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if args.num_agents != 2:
        raise ValueError("num_agents must be 2 for 1v1 combat")
    if args.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.replay_size <= 0:
        raise ValueError("replay_size must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not (0 < args.gamma <= 1):
        raise ValueError("gamma must be in (0, 1]")
    if not (0 < args.tau <= 1):
        raise ValueError("tau must be in (0, 1]")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative")
    if args.policy_pool_size <= 0:
        raise ValueError("policy_pool_size must be positive")
    if not (0 <= args.play_against_latest_ratio <= 1):
        raise ValueError("play_against_latest_ratio must be in [0, 1]")
    if args.k_factor <= 0:
        raise ValueError("k_factor must be positive")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if args.obs_dim <= 0:
        raise ValueError("obs_dim must be positive")
    if args.act_dim <= 0:
        raise ValueError("act_dim must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.save_interval <= 0:
        raise ValueError("save_interval must be positive")
    if args.device not in ["cuda:0", "cpu"]:
        raise ValueError("device must be 'cuda:0' or 'cpu'")

    # 将参数转换为字典以与 SAC 类兼容
    args_dict = {
        'num_envs': args.num_envs,
        'num_agents': args.num_agents,
        'max_episodes': args.max_episodes,
        'max_steps': args.max_steps,
        'replay_size': args.replay_size,
        'batch_size': args.batch_size,
        'gamma': args.gamma,
        'tau': args.tau,
        'lr': args.lr,
        'alpha': args.alpha,
        'policy_pool_size': args.policy_pool_size,
        'play_against_latest_ratio': args.play_against_latest_ratio,
        'k_factor': args.k_factor,
        'eval_interval': args.eval_interval,
        'obs_dim': args.obs_dim,
        'act_dim': args.act_dim,
        'seed': args.seed,
        'save_interval': args.save_interval,
        'device': torch.device(args.device)
    }

    return args_dict

# 策略评估函数
def evaluate_against_pool(red_agent, pool, env, num_matches=5, k_factor=16):
    wins = 0
    total_matches = num_matches * pool.size()
    for idx, opp in enumerate(pool.policies):
        opp_elo = pool.elos[idx]
        red_elo = pool.elos[-1] if pool.elos else 1000  # 默认初始 ELO
        for _ in range(num_matches):
            obs = env.reset()
            red_obs = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0).to(device)
            blue_obs = torch.tensor(obs[1], dtype=torch.float32).unsqueeze(0).to(device)
            episode_reward = 0

            for step in range(1000):
                red_action, _ = red_agent.actor.sample(red_obs)
                blue_action, _ = opp.actor.sample(blue_obs)
                actions = torch.stack([red_action, blue_action], dim=0).detach().cpu().numpy().squeeze(1)
                no_shoot = np.zeros((2, 1))  # 2 agents, each with one "no shoot" flag
                actions = np.hstack([actions, no_shoot])
                next_obs, rewards, done, info = env.step(actions)

                episode_reward += rewards[0]
                if done.all():
                    # 更新 ELO 分数
                    expected_red = 1 / (1 + 10 ** ((opp_elo - red_elo) / 400))
                    score = 1 if rewards[0] > rewards[1] else 0
                    red_elo += k_factor * (score - expected_red)
                    opp_elo += k_factor * ((1 - score) - (1 - expected_red))
                    pool.update_elo(idx, opp_elo)
                    if score == 1:
                        wins += 1
                    break

                red_obs = torch.tensor(next_obs[0], dtype=torch.float32).unsqueeze(0).to(device)
                blue_obs = torch.tensor(next_obs[1], dtype=torch.float32).unsqueeze(0).to(device)

    win_rate = wins / total_matches if total_matches > 0 else 0
    # logger.info(f"Win rate against pool: {win_rate:.3f}, Red ELO: {red_elo:.1f}")
    return win_rate, red_elo

# 主训练函数
def train(args):
    # 初始化环境
    env = SingleCombatEnv("1v1/ShootMissile/Selfplay")
    env.seed(args["seed"])
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    # 初始化智能体和缓冲区
    red_agent = SAC(obs_dim=args["obs_dim"], act_dim=args["act_dim"], args=args)
    blue_agent = SAC(obs_dim=args["obs_dim"], act_dim=args["act_dim"], args=args)
    # red_agent.load_model("checkpoints/episode_29999_20250604_012649.pth")
    # blue_agent.load_model("checkpoints/episode_29999_20250604_012649.pth")
    buffer_red = ReplayBuffer(args["replay_size"], args)
    buffer_blue = ReplayBuffer(args["replay_size"], args)
    pool = PolicyPool(max_size=args["policy_pool_size"])
    pool.add(red_agent, elo=1000)  # 初始策略加入池中

    rewards_per_episode = []
    elo_scores = []

    for episode in range(args["max_episodes"]):
        obs = env.reset()
        episode_reward_red = 0
        episode_reward_blue = 0

        # 选择蓝方对手
        if random.random() < args["play_against_latest_ratio"]:
            blue_policy = blue_agent  # 使用最新蓝方策略
        else:
            blue_policy = pool.sample()  # 从池中随机选择

        for step in range(args["max_steps"]):
            red_obs = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0).to(device)
            blue_obs = torch.tensor(obs[1], dtype=torch.float32).unsqueeze(0).to(device)

            red_action, red_action_prob = red_agent.actor.sample(red_obs)
            blue_action, blue_action_prob = blue_policy.actor.sample(blue_obs)

            actions = torch.stack([red_action, blue_action], dim=0).detach().cpu().numpy().squeeze(1)
            no_shoot = np.zeros((2, 1))  # 2 agents, each with one "no shoot" flag
            actions = np.hstack([actions, no_shoot])  # 添加不射击的动作
            next_obs, rewards, done, info = env.step(actions)

            # 存储经验
            buffer_red.store(
                red_obs.squeeze(0).detach().cpu().numpy(),
                red_action.squeeze(0).detach().cpu().numpy(),
                rewards[0],
                next_obs[0],
                done[0]
            )
            buffer_blue.store(
                blue_obs.squeeze(0).detach().cpu().numpy(),
                blue_action.squeeze(0).detach().cpu().numpy(),
                rewards[1],
                next_obs[1],
                done[1]
            )

            # episode_reward_red += float(rewards[0])
            # episode_reward_blue += float(rewards[1])
            episode_reward_red += rewards[0].item()
            episode_reward_blue += rewards[1].item()

            obs = next_obs

            if done.all():
                break

        rewards_per_episode.append(episode_reward_red)
        # logger.info(f"Episode {episode}, Red Reward: {episode_reward_red:.2f}, Blue Reward: {episode_reward_blue:.2f}")

        # 更新策略
        if buffer_red.size > args["batch_size"]:
            red_agent.update(buffer_red, args["batch_size"])
        if buffer_blue.size > args["batch_size"]:
            blue_agent.update(buffer_blue, args["batch_size"])

        # 定期评估并更新策略池
        if episode % 200 == 0 and episode > 0:
            win_rate, red_elo = evaluate_against_pool(red_agent, pool, env)
            elo_scores.append(red_elo)
            if win_rate > 0.5:
                pool.add(red_agent, elo=red_elo)
                logger.info(f"Added policy to pool. Pool size: {pool.size()}")
            if random.random() < 0.5:  # 随机更新蓝方策略
                pool.add(blue_agent, elo=1000)
                logger.info(f"Added blue policy to pool. Pool size: {pool.size()}")

        # 保存模型
        if (episode+1) % args["save_interval"] == 0 and episode > 0:
            # 直接把 episode 写进文件名
            red_agent.save_model(f"checkpoints/episode_{episode+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

            logger.info(f"Saved models at episode {episode+1}")

    # 绘图：奖励和 ELO 分数趋势
    window_size = 100
    mean_rewards = [np.mean(rewards_per_episode[i:i + window_size]) for i in range(0, len(rewards_per_episode), window_size)]
    std_rewards = [np.std(rewards_per_episode[i:i + window_size]) for i in range(0, len(rewards_per_episode), window_size)]
    episodes = list(range(window_size, len(rewards_per_episode) + 1, window_size))

    plt.figure(figsize=(12, 5))
    plt.plot(episodes, mean_rewards, label='Mean Reward (per 100 episodes)')
    plt.fill_between(episodes,
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Trend')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_trends.png')
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    train(args)

