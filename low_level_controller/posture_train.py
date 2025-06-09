import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import argparse
import logging

from low_level_controller.algo.policy_pool import PolicyPool
from envs.JSBSim.envs import SingleCombatEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 环境封装为 gym.Env 格式
class SB3SingleCombatEnv(gym.Env):
    def __init__(self, opponent_policy=None):
        super(SB3SingleCombatEnv, self).__init__()
        self.env = SingleCombatEnv("1v1/ShootMissile/Selfplay")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.opponent_policy = opponent_policy

    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        return obs[0]

    def step(self, red_action):
        if self.opponent_policy:
            blue_action, _ = self.opponent_policy.predict(self.last_obs[1], deterministic=True)
        else:
            blue_action = np.random.uniform(-1, 1, size=(4,))
        actions = np.vstack([red_action, blue_action])
        actions = np.hstack([actions, np.zeros((2, 1))])  # 加 no-shoot 标志
        next_obs, rewards, done, info = self.env.step(actions)
        self.last_obs = next_obs
        return next_obs[0], rewards[0], done[0], {}

    def seed(self, seed):
        self.env.seed(seed)

# 2. 评估策略对抗策略池
def evaluate_against_pool(model, pool, num_matches=5, k_factor=16):
    wins = 0
    red_elo = pool.elos[-1] if pool.elos else 1000
    total_matches = num_matches * pool.size()

    for idx, opp_model in enumerate(pool.policies):
        opp_elo = pool.elos[idx]
        env = SB3SingleCombatEnv(opponent_policy=opp_model)
        for _ in range(num_matches):
            obs = env.reset()
            done = False
            reward_red = 0
            reward_blue = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _ = env.step(action)
                reward_red += r
            score = 1 if reward_red > reward_blue else 0
            expected = 1 / (1 + 10 ** ((opp_elo - red_elo) / 400))
            red_elo += k_factor * (score - expected)
            opp_elo += k_factor * ((1 - score) - (1 - expected))
            pool.update_elo(idx, opp_elo)
            if score == 1:
                wins += 1

    return wins / total_matches if total_matches > 0 else 0, red_elo

# 3. 参数设置
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--steps_per_episode', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--policy_pool_size', type=int, default=5)
    parser.add_argument('--play_against_latest_ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    return parser.parse_args()

# 4. 主训练函数
def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建初始环境
    env_fn = lambda: SB3SingleCombatEnv()
    vec_env = DummyVecEnv([env_fn])

    model = SAC("MlpPolicy", vec_env,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                buffer_size=args.replay_size,
                gamma=args.gamma,
                tau=args.tau,
                verbose=0,
                seed=args.seed)

    policy_pool = PolicyPool(max_size=args.policy_pool_size)
    policy_pool.add(model, elo=1000)  # 添加初始策略

    rewards_per_episode = []
    elo_scores = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        total_reward = 0
        for _ in range(args.steps_per_episode):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = vec_env.step(action)
            total_reward += reward[0]
            if done[0]:
                break
        rewards_per_episode.append(total_reward)

        model.learn(total_timesteps=args.steps_per_episode, reset_num_timesteps=False, log_interval=10)

        if ep % args.eval_interval == 0 and ep > 0:
            win_rate, red_elo = evaluate_against_pool(model, policy_pool)
            elo_scores.append(red_elo)
            logger.info(f"[Episode {ep}] Win Rate: {win_rate:.2f}, ELO: {red_elo:.1f}")
            if win_rate > 0.5:
                policy_pool.add(model, elo=red_elo)

        if (ep + 1) % args.save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model.save(f"checkpoints/red_sac_{ep+1}_{timestamp}")
            logger.info(f"Saved model at episode {ep+1}")

    # 绘图
    plt.figure(figsize=(12, 5))
    window = 100
    means = [np.mean(rewards_per_episode[i:i+window]) for i in range(0, len(rewards_per_episode), window)]
    stds = [np.std(rewards_per_episode[i:i+window]) for i in range(0, len(rewards_per_episode), window)]
    xs = list(range(window, len(rewards_per_episode) + 1, window))
    plt.plot(xs, means, label='Mean Reward')
    plt.fill_between(xs, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2)
    plt.title('Reward Trend')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_rewards.png")
    plt.show()

if __name__ == "__main__":
    main()
