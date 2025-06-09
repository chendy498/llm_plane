import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import logging

from low_level_controller.algo.sac import SAC
from low_level_controller.algo.replay_buffer import ReplayBuffer
from low_level_controller.algo.policy_pool import PolicyPool
from envs.JSBSim.envs import SingleCombatEnv

# Set logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser("SAC Self-Play Training for Missile Evade Task")
    parser.add_argument('--max_episodes', type=int, default=50000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--policy_pool_size', type=int, default=5)
    parser.add_argument('--play_against_latest_ratio', type=float, default=0.7)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--k_factor', type=int, default=16)
    parser.add_argument('--obs_dim', type=int, default=29)
    parser.add_argument('--act_dim', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def evaluate_against_pool(agent, pool, env, num_matches=5, k_factor=16):
    wins = 0
    total_matches = num_matches * pool.size()
    red_elo = pool.elos[-1] if pool.elos else 1000

    for idx, opponent in enumerate(pool.policies):
        opp_elo = pool.elos[idx]
        for _ in range(num_matches):
            obs = env.reset()
            done = False
            red_obs = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
            blue_obs = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)

            for _ in range(1000):
                red_action, _ = agent.actor.sample(red_obs)
                blue_action, _ = opponent.actor.sample(blue_obs)
                actions = torch.cat([red_action, blue_action], dim=0).cpu().numpy()
                actions = np.hstack([actions, np.zeros((2, 1))])  # no-shoot

                next_obs, rewards, done, _ = env.step(actions)
                if done.all():
                    score = 1 if rewards[0] > rewards[1] else 0
                    expected_red = 1 / (1 + 10 ** ((opp_elo - red_elo) / 400))
                    red_elo += k_factor * (score - expected_red)
                    opp_elo += k_factor * ((1 - score) - (1 - expected_red))
                    pool.update_elo(idx, opp_elo)
                    if score == 1:
                        wins += 1
                    break
                red_obs = torch.FloatTensor(next_obs[0]).unsqueeze(0).to(device)
                blue_obs = torch.FloatTensor(next_obs[1]).unsqueeze(0).to(device)

    win_rate = wins / total_matches if total_matches > 0 else 0
    return win_rate, red_elo

def train(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup environment
    env = SingleCombatEnv("1v1/ShootMissile/Selfplay")
    env.seed(args.seed)

    # Setup agents
    red_agent = SAC(args.obs_dim, args.act_dim, vars(args))
    blue_agent = SAC(args.obs_dim, args.act_dim, vars(args))
    red_agent.load_model("checkpoints/episode_29999_20250604_012649.pth")
    blue_agent.load_model("checkpoints/episode_29999_20250604_012649.pth")
    buffer_red = ReplayBuffer(args.replay_size, vars(args))
    # buffer_blue = ReplayBuffer(args.replay_size, vars(args))
    # policy_pool = PolicyPool(args.policy_pool_size)
    # policy_pool.add(red_agent, elo=1000)

    rewards_per_episode = []
    elo_scores = []

    for ep in range(args.max_episodes):
        obs = env.reset()
        ep_reward = 0

        # if random.random() < args.play_against_latest_ratio:
        #     blue_policy = blue_agent
        # else:
        #     blue_policy = policy_pool.sample()

        for step in range(args.max_steps):
            red_obs = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0).to(device)
            blue_obs = torch.tensor(obs[1], dtype=torch.float32).unsqueeze(0).to(device)

            red_action, red_action_prob = red_agent.actor.sample(red_obs)
            blue_action, blue_action_prob = blue_agent.actor.sample(blue_obs)

            actions = torch.stack([red_action, blue_action], dim=0).detach().cpu().numpy().squeeze(1)
            thetaT = obs[1][15]  # obs[1][15]是蓝方的thetaT,航迹角
            d = obs[1][13]  # obs[1][14]是蓝方的d,距离
            blue_shoot = 1 if (abs(thetaT) < 0.3 and d < 0.4) else 0
            shoot_flags = np.array([[0], [blue_shoot]])

            actions = np.hstack([actions, shoot_flags])
            # no_shoot = np.zeros((2, 1))  # 2 agents, each with one "no shoot" flag
            # actions = np.hstack([actions, no_shoot])  # 添加不射击的动作
            next_obs, rewards, done, info = env.step(actions)

            # 蓝方发射逻辑
            # blue_angle = obs[1][15]  # 根据你的env实际结构确认索引
            #
            # blue_shoot = 1 if abs(blue_angle) < 10 else 0

            buffer_red.store(
                red_obs.squeeze(0).detach().cpu().numpy(),
                red_action.squeeze(0).detach().cpu().numpy(),
                rewards[0],
                next_obs[0],
                done[0]
            )
            ep_reward += rewards[0]
            obs = next_obs

            if done.all():
                break

        rewards_per_episode.append(ep_reward)

        if buffer_red.size > args.batch_size:
            red_agent.update(buffer_red, args.batch_size)
        # if buffer_blue.size > args.batch_size:
        #     blue_agent.update(buffer_blue, args.batch_size)

        # Evaluate & policy pool update
        # if ep > 0 and ep % args.eval_interval == 0:
        #     win_rate, red_elo = evaluate_against_pool(red_agent, policy_pool, env)
        #     elo_scores.append(red_elo)
        #     logger.info(f"[Ep {ep}] Win Rate: {win_rate:.3f}, Red ELO: {red_elo:.1f}")
        #     if win_rate > 0.5:
        #         policy_pool.add(red_agent, elo=red_elo)
        #         logger.info("Red agent added to policy pool.")
        #     if random.random() < 0.5:
        #         policy_pool.add(blue_agent, elo=1000)
        #         logger.info("Blue agent added to policy pool.")

        if (ep + 1) % args.save_interval == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f"checkpoints/episode_{ep+1}_{timestamp}.pth"
            red_agent.save_model(model_path)
            logger.info(f"Saved model at episode {ep+1} to {model_path}")

    # Plotting
    window = 100
    means = [np.mean(rewards_per_episode[i:i+window]) for i in range(0, len(rewards_per_episode), window)]
    stds = [np.std(rewards_per_episode[i:i+window]) for i in range(0, len(rewards_per_episode), window)]
    episodes = list(range(window, len(rewards_per_episode)+1, window))

    plt.figure()
    plt.plot(episodes, means, label="Mean Reward")
    plt.fill_between(episodes, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.3)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Missile Evade Training Reward over Time")
    plt.legend()
    plt.savefig("missile_evade_training_trends.png")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    train(args)
