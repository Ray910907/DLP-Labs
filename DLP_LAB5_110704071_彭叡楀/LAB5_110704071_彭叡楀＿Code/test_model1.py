import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import imageio
import os
import argparse

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    model = DQN(output_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    final_score = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        frames = []

        while not done:
            frame = env.render()
            frames.append(frame)

            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_tensor).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        out_path = os.path.join(args.output_dir, f"cartpole_eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f.astype(np.uint8))
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        final_score += total_reward
    
    print(f"Final AVG Score:{final_score / args.episodes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./cartpole_eval_videos")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)
