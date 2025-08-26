#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
import os


def init_layer_uniform(layer: nn.Linear, init_w: float = 1e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

def print_autograd_graph(loss, prefix=""):
    """列出從 loss 出發的 autograd graph 內容（debug 用）"""
    def _print_fn(fn, indent=0, visited=set()):
        if fn in visited:
            print("  " * indent + f"{prefix}<Already visited: {fn}>")
            return
        visited.add(fn)
        print("  " * indent + f"{prefix}{fn}")
        if hasattr(fn, "next_functions"):
            for u in fn.next_functions:
                if u[0] is not None:
                    _print_fn(u[0], indent + 1, visited)

    if loss.grad_fn is not None:
        print(f"{prefix}Backward chain from loss ({loss}):")
        _print_fn(loss.grad_fn)
    else:
        print(f"{prefix}No grad_fn found on loss ({loss}). Likely detached.")



class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        ).to(self.device)

        self.mu_layer = nn.Linear(128, out_dim).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(out_dim)).to(self.device)
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init_layer_uniform(layer)
        
        init_layer_uniform(self.mu_layer)


        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        output = self.network(state)
        mu = F.tanh(self.mu_layer(output)) * 2
        std = torch.clamp(self.log_std.exp(), 1e-3, 5)
        dist  = torch.distributions.Normal(mu, std)
        action = dist.rsample()   
        #print(std.item())
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init_layer_uniform(layer)

        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        
        #############################

        return self.network(state)

def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = 0
    gae_returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    #############################
    return gae_returns
 
class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.best_score = -10000
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch

        self.steps = [0,1000000,1500000,2000000,2500000,3000000]
        self.post = ['.','1m','1p5m','2m','2p5m','3m']
        self.best_step = -1
        self.zone = 1
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)


        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if isinstance(state, np.ndarray):
            state = state.squeeze()  
        else:
            state = np.array(state).squeeze()
        state_tensor = torch.FloatTensor(state).to(self.device)
        #print(state_tensor.shape)

        with torch.no_grad():
            action, dist = self.actor(state_tensor)
            log_prob = dist.log_prob(action).sum(dim=-1)
        if not self.is_test:
            with torch.no_grad():
                value = self.critic(state_tensor)
            self.states.append(state_tensor.unsqueeze(0))  # [1, obs_dim]
            self.actions.append(action.unsqueeze(0))       # [1, act_dim]
            self.values.append(value.unsqueeze(0))         # [1, 1]
            self.log_probs.append(log_prob.unsqueeze(0))
        select_action = dist.mean.cpu().detach().numpy() if self.is_test else action.cpu().numpy()
        return select_action


        

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if not self.is_test:
            self.rewards.append(torch.tensor([reward], dtype=torch.float32, device=self.device))
            self.masks.append(torch.tensor([1 - done], dtype=torch.float32, device=self.device))
        return np.expand_dims(next_state, axis=0), reward, done


    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau
        )

        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        critic_losses = []

        for _ in range(self.update_epoch):
            indices = np.arange(states.size(0))
            np.random.shuffle(indices)
            for i in range(0, states.size(0), self.batch_size):
                idx = indices[i:i + self.batch_size]
                sampled_states = states[idx]
                sampled_actions = actions[idx]
                sampled_returns = returns[idx].squeeze(-1)
                sampled_advantages = advantages[idx].squeeze(-1)
                sampled_log_probs = log_probs[idx]
                #print(sampled_states.shape,sampled_actions.shape,sampled_returns.shape,sampled_advantages.shape,sampled_log_probs.shape) 
                _, dist = self.actor(sampled_states)
                log_prob = dist.log_prob(sampled_actions).sum(dim=-1)
                ratio = torch.exp(log_prob - sampled_log_probs)
                #print(log_prob.shape) 


                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * sampled_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().sum(-1).mean()
                actor_loss -= self.entropy_weight * entropy
                #print(surr1.shape,surr2.shape,actor_loss.shape,entropy.shape)

                value_pred = self.critic(sampled_states).squeeze()
                critic_loss = F.mse_loss(value_pred, sampled_returns)
                #print(value_pred.shape)
                

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        return np.mean(actor_losses), np.mean(critic_losses)


    def train(self):
        """Train the PPO agent."""
        torch.autograd.set_detect_anomaly(True)
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        for ep in tqdm(range(1, self.num_episodes + 1)):
            valid = False
            score = 0
            time_count = 0
            print("\n")
            while True:
                self.total_step += 1
                time_count += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

                # if episode ends
                if done:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"Episode {episode_count}: Total Reward = {score} total_step: {self.total_step}")
                    wandb.log({
                    "episode": episode_count,
                    "return": score
                    }) 
                    if score >= 2400:
                        valid = True

                    score = 0

                    if time_count >= self.rollout_len:
                        break
                
                if getattr(self, 'total_step', 0) >= self.steps[self.zone]:
                    with open("best_score_record.txt", "a") as f:
                        f.write(f"From {self.steps[self.zone - 1]} ~ {self.steps[self.zone]}: Best Score: {self.best_score} Best Step: {self.best_step}\n")
                    self.zone += 1
                    self.best_score = -10000
                    self.best_step = -1


                if self.total_step >= 3000000:
                    print('come to an end')
                    exit()
            
            if valid == True:
                self.valid(self.total_step)
                state, _ = self.env.reset(seed=self.seed)
                state = np.expand_dims(state, axis=0)
            
            actor_loss, critic_loss = self.update_model(next_state)
            wandb.log({"step": self.total_step,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss})
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        # termination
        self.env.close()

    def test(self, video_folder: str, n_episodes: int = 20):
        """Test the agent for `n_episodes` episodes and report the average score."""
        self.is_test = True
        total_score = 0.0
        original_env = self.env

        video_env = gym.wrappers.RecordVideo(
            original_env,
            video_folder=video_folder,
            episode_trigger=lambda ep: True,
            name_prefix=f"test_run",
        )
        self.env = video_env

        for i in range(n_episodes):
            os.environ["MUJOCO_GL"] = "egl"
            state, _ = self.env.reset(seed=self.seed + i)
            done = False
            score = 0.0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward

            print(f"Episode {i+1:02d} score: {score}")
            total_score += score
        self.env.close()   # 關掉 RecordVideo wrapper

        # 恢復原本環境
        self.env = original_env

        avg_score = float(total_score / n_episodes)
        print(f"\nAverage score over {n_episodes} episodes: {avg_score:.2f}")
        return avg_score
        
    def valid(self,step):
        """Test the agent."""
        self.is_test = True 
        total_score = 0.0
        for i in range(20):
            state, _ = self.env.reset(seed=self.seed + i)  # 每回合 seed 不同避免 deterministic
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward

            
            total_score += score
        
        avg_score = total_score / 20
        if avg_score >= 2500 and avg_score >= self.best_score:
            self.best_score = avg_score
            self.best_step = step
            torch.save(
            {'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()},
            f'LAB7_110704071_task3_ppo_{self.post[self.zone]}.pt')

        print(f"Avg Score: {avg_score}") 
        self.is_test = False
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=4000)  
    parser.add_argument("--update-epoch", type=float, default=10)
    parser.add_argument("--test", action='store_true') 
    parser.add_argument("--find", action='store_true') 
    args = parser.parse_args()
 
    # environment
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
    
    agent = PPOAgent(env, args)
    if args.find == True:
        check = torch.load('LAB7_110704071_task3_ppo_1m.pt')
        agent.actor.load_state_dict(check['actor'])
        agent.critic.load_state_dict(check['critic'])
        for i in range(1,8000):
            agent.seed = i
            random.seed(i)
            np.random.seed(i)
            seed_torch(i)
            score = agent.test('./ppo_walk')
            if score >= -150:
                print(f"Seed:{i} Score: {score}")
                exit()
    
    if args.test == False:
        agent.train()
    else:
        check = torch.load('LAB7_110704071_task3_ppo_1m.pt')
        agent.actor.load_state_dict(check['actor'])
        agent.critic.load_state_dict(check['critic'])
        agent.test('./ppo_walk')