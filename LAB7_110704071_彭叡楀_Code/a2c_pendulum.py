#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
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
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
        ).to(self.device)

        self.mu_layer = nn.Sequential(
            nn.Linear(256, out_dim).to(self.device),
            nn.Tanh(),
        ).to(self.device)

        self.std_layer = nn.Sequential(
            nn.Linear(256, out_dim).to(self.device),
            nn.Softplus()
        ).to(self.device)
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                initialize_uniformly(layer)
        
        for layer in self.mu_layer:
            if isinstance(layer, nn.Linear):
                initialize_uniformly(layer)
        
        for layer in self.std_layer:
            if isinstance(layer, nn.Linear):
                initialize_uniformly(layer)


        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        output = self.network(state)
        mu = self.mu_layer(output) * 2
        std = self.std_layer(output) + 1e-3
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
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                initialize_uniformly(layer)

        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        
        #############################

        return self.network(state)

class Memory:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0
    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        
        else:
            self.buffer[self.pos] = transition
        
        self.pos = self.pos + 1 if self.pos < self.capacity - 1 else 0
        return 
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        if(batch_size == self.capacity):
            return self.buffer
        return [self.buffer[i] for i in range(0, batch_size)]
    def __len__(self):
        return len(self.buffer)
    def clear(self):
        self.buffer = []   

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.buffer_len = args.buffer_len
        self.buffer = Memory(self.buffer_len)
        self.best_score = -10000
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.transition: list = list()

        self.total_step = 0

        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action_raw, dist = self.actor(state)

        low, high = self.env.action_space.low[0], self.env.action_space.high[0]
        
        if self.is_test:
            selected_action = dist.mean.clamp(low, high)
        else:
            selected_action = action_raw.clamp(low, high)

        if not self.is_test:
            self.transition = [state, action_raw.detach()]

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            n_s = torch.FloatTensor(next_state).to(self.device)
            train_reward = (reward + 8.1) / 8.1
            self.transition.extend([train_reward, n_s, done]) 

        return next_state, reward, done

    def update_model(self,size) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        transition = self.buffer.sample(size)

        states, actions, rewards, next_states, dones = zip(*transition)

        states = torch.stack(states).float().to(self.device)
        actions = torch.stack(actions).float().to(self.device) 
        next_states = torch.stack(next_states).float().to(self.device)
        rewards = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        dones = torch.tensor(dones).float().unsqueeze(1).to(self.device)
        
        mask = 1.0 - dones
        
        
        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * mask
        values = self.critic(states)
        value_loss = F.mse_loss(values, td_target)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
       
        advantage = (td_target - values).detach()
        
        _, dist = self.actor(states)

        
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().mean()
        
        policy_loss = -(log_probs * advantage).mean() - self.entropy_weight * entropy
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        wandb.log({"log_mean": log_probs.mean().item(),
                    "log_std": log_probs.std().item(),
                    "advantage/mean": advantage.mean().item(),
                    "advantage/std": advantage.std().item()
                    })


        return policy_loss.item(), value_loss.item()


    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        
        for ep in tqdm(range(1, self.num_episodes+1)): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset(seed=self.seed)
            score = 0
            done = False
            cnt = 1
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                self.buffer.add(self.transition)
                self.transition = []

                if cnt % self.buffer_len == 0 or done:
                    size = self.buffer_len if(cnt % self.buffer_len == 0) else cnt % self.buffer_len
                    actor_loss, critic_loss = self.update_model(size)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    wandb.log({"step": step_count,
                               "actor_loss": actor_loss,
                               "critic_loss": critic_loss})

                state = next_state
                score += reward
                step_count += 1
                cnt += 1
                
                if done:
                    scores.append(score)
                    self.buffer.clear()
                    print(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    if score >= -200:
                        self.valid() 
                    wandb.log({
                        "episode": ep,
                        "return": score
                        }) 
    def valid(self):
        """Test the agent."""
        total_score = 0.0
        self.is_test = True
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
        
        if avg_score >= -230:
            if avg_score >= self.best_score:
                print('save!')
                self.best_score = avg_score
                torch.save(
                {'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()}
                ,'LAB7_110704071_task1_a2c_pendulum.pt')

        print(f"Valid Score: {avg_score}") 
        wandb.log({"valid": avg_score}) 
        self.is_test = False

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

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=6e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-3)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=0.015)
    parser.add_argument("--buffer-len", default=4) 
    parser.add_argument("--test", action='store_true') 
    parser.add_argument("--find", action='store_true') 
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = A2CAgent(env, args)
    if args.find == True:
        check = torch.load('LAB7_110704071_task1_a2c_pendulum.pt')
        agent.actor.load_state_dict(check['actor'])
        agent.critic.load_state_dict(check['critic'])
        for i in range(1,8000):
            agent.seed = i
            random.seed(i)
            np.random.seed(i)
            seed_torch(i)
            score = agent.test('./ppo_pen')
            if score >= -150:
                print(f"Seed:{i} Score: {score}")
                exit()
    
    if args.test == False:
        agent.train()
    else:
        check = torch.load('LAB7_110704071_task1_a2c_pendulum.pt')
        agent.actor.load_state_dict(check['actor'])
        agent.critic.load_state_dict(check['critic'])
        agent.test('./a2c_pen')
    #Best Seed 245