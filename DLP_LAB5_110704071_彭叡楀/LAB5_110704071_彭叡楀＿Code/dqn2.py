# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import namedtuple,deque
import wandb
import argparse
import time
gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self,num_states ,num_actions):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########
        print(num_states,num_actions)
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        #print(x.shape)
        return self.network(x / 255.0)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class MultiStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []

    def add(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) < self.n_step:
            return None

        total_reward, next_state, finish = 0, None, False
        for id, batch in enumerate(self.buffer):
            state, action, reward, next, done = batch
            total_reward += (self.gamma ** id) * reward
            next_state = next
            if done:
                finish = True
                break

        state, action, _, _, _ = self.buffer[0]
        self.buffer.pop(0)
        
        if finish:
            self.buffer.clear()
        
        return (state, action, total_reward, next_state, finish)

    def reset(self):
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, steps=3):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.steps = steps

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = (abs(error) + 1e-7) ** self.alpha
        self.pos = self.pos + 1 if(self.pos < self.capacity - 1) else 0
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        prior = self.priorities if(len(self.buffer) == self.capacity) else self.priorities[:self.pos]
        prob = prior / prior.sum()
        
        id = random.choices(range(len(self.buffer)), weights=prob, k=batch_size)

        weight = (len(self.buffer) * prob[id]) ** (-self.beta)
        weight = torch.tensor(weight / weight.max(), dtype=torch.float32)

        ########## END OF YOUR CODE (for Task 3) ########## 
        return [self.buffer[i] for i in id], id, weight
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for i in range(len(indices)):
            id = indices[i]
            error = errors[i]
            self.priorities[id] = (abs(error) + 1e-7) ** self.alpha

        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def __len__(self):
        return len(self.buffer)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0
    def add(self, transition):
        ########## YOUR CODE HERE (for Task 1) ########## 
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        
        else:
            self.buffer[self.pos] = transition
        
        self.pos = self.pos + 1 if self.pos < self.capacity - 1 else 0
        ########## END OF YOUR CODE (for Task 1) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 1) ########## 
        return random.sample(self.buffer, batch_size)
        ########## END OF YOUR CODE (for Task 1) ##########
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array", repeat_action_probability=0.1)
        self.test_env = gym.make(env_name, render_mode="rgb_array", repeat_action_probability=0.1)
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        self.state_dim = self.env.observation_space.shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(self.state_dim,self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.state_dim,self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.PER = args.PER
        self.DDQN = args.DDQN
        self.STEP = args.STEP
        self.step = args.n_step


        self.memory = PrioritizedReplayBuffer(args.memory_size) if args.PER else ReplayBuffer(args.memory_size)
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.td = 1.0
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.multi_step_buffer = MultiStepBuffer(n_step=self.step, gamma=self.gamma)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=4000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            
            self.multi_step_buffer.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)

                if args.STEP:
                    multi_step_transition = self.multi_step_buffer.add((state, action, reward, next_state, done))
                    if multi_step_transition:
                        if self.PER:
                            self.memory.add((state, action, reward, next_state, done),1.0)
                        else:
                            self.memory.add((state, action, reward, next_state, done))

                else:
                    if self.PER:
                        self.memory.add((state, action, reward, next_state, done),1.0)
                    else:
                        self.memory.add((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 200000 == 0:
                    model_path = os.path.join(self.save_dir, f"LAB5_110704071_task3_pong{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved checkpoint to {model_path} at step {self.env_count}")

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward >= self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        if self.PER:
            transitions, indices, weights = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            weights = weights.to(self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
        
        ########## END OF YOUR CODE ##########
        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(batch.state).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(batch.next_state).astype(np.float32)).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            if self.DDQN:
                next_actions = self.q_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(1)[0]
            
        
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        tds = q_values - target_q_values
        loss = (tds * tds * weights).mean() if self.PER else F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ########## END OF YOUR CODE ##########  
    
        if self.PER:
            tds.detach().cpu().numpy()
            self.memory.update_priorities(indices, tds)
        
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run-enhance")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.8)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=4)
    parser.add_argument("--STEP",action="store_true")
    parser.add_argument("--PER",action="store_true")
    parser.add_argument("--DDQN",action="store_true")

    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-PONG", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()