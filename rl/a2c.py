import copy

import gym
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 1000

NUM_PROCESSES = 16
NUM_ADVANCED_STEP = 5

value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(num_steps + 1, num_processes, 4)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.index = 0

    def insert(self, current_obs, action, reward, mask):
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * \
                GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)
        self.critic = nn.Linear(n_mid, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)
        actor_output = self.actor(h2)
        return critic_output, actor_output

    def act(self, x):
        value, actor_output = self(x)
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)
        return action
    
    def get_value(self, x):
        value, actor_output = self(x)
        return value

    def evaluate_actions(self, x, actions):
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

    def update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        value, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 4),
            rollouts.actions.view(-1, 1)
        )

        values = values.view(num_stpes, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        advantages = rollouts.returns[:-1] - values

        value_loss = advantages.pow(2).mean()
        action_gain = (action_log_probs * advantages.detach()).mean()

        total_loss = (value_loss * value_loss_coef - action_gain - entropy * entropy_coef)

        self.actor_critic.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        self.optimizer.step()


class Environment:
    def run(self):
        envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]

        n_in = envs[0].observation_space.shape[0]
        n_out = envs[0].action_space.n
        n_mid = 32
        actor_critic = Net(n_in, n_mid, n_out)

        global_brain = Brain(actor_critic)

        obs_shape = n_in
        current_obs = torch.zeros(NUM_PROCESSES, obs_shape)
        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])
        final_rewards = torch.zeros([NUM_PROCESSES, 1])
        obs_np = np.zeros([NUM_PROCESSES, 1])
        reward_np = np.zeros([NUM_PROCESSES, 1])
        done_np = np.zeros([NUM_PROCESSES, 1])
        each_step = np.zeros(NUM_PROCESSES)
        episode = 0

        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()
        current_obs = obs

        rollouts.observations[0].copy_(current_obs)

        for j in range(NUM_EPISODES * NUM_PROCESSES):
            for step in range(NUM_ADVANCED_STEP):
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                actions = action.squeeze(1).numpy()

                for i in range(NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(actions[i])

                    if done_np[i]:
                        if i == 0:
                            print('%d Episode: Finished after %d steps' %
                                (episode, each_step[i] + 1))
                            episode += 1

                        if each_step[i] < 195:
                            reward_np[i] = -1.0
                        else:
                            reward_np[i] = 1.0

                        each_step[i] = 0
                        obs_np[i] = envs[i].reset()

                    else:
                        reward_np[i] = 0.0
                        each_step[i] += 1
                
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards

                episode_rewards *= masks
                current_obs *= masks

                obs = torch.from_numpy(obs_np).float()
                current_obs = obs

                rollouts.insert(current_obs, action.data, reward, masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1]).detach()

            rollouts.compute_returns(next_value)

            global_brain.update(rollouts)
            rollouts.after_update()

            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print('Success!!')
                break


if __name__ == '__main__':
    cartpole_env = Environment()
    cartpole_env.run()
