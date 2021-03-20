import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np

GAMMA = 0.99
EPS = np.finfo(np.float32).eps.item()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc0 = nn.Linear(25, 64)
        self.fc1 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        action = F.softmax(self.actor(x), dim=0)
        critic = self.critic(x)
        return action, critic


# TODO: Dynamic model architecture
class Brain:
    def __init__(self):
        self.brain = Model()
        self.rewards_memories = []
        self.act_probabilities_memories = []
        self.critic_values_memories = []
        # TODO: Dynamic learning rate
        self.optimizer = opt.Adam(self.brain.parameters(), lr=0.001)
        self.huber = torch.nn.SmoothL1Loss()
        self.perception = None

    # TODO: Maybe normalize inputs?
    def init_perception(self, perception):
        # TODO: See if we can work with half-precision
        self.perception = torch.flatten(torch.from_numpy(perception.astype(np.float32)))

    def decision(self):
        a_probabilities, critic = self.brain.forward(self.perception)
        action = torch.multinomial(a_probabilities, 1)
        a_log = torch.log(a_probabilities[action])
        self.act_probabilities_memories.append(a_log)
        self.critic_values_memories.append(critic)
        return action.item()

    def remember_reward(self, reward, perception):
        self.rewards_memories.append(reward)
        if perception is not None:
            self.init_perception(perception[0])

    def dream(self):
        returns = []
        discounted_sum = 0
        for r in self.rewards_memories[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + EPS)
        returns = returns.tolist()

        history = zip(self.act_probabilities_memories, self.critic_values_memories, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(
                self.huber(torch.unsqueeze(value, dim=0), torch.Tensor([ret]))
            )
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        loss_value.backward()
        self.optimizer.step()

        # Clear memory and gradients
        self.act_probabilities_memories.clear()
        self.critic_values_memories.clear()
        self.rewards_memories.clear()
        self.optimizer.zero_grad()
