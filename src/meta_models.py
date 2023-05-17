import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from numpy.random import Generator, PCG64

rng = np.random.default_rng()


# Fully connected neural network with one LSTM hidden layer
class MetaLSTM(nn.Module):
    def __init__(self, observation_size, n_actions, hidden_size=48):
        super(MetaLSTM, self).__init__()

        self.observation_size = observation_size
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=(observation_size + n_actions + 1), hidden_size=hidden_size)
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self.initial_hidden = nn.Parameter(torch.zeros(1, 1, hidden_size))  # h0
        self.initial_cell = nn.Parameter(torch.zeros(1, 1, hidden_size))  # c0

    def forward(self, observation, prev_reward, prev_action, hidden=None, cell=None):
        """
        Inputs:
            observation: Tensor,
            prev_reward: int,
            prev_action: int,

        Outputs:
            value: float,
            action_prob: Tensor[action_number]
        """
        observation = torch.FloatTensor(observation).view(-1)
        prev_action = F.one_hot(torch.tensor(prev_action), self.n_actions).view(-1)
        prev_reward = torch.FloatTensor([prev_reward])

        lstm_input = torch.cat((observation, prev_reward, prev_action)).view(1, 1, -1)
        if hidden is None and cell is None:
            output, (hidden, cell) = self.lstm(lstm_input, (self.initial_hidden, self.initial_cell))
        else:
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        return self.critic(output), F.softmax(self.actor(output), dim=-1), (hidden, cell)


class MetaAgent:
    """
    The training loops over:
    -> episodes: num_iterations
    -----------> bandits: 3 orders of bandits randomly shuffled through rng.shuffle(bandits)
    --------------------> steps: 80
    """

    def __init__(self, model, optimizer, max_grad_norm=1.0, discount_factor=0.9, beta_v=0.05, beta_e=0.05):
        self.model = model
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.beta_v = beta_v
        self.beta_e = beta_e
        self.rewards_history = defaultdict(list)
        self.losses_history = defaultdict(list)
        self.stats = defaultdict(lambda: defaultdict(list))
        self.max_grad_norm = max_grad_norm

    def get_discounted_returns(self, rewards, values):
        returns = []
        advantages = []
        advantage = 0
        discounted_return = 0  # values[-1].item()
        for t in reversed(range(len(rewards))):
            discounted_return = rewards[t] + self.discount_factor * discounted_return

            v = values[t]
            if t == len(rewards) - 1:
                next_v = 0  # values[-1].item()
            else:
                next_v = values[t + 1]
            td_error = rewards[t] + self.discount_factor * next_v - v
            advantage = td_error + self.discount_factor * advantage

            returns.insert(0, discounted_return)
            advantages.insert(0, advantage)
        return torch.tensor(returns), torch.tensor(advantages)

    def learn(self, rewards, values, log_prob_actions, entropies):
        self.model.train()
        self.optimizer.zero_grad()
        values = torch.cat(values)
        log_prob_actions = torch.cat(log_prob_actions)
        entropies = torch.stack(entropies)

        returns, advantages = self.get_discounted_returns(rewards, values)
        policy_loss = -(log_prob_actions * advantages.detach()).sum()
        value_loss = F.smooth_l1_loss(values, returns, reduction='sum')
        loss = policy_loss + self.beta_v * value_loss - self.beta_e * entropies.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def train(self, bandits, num_iterations, save_model_every=500):
        self.rewards_history = defaultdict(list)
        self.losses_history = defaultdict(list)
        writer = SummaryWriter()
        t_range = tqdm(range(num_iterations))
        global_step = 0
        for ep in t_range:
            # print(f'current epoch: {ep}')
            rng.shuffle(bandits)
            for bandit in bandits:
                log_prob_actions = []
                values = []
                rewards = []
                entropies = []
                actions = []
                reward = 0.0
                action = torch.tensor(0)
                hidden = None
                cell = None
                bandit.reset()
                for step in range(bandit.maxtimestep):
                    value, action_space, (hidden, cell) = self.model([step], reward, action.item(), hidden=hidden,
                                                                     cell=cell)
                    action_distribution = distributions.Categorical(action_space)
                    action = action_distribution.sample()
                    log_prob_action = action_distribution.log_prob(action)
                    log_prob_actions.append(log_prob_action.squeeze(0))
                    entropies.append(action_distribution.entropy())

                    reward, done, _, corr = bandit.act(action.item())

                    values.append(value.view(-1))
                    rewards.append(reward)
                    actions.append(action.item())

                loss = self.learn(rewards, values, log_prob_actions, entropies)

                t_range.set_description("Current loss: {}".format(loss))
                self.rewards_history[bandit.name].append(sum(rewards) / len(rewards))
                self.losses_history[bandit.name].append(loss)

                # print (f'Current bandit: {bandit.name} with P = {bandit.P}')
                # ############# TENSORBOARD ########################

                global_step = global_step + 1
                writer.add_scalars("/training/rewards", {bandit.name: self.rewards_history[bandit.name][-1]},
                                   global_step)
                writer.add_scalars("/training/losses", {bandit.name: self.losses_history[bandit.name][-1]}, global_step)

                # ##################################################
            if ep % save_model_every == 0:
                print(f'Saving model after {ep} episodes of training')
                torch.save(self.model.state_dict(), f'meta-train-{ep}.pt')

        writer.flush()
        writer.close()

    def evaluate(self, bandits, num_iterations):
        print("NewNew")
        with torch.no_grad():
            self.rewards_history = defaultdict(list)
            self.stats = defaultdict(lambda: defaultdict(list))
            writer = SummaryWriter()
            t_range = tqdm(range(num_iterations))
            for ep in t_range:
                for bandit in bandits:
                    rewards = []
                    actions = []
                    corrects = []
                    hiddens = []
                    states =[]
                    cells = []
                    reward = 0.0
                    action = torch.tensor(0)
                    state = 0
                    hidden = None
                    cell = None
                    bandit.reset()
                    for step in range(bandit.maxtimestep):
                        value, action_space, (hidden, cell) = self.model([step], reward, action.item(), hidden=hidden,
                                                                         cell=cell)
                        action_distribution = distributions.Categorical(action_space)
                        action = action_distribution.sample()

                        reward, done, _, correct = bandit.act(action.item())

                        rewards.append(reward)
                        actions.append(action.item())
                        corrects.append(correct)
                        hiddens.append(hidden.detach().squeeze().numpy())
                        cells.append(cell.detach().squeeze().numpy())

                        if reward == 1:
                            state = action.item()
                        elif reward == 0:
                            state = 1 - action.item()
                        states.append(state)

                    self.stats['rewards'][bandit.name].append(rewards)  # num_iterationsX80
                    self.stats['actions'][bandit.name].append(actions)  # num_iterationsX80
                    self.stats['corrects'][bandit.name].append(corrects)  # num_iterationsX80
                    self.stats['states'][bandit.name].append(states)  # num_iterationsX80
                    self.stats['LSTMhidden'][bandit.name].append(hiddens)  # num_iterationsX80X48
                    self.stats['LSTMcell'][bandit.name].append(cells)  # num_iterationsX80X48
                    self.stats['P'][bandit.name].append(bandit.P)  # num_iterationsX1

                    # t_range.set_description(f"Evaluation on bandits")
                    self.rewards_history[bandit.name].append(sum(rewards)/len(rewards))

                    # print (f'Epoch [{ep}/{100}], bandit {bandit}], Loss: {loss:.4f}')
                    # ############# TENSORBOARD ########################

                writer.add_scalars("/test/rewards", {'0th_order': self.rewards_history['0th_order'][-1],
                                                    '1st_order': self.rewards_history['1st_order'][-1],
                                                    '2nd_order': self.rewards_history['2nd_order'][-1]}, ep)
        writer.close()