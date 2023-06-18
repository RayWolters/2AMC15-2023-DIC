import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from agents import BaseAgent
from collections import deque
np.set_printoptions(threshold=sys.maxsize)

device = torch.device("mps")

class DQN(nn.Module):
    def __init__(self, input_shape):
        super(DQN, self).__init__()

        # Input shape should be (channels, height, width)
        self.input_shape = input_shape

        # Define the Convolutional layers
        self.conv = nn.Sequential(
            # First Convolutional Layer
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1),  # IMPORTANT FOR TESTING AND TWEAKING
            nn.ReLU(),

            # Second Convolutional Layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # IMPORTANT FOR TESTING AND TWEAKING
            nn.ReLU(),

            # # Max Pooling Layer
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # Third Convolutional Layer
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # IMPORTANT FOR TESTING AND TWEAKING
            nn.ReLU()
        )

        # Calculate the output size after Convolutional layers
        conv_out_size = self._get_conv_out(input_shape)

        # Define the Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),  # IMPORTANT FOR TESTING AND TWEAKING
            # nn.ReLU(),
            # nn.Linear(128, 128),  # IMPORTANT FOR TESTING AND TWEAKING
            nn.ReLU(),
            nn.Linear(128, 9)  # IMPORTANT FOR TESTING AND TWEAKING
        )

    def _get_conv_out(self, shape):
        # Helper function to calculate the output dimensions after Conv layers
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Forward pass through Convolutional layers
        x = self.conv(x.to(device))
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Forward pass through Fully Connected layers
        x = self.fc(x.to(device))
        return x


class DQLAgent(BaseAgent):
    def __init__(
            self,
            agent_number,
            input_dim,
            alpha=0.001,  # IMPORTANT FOR TESTING AND TWEAKING
            gamma=0.99,  # IMPORTANT FOR TESTING AND TWEAKING
            epsilon=1.0,
            training=True,
            ddqn=True
    ):
        """
            Q-Learning agent for grid cleaning.
            Args:
                agent_number: The index of the agent in the environment.
                alpha: Learning rate (default: 0.1).
                gamma: Discount factor (default: 0.9).
                epsilon: Exploration rate (default: 0).
                training: Whether agent is in training mode (default: True)
        """
        super().__init__(agent_number)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.ddqn = ddqn

        self.epsilon_min = 0.1

        self.already_visited = set()
        self.cleaned_tiles = set()
        self.grid_state = None
        self.last_state = None
        self.second_last_state = None
        self.batch_size = 100  # IMPORTANT FOR TESTING AND TWEAKING

        self.input_dim = input_dim
        self.dqn = DQN(self.input_dim).to(device)
        self.target_dqn = DQN(self.input_dim).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=3000)  # IMPORTANT FOR TESTING AND TWEAKING

    def process_reward(
            self,
            observation: np.ndarray,
            reward: float,
            info: dict,
            state: np.ndarray,
            action: int,
            old_state: np.ndarray
    ):
        """
            Process reward based on given reward
            Args:
                observation: Observation corresponding to reward.
                reward: Reward gained.
                info: Info corresponding to reward.
                state: State corresponding to reward.
                action: Action that reward was earned on.
                old_state: Old state before reward.
        """
        agent_pos = info['agent_pos'][self.agent_number]
        flat_state = tuple(state.flatten())

        if info['agent_charging'][self.agent_number]:
            reward = 1
            return reward

        # If not making a move, give bad reward
        if action == 4:
            reward = -1
            return reward

        # If returning to previous state, give bad reward
        if not self.second_last_state:
            self.second_last_state = flat_state
        elif not self.last_state:
            self.last_state = flat_state
        else:
            if flat_state == self.second_last_state and \
                    info['agent_moved'][self.agent_number]:
                reward = -0.9
                return reward
            elif not info['agent_moved'][self.agent_number]:
                return reward
            self.second_last_state = self.last_state
            self.last_state = flat_state

        # If moving to state that agent has already been in, give bad reward
        if flat_state in self.already_visited and \
                info['agent_moved'][self.agent_number]:
            reward = -0.5
            return reward

        # If dirt cleaned with move, update grid_state and add tile to
        # cleaned tiles
        if reward == 0.9:
            self.cleaned_tiles.add(agent_pos)
            self.grid_state[agent_pos[0]][agent_pos[1]] = 3

        return reward

    def take_action(self, observation, info):
        state = self.get_state_from_info(observation, info)
        self.already_visited.add(tuple(state.flatten()))

        if np.random.uniform() < self.epsilon and self.training:
            # Exploration: Select a random action
            action = random.randint(0, 8)
        else:
            # Exploitation: Select the action with the highest Q-value
            with torch.no_grad():
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.dqn(state_tensor)
                action = q_values.argmax().item()

        return action

    def decay_epsilon(self, iters):
        if self.epsilon > self.epsilon_min and self.training:
            self.epsilon -= (1/iters)

    def update_q_values(self):
        if len(self.memory) < self.batch_size:
            # Not enough samples in memory to create a batch
            return

        batch = random.sample(self.memory, self.batch_size)

        # Process the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.stack(next_states),
                                   dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Compute current Q values
        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(
            1)).squeeze()

        if self.ddqn:
            # Compute next Q values using the online network for selecting the action
            # and the target network for evaluating the action
            with torch.no_grad():
                # Use online network to select actions
                selected_actions = self.dqn(next_states).max(1)[1].unsqueeze(1)
                # Use target network to evaluate the selected actions
                next_q_values = self.target_dqn(next_states).gather(1,
                                                                    selected_actions).squeeze()
        else:
            # Compute next Q values using the target network
            with torch.no_grad():
                next_q_values = self.target_dqn(next_states).max(1)[0]


        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss and perform optimization
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_state_from_info(self, observation: np.ndarray,
                            info: dict) -> np.ndarray:
        """
        Get state from given observation and info for use with CNN.

        Args:
            observation: Observation to compute state from.
            info: Info to compute position from.
        """

        # Extract agent position from info
        agent_pos = info['agent_pos'][self.agent_number]

        # Initialize grid state if it is None
        if self.grid_state is None:
            self.grid_state = self.get_dirtless_grid(observation)

        # Make a copy of the grid to avoid modifying the original
        grid_state = np.copy(self.grid_state)

        # Create channels
        # Channel 1: Walls and obstacles (1 for walls/obstacles, 0 otherwise)
        walls_channel = np.where(grid_state == 1, 1, 0) + np.where(
            grid_state == 2, 1, 0)

        # empty_channel = np.where(grid_state == 0, 1, 0)
        dirt_channel = np.where(grid_state == 3, 1, 0)
        charge_channel = np.where(grid_state == 4, 1, 0)

        agent_pos_channel = np.zeros_like(grid_state)
        agent_pos_channel[agent_pos[0], agent_pos[1]] = 1

        final_state = np.stack(
            [walls_channel, dirt_channel, charge_channel,
             agent_pos_channel], axis=0)

        return final_state

    def get_dirtless_grid(
            self,
            observation
    ) -> np.array:
        """
            Get dirtless grid from given observation.
            Args:
                observation: Grid to compute dirtless grid from.
        """
        grid = observation.copy()
        dirt_mask = (grid == 3)
        grid[dirt_mask] = 0
        return grid

    def reset_parameters(self) -> None:
        """
            Reset agent parameters for when grid is completed.
        """
        self.already_visited = set()
        self.cleaned_tiles = set()
        self.last_state = None
        self.second_last_state = None
        self.grid_state = None

    def remember(self, state, action, reward, next_state, done):
        if done:
            self.memory.append((state, action, reward, next_state, 1))
        else:
            self.memory.append((state, action, reward, next_state, 0))

    def synchronize_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
