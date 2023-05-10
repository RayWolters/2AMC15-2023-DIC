import numpy as np
from random import randint
from agents import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, agent_number, alpha=0.3, gamma=0.9, epsilon=0.2):
        """Q-Learning agent for grid cleaning.

        Args:
            agent_number: The index of the agent in the environment.
            alpha: Learning rate (default: 0.1)
            gamma: Discount factor (default: 0.9)
            epsilon: Exploration rate (default: 0.1)
        """
        super().__init__(agent_number)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: dict) -> int:
        state = self.get_state_from_info(observation, info)
        if np.random.uniform() < self.epsilon:
            # Exploration: Select a random action
            action = randint(0, 4)
        else:
            # Exploitation: Select the action with the highest Q-value
            action = self._get_best_action(state)
        return action

    def update_q_values(self, state: tuple, action: int, reward: float,
                        next_state: tuple):
        q_value = self.q_table.get((state, action), 0.0)
        max_q_value = max(self.q_table.get((next_state, a), 0.0) for a in range(5))
        new_q_value = q_value + self.alpha * (reward + self.gamma * max_q_value - q_value)
        self.q_table[(state, action)] = new_q_value

    def get_state_from_info(self, observation: np.ndarray, info: dict) -> tuple:
        # Extract the relevant information from the observation and info dictionary
        agent_pos = info["agent_pos"][self.agent_number]
        # Define the state representation by including the agent's position
        state = (tuple(observation.flatten()), agent_pos)
        return state

    def _get_best_action(self, state: tuple) -> int:
        # Get the action with the highest Q-value for the given state
        q_values = [self.q_table.get((state, a), 0.0) for a in range(5)]
        print(q_values)
        return int(np.argmax(q_values))
