import numpy as np
from random import randint
from agents import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
            self,
            agent_number,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.3,
            training=True
    ):
        """Q-Learning agent for grid cleaning.

        Args:
            agent_number: The index of the agent in the environment.
            alpha: Learning rate (default: 0.1)
            gamma: Discount factor (default: 0.9)
            epsilon: Exploration rate (default: 0)
            training: Whether agent is in training mode (default: True)
        """
        super().__init__(agent_number)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.training = training
        self.already_visited = set()
        self.cleaned_tiles = set()
        self.last_state = None
        self.second_last_state = None

    def process_reward(
            self,
            observation: np.ndarray,
            reward: float,
            info: dict,
            state: tuple,
            action: int
    ):
        agent_pos = info['agent_pos'][self.agent_number]
        if action == 4:
            reward = -1000
            return reward

        if not self.second_last_state:
            self.second_last_state = state
        elif not self.last_state:
            self.last_state = state
        else:
            if state == self.second_last_state and \
                    info['agent_moved'][self.agent_number]:
                reward = -4
                return reward
            elif not info['agent_moved'][self.agent_number]:
                return reward
            self.second_last_state = self.last_state
            self.last_state = state

        if state in self.already_visited and \
                info['agent_moved'][self.agent_number]:
            reward = -2
            return reward

        if reward == 10:
            self.cleaned_tiles.add(agent_pos)
            self.grid_state[agent_pos[0]][agent_pos[1]] = 3

        return reward

    def take_action(
            self,
            observation: np.ndarray,
            info: dict
    ) -> int:
        state = self.get_state_from_info(observation, info)
        self.already_visited.add(state)

        if np.random.uniform() < self.epsilon and self.training:
            # Exploration: Select a random action
            action = randint(0, 4)
        else:
            # Exploitation: Select the action with the highest Q-value
            action = self._get_best_action(state)
        return action

    def update_q_values(
            self,
            state: tuple,
            action: int,
            reward: float,
            next_state: tuple
    ) -> None:
        q_value = self.q_table.get((state, action), 0.0)
        max_q_value = max(self.q_table.get((next_state, a), 0.0)
                          for a in range(5))
        new_q_value = q_value + self.alpha * \
            (reward + self.gamma * max_q_value - q_value)

        self.q_table[(state, action)] = new_q_value

    def get_state_from_info(
            self,
            observation: np.ndarray,
            info: dict
    ) -> tuple:
        # Extract the relevant information from the info dictionary
        agent_pos = info['agent_pos'][self.agent_number]

        surroundings = self._get_surroundings(observation, agent_pos)
        number_of_cleaned_tiles = len(self.cleaned_tiles)
        cleaned_tiles = tuple(self.cleaned_tiles)

        # Define the state representation by including the agent's position, 
        # surroundings and (amount of tiles cleaned or locations of cleaned
        # tiles)
        state = (cleaned_tiles, surroundings, agent_pos)
        return state

    def reset_parameters(self) -> None:
        self.already_visited = set()
        self.cleaned_tiles = set()
        self.last_state = None
        self.second_last_state = None

    def _get_best_action(
            self,
            state: tuple
    ) -> int:
        # Get the action with the highest Q-value for the given state
        q_values = [self.q_table.get((state, a), 0.0) for a in range(5)]
        if all(v == 0.0 for v in q_values) and self.training:
            return randint(0, 4)
        return int(np.argmax(q_values))

    @staticmethod
    def _get_surroundings(
            obs: np.ndarray,
            pos: tuple,
            visibility_radius: int = 1
    ) -> tuple:
        i, j = pos
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        surroundings = []

        for direction in directions:
            for r in range(1, visibility_radius + 1):
                new_i, new_j = i + r * direction[0], j + r * direction[1]

                # Check if the new indices are within the boundaries of
                # the grid
                if new_i < 0 or new_i >= obs.shape[0] or new_j < 0 or \
                        new_j >= obs.shape[1]:
                    break

                # If the cell is a wall, add it and stop looking further in
                # this direction
                if obs[new_i, new_j] == 1 or obs[new_i, new_j] == 2:
                    surroundings.append(obs[new_i, new_j])
                    break

                surroundings.append(obs[new_i, new_j])

        return tuple(surroundings)

    # @staticmethod
    # def _get_surroundings(
    #         obs: np.ndarray,
    #         pos: tuple
    # ) -> tuple:
    #     i, j = pos
    #     surroundings = [obs[i - 1, j],
    #                     obs[i, j - 1],
    #                     obs[i, j + 1],
    #                     obs[i + 1, j]]
    #
    #     # Visibility of 2 tiles, but can not see through walls/obstacles
    #     if obs[i - 1, j] != 1 and obs[i - 1, j] != 2:
    #         surroundings.append(obs[i - 2, j])
    #     if obs[i + 1, j] != 1 and obs[i + 1, j] != 2:
    #         surroundings.append(obs[i + 2, j])
    #     if obs[i, j - 1] != 1 and obs[i, j - 1] != 2:
    #         surroundings.append(obs[i, j - 2])
    #     if obs[i, j + 1] != 1 and obs[i, j + 1] != 2:
    #         surroundings.append(obs[i, j + 2])
    #
    #     return tuple(surroundings)
