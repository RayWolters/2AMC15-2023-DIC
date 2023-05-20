import numpy as np
from random import randint
from agents import BaseAgent
from .pathfinding import calulate_path_to_charger


class QLearningAgent(BaseAgent):
    def __init__(self, agent_number, alpha=0.1, gamma=0.9, epsilon=0):
        """Q-Learning agent for grid cleaning.

        Args:
            agent_number: The index of the agent in the environment.
            alpha: Learning rate (default: 0.1)
            gamma: Discount factor (default: 0.9)
            epsilon: Exploration rate (default: 0)
        """
        super().__init__(agent_number)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.charger_pos = None
        self.way_back = None
        self.already_visited = set()
        self.clean_tiles = set()
        self.clean_tiles_int = 0

    def process_reward(
            self,
            observation: np.ndarray,
            reward: float,
            info: dict
    ):
        if info['agent_pos'][self.agent_number] in self.already_visited and \
                info['agent_moved'][self.agent_number]:
            reward = -5

        if reward > 10:
            self.clean_tiles.add(info['agent_pos'][self.agent_number])
            self.clean_tiles_int += 1

        return reward

    def take_action(
            self,
            observation: np.ndarray,
            info: dict
    ) -> int:
        state = self.get_state_from_info(observation, info)
        self.already_visited.add(info['agent_pos'][self.agent_number])

        if 3 not in observation.flatten():
            if not self.way_back:
                self.way_back = calulate_path_to_charger(
                    observation,
                    info,
                    self.agent_number,
                    self.charger_pos
                )
            action = self.way_back.pop(0)
            return action

        if np.random.uniform() < self.epsilon:
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
        agent_pos = info["agent_pos"][self.agent_number]

        surroundings = self._get_surroundings(observation, agent_pos)
        # surroundings = tuple(observation.flatten())

        # clean_tiles = tuple(self.clean_tiles)
        # clean_tiles = self.clean_tiles_int

        # Define the state representation by including the agent's position
        state = (surroundings, agent_pos)
        return state

    def reset_parameters(self) -> None:
        self.way_back = None
        self.already_visited = set()
        self.clean_tiles = set()
        self.clean_tiles_int = 0

    def _get_best_action(
            self,
            state: tuple
    ) -> int:
        # Get the action with the highest Q-value for the given state
        q_values = [self.q_table.get((state, a), 0.0) for a in range(5)]
        return int(np.argmax(q_values))

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

    @staticmethod
    def _get_surroundings(
            obs: np.ndarray,
            pos: tuple,
            visibility_radius: int = 2
    ) -> tuple:
        i, j = pos
        directions = [(0, 1), (0, -1), (1, 0),
                      (-1, 0)]  # right, left, down, up
        surroundings = []

        for direction in directions:
            for r in range(1, visibility_radius + 1):
                new_i, new_j = i + r * direction[0], j + r * direction[1]

                # Check if the new indices are within the boundaries of the grid
                if new_i < 0 or new_i >= obs.shape[0] or new_j < 0 or new_j >= \
                        obs.shape[1]:
                    break

                # If the cell is a wall, add it and stop looking further in this direction
                if obs[new_i, new_j] == 1 or obs[new_i, new_j] == 2:
                    surroundings.append(obs[new_i, new_j])
                    break

                surroundings.append(obs[new_i, new_j])

        return tuple(surroundings)

