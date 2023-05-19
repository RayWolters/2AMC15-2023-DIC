from collections import deque
import numpy as np

def calulate_path_to_charger(observation, info, agent_number, charger_pos):
    # Create a grid representation to keep track of visited cells and obstacles
    grid = np.copy(observation)

    # Define the actions for moving in each direction (up, down, left, right)
    actions = [0, 1, 2, 3]

    # Get the agent's current position
    agent_pos = info['agent_pos'][agent_number]

    # Create a queue for BFS traversal
    queue = deque()
    queue.append(agent_pos)

    # Create a dictionary to store the parent cells for constructing the path
    parent = {agent_pos: None}

    # Perform BFS to find the shortest path to the charger
    while queue:
        current_pos = queue.popleft()

        # Check if the current position is the charger
        if current_pos == charger_pos:
            break

        # Explore the neighboring cells
        for action in actions:
            next_pos = _get_next_position(current_pos, action)

            # Check if the next position is valid and unvisited
            if _is_valid_position(next_pos,
                                       grid) and next_pos not in parent:
                queue.append(next_pos)
                parent[next_pos] = current_pos
                grid[next_pos[0], next_pos[
                    1]] = 5  # Mark the cell as visited

    # Reconstruct the path from the charger to the agent's position
    path = []
    current_pos = charger_pos

    while current_pos != agent_pos:
        parent_pos = parent[current_pos]
        action = _get_action_from_positions(parent_pos, current_pos)
        path.append(action)
        current_pos = parent_pos

    # Reverse the path to get the actions from the agent's position to the charger
    path.reverse()

    return path


def _get_next_position(pos, action):
    # Calculate the next position based on the current position and the action
    if action == 0:  # Up
        return pos[0] - 1, pos[1]
    elif action == 1:  # Down
        return pos[0] + 1, pos[1]
    elif action == 2:  # Left
        return pos[0], pos[1] + 1
    elif action == 3:  # Right
        return pos[0], pos[1] - 1


def _is_valid_position(pos, grid):
    # Check if the position is within the grid boundaries and not an obstacle (2) or wall (1)
    x, y = pos
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and \
           grid[x, y] not in [1, 2]


def _get_action_from_positions(prev_pos, current_pos):
    # Determine the action based on the previous position and the current position
    if prev_pos[0] == current_pos[0] - 1:  # Up
        return 3
    elif prev_pos[0] == current_pos[0] + 1:  # Down
        return 2
    elif prev_pos[1] == current_pos[1] - 1:  # Left
        return 0
    elif prev_pos[1] == current_pos[1] + 1:  # Right
        return 1