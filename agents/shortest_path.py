import numpy as np
from collections import deque
from itertools import permutations

# Define the movement directions (up, down, left, right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# Breadth-first search (BFS) function to find the shortest path between two points
# Breadth-first search (BFS) function to find the shortest path between two points
def bfs(grid, start, end):
    queue = deque([(*start, 0)])
    visited = set([start])
    while queue:
        x, y, steps = queue.popleft()
        if (x, y) == end:
            return steps + 1 # plus 1 to count the first step
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != 1 and (nx, ny) not in visited:
                queue.append((nx, ny, steps + 1))
                visited.add((nx, ny))
    return float('inf')


# Function to compute the most efficient path
def compute_path(arr, start):
    # Find the points to visit (points with value 3) and the end point (value 4)
    points_to_visit = [(i, j) for i in range(len(arr)) for j in
                       range(len(arr[0])) if arr[i][j] == 3]
    end_point = [(i, j) for i in range(len(arr)) for j in range(len(arr[0])) if
                 arr[i][j] == 4][0]

    # Calculate the shortest path between all pairs of points to visit (including the end point)
    shortest_paths = {(p1, p2): bfs(arr, p1, p2) for p1 in points_to_visit for
                      p2 in points_to_visit + [end_point]}

    # Find the shortest path that visits all points to visit and ends at the end point using dynamic programming
    min_path, min_distance = None, float('inf')
    for perm in permutations(range(len(points_to_visit))):
        path = list(perm)
        distance = sum(shortest_paths[(
        points_to_visit[path[i - 1]], points_to_visit[path[i]])] for i in
                       range(1, len(path)))
        # add the distances from the start point to the first point to visit and from the last point to visit to the end point
        distance += bfs(arr, start, points_to_visit[path[0]]) + shortest_paths[
            (points_to_visit[path[-1]], end_point)]
        if distance < min_distance:
            min_path, min_distance = path, distance

    # Convert the path from indices to actual points
    actual_path = [start] + [points_to_visit[i] for i in min_path] + [
        end_point]

    return actual_path, min_distance-2
