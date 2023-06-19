"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import time
from tqdm import trange

try:
    from world import Environment

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.greedy_agent import GreedyAgent
    from agents.random_agent import RandomAgent
    from agents.q_learning_agent import QLearningAgent
    from agents.dqn_and_ddqn_agent import DQLAgent
    from agents.duel_and_ddqn_agent import DuelQLAgent
    from agents.dqn_per_agent import PERDQLAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from world import Environment

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.greedy_agent import GreedyAgent
    from agents.random_agent import RandomAgent
    from agents.q_learning_agent import QLearningAgent

def plot(y):
    x = np.arange(len(y))
    plt.plot(x, y)
    # Adding labels and title
    plt.xlabel('The times of reset the envrionment')
    plt.ylabel('The iters used in before reset ')
    # Displaying the chart
    plt.show()

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--out", type=Path, default=Path("results/"),
                   help="Where to save training results.")

    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, out: Path, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        start_pos = [(2, 1)]

        env = Environment(grid, no_gui, n_agents=1, agent_start_pos=start_pos,
                          reward_fn=Environment.simple_reward_function,
                          sigma=sigma, target_fps=fps, random_seed=random_seed)
        obs, info = env.get_observation()

        # Set up the agents from scratch for every grid
        # Add your agents here

        channels_used = 4

        agents = [
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0]))
            ),
            DuelQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0]))
            ),
            PERDQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0]))
            ),
        ]

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            start_time = time.time()
            for _ in trange(iters):
                old_state = agent.get_state_from_info(
                    obs, info)  # Convert observation to state
                action = agent.take_action(obs, info)

                agent.decay_epsilon(iters)

                obs, reward, terminated, info, actual_action = env.step([action])
                new_state = agent.get_state_from_info(
                    obs, info)  # Convert next observation to state

                reward = agent.process_reward(
                    obs,
                    reward,
                    info,
                    new_state,
                    actual_action,
                    old_state
                )

                agent.remember(old_state, actual_action, reward, new_state, terminated)
                agent.update_q_values()

                if _ % (iters/40) == 0:  # IMPORTANT FOR TESTING AND TWEAKING
                    agent.synchronize_target_network()

                # If the agent is terminated, we reset the env.
                if terminated:
                    obs, info, world_stats = env.reset()
                    agent.reset_parameters()

            training_time = time.time() - start_time

            # Reset parameters and disable training mode
            agent.reset_parameters()
            agent.training = False

            obs, info, world_stats = env.reset()
            print(world_stats)
            Environment.evaluate_agent(grid, [agent], 1000, out, training_time,
                                       sigma, agent_start_pos=start_pos)
            # Environment.evaluate_plot(grid, [agent], 10000, out,
            #                           sigma, agent_start_pos=None)



if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.out,
         args.random_seed)
