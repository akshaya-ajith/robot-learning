import gymnasium as gym
import panda_gym

import time
import numpy as np

from utils import *

def reach(
    env
): 

    """
    Panda robot arm reaches the goal
    args: 
        - env: gym environment
    returns: 
        - success: successfully reaches the goal or not
        - traj: the ee's trajectory
        - goal: the ee's goal position
    """

    traj = []
    state, info = env.reset()
    traj.append(state['achieved_goal']) # ee's position
    K = 0.8

    success, iteration = False, 0
    while not success and iteration < 500: 
        error = state['desired_goal'] - state['achieved_goal']
        action = K * error

        action = np.clip(action, -1.0, 1.0)

        state, reward, terminated, truncated, info = env.step(action)
        traj.append(state['achieved_goal']) # ee's position
        # slow down the motion
        time.sleep(0.01)

        if np.linalg.norm(state['desired_goal'] - state['achieved_goal']) <= 1e-2: 
            success = True

        iteration += 1

    return success, np.array(traj), state['desired_goal']

if __name__ == '__main__': 
    env = gym.make('PandaReach-v3', render_mode="human")

    epi_num, total_success = 20, 0.
    for _ in range(epi_num): 
        success, traj, goal = reach(env)
        # plot the trajectory, for debugging
        #plotTrajectory(traj, goal)
        total_success += success

    print('PandaReach-v3 success rate: ', total_success/epi_num)
