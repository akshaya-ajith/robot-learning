import gymnasium as gym
import panda_gym

import time
import numpy as np

from utils import *

def pickPlace(
    env
): 

    """
    Panda robot arm picks and places the box
    args: 
        - env: gym environment
    returns: 
        - success: successfully places the box or not
        - traj: the object's trajectory
        - goal: the object's goal position
    """

    traj = []
    state, info = env.reset()
    traj.append(state['achieved_goal']) # object's position

    success, iteration = False, 0
    while not success and iteration < 500: 
        # TODO: your code here

        state, reward, terminated, truncated, info = env.step(action)
        traj.append(state['achieved_goal']) # object's position
        # slow down the motion
        time.sleep(0.01)

        success = info['is_success']

        iteration += 1

    return success, np.array(traj), state['desired_goal']

if __name__ == '__main__': 
    env = gym.make('PandaPickAndPlace-v3', render_mode="human")

    epi_num, total_success = 20, 0.
    for _ in range(epi_num): 
        success, traj, goal = pickPlace(env)
        # plot the trajectory, for debugging
        # plotTrajectory(traj, goal)
        total_success += success

    print('PandaPickAndPlace-v3 success rate: ', total_success/epi_num)
