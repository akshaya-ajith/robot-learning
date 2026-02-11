import gymnasium as gym
import panda_gym

import time
import numpy as np

from utils import *

def push(env): 
    """
    Panda robot arm pushes the box
    args: 
        - env: gym environment
    returns: 
        - success: successfully pushes the box or not
        - traj: the object's trajectory
        - goal: the object's goal position
    """

    traj = []
    state, info = env.reset()
    print("=== OBSERVATION STRUCTURE ===")
    print("Keys:", state.keys())
    print("observation shape:", state['observation'].shape)
    print("observation:", state['observation'])
    print("achieved_goal:", state['achieved_goal'])
    print("desired_goal:", state['desired_goal'])
    print("============================")
    traj.append(state['achieved_goal'])

    phase = "APPROACH"

    OFFSET = 0.08
    PUSH_DISTANCE = 0.05
    THRESHOLD = 0.03  
    K = 0.8
    GOAL_THRESHOLD = 0.05
    

    success, iteration = False, 0
    while not success and iteration < 500: 
        ee_pos = state['observation'][0:3]         
        object_pos = state['achieved_goal']         
        goal_pos = state['desired_goal']           
        
        push_dir = goal_pos - object_pos
        dist_to_goal = np.linalg.norm(push_dir)
        
        if dist_to_goal > 1e-6:
            push_dir = push_dir / dist_to_goal
        
        # check if we're on the correct side of the object
        ee_to_obj = object_pos - ee_pos
        dot_product = np.dot(ee_to_obj, push_dir)

        # if dot_product < 0, we're in front of the object (wrong side)
        if dot_product < -0.01:
            phase = "REPOSITION"
        
        if phase == "REPOSITION":
            target = object_pos - push_dir * OFFSET
            
            if np.linalg.norm(ee_pos - target) < THRESHOLD and dot_product > 0:
                phase = "APPROACH"
        
        elif phase == "APPROACH":
            target = object_pos - push_dir * OFFSET
            
            if np.linalg.norm(ee_pos - target) < THRESHOLD:
                phase = "PUSH"
        
        else:  # PUSH phase
            if dist_to_goal > GOAL_THRESHOLD:
                target = object_pos + push_dir * PUSH_DISTANCE
            else:
                target = goal_pos
        
        # proportional controller
        error = target - ee_pos
        action = K * error
        action = np.clip(action, -1.0, 1.0)

        state, reward, terminated, truncated, info = env.step(action)
        traj.append(state['achieved_goal'])
        time.sleep(0.01)

        success = info['is_success']
        iteration += 1

    return success, np.array(traj), state['desired_goal']

if __name__ == '__main__': 
    env = gym.make('PandaPush-v3', render_mode="human")

    epi_num, total_success = 20, 0.
    for _ in range(epi_num): 
        success, traj, goal = push(env)
        # plot the trajectory, for debugging
        # plotTrajectory(traj, goal)
        total_success += success

    print('PandaPush-v3 success rate: ', total_success/epi_num)