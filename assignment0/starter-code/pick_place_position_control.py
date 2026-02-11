import gymnasium as gym
import panda_gym

import time
import numpy as np

from utils import *

def pickPlace(env): 

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

    # State machine
    phase = "APPROACH"
    object_height = state['achieved_goal'][2]
    
    # Constants
    K = 0.8                     
    SAFE_HEIGHT = 0.15          
    GRASP_HEIGHT = object_height / 2         
    PLACE_HEIGHT = object_height         
    THRESHOLD = 0.03            
    GRASP_WAIT = 10             
    RELEASE_WAIT = 10           
    
    wait_counter = 0

    success, iteration = False, 0
    while not success and iteration < 500: 
        ee_pos = state['observation'][0:3]      
        object_pos = state['achieved_goal']      
        goal_pos = state['desired_goal']         
        
        if phase == "APPROACH":
            # Phase 1: Move above the object
            target = np.array([object_pos[0], object_pos[1], SAFE_HEIGHT])
            gripper_action = 1  # Open gripper
            
            # check if ready to descend
            if np.linalg.norm(ee_pos - target) < THRESHOLD:
                phase = "DESCEND"
        
        elif phase == "DESCEND":
            # Phase 2: Lower to object height
            target = np.array([object_pos[0], object_pos[1], GRASP_HEIGHT])
            
            # check if reached object height
            if ee_pos[2] < GRASP_HEIGHT + THRESHOLD:
                phase = "GRASP"
                wait_counter = 0
        
        elif phase == "GRASP":
            # Phase 3: Close gripper around object
            target = ee_pos  
            gripper_action = -1 
            
            wait_counter += 1
            # wait long enough for grasp
            if wait_counter > GRASP_WAIT:
                phase = "LIFT"
        
        elif phase == "LIFT":
            # Phase 4: Lift object up to safe height
            target = np.array([object_pos[0], object_pos[1], SAFE_HEIGHT])
            gripper_action = -1  # Keep gripper closed
            
            # check if reached safe height
            if ee_pos[2] > SAFE_HEIGHT - THRESHOLD:
                phase = "MOVE_TO_GOAL"
        
        elif phase == "MOVE_TO_GOAL":
            # Phase 5: Move horizontally to goal position
            target = np.array([goal_pos[0], goal_pos[1], SAFE_HEIGHT])
            gripper_action = -1  # Keep gripper closed
            
            # check if close to goal horizontally
            if np.linalg.norm(ee_pos[:2] - goal_pos[:2]) < THRESHOLD:
                phase = "LOWER_AND_RELEASE"
                wait_counter = 0
        
        elif phase == "LOWER_AND_RELEASE":
            # Phase 6: Lower to place height and release
            target = np.array([goal_pos[0], goal_pos[1], PLACE_HEIGHT])
            
            # lower first, then release
            if ee_pos[2] < PLACE_HEIGHT + THRESHOLD:
                gripper_action = 1  # open gripper to release
                wait_counter += 1
                # wait a bit after releasing
                if wait_counter > RELEASE_WAIT:
                    phase = "DONE"
            else:
                gripper_action = -1  # Keep closed while lowering
        
        # Proportional controller for position
        error = target - ee_pos
        position_action = K * error
        position_action = np.clip(position_action, -1.0, 1.0)
        
        # Combine position and gripper actions
        action = np.append(position_action, gripper_action)

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