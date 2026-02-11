import matplotlib
matplotlib.use('MacOSX')
import numpy as np
import matplotlib.pyplot as plt


def plotTrajectory(
    traj: np.ndarray, 
    goal: np.ndarray, 
): 

    """
    args: 
        - traj: (n, 3)
        - goal: (3,)
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the goal in green
    ax.scatter(goal[0], goal[1], goal[2], color='g', marker='.', s=50)

    # plot the trajectory in blue
    for idx in range(traj.shape[0]): 
        ax.scatter(traj[idx,0], traj[idx,1], traj[idx,2], color='b', marker='.', s=30)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
