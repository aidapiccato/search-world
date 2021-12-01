import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
def plot_maze(trial, ax, other=None):
    orig_maze_array = trial.env._maze

    prey_pos = trial.env._state_space[trial['init_state']['target_state']]
    agent_pos = [trial.env._state_space[i] for i in trial.agent_state]
    agent_init_pos = agent_pos[0]
    ax.imshow(np.flip(orig_maze_array), cmap='binary_r')
    prey = plt.Circle(np.flip(prey_pos), color='y', radius=0.25)
    cmap = [cm.get_cmap('Greens_r', len(agent_pos))(i) for i in range(len(agent_pos))]
    ax.add_patch(prey)    
    ax.add_patch(plt.Circle((agent_init_pos[1], agent_init_pos[0]), color='Orange', facecolor=None, radius=0.25))
    agent_pos = np.stack(agent_pos)
    if other is not None:
        ax.plot(agent_pos[:, 1]-0.1 - np.linspace(0, 0.2, len(agent_pos)), agent_pos[:, 0], 'o-', c='green', alpha=0.8)
    else:
        ax.scatter(agent_pos[:, 1], agent_pos[:, 0], c=cmap,  s=20)
    if other is not None:
        other_pos = [trial.env._state_space[i] for i in other]
        other_pos = np.stack(other_pos)
        cmap = [cm.get_cmap('Reds_r', len(agent_pos))(i) for i in range(len(other_pos))]
        ax.plot(other_pos[:, 1]+0.1 + np.linspace(0, 0.2, len(other_pos)), other_pos[:, 0], 'o-', c='red', alpha=0.8)
    ax.set_yticks([])
    ax.set_xticks([])

