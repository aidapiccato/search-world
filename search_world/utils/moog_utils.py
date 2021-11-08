import numpy as np
import sys
import search_world
from search_world.envs.maze import Maze
from search_world.utils.maze_utils import symm_corr


def moog_generator():
    maze = Maze(max_steps=100, maze_gen_func=symm_corr, maze_gen_func_kwargs={'length': 5,  'n_corr': 4, 'target_pos': 3, 'agent_init_pos': 1})
    maze.reset()
    reference_states = maze._state_space._coordinates.astype('float64')
    reference_edges = []
    for s in maze._state_space:
        for a in maze._action_space:
            ns = maze._transition_model(state=s, action=a)
            if s != ns:
                reference_edges.append([np.vstack((maze._state_space[s], maze._state_space[ns]))])

    reference_edges = np.vstack(reference_edges)

    return dict(
        reference_nodes=reference_states,
        reference_edges=reference_edges,
        search_nodes=[],
        search_edges=[],
        target_nodes_search=[],
        target_nodes_reference=[]
    )