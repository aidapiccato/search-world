import numpy as np
import sys

def symmetric_corridors(length, n_corridors, target_position, agent_initial_position):
    """Generates maze of symmetric corridor type, with no informative nodes

    Args:
        length ([type]): [description]
        n_corridors ([type]): [description]
        target_position ([type]): [description]
        agent_position ([type]): [description]

    Returns:
        dict: dictionary containing maze and initial conditions
    """ 

    # x- and y-coordinates of empty nodes
    corridors_x = np.arange(1, 2 * n_corridors, 2)  
    corridors_y = np.arange(1, length+1)

    maze = np.ones((length + 2, 2 * n_corridors + 1))    

    # creating vertical corridors 
    for x in corridors_x:
        maze[corridors_y, x] = np.zeros(length)

    # making central horizontal corridor 
    maze[int(np.ceil(length/2)), 1:2 * n_corridors] = 0

    # no informative nodes 
    inf_positions = np.empty((0, 2))
    

    # creating set of all possible start states for agents    
    states = np.argwhere(maze == 0)
    # finding agent and target position by indexing into states
    target_position = states[target_position]
    agent_initial_position = states[agent_initial_position]

    return dict(inf_positions=inf_positions, maze=maze, target_position=target_position, agent_initial_position=agent_initial_position)


def hallway(length, target_position, agent_initial_position):
    """Generates hallway maze of given length, with target and agent located at given position

    Args:
        length (int): length of hallway
        target_position (int): position of target
        agent_initial_position (int): initial position of agent
    
    Returns:
        dict: dictionary containing maze and initial condition information
    """
    maze = np.ones((3, length + 2))
    maze[1, 1:length+1] = np.zeros((1, length))
    states = np.argwhere(maze == 0)
    target_position = states[target_position]
    agent_initial_position = states[agent_initial_position]
    inf_positions = np.empty((0, 2))
    
    return dict(inf_positions=inf_positions, maze=maze, target_position=target_position, agent_initial_position=agent_initial_position)



def leaf_hallway(length, target_position, agent_initial_position):
    """Generates maze of hallway type, consisting of long corridor with offshoot leaves. Some leaves contain informative nodes    

    Returns:
        dict: dictionary containing maze and initial condition
    """
    leaf_nodes = np.arange(1, length, 2)
    # creating main empty corridor and leaves 
    maze = np.ones((2, length))
    maze[0] = np.zeros((1, length))
    maze[1, leaf_nodes] = 0
    # randomly selecting leaf node to be target node        
    target_position = leaf_nodes[target_position]
    # randomly selecting leaf nodes to be informative nodes
    num_inf_positions = np.random.randint(low=0, high=len(leaf_nodes))
    inf_positions = leaf_nodes[leaf_nodes != target_position]
    inf_positions = np.random.choice(inf_positions, size=num_inf_positions, replace=False)
    if num_inf_positions == 0:
        inf_positions = np.empty((0, 2))

    # adding horizontal and vertical padding
    maze = np.vstack([np.ones((1, length)), maze, np.ones((1, length))])
    maze = np.hstack([np.ones((4, 1)), maze, np.ones((4, 1))])
    target_position = np.asarray([1, target_position])
    target_position += [1, 1]  

    inf_positions = np.vstack([np.asarray([1, inf_pos]) + [1, 1] for inf_pos in inf_positions])

    agent_initial_position = np.asarray([1, np.arange(length)[agent_initial_position] + 1])        
    return dict(maze=maze, target_position=target_position, inf_positions=inf_positions, agent_initial_position=agent_initial_position)

def _find_shortest_path_is_valid(maze, visited, x, y):
        return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and not (maze[x][y] == 1 or visited[x][y])

def _find_shortest_path_helper(maze, visited, x, y, dest, min_dist=sys.maxsize, dist=0):

        # if you've reached destination, return minimum
        if np.all((x, y) == dest):
            return min(dist, min_dist)

        visited[x][y] = 1

        if _find_shortest_path_is_valid(maze, visited, x+1, y):
            min_dist = _find_shortest_path_helper(maze, visited, x + 1, y, dest, min_dist, dist+1)

        if _find_shortest_path_is_valid(maze, visited, x, y+1):
            min_dist = _find_shortest_path_helper(maze, visited, x, y+1, dest, min_dist, dist+1)

        if _find_shortest_path_is_valid(maze, visited, x-1, y):
            min_dist = _find_shortest_path_helper(maze, visited, x-1, y, dest, min_dist, dist+1)

        if _find_shortest_path_is_valid(maze, visited, x, y-1):
            min_dist = _find_shortest_path_helper(maze, visited, x, y-1, dest, min_dist, dist+1)

        visited[x][y] = 0

        return min_dist



def find_longest_path(maze, target_position, agent_initial_position):
    # TODO: Document find_longest_path
    # v = self._state_space.shape[0]
    # e = (self._maze_gen_func_kwargs['length'] - 1) * self._maze_gen_func_kwargs['n_corridors'] + self._maze_gen_func_kwargs['n_corridors']
    # return 2 * v * e
    return 2 * np.argwhere(maze == 0)

def find_shortest_path(maze, target_position, agent_initial_position):
    # TODO: Document _find_shortest_path
    s_x, s_y = agent_initial_position        
    d_x, d_y = target_position

    visited = np.zeros_like(maze)

    min_dist = _find_shortest_path_helper(maze, visited, s_x, s_y, (d_x, d_y))
    return min_dist
