import numpy as np

def symmetric_corridors(length, n_corridors, target_position, agent_position):
    """Generates maze of symmetric corridor type, with no informative nodes

    Returns:
        dict: dictionary containing maze and initial conditions
    """
    corridors_x = np.arange(1, 2 * n_corridors, 2)  
    corridors_y = np.arange(1, length+1)
    target_corridor = corridors_x[target_position]
    target_height = corridors_y[target_position]
    agent_corridor = corridors_x[agent_position]
    agent_height = corridors_y[agent_position]
    while np.all([agent_corridor, agent_height] 
        == [target_corridor, target_height]):
            agent_corridor = np.random.choice(corridors_x)
            agent_height = np.random.choice(corridors_y)
    maze = np.ones((length + 2, 2 * n_corridors + 1))    
    for x in corridors_x:
        maze[corridors_y, x] = np.zeros(length)
    maze[int(np.ceil(length/2)), 1:2 * n_corridors] = 0
    inf_positions = np.empty((0, 2))
    target_position = np.asarray([target_height, target_corridor])
    agent_position = np.asarray([agent_height, agent_corridor])
    return dict(inf_positions=inf_positions, maze=maze, target_position=target_position, agent_position=agent_position)



def hallway(length, target_position, agent_position):
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

    agent_position = np.asarray([1, np.arange(length)[agent_position] + 1])        
    return dict(maze=maze, target_position=target_position, inf_positions=inf_positions, agent_position=agent_position)
