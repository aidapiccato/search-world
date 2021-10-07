import search_world
import numpy as np


class GridActionSpace(search_world.Space):
    """A grid world action space with 4 movements
    """
    def __init__(self, seed=None):
        self._action_space = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        super().__init__(len(self._action_space), np.ndarray, seed)

    def sample(self):
        return self._action_space[np.random.choice(len(self._action_space))]
    
    def contains(self, a):
        a = np.asarray(a)
        return (-1 <= a).all() and (a <= 1).all()


class Hallway(search_world.Env):
    def __init__(self) -> None:
        """Constructor for hallway class. 
        """        
        super().__init__()
        self.action_space = GridActionSpace()
        # TODO: Grid world observation space. All possible configurations of walls below, above, right, and left, as well as color of the current node (to make informative nodes possible)
        self._maze = None

    def _observation(self) -> object:
        """Generates observation based on current location of agent. The observation is a binary vector of length 4 corresponding adjacent nodes have a wall, 0 otherwise. 

        Returns:
            object: Observation corresponding to current position of agent.
        """ 
        adjacent_nodes = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]]) + self._agent_position
        adjacent_nodes = [self._maze[node_x, node_y] for (node_x, node_y) in adjacent_nodes]
        return adjacent_nodes


    def step(self, action):
        """ Executes one time step within the environment. 

        Args:
            action (object): Object corresponding to agent's current action            
        
        Returns:
            observation (object): observation corresponding to novel position in environment
            reward (float): positive reward if agent has found target, 0 otherwise
            done (bool): True if agent has found target or environment times out, False otherwise.
            info (dict): auxiliary information
        """ 
        self._take_action(action)

        done = False
        reward = 0

        if np.all(self._agent_position == self._target_position):
            done = True
            reward = 10

        obs = self._observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        """Updates agent position. 

        Args:
            action (object): action performed by agent.
        """

        if action not in self.action_space:
            return ValueError("Agent action not in Maze environment action space")
        new_agent_position = self._agent_position + action
        
        # only update position if tentative new position is a valid one 
        if self._is_valid(new_agent_position):
            self._agent_position = new_agent_position

    def _is_valid(self, position):
        """Returns true if given position is valid for agent occupancy

        Args:
            position (object): coordinates of object whose position is being checked
        """
        if (0 <= position).all() and (position < self._maze.shape).all():
            return self._maze[position[0], position[1]] != 1
        return False
    
    def render(self, mode="human"):
        # TODO: Implement render methods
        pass

    def reset(self):
        """Generates new maze, target position, and agent position
        Returns:
            observation (object): observation corresponding to initial state
        """
        # TODO: Move all of this into a maze-generating function
        self._length = np.random.randint(low=3, high=12)
        leaf_nodes = np.arange(1, self._length, 2)
        # creating main empty corridor and leaves 
        self._maze = np.ones((2, self._length))
        self._maze[0] = np.zeros((1, self._length))
        self._maze[1, leaf_nodes] = 0
        # randomly selecting leaf node to be target node
        self._target_position = np.asarray([1, np.random.choice(leaf_nodes)])
        # adding horizontal and vertical padding
        self._maze = np.vstack([np.ones((1, self._length)), self._maze, np.ones((1, self._length))])
        self._maze = np.hstack([np.ones((4, 1)), self._maze, np.ones((4, 1))])
        self._target_position += [1, 1]  
        # start agent somewhere along main corridor
        self._agent_position = np.asarray([1, np.random.choice(self._length) + 1])
        return self._observation()
        








