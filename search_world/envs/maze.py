import io
import search_world
import numpy as np
import matplotlib.pyplot as plt
from search_world.utils.maze_utils import find_shortest_path, find_longest_path
from search_world.utils.pomdp_utils import run

class MazeStateSpace(object):
    def __init__(self, maze) -> None:
        self._maze = maze        
        self._coordinates = np.vstack(np.where(self._maze == 0)).T
        self._coordinates_to_states = {str(coor): i for (i, coor) in enumerate(self._coordinates)}
        self._state_space = np.arange(len(self._coordinates))
        self._n_states = len(self._state_space)

    def __len__(self):
        return len(self._state_space)

    def __getitem__(self, index):
        return self._coordinates[index]

    def __iter__(self):
        return iter(self._state_space)

    def __call__(self, coordinate):
        """Maps coordinates to state index"""
        if str(coordinate) in self._coordinates_to_states:
            return self._coordinates_to_states[str(coordinate)]
        return None

class MazeObservationSpace(object):
    """Grid world observation space"""

    def __init__(self, state_space, observation_func) -> None:
        self._state_space = state_space
        self._observation_func = observation_func 
        self._observations = list(set([self._observation_func(self._state_space[state]) for state in self._state_space]))
        self._observations_to_key = {observation: i for (i, observation) in enumerate(self._observations)}
        self._n_observations = len(self._observations)
        self._observation_space = np.arange(self._n_observations)

    def __len__(self):
        return len(self._observation_space)

    def __getitem__(self, index):
        return self._observations[index]

    def __iter__(self):
        return iter(self._observation_space)

    def __call__(self, observation):
        if observation in self._observations_to_key:
            return self._observations_to_key[observation]
        return None    

        
class MazeObservationModel(object):
    """Grid world observation model. 
    """
    def __init__(self, observation_space, state_space, observation_func_prob, observation_func): 
        self._observation_space = observation_space
        self._state_space = state_space
        self._observation_func_prob = observation_func_prob
        self._observation_func = observation_func
        self._observation_func = {state: self._observation_space(self._observation_func(self._state_space[state])) for state in self._state_space}
        self._histogram = {state: {observation: self._observation_func_prob(self._state_space[state], self._observation_space[observation]) 
            for observation in self._observation_space}
            for state in self._state_space}

    def prob(self, observation, state):
        return self._histogram[state][observation]

    def __call__(self, state):
        return self._observation_func[state]

    def generate_solver_input(self):
        output = io.StringIO()
        output.write("O: *\n")
        for state in self._state_space:
            probs = [str(float(self._histogram[state][observation])) for observation in self._observation_space] 
            output.write(" ".join(probs))
            output.write('\n')
        string = output.getvalue()
        output.close()
        return string

class MazeRewardModel(object):
    """Grid world reward model"""
    def __init__(self, state_space, reward_func):
        self._state_space = state_space
        self._reward_map = {state: reward_func(self._state_space[state]) for state in state_space}

    def __call__(self, state):
        return self._reward_map[state]

    def generate_solver_input(self):
        output = io.StringIO()
        for end_state in self._state_space:
            output.write("R: * : * : {} : * {}\n".format(end_state, self._reward_map[end_state]))
        string = output.getvalue()
        output.close()
        return string

class MazeTransitionModel(object):
    """Grid world transition model
    """
    def __init__(self, state_space, action_space, transition_func_prob, transition_func) -> None:
        self._state_space = state_space
        self._action_space = action_space
        self._transition_func_prob = transition_func_prob
        self._transition_func = transition_func
        self._histogram = {state: {action: {next_state: self._transition_func_prob(self._state_space[state], self._action_space[action], self._state_space[next_state]) for next_state in self._state_space} for action in self._action_space} for state in self._state_space}
         
    def __call__(self, state, action):
        if action is None:
            return self._state_space(self._transition_func(self._state_space[state], None))
        return self._state_space(self._transition_func(self._state_space[state], self._action_space[action]))
        
    def prob(self, next_state, state, action):
        if action is None:
            return next_state == state
        return self._histogram[state][action][next_state]
    
    def generate_solver_input(self):
        output = io.StringIO()
        for action in self._action_space:
            output.write("T: {}\n".format(int(action)))            
            for state in self._state_space:
                probs = [str(float(self._histogram[state][action][next_state])) for next_state in self._state_space] 
                output.write(" ".join(probs))
                output.write('\n')
        string = output.getvalue()
        output.close()
        return string

class MazeActionSpace(search_world.Space):
    """A grid world action space with 4 movements
    """
    def __init__(self):
        self._actions = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        self._action_map = {str(action_val): i for (i, action_val) in enumerate(self._actions)}
        self._action_space = np.arange(len(self._actions))
    def __len__(self):
        return len(self._action_space)
    def sample(self):
        return np.random.choice(self._action_space)
    
    def __getitem__(self, index):
        return self._actions[index]

    def __call__(self, action):
        if str(action) in self._action_map:
            return self._action_map[str(action)]
        return None

    def __iter__(self):
        return iter(self._action_space)

    def contains(self, a):
        return a in self._action_space

class Maze(search_world.Env):
    def __init__(self, maze_gen_func, maze_gen_func_kwargs, max_steps) -> None:
        """Constructor for maze class

        Args:
            _maze_gen_func (func): Function to generate mazes
        """
        super().__init__() 
        self._max_steps = max_steps
        self._maze_gen_func = maze_gen_func
        self._maze_gen_func_kwargs = maze_gen_func_kwargs
        self._maze = None


    def _observation(self, state) -> object:
        """Generates observation for given coordinate position. Useful for constructing entire observation space

        Args:
            state (tuple of ints): coordinates from which to generate observation. 
        """
        adjacent_nodes = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]]) + state
        adjacent_nodes = [self._maze[node_x, node_y] for (node_x, node_y) in adjacent_nodes]
        is_inf = np.any(np.all(np.isin(self._inf_positions, state, True), axis=1))
        is_target = np.all(state == self._target_position)
        return (adjacent_nodes, is_inf, is_target)

    def _agent_observation(self) -> object:
        """Generates observation based on current location of agent. The observation is a binary vector of length 4 corresponding adjacent nodes have a wall, 0 otherwise. 

        Returns:
            object: Observation corresponding to current position of agent.
        """ 
        return self._observation_model(state=self._agent_state)

    def agent_reward(self):
        return self._reward_model(state=self._agent_state)

    def name(self):
        id_vals = []
        for k in self._maze_gen_func_kwargs.keys(): 
            id_vals.append(str(k))
            id_vals.append(str(self._maze_gen_func_kwargs[k]))
        id_vals = "".join(id_vals)
        return '{}{}'.format(self._maze_gen_func.__name__, id_vals)

    def info(self):
        info =  self._maze_gen_func_kwargs
        info.update({'agent_initial_state': self._agent_initial_state, 
            'target_state': self._target_state, 'maze': self._maze})
        return info
    
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
        reward = self._reward_model(self._agent_state)


        if np.all(self._agent_state == self._target_state):
            done = True

        obs = self._agent_observation()

        self._num_steps += 1

        if self._num_steps >= self._max_steps:
            done = True
        
        info = {'agent_state': self._agent_state}
        return obs, reward, done, info



    def _take_action(self, action):
        """Updates agent state. 

        Args:
            action (object): action performed by agent.
        """
        # import pdb; pdb.set_trace()
        if action is None or action in self._action_space:
            self._agent_state = self._transition_model(state=self._agent_state, action=action)
        else:
            raise ValueError("Agent action not in Maze environment action space")

    def _generate_solver_input(self):
        output = io.StringIO()
        # writing preamble
        output.write('discount: {}\n'.format(0.9))
        output.write('values: reward\n')
        output.write('states: {}\n'.format(len(self._state_space)))
        output.write('actions: {}\n'.format(len(self._action_space)))
        output.write('observations: {}\n'.format(len(self._observation_space)))
        start_state = [str(1/len(self._state_space))] * len(self._state_space)
        start_state = " ".join(start_state)
        output.write('start: {}\n'.format(start_state))
        output.write(self._observation_model.generate_solver_input())
        output.write(self._transition_model.generate_solver_input())
        output.write(self._reward_model.generate_solver_input())
        text = output.getvalue()
        return text

    def render(self, ax=None, mode="human"):
        agent_position = self._state_space[self._agent_state]
        target_position = self._state_space[self._target_state]
        if ax is None:
            ax = plt.gca()
        ax.imshow(self._maze, cmap='gray')
        target = plt.Circle((target_position[1], target_position[0]), radius=0.5, color='yellow')
        ax.add_artist(target)
        agent = plt.Circle((agent_position[1], agent_position[0]), radius=0.5)        
        # TODO: Add rendering capacity for informative items
        # for inf_pos in self._inf_positions:
        #     inf = plt.Circle((inf_pos[1], inf_pos[0]), radius=0.5, color='red')
        #     ax.add_artist(inf)
        ax.add_artist(agent)
        ax.set_title('obs = {}, reward={}'.format(self._agent_observation(), self.agent_reward()))
        
    def reset(self):
        """Generates new maze, target position, and agent position
        Returns:
            observation (object): observation corresponding to initial state
        """
        # creating maze and setting initial conditions
        maze = self._maze_gen_func(**self._maze_gen_func_kwargs)
        self._maze = maze['maze']
        self._target_position = maze['target_pos']        
        self._inf_positions = maze['inf_positions']
        self._agent_init_pos = maze['agent_init_pos']

        
        def _observation_func(coor) -> object:
            # TODO: Add possibility of informative observations
            """Generates observation for given coordinate position. Useful for constructing entire observation space

            Args:
                state (tuple of ints): coordinates from which to generate observation. 
            """
            adjacent_nodes = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]]) + coor
            adjacent_nodes = tuple([self._maze[node_x, node_y] for (node_x, node_y) in adjacent_nodes])
            # is_inf = np.any(np.all(np.isin(self._inf_positions, coor, True), axis=1))
            is_target = np.all(coor == self._target_position)
            return adjacent_nodes + (is_target, )

       

        def _observation_func_prob(coor, obs):
            true_obs = _observation_func(coor)
            return true_obs == obs

        def _is_valid(coor):
            """Returns true if given position is valid for agent occupancy

            Args:
                position (object): coordinates of object whose position is being checked
            """
            if (0 <= coor).all() and (coor < self._maze.shape).all():
                return self._maze[coor[0], coor[1]] != 1
            return False

        def _transition_func(coor, action):             
            if action is None:
                return coor
            new_coor = coor + action
            if _is_valid(new_coor):
                return new_coor
            return coor

        def _transition_func_prob(coor, action, new_coor):
            true_coor = _transition_func(coor, action)
            return np.all(true_coor == new_coor)              

        def _reward_func(coor):
            return np.all(coor == self._target_position) * 10 + (1 - np.all(coor == self._target_position)) * -3

        # creating state and observation spaces
        self._state_space = MazeStateSpace(self._maze)
        self._observation_space = MazeObservationSpace(state_space=self._state_space, observation_func=_observation_func) # [self._observation(state) for state in self._state_space]       
        self._action_space = MazeActionSpace()        
        self._observation_model = MazeObservationModel(state_space=self._state_space, observation_space=self._observation_space, observation_func_prob=_observation_func_prob, observation_func=_observation_func)
        self._transition_model = MazeTransitionModel(state_space=self._state_space, action_space=self._action_space, transition_func_prob=_transition_func_prob, transition_func=_transition_func)

        self._target_state = self._state_space(self._target_position)
        self._agent_initial_state = self._state_space(self._agent_init_pos)
        self._agent_state = self._agent_initial_state

        self._reward_model = MazeRewardModel(state_space=self._state_space, reward_func=_reward_func)
        self._min_dist = find_shortest_path(self._maze, self._target_position, self._agent_init_pos)
        self._max_dist = find_longest_path(self._maze, self._target_position, self._agent_init_pos)
        self._num_steps = 0

        return self.step(None)
