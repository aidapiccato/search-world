# TODO: Create wrapper class. See here - https://alexandervandekleut.github.io/gym-wrappers/ - for why.
from abc import abstractmethod


class Env(object):
    """Abstract class for implementing an arbitrary environment dynamics. Can be partially or fully observed. Adapted from OpenAI Gym's abstract Env class. (https://github.com/openai/gym/blob/master/gym/core.py) """

    @abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of the episode is reached, you are responsible for calling reset() to reset environment's state

        Args:
            action (object):  action provided by agent

        Returns:
            observation (object):  agent's observation of current environmnet
            reward (float): amount of reward returned after previous action
            done (bool): whether episode has ended, in which case further step() calls will return undefined results
            info (dict): auxiliary information
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the environment to an initial state and returns an initial observation. Should not reset environment's random number variables

        Returns:
            observation (object): the initial observation
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator 

        Args:
            seed (list<int>, optional): returns list of seeds used in env's random number generators. Defaults to None.
        """
        return
