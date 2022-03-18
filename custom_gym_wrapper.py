
import numpy as np
from gym import spaces
from gym.core import Env

from robosuite.wrappers import Wrapper

from robosuite.utils.camera_utils import get_real_depth_map


class CustomGymWrapper(Wrapper, Env):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module
    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.
    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        assert keys is not None, (
            "You need to specifi which observation keys to use when using the CustomGymWrapper "
        )
        self.keys = keys
          

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        temp_dict = {key: obs[key].shape for key in self.keys}
        for key in self.keys:
          low = -np.inf
          high = np.inf
          dtype = np.float32
          for cam_name in self.env.camera_names:
            if cam_name in key:
              low = 0
              high = 255
              dtype = np.uint8       
          temp_dict[key] = spaces.Box(low = low,high = high, shape=(obs[key].shape),dtype= dtype)
        self.observation_space = spaces.Dict(temp_dict)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low= np.float32(low), high=np.float32(high))
        #spaces.Dict().contains

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and unnormalizes depth images and 
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            Dict
        """
        ob_lst = {}
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                        #finding depth observables and chaning them from normalized too 
                #if "image" in key:
                  #rgb_image = obs_dict[key]
                if "depth" in key:
                  old_array = obs_dict[key]
                  #depth_map = np.clip(get_real_depth_map(self.sim, old_array)*(255/3), 0,255)    ## mapps from 0-3 to 0-255 and cuts all values over 255
                  obs_dict[key] = np.clip(get_real_depth_map(self.sim, old_array)*(255/3), 0,255).astype(np.uint8)
                ob_lst[key] =obs_dict[key]
        #if depth_map and rgb_image:
         # ob_lst[]
        
        return ob_lst

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.
        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.
        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:
                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        """
        Utility function to set numpy seed
        Args:
            seed (None or int): If specified, numpy seed to set
        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward
        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]
        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()