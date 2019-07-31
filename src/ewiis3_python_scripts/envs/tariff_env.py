import gym
from gym import spaces
import numpy as np
import ewiis3DatabaseConnector as db
import numpy

class TariffEnv(gym.Env):
    def __init__(self):
        # define actions
        # define states
        print('TariffEnv init called')
        self.action_space = spaces.Discrete(5)
        # self.observation_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))
        self.observation_space = spaces.Discrete(9)
        pass

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = self._get_reward()
        ob = self._observe_state()
        # episode_over = self.status != hfo_py.IN_GAME
        episode_over = False
        return ob, reward, episode_over, {}

    def reset(self):
        pass

    def render(self):
        pass

    def _take_action(self, action):
        # Todo: wait here
        # take action with broker
        # check database for new state
        # continue
        pass

    def _observe_state(self):
        return np.random.randint(0, self.observation_space.n)

    def _get_reward(self):
        """ Reward is given for XY. """
        all_gameIds = db.get_running_gameIds()
        game_id = all_gameIds[0]
        df_prosumption = db.load_consumption_tariff_prosumption(game_id, 24)
        df_prosumption["grid_prosumption"] = df_prosumption["totalProduction"] - df_prosumption["totalConsumption"]
        df_earnings = db.load_consumption_tariff_earnings(game_id, 24)

        #print(df_prosumption.columns)
        print(numpy.corrcoef(df_prosumption["grid_prosumption"], df_prosumption["SUM(t.kWh)"]))
        #print(df_earnings.head())
        return np.random.randint(0, 5)
