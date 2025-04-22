import os
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import gym_neu_racing.envs.map as map
import gym_neu_racing.motion_models as motion_models
import gym_neu_racing.sensor_models as sensor_models


class MazeEnv(gym.Env):
    """gym env for navigating reproduction of the papers maze 
    from a given start point to end point
    """

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # path to envs dir maze.png map
        dir_path = os.path.dirname(os.path.realpath(__file__))
        map_dir = os.path.join(dir_path, "..", "_static", "maps")
        map_filename = os.path.join(map_dir, "maze.png")

        # load maze map adjust map size, resolution
        self.map = map.Map(10, 10, 0.1, map_filename=map_filename)

        # define motion and sensor models
        self.motion_model = motion_models.Unicycle()
        self.sensor_models = {
            "state": sensor_models.StateFeedback(),
            # "lidar": sensor_models.Lidar2D(self.map),
        }

        # start, end points in world coords
        self.start_pos = np.array([-3.0, -3.0, 0.0])    # (x, y, heading)
        self.goal_pos = np.array([3.7, 3.0])         # (x, y), ignoring heading for goal

        #observation and action spaces come from motion/sensor models
        self.observation_space = spaces.Dict(
            {
                key: sensor.observation_space
                for (key, sensor) in self.sensor_models.items()
            }
        )
        self.action_space = self.motion_model.action_space
        self.state = None
        self.action = np.array([0.0, 0.0])
        self.dt = 0.1
        # success threshold for reaching goal
        self.success_threshold = 0.2

    @property
    def motion_model(self):
        """get motion_model"""
        return self._motion_model

    @motion_model.setter
    def motion_model(self, motion_model_to_use):
        self._motion_model = motion_model_to_use
        self.action_space = motion_model_to_use.action_space

    @property
    def sensor_models(self):
        """get sensor_models"""
        return self._sensor_models

    @sensor_models.setter
    def sensor_models(self, sensor_models_to_use):
        self._sensor_models = sensor_models_to_use
        self.observation_space = spaces.Dict(
            {
                key: sensor.observation_space
                for (key, sensor) in sensor_models_to_use.items()
            }
        )

    def _get_obs(self):
        """Ccllect observations from sensor models"""
        observation = {}
        for key, sensor in self.sensor_models.items():
            observation[key] = sensor.step(
                self.state.copy(), self.action.copy()
            )
        return observation

    def _get_info(self):
        """return debug info or additional data for analysis"""
        return {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, Any], dict]:
        super().reset(seed=seed)

        # start robot near self.start_pos with some small random perturbation
        self.state = self.start_pos.copy()
        self.state[0] += np.random.uniform(-0.3, 0.3)  # small x-perturbation
        self.state[1] += np.random.uniform(-0.3, 0.3)  # small y-perturbation
        self.state[2] += np.random.uniform(-0.5, 0.5)  # small heading perturbation

        assert (
            self.check_if_agent_in_free_space()
        ), "agent outside map or in occupied space"

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.action = action.copy()
        prev_state = self.state.copy()
        self.state = self.motion_model.step(prev_state, action, dt=self.dt)
        observation = self._get_obs()

        # check if agent in free space
        if not self.check_if_agent_in_free_space():
            # in occupied space
            reward = -100
            terminated = True
        # check if agent reached goal
        elif self.agent_reached_goal():
            reward = 100
            terminated = True
        else:
            # keep exploring
            reward = -1  # penalty to encourage efficiency
            terminated = False

        return observation, reward, terminated, False, self._get_info()

    def check_if_agent_in_free_space(self):
        """verify agent is inside map, not occupied space"""
        grid_coords, in_map = self.map.world_coordinates_to_map_indices(
            self.state[0:2]
        )
        if not in_map:
            print("Outside of map boundaries.")
            return False
        if self.map.static_map[grid_coords[0], grid_coords[1]]:
            print("In occupied space.")
            return False
        return True

    def agent_reached_goal(self):
        """check if agent is within certain distance of the goal."""
        dist_to_goal = np.linalg.norm(self.state[0:2] - self.goal_pos)
        return dist_to_goal < self.success_threshold