from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env import utils
from highway_env.utils import near_split
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.action import Action
from typing import TypeVar
import numpy as np

Observation = TypeVar("Observation")


class EnvHighwayAdv(HighwayEnv):

    def __init__(
        self,
        config: dict = None,
        args=None,
        is_change_reward=False,
        render_mode: str | None = None,
    ) -> None:
        # super().__init__(config, render_mode)

        self.is_change_reward = is_change_reward

        collision_reward_t = -1.0
        right_lane_reward_t = 0.1
        high_speed_reward_t = 0.4
        lane_change_reward_t = 0.0
        reward_speed_range_t = [20, 30]

        lanes_count_t = 4
        vehicles_count_t = 50
        ego_count_t = 2
        vehicles_density_t = 1
        speed_limit_t = 30
        offroad_terminal_t = False

        if is_change_reward:
            collision_reward_t = np.random.uniform(-1, -0.1)
            right_lane_reward_t = np.random.uniform(0.1, 1)
            high_speed_reward_t = np.random.uniform(0.1, 1)
            lane_change_reward_t = np.random.uniform(-0.5, 0.5)
            reward_speed_range_t = [
                np.random.randint(17, 22),
                np.random.randint(23, 30),
            ]
        else:
            lanes_count_t = np.random.randint(3, 7)
            vehicles_count_t = np.random.randint(30, 90)
            ego_count_t = np.random.uniform(0.5, 2)
            vehicles_density_t = np.random.uniform(0.5, 2)
            speed_limit_t = np.random.randint(25, 50)
            offroad_terminal_t = args.offroad_terminal

        # Configuration
        self.config = self.default_config()
        self.configure(config)
        self.configure(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": lanes_count_t,
                "vehicles_count": vehicles_count_t,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": ego_count_t,
                "vehicles_density": vehicles_density_t,
                "collision_reward": collision_reward_t,
                "right_lane_reward": right_lane_reward_t,
                "high_speed_reward": high_speed_reward_t,
                "lane_change_reward": lane_change_reward_t,
                "reward_speed_range": reward_speed_range_t,
                "normalize_reward": True,
                "offroad_terminal": offroad_terminal_t,
                "speed_limit": speed_limit_t,
            }
        )

        # print(self.config)

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.enable_auto_render = False

        self.reset()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""

        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"],
                speed_limit=self.config["speed_limit"],
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        speed_t = np.random.randint(5, self.config["speed_limit"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=speed_t,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def step(
        self, action: int | np.ndarray
    ) -> utils.Tuple[Observation | float | bool | dict]:

        # if self.is_change_reward:

        #     # if self.road is None or self.vehicle is None:
        #     #     raise NotImplementedError(
        #     #         "The road and vehicle must be initialized in the environment implementation"
        #     #     )

        #     # self.time += 1 / self.config["policy_frequency"]
        #     # self._simulate(action)

        #     # obs = self.observation_type.observe()
        #     # reward = self.reward_change(action)
        #     # terminated = self._is_terminated()
        #     # truncated = self._is_truncated()
        #     # info = self._info(obs, action)
        #     # if self.render_mode == "human":
        #     #     self.render()

        #     # return obs, reward, terminated, truncated, info

        #     obs, reward, terminated, truncated, info = super().step(action)
        #     reward = self.reward_change(action)

        #     return obs, reward, terminated, truncated, info
        # else:
        #     return super().step(action)

        obs, reward, terminated, truncated, info = super().step(action)
        if self.is_change_reward:
            reward = self.reward_change(action)
        return obs, reward, terminated, truncated, info

    def reward_change(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        if bool(rewards["on_road_reward"]):
            reward *= rewards["on_road_reward"]
            return reward
        else:
            return -1.0
