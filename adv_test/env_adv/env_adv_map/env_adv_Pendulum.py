from gymnasium.envs.classic_control.pendulum import PendulumEnv
import numpy as np
from gymnasium import spaces


class EnvPendulumAdv(PendulumEnv):

    def __init__(
        self,
        is_change_reward,
        render_mode: str | None = None,
        g=10,
        args=None,
    ):
        super().__init__(render_mode, g)
        self.is_change_reward = is_change_reward

        if is_change_reward is False:
            self.max_speed = np.random.uniform(2, 20)
            self.max_torque = np.random.uniform(0.5, 4)
            # self.dt = 0.05
            self.g = np.random.uniform(1, 20)
            self.m = np.random.uniform(0.2, 4)
            self.l = np.random.uniform(0.2, 2)

            high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)

            self.action_space = spaces.Box(
                low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
            )

            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.steps = 1
        self.max_episode_steps = 500

        self.last_u = 0.0
        self.last_angel = 2 * np.pi

    def step(self, u):

        if self.is_change_reward:
            th, thdot = self.state  # th := theta

            g = self.g
            m = self.m
            l = self.l
            dt = self.dt

            u = np.clip(u, -self.max_torque, self.max_torque)[0]

            # time_penalty = 0.01  # 时间惩罚
            # energy_consumption = 0.005 * (u - self.last_u) ** 2
            stability_bonus = 0 if np.abs(th) < 0.1 else 1.0
            goal_reward = (
                0.0
                if np.abs(angle_normalize(th)) < 0.05 and np.abs(thdot) < 0.1
                else 1.0
            )

            smooth_for_duration = np.abs(self.last_angel - np.abs(angle_normalize(th)))

            keep_upright = (
                0.0
                if self.last_angel < 0.05 and np.abs(angle_normalize(th)) < 0.05
                else 2.0
            )

            costs_t = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

            costs = (
                # time_penalty
                # energy_consumption
                stability_bonus
                + goal_reward
                + smooth_for_duration
                + keep_upright
                + costs_t
            )
            self.last_u = u  # for rendering
            self.last_angel = np.abs(angle_normalize(th))

            newthdot = (
                thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
            )
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
            newth = th + newthdot * dt

            self.state = np.array([newth, newthdot])

            if self.render_mode == "human":
                self.render()

            self.steps += 1
            truncated = self.steps >= self.max_episode_steps
            terminated = False

            done = terminated | truncated
            if done:
                self.reset()
                self.steps = 1
                self.last_u = 0.0
                self.last_angel = 2 * np.pi
            return self._get_obs(), -costs, terminated, truncated, {}
        else:
            state, reward, terminated, truncated, info = super().step(u)
            self.steps += 1
            truncated = self.steps >= self.max_episode_steps

            done = terminated | truncated
            if done:
                self.reset()
                self.steps = 1
                self.last_u = 0.0
                self.last_angel = 2 * np.pi

        return state, reward, terminated, truncated, info


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
