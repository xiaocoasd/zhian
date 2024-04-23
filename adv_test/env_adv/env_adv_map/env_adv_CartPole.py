from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np
import math
from gymnasium import logger
from gymnasium import spaces


class EnvCartPoleAdv(CartPoleEnv):

    def __init__(self, is_change_reward, render_mode: str | None = None, args=None):
        super().__init__(render_mode)
        self.is_change_reward = is_change_reward
        if is_change_reward is False:
            self.gravity = np.random.uniform(1, 50)
            self.masscart = np.random.uniform(0.1, 5)
            self.masspole = np.random.uniform(0.01, 5)
            self.total_mass = self.masspole + self.masscart

            self.length = np.random.uniform(0.1, 1)
            self.polemass_length = self.masspole * self.length

            self.force_mag = np.random.uniform(1, 60)
            high = np.array(
                [
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    self.theta_threshold_radians * 2,
                    np.finfo(np.float32).max,
                ],
                dtype=np.float32,
            )

            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.steps = 1
        self.max_episode_steps = 500

    def step(self, action):
        if self.is_change_reward:
            # print("*******")
            # print(self.steps)
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid"
            assert self.state is not None, "Call reset before using step method."

            x, x_dot, theta, theta_dot = self.state
            force = self.force_mag if action == 1 else -self.force_mag
            costheta = math.cos(theta)
            sintheta = math.sin(theta)

            temp = (
                force + self.polemass_length * theta_dot**2 * sintheta
            ) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            if self.kinematics_integrator == "euler":
                x = x + self.tau * x_dot
                x_dot = x_dot + self.tau * xacc
                theta = theta + self.tau * theta_dot
                theta_dot = theta_dot + self.tau * thetaacc
            else:  # semi-implicit euler
                x_dot = x_dot + self.tau * xacc
                x = x + self.tau * x_dot
                theta_dot = theta_dot + self.tau * thetaacc
                theta = theta + self.tau * theta_dot

            self.state = (x, x_dot, theta, theta_dot)

            terminated = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )

            self.steps += 1
            truncated = self.steps >= self.max_episode_steps
            done = terminated | truncated
            if done:
                self.reset()
                self.steps = 1

            if not (terminated | truncated):

                x_rate = 1 - abs(x) / self.x_threshold
                angle_rate = 1 - abs(theta) / self.theta_threshold_radians
                reward = (x_rate + angle_rate) / 2
                reward = round(reward, 1)
            elif self.steps_beyond_terminated is None:
                # Pole just fell!
                self.steps_beyond_terminated = 0
                reward = 1.0
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1
                reward = 0.0

            if self.render_mode == "human":
                self.render()
            return (
                np.array(self.state, dtype=np.float32),
                reward,
                terminated,
                truncated,
                {},
            )
        else:
            state, reward, terminated, truncated, info = super().step(action)
            self.steps += 1
            truncated = self.steps >= self.max_episode_steps

            done = terminated | truncated
            if done:
                self.reset()
                self.steps = 1

            return state, reward, terminated, truncated, info
