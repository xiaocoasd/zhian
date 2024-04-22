import os
from abc import ABC

from adv_test.env_adv.env_utils import env_adv_make
from adv_test.env_adv.env_adv_execute import EnvAdvExe

import time
import numpy as np

from tianshou.env import DummyVectorEnv
from tianshou.data import Batch

def write_to_file(text):
    # 检查文件是否存在
    if not os.path.exists("view/display.txt"):
        # 如果文件不存在，创建文件
        with open("view/display.txt", 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        # 如果文件已存在，在末尾追加内容
        with open("view/display.txt", 'a', encoding='utf-8') as f:
            f.write(text)
class EnvAdv(ABC):

    def __init__(
        self,
        env_name,
        render_mode,
        policy,
        is_rdm,
        # env_stats_steps,
        # rew_change_steps,
        env_stats_episode,
        rew_change_episode,
        args=None,
        env=None,  # 当不是gym环境时传入
    ) -> None:
        super().__init__()

        self.args = args

        self.env_name = env_name
        self.env = env
        self.render_mode = render_mode
        self.env_adv = None
        self.data = Batch(
            state={}, obs={}, act={}, rew={}, done={}, info={}, obs_next={}, policy={}
        )
        self.policy = policy
        self.is_rdm = is_rdm

        # self.env_stats_steps = env_stats_steps
        # self.rew_change_steps = rew_change_steps

        self.env_stats_episode = env_stats_episode
        self.rew_change_episode = rew_change_episode

        self.env_adv = DummyVectorEnv(
            [
                lambda: env_adv_make(
                    env_name=self.env_name,
                    render_mode=self.render_mode,
                    is_change_reward=False,
                    args=self.args,
                    env=self.env,
                )
            ]
        )

    def run(self):
        self.reset()

        env_stats_frames_array = []
        env_stats_rew_array = []

        rew_change_frames_array = []
        rew_change_rew_array = []

        start_time = time.time()

        # 改变环境属性
        episode_count = 0
        while episode_count < self.env_stats_episode:
            self.env_adv = DummyVectorEnv(
                [
                    lambda: env_adv_make(
                        env_name=self.env_name,
                        render_mode=self.render_mode,
                        is_change_reward=False,
                        args=self.args,
                        env=self.env,
                    )
                ]
            )
            self.reset()

            frames, rew = EnvAdvExe(
                self.policy,
                self.is_rdm,
                self.env_adv,
                self.data,
            ).run()

            env_stats_rew_array.append(rew)
            env_stats_frames_array.append(frames)
            episode_count += 1
            print("env_stats_adv {} episode rew:{}".format(episode_count, rew))
            write_to_file("改变环境参数后，第{}回合结束，模型奖励为{}\n".format(episode_count, rew))
        # 改变奖励结构
        episode_count = 0
        self.env_adv = DummyVectorEnv(
            [
                lambda: env_adv_make(
                    env_name=self.env_name,
                    render_mode=self.render_mode,
                    is_change_reward=True,
                    args=self.args,
                    env=self.env,
                )
            ]
        )
        while episode_count < self.rew_change_episode:
            self.reset()

            frames, rew = EnvAdvExe(
                self.policy,
                self.is_rdm,
                self.env_adv,
                self.data,
            ).run()

            rew_change_frames_array.append(frames)
            rew_change_rew_array.append(rew)
            episode_count += 1
            print("rew_change_adv {} episode rew:{}".format(episode_count, rew))
            write_to_file("改变奖励结构后，第{}回合结束，模型奖励为{}\n".format(episode_count, rew))

        end_time = time.time()
        print("execute_one_epoch  time :", end_time - start_time)

        env_stats_rdm_index = np.random.choice(len(env_stats_rew_array), replace=False)
        rew_change_rdm_index = np.random.choice(
            len(rew_change_rew_array), replace=False
        )

        # env_stats_mean_rew = np.mean(env_stats_rew_array)
        # rew_change_mean_rew = np.mean(rew_change_rew_array)

        # env_stats_frames = env_stats_frames_array[env_stats_rdm_index]
        # rew_change_frames = rew_change_frames_array[rew_change_rdm_index]

        return (
            env_stats_frames_array[env_stats_rdm_index],
            rew_change_frames_array[rew_change_rdm_index],
            np.mean(env_stats_rew_array),
            np.mean(rew_change_rew_array),
        )

    def reset(self) -> None:
        self.data = Batch(
            state={},
            obs={},
            act={},
            rew={},
            done={},
            info={},
            policy={},
        )
        self.reset_env()

    def reset_env(self) -> None:
        obs, info = self.env_adv.reset()

        self.data.obs = obs
        self.data.info = info
