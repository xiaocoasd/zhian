from abc import ABC
import gymnasium as gym
import torch.nn as nn
import random as rd
import torch
import numpy as np

from copy import deepcopy

from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_numpy

from adv_test.algo_adv.algo_obs_gene import AdvAtkObsGene
from adv_test.utils import excute_for_one_episode

import sys
import math
from copy import deepcopy


class AdvAtkExe(ABC):

    def __init__(
        self,
        policy: BasePolicy,
        env: gym.Env,
        adv_obs_gene: AdvAtkObsGene,
        net: nn.Module,
        test_steps=None,
        test_episode=None,
    ) -> None:
        super().__init__()
        self.net = net
        self.policy = deepcopy(policy)
        self.env = env
        self.data = Batch(
            state={}, obs={}, act={}, rew={}, done={}, info={}, obs_next={}, policy={}
        )
        self.adv_obs_gene = adv_obs_gene
        self.test_steps = test_steps
        self.test_episode = test_episode

    def run(self):
        frequence_counts = 10

        atk_rew_array = np.arange(frequence_counts)
        atk_pos_array = np.arange(frequence_counts, dtype=object)
        atk_frames_array = np.arange(frequence_counts, dtype=object)

        atk_rew_all_array = np.arange(frequence_counts, dtype=object)

        atk_succ_rate_array = np.arange(frequence_counts)

        atk_act_array = []
        # asr_mean = 0.0

        if self.test_steps is None and self.test_episode is None:
            print("需要传递一个")
            sys.exit()
        elif self.test_steps is not None and self.test_episode is not None:
            print("只能传递一个")
            sys.exit()
        elif self.test_steps is not None:
            # 攻击得到结果
            count = 1
            while count <= self.test_steps:

                self.reset()
                (
                    atk_frames,
                    atk_rew,
                    atk_steps,
                    succ_atk,
                    nums_atk,
                    atk_logits_array,
                    atk_obs_input,
                    atk_act_output,
                ) = excute_for_one_episode(
                    policy_t=self.policy,
                    env=self.env,
                    data=self.data,
                    net=self.net,
                    is_rdm=True,
                    is_atk=False,
                    atk_frequence=0.1,
                    adv_obs_gene=self.adv_obs_gene,
                )
                atk_mean_rew += atk_rew

                atk_frames_array.append(atk_frames)

                atk_pos_array.append(atk_logits_array)
                succ_rate = 100 * succ_atk / nums_atk

                print("第一回合结束，攻击成功率为：{:.2f}".format(succ_rate))
                atk_succ_rate_t.append(succ_rate)
                count += atk_steps

        elif self.test_episode is not None:

            for i in range(0, frequence_counts):
                atk_frequence = (i + 1) * (1 / frequence_counts)
                print("攻击频率为{:.2f}，测试开始".format(atk_frequence))
                atk_mean_rew = 0.0
                atk_rew_all = np.arange(self.test_episode)

                atk_frames_array_t = []
                atk_pos_array_t = []
                atk_succ_rate_t = []
                atk_act_array_t = []
                episode_count = 0
                while episode_count < self.test_episode:

                    # 攻击得到结果
                    self.reset()
                    (
                        atk_frames,
                        atk_rew,
                        atk_steps,
                        succ_atk,
                        nums_atk,
                        atk_logits_array,
                        atk_obs_input,
                        atk_act_output,
                    ) = excute_for_one_episode(
                        policy_t=self.policy,
                        env=self.env,
                        data=self.data,
                        net=self.net,
                        is_rdm=False,
                        is_atk=True,
                        atk_frequence=atk_frequence,
                        adv_obs_gene=self.adv_obs_gene,
                    )

                    atk_mean_rew += atk_rew
                    atk_rew_all[episode_count] = atk_rew

                    atk_frames_array_t.append(atk_frames)
                    atk_pos_array_t.append(atk_logits_array)

                    if nums_atk > 0:
                        succ_rate = 100 * succ_atk / nums_atk
                    else:
                        succ_rate = 0.0
                    atk_succ_rate_t.append(succ_rate)

                    atk_act_array_t.append(atk_act_output)

                    print(
                        "第{}回合结束，攻击次数:{},攻击成功次数：{},攻击成功率为：{:.2f}%".format(
                            (episode_count + 1),
                            nums_atk,
                            succ_atk,
                            succ_rate,
                        )
                    )
                    episode_count += 1

                atk_rew_array[i] = atk_mean_rew / self.test_episode
                atk_rew_all_array[i] = atk_rew_all
                # print(atk_rew_array[i])
                atk_rdm_index = np.random.choice(len(atk_frames_array_t), replace=False)
                atk_frames_array[i] = atk_frames_array_t[atk_rdm_index]
                atk_pos_array[i] = atk_pos_array_t[atk_rdm_index]
                atk_succ_rate_array[i] = np.mean(atk_succ_rate_t)
                atk_act_array.append(atk_act_array_t[atk_rdm_index])

                print("攻击频率为{:.2f}，测试完成".format(atk_frequence))

            min_rew_index = np.argmin(atk_rew_array)
            max_rew_index = np.argmax(atk_rew_array)

            mean = np.mean(atk_rew_array)
            med_rew_index = (np.abs(atk_rew_array - mean)).argmin()
            asr_mean = np.mean(atk_succ_rate_array)

        return (
            atk_frames_array[min_rew_index],
            atk_frames_array[med_rew_index],
            atk_frames_array[max_rew_index],
            atk_rew_array[min_rew_index],
            atk_rew_array[med_rew_index],
            atk_rew_array[max_rew_index],
            atk_rew_all_array[med_rew_index],
            atk_rew_array,
            atk_pos_array[min_rew_index],
            atk_pos_array[med_rew_index],
            atk_pos_array[max_rew_index],
            atk_act_array[min_rew_index],
            atk_act_array[med_rew_index],
            atk_act_array[max_rew_index],
            atk_succ_rate_array,
            asr_mean,
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
        obs, info = self.env.reset()

        self.data.obs = obs
        self.data.info = info
