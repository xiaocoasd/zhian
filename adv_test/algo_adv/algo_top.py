from abc import ABC
import torch.nn as nn
import torch

from adv_test.algo_adv.algo_obs_gene import AdvAtkObsGene
from adv_test.algo_adv.algo_adv_execute import AdvAtkExe

from typing import List

import numpy as np


# 返回样本扰动
class AlgoAtkAdv(ABC):

    def __init__(
        self,
        # my_window,
        # ready,
        args,
        policy,
        net: nn.Module,
        env,
    ) -> None:
        super().__init__()
        # self.my_window = my_window
        self.args = args
        self.policy = policy
        self.net = net
        self.env = env
        # self.ready = ready

    def run(self):

        counts = 10

        atk_succ_rate_array = np.arange(counts, dtype=object)
        asr_mean_array = np.arange(counts)

        atk_low_frames_array = np.arange(counts, dtype=object)
        atk_low_rew_array = np.arange(counts)
        atk_low_pos_array = np.arange(counts, dtype=object)
        atk_low_act_array = []

        atk_med_frames_array = np.arange(counts, dtype=object)
        atk_med_rew_array = np.arange(counts)
        atk_med_pos_array = np.arange(counts, dtype=object)
        atk_med_act_array = []

        atk_high_frames_array = np.arange(counts, dtype=object)
        atk_high_rew_array = np.arange(counts)
        atk_high_pos_array = np.arange(counts, dtype=object)
        atk_high_act_array = []

        atk_rew_array = np.arange(counts, dtype=object)

        for i in range(0, counts):
            atk_eps = (i + 1) * (1 / counts)

            print("攻击强度为{}，测试开始".format(atk_eps))
            # self.my_window.test_progress.append("\n攻击强度为{:.1f}，测试开始\n".format(atk_eps))
            atk_obs_gene = AdvAtkObsGene(
                args=self.args,
                net=self.net,
                atk_eps=atk_eps,
            )

            (
                atk_low_frames,
                atk_med_frames,
                atk_high_frames,
                atk_low_rew,
                atk_med_rew,
                atk_high_rew,
                atk_all_rew,
                atk_low_pos,
                atk_med_pos,
                atk_high_pos,
                atk_low_act,
                atk_med_act,
                atk_high_act,
                atk_succ_rate,
                asr_mean,
            ) = AdvAtkExe(
                # my_window=self.my_window,
                # ready=self.ready,
                policy=self.policy,
                env=self.env,
                adv_obs_gene=atk_obs_gene,
                net=self.net,
                test_steps=self.args.atk_steps,
                test_episode=self.args.atk_episode,
            ).run()

            atk_low_frames_array[i] = atk_low_frames
            atk_low_rew_array[i] = atk_low_rew
            atk_low_pos_array[i] = atk_low_pos
            atk_med_frames_array[i] = atk_med_frames
            atk_med_rew_array[i] = atk_med_rew
            atk_med_pos_array[i] = atk_med_pos
            atk_high_frames_array[i] = atk_high_frames
            atk_high_rew_array[i] = atk_high_rew
            atk_high_pos_array[i] = atk_high_pos

            atk_low_act_array.append(atk_low_act)
            atk_med_act_array.append(atk_med_act)
            atk_high_act_array.append(atk_high_act)

            atk_succ_rate_array[i] = atk_succ_rate
            asr_mean_array[i] = asr_mean
            atk_rew_array[i] = atk_all_rew
            print("攻击强度为{}，测试完成".format(atk_eps))

        min_rew_index = np.argmin(atk_low_rew_array)
        max_rew_index = np.argmax(atk_high_rew_array)

        mean = np.mean(atk_med_rew_array)
        med_rew_index = (np.abs(atk_med_rew_array - mean)).argmin()

        print(atk_low_rew_array[min_rew_index])
        print(atk_high_rew_array[max_rew_index])
        print(atk_med_rew_array[med_rew_index])

        atk_low_frames = atk_low_frames_array[min_rew_index]
        atk_low_rew = atk_low_rew_array[min_rew_index]
        atk_low_pos = atk_low_pos_array[min_rew_index]
        atk_low_act = atk_low_act_array[min_rew_index]

        atk_med_frames = atk_med_frames_array[med_rew_index]
        atk_med_rew = atk_med_rew_array[med_rew_index]
        atk_med_pos = atk_med_pos_array[med_rew_index]
        atk_med_act = atk_med_act_array[med_rew_index]

        atk_high_frames = atk_high_frames_array[max_rew_index]
        atk_high_rew = atk_high_rew_array[max_rew_index]
        atk_high_pos = atk_high_pos_array[max_rew_index]
        atk_high_act = atk_high_act_array[max_rew_index]
        print(len(atk_high_act))

        asr_mean_mean = np.mean(asr_mean_array)

        return (
            atk_low_frames,
            atk_med_frames,
            atk_high_frames,
            atk_med_rew,
            atk_low_pos,
            atk_med_pos,
            atk_high_pos,
            atk_low_act,
            atk_med_act,
            atk_high_act,
            atk_succ_rate_array,
            atk_rew_array,
            asr_mean_mean,
        )
