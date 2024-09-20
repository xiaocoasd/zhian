import os
from abc import ABC
import torch.nn as nn

import gymnasium as gym


from adv_test.algo_adv.algo_obs_gene import AdvAtkObsGene
from adv_test.algo_adv.algo_adv_execute import AdvAtkExe

from typing import List

import numpy as np

def write_to_file(text):
    # 检查文件是否存在
    if not os.path.exists("../view/display.txt"):
        # 如果文件不存在，创建文件
        with open("../view/display.txt", 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        # 如果文件已存在，在末尾追加内容
        with open("../view/display.txt", 'a', encoding='utf-8') as f:
            f.write(text)
# 返回样本扰动
class AlgoAtkAdv(ABC):

    def __init__(
        self,
        args,
        policy,
        net: nn.Module,
        env: gym.Env,
    ) -> None:
        super().__init__()

        self.args = args
        self.policy = policy
        self.net = net
        self.env = env

        self.high = env.observation_space[0].high
        self.low = env.observation_space[0].low

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

        atk_med_all_rew_array = np.arange(counts, dtype=object)

        for i in range(0, counts):
            atk_eps = (i + 1) * (1 / counts)

            print("攻击强度为{:.2f}，测试开始".format(atk_eps))
            write_to_file("攻击强度为{:.2f}，测试开始\n".format(atk_eps))
            atk_obs_gene = AdvAtkObsGene(
                args=self.args,
                net=self.net,
                atk_eps=atk_eps,
                high=self.high,
                low=self.low,
            )

            (
                atk_low_frames,
                atk_med_frames,
                atk_high_frames,
                atk_low_rew,
                atk_med_rew,
                atk_high_rew,
                atk_med_all_rew,
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
            atk_med_all_rew_array[i] = atk_med_all_rew
            print("攻击强度为{:.2f}，测试完成".format(atk_eps))

        min_rew_index = np.argmin(atk_low_rew_array)
        max_rew_index = np.argmax(atk_high_rew_array)

        mean = np.mean(atk_med_rew_array)
        med_rew_index = (np.abs(atk_med_rew_array - mean)).argmin()

        # print(atk_low_rew_array[min_rew_index])
        # print(atk_high_rew_array[max_rew_index])
        # print(atk_med_rew_array[med_rew_index])

        atk_low_frames = atk_low_frames_array[min_rew_index]
        atk_low_rew = atk_low_rew_array[min_rew_index]
        atk_low_pos = atk_low_pos_array[min_rew_index]
        atk_low_act = atk_low_act_array[min_rew_index]
        atk_low_pos_all = [item for sublist in atk_low_pos_array for item in sublist]
        atk_low_act_all = [item for sublist in atk_low_act_array for item in sublist]

        atk_med_frames = atk_med_frames_array[med_rew_index]
        atk_med_rew = atk_med_rew_array[med_rew_index]
        atk_med_pos = atk_med_pos_array[med_rew_index]
        atk_med_act = atk_med_act_array[med_rew_index]
        atk_med_pos_all = [item for sublist in atk_med_pos_array for item in sublist]
        atk_med_act_all = [item for sublist in atk_med_act_array for item in sublist]

        atk_high_frames = atk_high_frames_array[max_rew_index]
        atk_high_rew = atk_high_rew_array[max_rew_index]
        atk_high_pos = atk_high_pos_array[max_rew_index]
        atk_high_act = atk_high_act_array[max_rew_index]
        atk_high_pos_all = [item for sublist in atk_high_pos_array for item in sublist]
        atk_high_act_all = [item for sublist in atk_high_act_array for item in sublist]

        atk_med_all_rew = atk_med_all_rew_array[med_rew_index]

        asr_mean_mean = np.mean(asr_mean_array)

        return (
            atk_low_frames,
            atk_med_frames,
            atk_high_frames,
            atk_med_rew,
            atk_med_all_rew,
            atk_low_pos_all,
            atk_med_pos_all,
            atk_high_pos_all,
            atk_low_act_all,
            atk_med_act_all,
            atk_high_act_all,
            atk_succ_rate_array,
            atk_rew_array,
            asr_mean_mean,
        )
