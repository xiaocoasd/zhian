from abc import ABC
import gymnasium as gym
import torch.nn as nn
import random as rd
import torch
import numpy as np
import time

import time

from copy import deepcopy

np.seterr(divide="ignore", invalid="ignore")
from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_numpy

from adv_test.algo_adv.algo_top import ImgAdvObsGene
from adv_test.net_adapter.net_top import NetAdapter
import math


class ExecutePerEpoch(ABC):
    def __init__(
            self,
            policy: BasePolicy,
            env: gym.Env,
            data: Batch,
            atk_eps: float,
            atk_frequence: float,
            steps: int,
            adv_obs_gene: ImgAdvObsGene,
            is_atk: bool,
            is_rdm: bool,
            net: nn.Module,
    ) -> None:
        super().__init__()
        self.net = net
        self.policy = deepcopy(policy)
        self.env = env
        self.data = data
        self.atk_eps = atk_eps
        self.atk_frequence = atk_frequence
        self.steps = steps
        self.is_atk = is_atk
        self.is_rdm = is_rdm
        self.adv_obs_gene = adv_obs_gene

    def run(self):

        frames = []
        best_frames = []

        count = 0
        nums_atk = 0
        succ_atk = 0
        rew_t = 0.0
        mean_rew = 0.0
        bout = 0
        best_rew = float("-inf")

        def calculate(numpy, length, out):
            sum = 0
            for i in range(0, length):
                a = numpy[0][i].item()
                if (a > 0):
                    sum += a
            for i in range(0, length):
                a = numpy[0][i].item()
                if (a > 0):
                    normalized_a = a / sum
                    out[i][count] = normalized_a
                if (a <= 0):
                    normalized_a = 0.001
                    out[i][count] = normalized_a
            return out[i][count]

        aa = self.net(self.data.obs)
        length = aa.size(1)
        mat = np.zeros((length, 100000))
        mat_2 = np.zeros((length, 100000))
        rdm_mat = np.zeros((length, 100000))
        ls = []
        ls_2 = []
        rdm_ls = []
        obs_fist = []
        act_fist = []
        adv_act_values = []
        obs_fist_values = []

        start_time = time.time()

        while True:
            if count >= self.steps:
                break
            # 更新状态
            # last_state = self.data.state
            last_state = self.data.policy.pop("hidden_state", None)

            frames.append(self.env.render())

            # 带入数据进行计算，并更新相关数据
            if self.is_rdm:
                rdm_act = [self.env.action_space[0].sample()]
                rdm_act_sample = self.policy.map_action_inverse(rdm_act)
                self.data.update(act=rdm_act_sample)
                rdm_aa = self.net(self.data.obs)
                length = rdm_aa.size(1)
                calculate(rdm_aa, length, rdm_mat)
            else:
                result = self.policy(self.data, last_state)
                aa = self.net(self.data.obs)
                length = aa.size(1)
                calculate(aa, length, mat)
                calculate(aa, length, mat_2)
                # 这是计算的这一个动作的概率
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)

                state = result.get("state", Batch())
                if state is not None:
                    policy.hidden_state = state

                act = to_numpy(result.act)
                self.data.update(policy=result.get("policy", Batch()), act=act)

            # 根据频率开始攻击并记录相关信息
            if self.is_rdm is False and self.is_atk:
                x = rd.uniform(0, 1)
                if x < self.atk_frequence:
                    ori_act = deepcopy(self.data.act)
                    ori_obs = deepcopy(self.data.obs)

                    adv_obs = self.adv_obs_gene.run(
                        ori_obs=ori_obs,
                        ori_act=ori_act,
                        atk_eps=self.atk_eps,
                    )

                    with torch.no_grad():
                        adv_obs = adv_obs.cpu().detach().numpy()
                        self.data.obs = adv_obs
                        result = self.policy(self.data, last_state=None)
                    adv_act = to_numpy(result.act)

                    bb = self.net(adv_obs)
                    length_2 = bb.size(1)
                    if isinstance(self.env.action_space[0], gym.spaces.Discrete):
                        if adv_act != ori_act:
                            succ_atk += 1
                            calculate(bb, length_2, mat_2)
                            act_fist.append(adv_act)
                            obs_fist.append(adv_obs.tolist())
                        else:
                            act_fist.append(ori_act)
                            obs_fist.append(ori_obs)

                    elif isinstance(self.env.action_space[0], gym.spaces.Box):
                        max_action = (self.env.action_space)[0].high
                        min_action = (self.env.action_space)[0].low

                        len_action = max_action - min_action
                        chan_action = abs(adv_act - ori_act)

                        rate_action = np.divide(chan_action, len_action)

                        mean_rate = np.mean(rate_action)
                        # print(rate_action)
                        # print(mean_rate)
                        if mean_rate >= 0.05:
                            succ_atk += 1
                            calculate(bb, length_2, mat_2)
                            act_fist.append(adv_act)
                            obs_fist.append(adv_obs.tolist())
                        else:
                            act_fist.append(ori_act)
                            obs_fist.append(ori_obs)
                    else:
                        print("No")
                        print(self.env.action_space)
                        print(gym.spaces.Discrete)
                        print(gym.spaces.Box)
                        print(isinstance(self.env.action_space, gym.spaces.Box))
                        print(isinstance(self.env.action_space, gym.spaces.Discrete))
                    nums_atk += 1
                    self.data.update(obs=adv_obs, act=adv_act)

            # 更新环境   ----使用的act是否需要map
            # act_remap = self.policy.map_action(self.data.act)
            # obs_next, rew, terminated, truncated, info = self.env.step(act_remap)

            obs_next, rew, terminated, truncated, info = self.env.step(self.data.act)
            done = terminated or truncated
            mean_rew += rew
            # 更新数据
            self.data.update(
                obs=obs_next,
                rew=rew,
                done=done,
                info=info,
            )

            # 检查是否出触发结束条件
            if done:
                bout+=1
                obs_next, info = self.env.reset()
                # print(self.data.rew)
                self.data.update(
                    obs=obs_next,
                    info=info,
                )
                if rew_t > best_rew:
                    best_rew = rew_t
                    best_frames = frames
                rew_t = 0.0
                frames = []
            else:
                rew_t = rew_t + rew

            # 跳出循环判断
            count = count + 1

            KL = 0
            KL_2 = 0
            rdm_KL = 0
            if count >= 2:
                for k in range(0, length):
                    KL += abs(
                        mat[k][count - 1]
                        * math.log((mat[k][count - 1] / mat[k][count - 2]))
                    )
                    KL_2 += abs(
                        mat_2[k][count - 1]
                        * math.log((mat_2[k][count - 1] / mat_2[k][count - 2]))
                    )
                    rdm_KL += abs(
                        rdm_mat[k][count - 1]
                        * math.log((rdm_mat[k][count - 1] / rdm_mat[k][count - 2]))
                    )

                    ls.append(KL)
                    ls_2.append(KL_2)
                    rdm_ls.append(rdm_KL)

        end_time = time.time()

        print("execute_one_epoch  time :", end_time - start_time)

        mean_rew = mean_rew / bout
        mean_KL = 0
        mean_KL_2 = 0
        mean_rdm_KL = 0
        len_lis = len(ls)
        len_lis_2 = len(ls_2)
        len_rdm_lis_2 = len(rdm_ls)
        for i in range(0, len_lis - 1):
            mean_KL += ls[i]
        mean_KL /= len_lis
        mean_KL *= 100000
        for i in range(0, len_lis_2 - 1):
            mean_KL_2 += ls_2[i]
        mean_KL_2 /= len_lis_2
        mean_KL_2 *= 100000

        for i in range(0, len_rdm_lis_2 - 1):
            mean_rdm_KL += rdm_ls[i]
        mean_rdm_KL /= len_rdm_lis_2
        mean_rdm_KL *= 100000

        adv_act_values = [item[0] for item in act_fist]
        obs_fist_values = [item[0] for item in obs_fist]
        # if adv_act_values:  # 如果列表不为空
        #     print(adv_act_values)
        # if obs_fist_values:  # 如果列表不为空
        #     print(obs_fist_values)

        # 使用列表推导式将其转换为扁平列表
        # for i in range(self.steps):
        # flat_frames[i] = frames[i][0]
        flat_frames = [sublist[0] for sublist in best_frames]

        return (
            flat_frames,
            best_rew,
            nums_atk,
            succ_atk,
            mean_rdm_KL,
            mean_KL,
            mean_KL_2,
            adv_act_values,
            obs_fist_values,
            mean_rew
        )
