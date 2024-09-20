from abc import ABC
import torch.nn as nn
import torch

from adv_test.algo_adv.algo_utils import make_img_atk
from typing import List


# 返回样本扰动
class AdvAtkObsGene(ABC):

    def __init__(
        self,
        args,
        net: nn.Module,
        atk_eps,
        high,
        low,
    ) -> None:
        super().__init__()

        self.args = args
        self.net = net
        self.atk_eps = atk_eps
        self.high = high
        self.low = low

    def run(self, ori_obs, ori_act: List[int]):

        obs_atk = make_img_atk(
            args=self.args,
            net=self.net,
            atk_eps=self.atk_eps,
            high=self.high,
            low=self.low,
        )

        ori_obs = torch.FloatTensor(ori_obs).to(self.args.device)
        ori_act = torch.tensor(ori_act).to(self.args.device)

        # if ori_obs.dim() > 2:
        #     ori_obs_t = ori_obs.view(1, -1)
        #     adv_obs_t = obs_atk.perturb(ori_obs_t, ori_act)
        #     adv_obs = torch.ones(1, len_i, len_j)
        #     for i in range(0, len_i):
        #         for j in range(0, len_j):
        #             adv_obs[0][i][j] = adv_obs_t[0][i * len_j + j]
        # else:
        #     adv_obs = obs_atk.perturb(ori_obs, ori_act)

        adv_obs = obs_atk.perturb(ori_obs, ori_act)

        return adv_obs
