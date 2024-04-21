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
    ) -> None:
        super().__init__()

        self.args = args
        self.net = net
        self.atk_eps = atk_eps

    def run(self, ori_obs, ori_act: List[int]):

        obs_atk = make_img_atk(
            args=self.args,
            net=self.net,
            atk_eps=self.atk_eps,
        )

        ori_obs = torch.FloatTensor(ori_obs).to(self.args.device)
        ori_act = torch.tensor(ori_act).to(self.args.device)

        adv_obs = obs_atk.perturb(ori_obs, ori_act)

        return adv_obs
