from abc import ABC

import gym
import torch.nn as nn
import torch

from adv_test.adv_sample_gene.algo_utils import make_img_atk
from typing import List


# 返回样本扰动
class ImgAdvObsGene(ABC):

    def __init__(
        self,
        net: nn.Module,
        atk_type: str,
        min_pixel: float,
        max_pixel: float,
        targeted: bool,
        device: str,
        iterations: int,
        env: gym.Env
    ) -> None:
        super().__init__()

        self.net = net
        self.atk_type = atk_type
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.targeted = targeted
        self.device = device
        self.env = env
        self.iterations = iterations
    # 返回攻击类 ，此类实现位于utils
    # def make_img_atk(
    #     self, net, atk_eps, atk_type, min_pixel=0.0, max_pixel=255.0, targeted=False
    # ):

    def run(self, ori_obs, ori_act: List[int], atk_eps):

        obs_atk = make_img_atk(
            net=self.net,
            atk_eps=atk_eps,
            atk_type=self.atk_type,
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
            targeted=self.targeted,
            env=self.env,
            iterations=self.iterations
        )

        ori_obs = torch.FloatTensor(ori_obs).to(self.device)
        ori_act = torch.tensor(ori_act).to(self.device)
        # adv_obs = None
        adv_obs = obs_atk.perturb(ori_obs, ori_act)

        return adv_obs
