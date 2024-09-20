from adv_test.algo_adv.algo_adv.advertorch_algo.GradientSignAttack import (
    GradientSignAttack,
)
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import batch_multiply


import torch.nn as nn


class GradientSignAttackT(GradientSignAttack):

    def __init__(
        self,
        predict,
        loss_fn=None,
        eps=0.3,
        clip_min=None,
        clip_max=None,
        targeted=False,
    ):

        super(GradientSignAttack, self).__init__(predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = xadv.grad.detach().sign()

        xadv = xadv + batch_multiply(self.eps, grad_sign)

        # xadv = clamp(xadv, self.clip_min, self.clip_max)

        # if xadv.dim() > 2:
        #     xadv_t = xadv.view(1, -1)
        #     clip_min_t = self.clip_min.view(-1)
        #     clip_max_t = self.clip_max.view(-1)
        # else:
        #     xadv_t = xadv
        #     clip_min_t = self.clip_min
        #     clip_max_t = self.clip_max

        if xadv.dim() > 2:
            len_i = len(xadv[0])
            len_j = len(xadv[0][0])
            for i in range(len_i):
                for j in range(len_j):
                    if xadv[0][i][j] < self.clip_min[i][j]:
                        xadv[0][i][j] = self.clip_min[i][j]
                    elif xadv[0][i][j] > self.clip_max[i][j]:
                        xadv[0][i][j] = self.clip_max[i][j]
        else:
            for i in range(len(xadv[0])):

                if xadv[0][i] < self.clip_min[i]:
                    xadv[0][i] = self.clip_min[i]
                if xadv[0][i] > self.clip_max[i]:
                    xadv[0][i] = self.clip_max[i]

        # for i in range(len(xadv[0])):
        #     if xadv[0][i] < self.clip_min[i]:
        #         xadv[0][i] = self.clip_min[i]
        #     if xadv[0][i] > self.clip_max[i]:
        #         xadv[0][i] = self.clip_max[i]
        return xadv.detach()
