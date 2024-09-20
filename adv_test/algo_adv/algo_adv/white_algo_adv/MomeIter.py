import numpy as np
import torch
import torch.nn as nn

from adv_test.algo_adv.algo_adv.advertorch_algo.utils import clamp
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import normalize_by_pnorm
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import batch_multiply
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import batch_clamp
from adv_test.algo_adv.algo_adv.advertorch_algo.iterative_projected_gradient import (
    MomentumIterativeAttack,
)


class MomentumIterativeAttackT(MomentumIterativeAttack):

    def __init__(
        self,
        predict,
        loss_fn=None,
        eps=0.3,
        nb_iter=40,
        decay_factor=1.0,
        eps_iter=0.01,
        clip_min=None,
        clip_max=None,
        targeted=False,
        ord=np.inf,
    ):

        super(MomentumIterativeAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max
        )
        self.eps = eps
        self.nb_iter = nb_iter
        self.decay_factor = decay_factor
        self.eps_iter = eps_iter
        self.targeted = targeted
        self.ord = ord
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):

        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)

        delta = nn.Parameter(delta)

        for i in range(self.nb_iter):

            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

            imgadv = x + delta
            outputs = self.predict(imgadv)
            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss
            loss.backward()

            g = self.decay_factor * g + normalize_by_pnorm(delta.grad.data, p=1)

            if self.ord == np.inf:
                delta.data += batch_multiply(self.eps_iter, torch.sign(g))
                delta.data = batch_clamp(self.eps, delta.data)
                # delta.data = (
                #     clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x
                # )
                if x.dim() > 2:
                    len_i = len(x[0])
                    len_j = len(x[0][0])
                    for i in range(len_i):
                        for j in range(len_j):

                            if x[0][i][j] + delta.data[0][i][j] < self.clip_min[i][j]:
                                delta.data[0][i][j] = self.clip_min[i][j] - x[0][i][j]

                            if x[0][i][j] + delta.data[0][i][j] > self.clip_max[i][j]:
                                delta.data[0][i][j] = self.clip_max[i][j] - x[0][i][j]
                else:
                    for i in range(len(delta.data[0])):
                        if x[0][i] + delta.data[0][i] < self.clip_min[i]:
                            delta.data[0][i] = self.clip_min[i] - x[0][i]

                        if x[0][i] + delta.data[0][i] > self.clip_max[i]:
                            delta.data[0][i] = self.clip_max[i] - x[0][i]
            elif self.ord == 2:
                delta.data += self.eps_iter * normalize_by_pnorm(g, p=2)

                delta.data *= clamp(
                    (self.eps * normalize_by_pnorm(delta.data, p=2) / delta.data),
                    max=1.0,
                )

                # delta.data = (
                #     clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x
                # )

                if x.dim() > 2:
                    len_i = len(x[0])
                    len_j = len(x[0][0])
                    for i in range(len_i):
                        for j in range(len_j):

                            if x[0][i][j] + delta.data[0][i][j] < self.clip_min[i][j]:
                                delta.data[0][i][j] = self.clip_min[i][j] - x[0][i][j]

                            if x[0][i][j] + delta.data[0][i][j] > self.clip_max[i][j]:
                                delta.data[0][i][j] = self.clip_max[i][j] - x[0][i][j]
                else:
                    for i in range(len(delta.data[0])):
                        if x[0][i] + delta.data[0][i] < self.clip_min[i]:
                            delta.data[0][i] = self.clip_min[i] - x[0][i]

                        if x[0][i] + delta.data[0][i] > self.clip_max[i]:
                            delta.data[0][i] = self.clip_max[i] - x[0][i]
            else:
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)

        rval = x + delta.data
        return rval
