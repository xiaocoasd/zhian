# PGDAttack

from adv_test.algo_adv.algo_adv.advertorch_algo.iterative_projected_gradient import (
    PGDAttack,
)
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import is_float_or_torch_tensor
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import batch_multiply
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import batch_clamp
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import normalize_by_pnorm
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import clamp_by_pnorm
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import batch_l1_proj

from copy import deepcopy


import torch
import torch.nn as nn
import numpy as np
from torch.distributions import laplace
from torch.distributions import uniform


def perturb_iterativeT(
    xvar,
    yvar,
    predict,
    nb_iter,
    eps,
    eps_iter,
    loss_fn,
    delta_init=None,
    minimize=False,
    ord=np.inf,
    clip_min=0.0,
    clip_max=1.0,
    l1_sparsity=None,
):

    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(xvar + delta)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            # delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data

            if xvar.dim() > 2:
                len_i = len(xvar[0])
                len_j = len(xvar[0][0])

                for i in range(len_i):
                    for j in range(len_j):
                        if xvar.data[0][i][j] + delta.data[0][i][j] < clip_min[i][j]:
                            delta.data[0][i][j] = clip_min[i][j] - xvar.data[0][i][j]
                        if xvar.data[0][i][j] + delta.data[0][i][j] > clip_max[i][j]:
                            delta.data[0][i][j] = clip_max[i][j] - xvar.data[0][i][j]

            else:

                for i in range(len(delta.data[0])):
                    if xvar.data[0][i] + delta.data[0][i] < clip_min[i]:
                        delta.data[0][i] = clip_min[i] - xvar.data[0][i]
                    if xvar.data[0][i] + delta.data[0][i] > clip_max[i]:
                        delta.data[0][i] = clip_max[i] - xvar.data[0][i]

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            # delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data

            if xvar.dim() > 2:
                len_i = len(xvar[0])
                len_j = len(xvar[0][0])

                for i in range(len_i):
                    for j in range(len_j):
                        if xvar.data[0][i][j] + delta.data[0][i][j] < clip_min[i][j]:
                            delta.data[0][i][j] = clip_min[i][j] - xvar.data[0][i][j]
                        if xvar.data[0][i][j] + delta.data[0][i][j] > clip_max[i][j]:
                            delta.data[0][i][j] = clip_max[i][j] - xvar.data[0][i][j]

            else:

                for i in range(len(delta.data[0])):
                    if xvar.data[0][i] + delta.data[0][i] < clip_min[i]:
                        delta.data[0][i] = clip_min[i] - xvar.data[0][i]
                    if xvar.data[0][i] + delta.data[0][i] > clip_max[i]:
                        delta.data[0][i] = clip_max[i] - xvar.data[0][i]

            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)

        elif ord == 1:
            grad = delta.grad.data
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            if xvar.is_cuda:
                delta.data = delta.data.cuda()
            # delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data

            if xvar.dim() > 2:
                len_i = len(xvar[0])
                len_j = len(xvar[0][0])

                for i in range(len_i):
                    for j in range(len_j):
                        if xvar.data[0][i][j] + delta.data[0][i][j] < clip_min[i][j]:
                            delta.data[0][i][j] = clip_min[i][j] - xvar.data[0][i][j]
                        if xvar.data[0][i][j] + delta.data[0][i][j] > clip_max[i][j]:
                            delta.data[0][i][j] = clip_max[i][j] - xvar.data[0][i][j]

            else:

                for i in range(len(delta.data[0])):
                    if xvar.data[0][i] + delta.data[0][i] < clip_min[i]:
                        delta.data[0][i] = clip_min[i] - xvar.data[0][i]
                    if xvar.data[0][i] + delta.data[0][i] > clip_max[i]:
                        delta.data[0][i] = clip_max[i] - xvar.data[0][i]
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    # x_adv = clamp(xvar + delta, clip_min, clip_max)

    x_adv = deepcopy(xvar)

    if x_adv.dim() > 2:
        len_i = len(xvar[0])
        len_j = len(xvar[0][0])

        for i in range(len_i):
            for j in range(len_j):
                if xvar.data[0][i][j] + delta.data[0][i][j] < clip_min[i][j]:
                    x_adv[0][i][j] = clip_min[i][j]
                elif xvar.data[0][i][j] + delta.data[0][i][j] > clip_max[i][j]:
                    x_adv[0][i][j] = clip_max[i][j]
                else:
                    x_adv[0][i][j] = xvar[0][i][j] + delta[0][i][j]

    else:

        for i in range(len(delta.data[0])):
            if xvar.data[0][i] + delta.data[0][i] < clip_min[i]:
                x_adv[0][i] = clip_min[i]
            elif xvar.data[0][i] + delta.data[0][i] > clip_max[i]:
                x_adv[0][i] = clip_max[i]
            else:
                x_adv[0][i] = xvar[0][i] + delta[0][i]

    return x_adv


def rand_init_deltaT(delta, x, ord, eps, clip_min, clip_max):

    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        # delta.data.uniform_(clip_min, clip_max)

        delta.data = torch.tensor(
            np.random.uniform(
                clip_min,
                clip_max,
                size=len(x[0]),
            )
        )

        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(loc=delta.new_tensor(0), scale=delta.new_tensor(1))
        delta.data = ini.sample(delta.data.shape)
        delta.data = normalize_by_pnorm(delta.data, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta.data *= ray

        # delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data

        if x.dim() > 2:
            len_i = len(x[0])
            len_j = len(x[0][0])

            for i in range(len_i):
                for j in range(len_j):
                    if x.data[0][i][j] + delta.data[0][i][j] < clip_min[i][j]:
                        delta.data[0][i][j] = clip_min[i][j] - x.data[0][i][j]
                    if x.data[0][i][j] + delta.data[0][i][j] > clip_max[i][j]:
                        delta.data[0][i][j] = clip_max[i][j] - x.data[0][i][j]

        else:

            for i in range(len(delta.data[0])):
                if x.data[0][i] + delta.data[0][i] < clip_min[i]:
                    delta.data[0][i] = clip_min[i] - x.data[0][i]
                if x.data[0][i] + delta.data[0][i] > clip_max[i]:
                    delta.data[0][i] = clip_max[i] - x.data[0][i]
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    # delta.data = clamp(x + delta.data, min=clip_min, max=clip_max) - x

    if x.dim() > 2:
        len_i = len(x[0])
        len_j = len(x[0][0])
        for i in range(len_i):
            for j in range(len_j):
                if x[0][i][j] + delta.data[0][i][j] < clip_min[i][j]:
                    delta.data[0][i][j] = clip_min[i][j] - x[0][i][j]
                if x[0][i][j] + delta.data[0][i][j] > clip_max[i][j]:
                    delta.data[0][i][j] = clip_max[i][j] - x[0][i][j]
    else:

        for i in range(len(delta.data[0])):
            if x[0][i] + delta.data[0][i] < clip_min[i]:
                delta.data[0][i] = clip_min[i] - x[0][i]
            if x[0][i] + delta.data[0][i] > clip_max[i]:
                delta.data[0][i] = clip_max[i] - x[0][i]

    return delta.data


class PGDAttackT(PGDAttack):

    def __init__(
        self,
        predict,
        loss_fn=None,
        eps=0.3,
        nb_iter=40,
        eps_iter=0.01,
        rand_init=True,
        clip_min=None,
        clip_max=None,
        ord=np.inf,
        l1_sparsity=None,
        targeted=False,
    ):

        super(PGDAttack, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None):

        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_deltaT(delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            # delta.data = clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x

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

        rval = perturb_iterativeT(
            x,
            y,
            self.predict,
            nb_iter=self.nb_iter,
            eps=self.eps,
            eps_iter=self.eps_iter,
            loss_fn=self.loss_fn,
            minimize=self.targeted,
            ord=self.ord,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval.data


class L2PGDAttackT(PGDAttackT):

    def __init__(
        self,
        predict,
        loss_fn=None,
        eps=0.3,
        nb_iter=40,
        eps_iter=0.01,
        rand_init=True,
        clip_min=None,
        clip_max=None,
        targeted=False,
    ):
        ord = 2
        super(L2PGDAttackT, self).__init__(
            predict=predict,
            loss_fn=loss_fn,
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            rand_init=rand_init,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=targeted,
            ord=ord,
        )


class SparseL1DescentAttackT(PGDAttackT):

    def __init__(
        self,
        predict,
        loss_fn=None,
        eps=0.3,
        nb_iter=40,
        eps_iter=0.01,
        rand_init=False,
        clip_min=None,
        clip_max=None,
        l1_sparsity=0.95,
        targeted=False,
    ):
        ord = 1
        super(SparseL1DescentAttackT, self).__init__(
            predict=predict,
            loss_fn=loss_fn,
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            rand_init=rand_init,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=targeted,
            ord=ord,
            l1_sparsity=l1_sparsity,
        )
