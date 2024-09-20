import torch
import numpy as np

from adv_test.algo_adv.algo_adv.advertorch_algo.utils import torch_arctanh
from adv_test.algo_adv.algo_adv.advertorch_algo.carlini_wagner import (
    CarliniWagnerL2Attack,
)


CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10


def calc_l2distsqT(x, y):
    d_t = (x - y) ** 2
    # print("*************")
    # print(d_t)
    d = np.array(d_t.to("cpu"))
    sum = 0.0

    if x.dim() > 2:
        len_i = len(x[0])
        len_j = len(x[0][0])
        for i in range(len_i):
            for j in range(len_j):
                sum += d[0][i][j]

    else:

        for i in range(len(d[0])):
            sum += d[0][i]

    return torch.FloatTensor([sum]).to(x.device)


def tanh_rescaleT(x, x_min=-1.0, x_max=1.0):

    if x.dim() > 2:
        x_tanh_t = torch.tanh(x)
        x_tanh = x_tanh_t.to("cpu").detach().numpy()
        len_i = len(x[0])
        len_j = len(x[0][0])

        result = torch.zeros(1, len_i, len_j).to(x.device)

        for i in range(len_i):
            for j in range(len_j):
                if x_max[i][j] - x_min[i][j] == float("inf"):
                    result[0][i][j] = (
                        x_tanh[0][i][j] * 0.5 * x_max[i][j]
                        + (x_max[i][j] + x_min[i][j]) * 0.5
                    )
                else:
                    result[0][i][j] = (
                        x_tanh[0][i][j] * 0.5 * (x_max[i][j] - x_min[i][j])
                        + (x_max[i][j] + x_min[i][j]) * 0.5
                    )
        return torch.Tensor(result).to(x.device)

    else:
        x_tanh_t = torch.tanh(x)
        x_tanh_tt = x_tanh_t.to("cpu")
        x_tanh = x_tanh_tt.detach().numpy()

        result = [[0.0 for _ in range(len(x[0]))]]
        for i in range(len(x[0])):
            if x_max[i] - x_min[i] == float("inf"):
                result[0][i] = (
                    x_tanh[0][i] * 0.5 * x_max[i] + (x_max[i] + x_min[i]) * 0.5
                )
            else:
                result[0][i] = (
                    x_tanh[0][i] * 0.5 * (x_max[i] - x_min[i])
                    + (x_max[i] + x_min[i]) * 0.5
                )
        return torch.Tensor(result).to(x.device)


class CarliniWagnerL2AttackT(CarliniWagnerL2Attack):

    def __init__(
        self,
        predict,
        num_classes,
        confidence=0,
        targeted=False,
        learning_rate=0.01,
        binary_search_steps=9,
        max_iterations=10000,
        abort_early=True,
        initial_const=1e-3,
        clip_min=None,
        clip_max=None,
        loss_fn=None,
    ):
        if loss_fn is not None:
            import warnings

            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = None

        super(CarliniWagnerL2Attack, self).__init__(
            predict, loss_fn, clip_min, clip_max
        )

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.targeted = targeted

    def _get_arctanh_x(self, x):

        if x.dim() > 2:
            len_i = len(x[0])
            len_j = len(x[0][0])

            result = torch.zeros(1, len_i, len_j).to(x.device)

            for i in range(len_i):
                for j in range(len_j):

                    if self.clip_max[i][j] - self.clip_min[i][j] == float("inf"):
                        # result[0][i][j] = (
                        #     x[0][i][j] - self.clip_min[i][j]
                        # ) / self.clip_max[i][j]
                        result[0][i][j] = 0.0
                    else:
                        result[0][i][j] = (x[0][i][j] - self.clip_min[i][j]) / (
                            self.clip_max[i][j] - self.clip_min[i][j]
                        )

            for i in range(len_i):
                for j in range(len_j):
                    if result[0][i][j] < self.clip_min[i][j]:
                        result[0][i][j] = self.clip_min[i][j]
                    elif result[0][i][j] > self.clip_max[i][j]:
                        result[0][i][j] = self.clip_max[i][j]

            # print("result")
            # print(result)
            return torch_arctanh(result * ONE_MINUS_EPS)
        else:

            result = torch.tensor([[0.0 for _ in range(len(x[0]))]]).to(x.device)

            for i in range(len(x[0])):
                if self.clip_max[i] - self.clip_min[i] == float("inf"):
                    result[0][i] = (x[0][i] - self.clip_min[i]) / self.clip_max[i]

                else:
                    result[0][i] = (x[0][i] - self.clip_min[i]) / (
                        self.clip_max[i] - self.clip_min[i]
                    )

            for i in range(len(x[0])):
                if result[0][i] < 0.0:
                    result[0][i] = -1
                elif result[0][i] > 1.0:
                    result[0][i] = 1
                else:
                    result[0][i] = result[0][i] * 2 - 1

            return torch_arctanh(result * ONE_MINUS_EPS)

    def _forward_and_update_delta(
        self, optimizer, x_atanh, delta, y_onehot, loss_coeffs
    ):

        # print("delta")
        # print(delta)
        # print("x_atanh")
        # print(x_atanh)

        optimizer.zero_grad()
        adv = tanh_rescaleT(delta + x_atanh, self.clip_min, self.clip_max)
        # print("adv")
        # print(adv)
        transimgs_rescale = tanh_rescaleT(x_atanh, self.clip_min, self.clip_max)
        output = self.predict(adv)
        l2distsq = calc_l2distsqT(adv, transimgs_rescale)
        # print("l2distsq")
        # print(l2distsq)
        loss = self._loss_fn(output, y_onehot, l2distsq, loss_coeffs)
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.data, output.data, adv.data

    def _update_if_smaller_dist_succeed(
        self,
        adv_img,
        labs,
        output,
        l2distsq,
        batch_size,
        cur_l2distsqs,
        cur_labels,
        final_l2distsqs,
        final_labels,
        final_advs,
    ):

        target_label = labs
        output_logits = output
        _, output_label = torch.max(output_logits, 1)

        mask = (l2distsq < cur_l2distsqs) & self._is_successful(
            output_logits, target_label, True
        )

        # print(cur_l2distsqs)
        # print(l2distsq)

        cur_l2distsqs[mask] = l2distsq[mask]  # redundant
        cur_labels[mask] = output_label[mask]

        mask = (l2distsq < final_l2distsqs) & self._is_successful(
            output_logits, target_label, True
        )
        final_l2distsqs[mask] = l2distsq[mask]
        final_labels[mask] = output_label[mask]

        final_advs[mask] = adv_img[mask]
