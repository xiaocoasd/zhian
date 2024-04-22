import torch
import torch.nn as nn

import numpy as np

from adv_test.algo_adv.algo_adv.advertorch_algo.utils import to_one_hot
from adv_test.algo_adv.algo_adv.advertorch_algo.utils import replicate_input
from adv_test.algo_adv.algo_adv.advertorch_algo.ead import ElasticNetL1Attack


DIST_UPPER = 1e10
COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000
NUM_CHECKS = 10


def calc_l2distsqT(x, y):
    d_t = (x - y) ** 2
    d = np.array(d_t.data)
    sum = 0.0

    for i in range(len(d[0])):
        sum += d[0][i]

    return torch.FloatTensor([sum])


def calc_l1distT(x, y):
    d = torch.abs(x - y)
    # print(d)

    return d.view(d.shape[0], -1).sum(dim=1)


class ElasticNetL1AttackT(ElasticNetL1Attack):

    def __init__(
        self,
        predict,
        num_classes,
        confidence=0,
        targeted=False,
        learning_rate=1e-2,
        binary_search_steps=9,
        max_iterations=10000,
        abort_early=False,
        initial_const=1e-3,
        clip_min=None,
        clip_max=None,
        beta=1e-2,
        decision_rule="EN",
        loss_fn=None,
    ):
        """ElasticNet L1 Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings

            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = None

        super(ElasticNetL1Attack, self).__init__(predict, loss_fn, clip_min, clip_max)

        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        self.beta = beta
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.targeted = targeted
        self.decision_rule = decision_rule

    def _fast_iterative_shrinkage_thresholding(self, x, yy_k, xx_k):

        zt = self.global_step / (self.global_step + 3)

        upper_t = [[0.0 for _ in range(len(x[0]))]]
        lower_t = [[0.0 for _ in range(len(x[0]))]]

        for i in range(len(x[0])):
            if yy_k[0][i] - self.beta > self.clip_max[i]:
                upper_t[0][i] = self.clip_max[i]
            else:
                upper_t[0][i] = yy_k[0][i] - self.beta

            if yy_k[0][i] + self.beta < self.clip_min[i]:
                lower_t[0][i] = self.clip_min[i]
            else:
                lower_t[0][i] = yy_k[0][i] + self.beta

        upper = torch.FloatTensor(upper_t)
        lower = torch.FloatTensor(lower_t)

        diff = yy_k - x
        cond1 = (diff > self.beta).float()
        cond2 = (torch.abs(diff) <= self.beta).float()
        cond3 = (diff < -self.beta).float()

        xx_k_p_1 = (cond1 * upper) + (cond2 * x) + (cond3 * lower)
        yy_k.data = xx_k_p_1 + (zt * (xx_k_p_1 - xx_k))
        return yy_k, xx_k_p_1

    def perturb(self, x, y=None):

        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        x = replicate_input(x)
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const

        final_dist = [DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size

        final_advs = x.clone()
        y_onehot = to_one_hot(y, self.num_classes).float()

        final_dist = torch.FloatTensor(final_dist).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):

            self.global_step = 0

            # slack vector from the paper
            yy_k = nn.Parameter(x.clone())
            xx_k = x.clone()

            cur_dist = [DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size

            cur_dist = torch.FloatTensor(cur_dist).to(x.device)
            cur_labels = torch.LongTensor(cur_labels).to(x.device)

            prevloss = PREV_LOSS_INIT

            if self.repeat and outer_step == (self.binary_search_steps - 1):
                loss_coeffs = coeff_upper_bound

            lr = self.learning_rate

            for ii in range(self.max_iterations):

                # reset gradient
                if yy_k.grad is not None:
                    yy_k.grad.detach_()
                    yy_k.grad.zero_()

                # loss over yy_k with only L2 same as C&W
                # we don't update L1 loss with SGD because we use ISTA
                output = self.predict(yy_k)
                l2distsq = calc_l2distsqT(yy_k, x)
                loss_opt = self._loss_fn(
                    output, y_onehot, None, l2distsq, loss_coeffs, opt=True
                )
                loss_opt.backward()

                # gradient step
                yy_k.data.add_(-lr, yy_k.grad.data)
                self.global_step += 1

                # ploynomial decay of learning rate
                lr = (
                    self.init_learning_rate
                    * (1 - self.global_step / self.max_iterations) ** 0.5
                )

                yy_k, xx_k = self._fast_iterative_shrinkage_thresholding(x, yy_k, xx_k)

                # loss ElasticNet or L1 over xx_k
                with torch.no_grad():
                    output = self.predict(xx_k)
                    l2distsq = calc_l2distsqT(xx_k, x)
                    l1dist = calc_l1distT(xx_k, x)

                    if self.decision_rule == "EN":
                        dist = l2distsq + (l1dist * self.beta)
                    elif self.decision_rule == "L1":
                        dist = l1dist
                    loss = self._loss_fn(
                        output, y_onehot, l1dist, l2distsq, loss_coeffs
                    )

                    if self.abort_early:
                        if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                            if loss > prevloss * ONE_MINUS_EPS:
                                break
                            prevloss = loss

                    self._update_if_smaller_dist_succeed(
                        xx_k.data,
                        y,
                        output,
                        dist,
                        batch_size,
                        cur_dist,
                        cur_labels,
                        final_dist,
                        final_labels,
                        final_advs,
                    )

            self._update_loss_coeffs(
                y,
                cur_labels,
                batch_size,
                loss_coeffs,
                coeff_upper_bound,
                coeff_lower_bound,
            )

        return final_advs
