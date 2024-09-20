import torch.nn as nn
import torch


from adv_test.algo_adv.algo_adv.advertorch_algo.utils import clamp
from adv_test.algo_adv.algo_adv.advertorch_algo.base import Attack
from adv_test.algo_adv.algo_adv.advertorch_algo.base import LabelMixin


class ZerothOrderOptimizationAttack(Attack, LabelMixin):
    """
    Zeroth Order Optimization attack using finite differences to approximate gradients.
    This method uses the outputs of the model to estimate gradients and performs an
    optimization step based on these estimated gradients.

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: perturbation size.
    :param nb_samples: number of samples for finite difference.
    :param delta: finite difference step size.
    :param clip_min: minimum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(
        self,
        predict,
        loss_fn=None,
        eps=0.3,
        nb_samples=20,
        delta=0.01,
        clip_min=0.0,
        clip_max=1.0,
        targeted=False,
    ):
        """
        Create an instance of the ZerothOrderOptimizationAttack.
        """
        super(ZerothOrderOptimizationAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max
        )

        self.eps = eps
        self.nb_samples = nb_samples
        self.delta = delta
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        a perturbation based on zeroth order optimization.

        :param x: input tensor.
        :param y: label tensor.
        - if None and self.targeted=False, compute y as predicted labels.
        - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.clone().requires_grad_(False)

        for i in range(self.nb_samples):
            random_direction = torch.randn_like(xadv)
            xplus = xadv + self.delta * random_direction
            xminus = xadv - self.delta * random_direction

            fplus = self.predict(xplus)
            fminus = self.predict(xminus)

            loss_plus = self.loss_fn(fplus, y)
            loss_minus = self.loss_fn(fminus, y)

            if self.targeted:
                grad_approx = (
                    (loss_minus - loss_plus) / (2 * self.delta) * random_direction
                )
            else:
                grad_approx = (
                    (loss_plus - loss_minus) / (2 * self.delta) * random_direction
                )

            xadv = xadv + self.eps * grad_approx.sign()

        # xadv = clamp(xadv, self.clip_min, self.clip_max)

        # for i in range(len(xadv[0])):

        #     if xadv[0][i] < self.clip_min[i]:
        #         xadv[0][i] = self.clip_min[i]
        #     if xadv[0][i] > self.clip_max[i]:
        #         xadv[0][i] = self.clip_max[i]

        return xadv.detach()
