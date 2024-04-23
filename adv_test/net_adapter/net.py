import torch.nn as nn


class TSDQNNetAdapter(nn.Module):
    """
    Tianshou models return (logits, state) while Advertorch models should return (logits).
    Hence, this class adapts Tianshou output to Advertorch output."""

    def __init__(self, policy):
        super().__init__()
        self.net = policy.model

    def forward(self, obs, state=None):
        # return self.net(s)[0]
        return self.net(obs)[0]


class TSA2CPPONetAdapter(nn.Module):
    """
    Adapt the output of A2C-PPO models to Advertorch required output (logits)."""

    def __init__(self, policy, device):
        super().__init__()
        self.net = policy.actor
        self.device = device
        # self.dist = policy.dist_fn

    def init_net(self, module, weight_init, bias_init, gain):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def forward(self, obs, state=None):
        logits, hidden = self.net.preprocess(obs, state)

        init_ = lambda m: self.init_net(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        linear = init_(
            nn.Linear(len(logits[0]), self.net.output_dim, device=self.device)
        )
        x = linear(logits)
        return x
