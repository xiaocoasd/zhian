from adv_test.net_adapter.net import TSDQNNetAdapter, TSA2CPPONetAdapter

import copy


def NetAdapter(policy_type, policy, device):

    if policy_type == "dqn":
        adv_net = TSDQNNetAdapter(copy.deepcopy(policy)).to(device)
    elif policy_type in ["ppo", "a2c"]:
        adv_net = TSA2CPPONetAdapter(copy.deepcopy(policy), device=device).to(device)
    adv_net.eval()
    return adv_net
