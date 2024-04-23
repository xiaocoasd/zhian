from tianshou.data import Batch, to_numpy
import numpy as np
from copy import deepcopy
import torch
import gymnasium as gym

import torch.nn.modules


def excute_for_one_episode(
    policy_t,
    env,
    data: Batch,
    net,
    is_rdm,
    is_atk,
    atk_frequence,
    adv_obs_gene,
):
    frames = []
    rew_t = 0
    steps = 0
    succ_atk = 0
    nums_atk = 0

    obs_input = []
    act_output = []
    obs_input_temp = [None]
    act_output_temp = [None]

    logits = net(data.obs)
    length = logits.size(1)

    logits_array = []
    logits_temp = np.zeros(length)

    while True:
        last_state = data.policy.pop("hidden_state", None)
        # print("**********")
        frames.append(env.render())
        # print("&&&&&&&&&&")
        if is_rdm:
            rdm_act = [env.action_space[0].sample()]
            rdm_act_sample = policy_t.map_action_inverse(rdm_act)
            data.update(act=rdm_act_sample)
        else:
            result = policy_t(data, last_state)
            policy = result.get("policy", Batch())
            assert isinstance(policy, Batch)
            state = result.get("state", Batch())
            if state is not None:
                policy.hidden_state = state
            act = to_numpy(result.act)
            data.update(policy=result.get("policy", Batch()), act=act)

            obs_input_temp = deepcopy(data.obs.tolist())
            act_output_temp = deepcopy(data.act.tolist())

            ori_logits = net(data.obs)
            # len = ori_logits.size(1)
            for i in range(0, length):
                logits_temp[i] = ori_logits[0][i].item()

        # 根据频率开始攻击并记录相关信息
        if is_rdm is False and is_atk:
            x = np.random.uniform(0, 1)
            if x < atk_frequence:
                ori_act = deepcopy(data.act)
                ori_obs = deepcopy(data.obs)

                adv_obs = adv_obs_gene.run(
                    ori_obs=ori_obs,
                    ori_act=ori_act,
                )

                # print(ori_obs)
                # print(adv_obs)

                with torch.no_grad():
                    adv_obs = adv_obs.cpu().detach().numpy()
                    data.obs = adv_obs
                    result = policy_t(data, last_state=None)
                adv_act = to_numpy(result.act)

                atk_logits = net(adv_obs)

                if isinstance(env.action_space[0], gym.spaces.Discrete):
                    if adv_act != ori_act:
                        succ_atk += 1
                        for i in range(0, length):
                            logits_temp[i] = atk_logits[0][i].item()
                        act_output_temp = deepcopy(adv_act.tolist())
                        obs_input_temp = deepcopy(adv_obs.tolist())

                elif isinstance(env.action_space[0], gym.spaces.Box):
                    max_action = (env.action_space)[0].high
                    min_action = (env.action_space)[0].low

                    len_action = max_action - min_action
                    chan_action = abs(adv_act - ori_act)

                    rate_action = np.divide(chan_action, len_action)
                    mean_rate = np.mean(rate_action)

                    if mean_rate >= 0.05:
                        succ_atk += 1
                        for i in range(0, length):
                            logits_temp[i] = atk_logits[0][i].item()
                        act_output_temp = deepcopy(adv_act.tolist())
                        obs_input_temp = deepcopy(adv_obs.tolist())
                else:
                    print("No")
                    print(env.action_space)
                    print(gym.spaces.Discrete)
                    print(gym.spaces.Box)
                    print(isinstance(env.action_space, gym.spaces.Box))
                    print(isinstance(env.action_space, gym.spaces.Discrete))

                nums_atk += 1

                data.update(obs=adv_obs, act=adv_act)

        act_output.append(deepcopy(act_output_temp[0]))
        obs_input.append(deepcopy(obs_input_temp[0]))
        logits_array.append(deepcopy(logits_temp))

        obs_next, rew, terminated, truncated, info = env.step(action=data.act)

        # print("*********")
        # print(obs_next)

        done = terminated | truncated
        # 更新数据
        data.update(
            obs=obs_next,
            rew=rew,
            done=done,
            info=info,
        )

        rew_t += rew
        steps += 1

        if any(done):
            break

    flat_frames = [sublist[0] for sublist in frames]
    return (
        flat_frames,
        rew_t,
        steps,
        succ_atk,
        nums_atk,
        logits_array,
        obs_input,
        act_output,
    )
