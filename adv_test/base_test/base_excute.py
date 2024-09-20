from adv_test.utils import excute_for_one_episode

from abc import ABC
from tianshou.data import Batch
from copy import deepcopy


class BaseTestExe(ABC):

    def __init__(
        self,
        policy,
        env,
        test_episode,
        net,
        atk_frequence,
        adv_obs_gene,
    ) -> None:
        super().__init__()

        self.policy = policy
        self.net = net
        self.env = env

        self.atk_frequence = atk_frequence
        self.adv_obs_gene = adv_obs_gene

        self.test_episode = test_episode

        self.data = Batch(
            state={}, obs={}, act={}, rew={}, done={}, info={}, obs_next={}, policy={}
        )

    def run(self):

        ori_frames_array = []
        rdm_frames_array = []

        ori_mean_rew = 0.0
        rdm_mean_rew = 0.0
        ori_all_rew = []

        ori_tree_input = []
        ori_tree_output = []

        ori_pos_array = []

        episode_count = 0
        while episode_count < self.test_episode:

            print("第{}回合测试开始".format(episode_count + 1))

            self.reset()
            (
                rdm_frames,
                rdm_rew,
                rdm_steps,
                rdm_succ_atk,
                rdm_nums_atk,
                rdm_logits_array,
                rdm_obs_input,
                rdm_act_output,
            ) = excute_for_one_episode(
                policy_t=deepcopy(self.policy),
                env=deepcopy(self.env),
                data=deepcopy(self.data),
                net=deepcopy(self.net),
                is_rdm=True,
                is_atk=False,
                atk_frequence=self.atk_frequence,
                adv_obs_gene=self.adv_obs_gene,
            )

            rdm_mean_rew += rdm_rew

            rdm_frames_array.append(rdm_frames)

            self.reset()
            (
                ori_frames,
                ori_rew,
                ori_steps,
                ori_succ_atk,
                ori_nums_atk,
                ori_logits_array,
                ori_obs_input,
                ori_act_output,
            ) = excute_for_one_episode(
                policy_t=deepcopy(self.policy),
                env=deepcopy(self.env),
                data=deepcopy(self.data),
                net=deepcopy(self.net),
                is_rdm=False,
                is_atk=False,
                atk_frequence=self.atk_frequence,
                adv_obs_gene=self.adv_obs_gene,
            )

            ori_mean_rew += ori_rew
            ori_all_rew.append(ori_rew)

            ori_frames_array.append(ori_frames)
            ori_tree_input.append(ori_obs_input)
            ori_tree_output.append(ori_act_output)
            ori_pos_array.append(ori_logits_array)

            print("第{}回合测试结束".format(episode_count + 1))

            episode_count += 1

        rdm_mean_rew = rdm_mean_rew / self.test_episode
        ori_mean_rew = ori_mean_rew / self.test_episode

        return (
            rdm_frames_array,
            ori_frames_array,
            rdm_mean_rew,
            ori_mean_rew,
            ori_all_rew,
            ori_tree_input,
            ori_tree_output,
            ori_pos_array,
        )

    def reset(self) -> None:
        self.data = Batch(
            state={},
            obs={},
            act={},
            rew={},
            done={},
            info={},
            policy={},
        )
        self.reset_env()

    def reset_env(self) -> None:
        obs, info = self.env.reset()

        self.data.obs = obs
        self.data.info = info
