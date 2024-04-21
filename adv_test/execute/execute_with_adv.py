from abc import ABC
import gymnasium as gym
import torch.nn as nn
import numpy as np


from tianshou.policy import BasePolicy
from tianshou.data import Batch

from adv_test.algo_adv.algo_top import ImgAdvObsGene
from adv_test.execute.execute_per_epoch import ExecutePerEpoch


class TestWithAdv(ABC):

    def __init__(
        self,
        args,
        myWindow,
        policy: BasePolicy,
        net: nn.Module,
        ori_env: gym.Env,
        atk_env: gym.Env,
    ) -> None:
        super().__init__()
        self.myWindow = myWindow
        self.args = args
        self.policy = policy
        self.net = net
        self.ori_env = ori_env
        self.atk_env = atk_env
        self.data = Batch(
            state={}, obs={}, act={}, rew={}, done={}, info={}, obs_next={}, policy={}
        )

        self.obs_atk = None

        # self.reset()

    def reset(self, is_atk: bool) -> None:
        self.data = Batch(
            state={},
            obs={},
            act={},
            rew={},
            done={},
            info={},
            policy={},
        )
        self.reset_env(is_atk)

    def reset_env(self, is_atk: bool) -> None:
        if is_atk:
            obs, info = self.atk_env.reset()
        else:
            obs, info = self.ori_env.reset()

        self.data.obs = obs
        self.data.info = info

    def run(self):

        epoch = 0
        # 指标
        nums_atk_array = np.arange(self.args.test_epoch)
        succ_atk_array = np.arange(self.args.test_epoch)
        rdm_rew_array = np.arange(self.args.test_epoch)
        ori_rew_array = np.arange(self.args.test_epoch)
        atk_rew_array = np.arange(self.args.test_epoch)

        rdm_frames = np.empty(self.args.test_epoch, dtype=object)
        ori_frames = np.empty(self.args.test_epoch, dtype=object)
        atk_frames = np.empty(self.args.test_epoch, dtype=object)

        rdm_mean_KL = np.arange(self.args.test_epoch)
        mean_KL_array = np.arange(self.args.test_epoch)
        mean_KL_2_array = np.arange(self.args.test_epoch)
        adv_act_values_array = []
        obs_fist_values_array = []
        mean_rew_array = []
        atk_eps_array=[]
        while epoch < self.args.test_epoch:

            # 指标
            nums_atk = 0
            succ_atk = 0
            # 一些需要的参数
            rdm_best_rew = 0.0
            mean_rdm_KL = 0.0
            ori_best_rew = 0.0
            atk_best_rew = 0.0
            mean_KL = 0.0
            mean_KL_2 = 0.0
            adv_obs_gene = ImgAdvObsGene(
                args=self.args,
                net=self.net,
            )
            self.args.atk_eps=0.2
            self.reset(is_atk=False)
            rdm_frame, rdm_best_rew, _, _, mean_rdm_KL, *_ = ExecutePerEpoch(
                policy=self.policy,
                env=self.ori_env,
                data=self.data,
                atk_eps=self.args.atk_eps,
                atk_frequence=self.args.atk_frequence,
                steps=self.args.step_per_epoch,
                adv_obs_gene=adv_obs_gene,
                is_atk=False,
                is_rdm=True,
                net=self.net,
            ).run()

            self.reset(is_atk=False)
            ori_frame, ori_best_rew, *_ = ExecutePerEpoch(
                policy=self.policy,
                env=self.ori_env,
                data=self.data,
                atk_eps=self.args.atk_eps,
                atk_frequence=self.args.atk_frequence,
                steps=self.args.step_per_epoch,
                adv_obs_gene=adv_obs_gene,
                is_atk=False,
                is_rdm=False,
                net=self.net,
            ).run()
            self.args.atk_eps = 1 *(epoch+1)/self.args.test_epoch
            atk_eps_array.append(self.args.atk_eps)
            self.reset(is_atk=True)
            (
                atk_frame,
                atk_best_rew,
                nums_atk,
                succ_atk,
                _,
                mean_KL,
                mean_KL_2,
                adv_act_values,
                obs_fist_values,
                mean_rew,
            ) = ExecutePerEpoch(
                policy=self.policy,
                env=self.atk_env,
                data=self.data,
                atk_eps=self.args.atk_eps,
                atk_frequence=self.args.atk_frequence,
                steps=self.args.step_per_epoch,
                adv_obs_gene=adv_obs_gene,
                is_atk=True,
                is_rdm=False,
                net=self.net,
            ).run()

            rdm_frames[epoch] = rdm_frame
            ori_frames[epoch] = ori_frame
            atk_frames[epoch] = atk_frame

            rdm_rew_array[epoch] = rdm_best_rew
            rdm_mean_KL[epoch] = mean_rdm_KL
            ori_rew_array[epoch] = ori_best_rew
            atk_rew_array[epoch] = atk_best_rew
            nums_atk_array[epoch] = nums_atk
            succ_atk_array[epoch] = succ_atk
            mean_KL_array[epoch] = mean_KL
            mean_KL_2_array[epoch] = mean_KL_2
            adv_act_values_array.append(adv_act_values)
            obs_fist_values_array.append(obs_fist_values)
            mean_rew_array.append(mean_rew)


            self.myWindow.test_progress.append("第{}轮测试完成\n   攻击次数:{}\n   成功攻击次数:{}\n   该环境下基础奖励:{}\n   模型原最佳奖励:{}\n   "
                                               "攻击后最佳奖励:{}\n"
                                               .format(epoch + 1, nums_atk, succ_atk, rdm_best_rew, ori_best_rew,
                                                       atk_best_rew))
            epoch = epoch + 1
        print("atk_eps_array")
        print(atk_eps_array)
        print("acr_array")
        print(succ_atk_array/nums_atk_array)
        print("mean_rew_array")
        print(mean_rew_array)
        return (
            rdm_frames,
            ori_frames,
            atk_frames,
            atk_eps_array,
            nums_atk_array,
            succ_atk_array,
            ori_rew_array,
            atk_rew_array,
            rdm_rew_array,
            rdm_mean_KL,
            mean_KL_array,
            mean_KL_2_array,
            adv_act_values_array,
            obs_fist_values_array,
            mean_rew_array
        )
