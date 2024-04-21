from abc import ABC
import gymnasium as gym
import numpy as np
from tianshou.env import DummyVectorEnv
import os
from copy import deepcopy

from adv_test.base_test.base_excute import BaseTestExe
from adv_test.algo_adv.algo_top import AlgoAtkAdv
from adv_test.net_adapter.net_top import NetAdapter
from adv_test.env_adv import EnvAdv
from adv_test.show.logits_plt import LogitsPlt
from adv_test.show.tree import TreeTop
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


class AdvTestTop(ABC):

    def __init__(
            self,
            args,
            policy,
            env: gym.Env
    ) -> None:
        super().__init__()
        self.policy = policy
        self.args = args
        if self.args.is_based_on_gym:
            self.env = DummyVectorEnv(
                [lambda: gym.make(self.args.task, render_mode="rgb_array")]
            )
        else:
            self.env = deepcopy(env)

        # self.reset()

    def run(self):
        print("test begin...")
        print("adv_sample_test begin...")
        # self.my_window.test_progress.append("开始测试 ！\n")
        asr_mean = 0.0
        robust_rate = 0.0
        rew_change_mean_rew = 0.0
        env_stats_mean_rew = 0.0
        net_t = NetAdapter(
            policy_type=self.args.policy_type,
            policy=deepcopy(self.policy),
            device=self.args.device,
        )
        # self.my_window.test_progress.append("正在进行冒烟测试......\n")
        print("冒烟测试开始")
        (
            rdm_frames_array,
            ori_frames_array,
            rdm_mean_rew,
            ori_mean_rew,
            ori_tree_input,
            ori_tree_output,
            ori_pos_array,
        ) = BaseTestExe(
            policy=self.policy,
            env=self.env,
            test_episode=1,
            net=net_t,
            atk_frequence=None,
            adv_obs_gene=None,
        ).run()
        # self.my_window.test_progress.append("冒烟测试完毕，正在打印结果...\n")
        rdm_rdm_index = np.random.choice(len(rdm_frames_array), replace=False)
        ori_rdm_index = np.random.choice(len(ori_frames_array), replace=False)

        rdm_rdm_frame = rdm_frames_array[rdm_rdm_index]
        ori_rdm_frame = ori_frames_array[ori_rdm_index]
        # 随机模型表现  "resource/random/rdm_result.gif"
        # self.my_window.test_progress.append("正在生成随机模型结果...\n")
        make_gif(
            rdm_rdm_frame,
            is_rdm=True,
            is_atk=False,
            is_env_adv=False,
            env_rew_adv=False,
            env_stats_adv=False,
        )
        # self.ready.emit("rdm_result")
        # self.my_window.test_progress.append("正在生成待测模型结果...\n")
        # 原模型"resource/origin/ori_result.gif"
        make_gif(
            ori_rdm_frame,
            is_rdm=False,
            is_atk=False,
            is_env_adv=False,
            env_rew_adv=False,
            env_stats_adv=False,
        )
        # self.ready.emit("ori_result")
        ori_tree_input_value = [item for sublist in ori_tree_input for item in sublist]
        ori_tree_output_value = [
            item for sublist in ori_tree_output for item in sublist
        ]
        # self.my_window.test_progress.append("正在生成模型行为决策树...\n")
        TreeTop(
            tree_input=ori_tree_input_value,
            tree_output=ori_tree_output_value,
        )
        # self.ready.emit("ori_tree")
        # self.my_window.test_progress.append("正在待测模型的行为预测图...\n")
        ori_logits = LogitsPlt(
            data=ori_pos_array[0],
            act_data=ori_tree_output[0],
        )
        ori_logits.plt(
            is_2d=True,
            is_3d=False,
            path="resource/act_pos/origin/result.png",
            title="action position for origin policy",
        )
        # self.ready.emit("act_pos_ori_result")
        print("冒烟测试结束")
        # self.my_window.test_progress.append("冒烟测试结束！\n")

        if (self.args.do_attack):
            print("对抗攻击开始")
            # self.my_window.test_progress.append("\n正在执行对抗攻击测试......\n")
            (
                atk_low_frames,
                atk_med_frames,
                atk_high_frames,
                atk_med_rew,
                atk_low_pos,
                atk_med_pos,
                atk_high_pos,
                atk_low_act,
                atk_med_act,
                atk_high_act,
                atk_succ_rate_array,
                atk_rew_array,
                asr_mean,
            ) = AlgoAtkAdv(
                # my_window=self.my_window,ready=self.ready,
                args=self.args, policy=self.policy, net=net_t, env=self.env
            ).run()

            print("对抗攻击结束")
            # self.my_window.test_progress.append("对抗攻击执行完毕！\n")
            # self.my_window.test_progress.append("正在生成结果\n")
            print("开始保存图片")
            make_gif(
                atk_low_frames,
                is_rdm=False,
                is_atk=True,
                is_env_adv=False,
                env_stats_adv=False,
                env_rew_adv=False,
                gif_code=1,
            )
            make_gif(
                atk_med_frames,
                is_rdm=False,
                is_atk=True,
                is_env_adv=False,
                env_stats_adv=False,
                env_rew_adv=False,
                gif_code=2,
            )
            make_gif(
                atk_high_frames,
                is_rdm=False,
                is_atk=True,
                is_env_adv=False,
                env_stats_adv=False,
                env_rew_adv=False,
                gif_code=3,
            )
            # self.ready.emit("atk_gif")
            succ_rate_value = np.concatenate(atk_succ_rate_array)
            rew_value = np.concatenate(atk_rew_array)
            # print(rew_value)
            # print(len(rew_value))
            sequence = np.arange(0.1, 1.1, 0.1)

            atk_frequence = np.tile(sequence, (100 // len(sequence) + 1))[:100]
            atk_eps = np.repeat(sequence, 10)
            rew_plt_value = [list(t) for t in zip(atk_eps, atk_frequence, rew_value)]
            asr_plt_value = [list(t) for t in zip(atk_eps, atk_frequence, succ_rate_value)]

            rew_plt_3d = LogitsPlt(data=rew_plt_value)
            asr_plt_3d = LogitsPlt(data=asr_plt_value)
            atk_logits_low = LogitsPlt(data=atk_low_pos, act_data=atk_low_act)
            atk_logits_med = LogitsPlt(data=atk_med_pos, act_data=atk_med_act)
            atk_logits_high = LogitsPlt(data=atk_high_pos, act_data=atk_high_act)
            # self.my_window.test_progress.append("正在渲染3D图......\n")
            reward_fig = rew_plt_3d.plt(
                is_2d=False,
                is_3d=True,
                path="resource/reward_3d/reward_3d.gif",
                title="reward for attack_eps and attack_frequence",
                x_label="atk_eps",
                y_label="atk_frequence",
                z_label="reward",
            )
            # self.ready.emit("reward_3d")
            asr_fig = asr_plt_3d.plt(
                is_2d=False,
                is_3d=True,
                path="resource/asr_3d/asr_3d.gif",
                title="asr for attack_eps and attack_frequence",
                x_label="atk_eps",
                y_label="atk_frequence",
                z_label="asr",
            )
            # self.ready.emit("asr_3d")
            # atk_logits_low.plt(
            #     is_2d=True,
            #     is_3d=False,
            #     path="resource/act_pos/attack/low_rew.png",
            #     title="action position after attack for low reward",
            # )
            atk_logits_med.plt(
                is_2d=True,
                is_3d=False,
                path="resource/act_pos/attack/medium_rew.png",
                title="action position after attack for medium reward",
            )
            # self.ready.emit("medium_rew")
            atk_logits_high.plt(
                is_2d=True,
                is_3d=False,
                path="resource/act_pos/attack/high_rew.png",
                title="action position after attack for high reward",
            )
            # self.ready.emit("high_rew")

            robust_rate = (atk_med_rew - rdm_mean_rew) / (ori_mean_rew - rdm_mean_rew)

            print(
                "在{}攻击算法下，平均攻击成功率为：{:.2f}%".format(
                    self.args.atk_type, asr_mean
                )
            )

            # print(
            #     "在{}攻击算法下，模型的相对鲁棒性为：{:.2f}%".format(
            #         self.args.atk_type, robust_rate
            #     )
            # )
            # if self.args.do_attack:
            # self.my_window.result_1.setText(str(round(np.round(robust_rate[0] * 100))))
            # self.my_window.result_2.setText(str(round((100 - asr_mean))))
            print("在{}攻击算法下，模型的相对鲁棒性为:".format(self.args.atk_type))
            print(robust_rate)

            print("对抗测试结束")
            # self.my_window.test_progress.append("对抗测试结束！\n")

        if (self.args.do_env_change):
            print("adv_env_test begin...")
            # self.my_window.test_progress.append("正在进行环境异化......\n")
            (
                env_stats_frames,
                rew_change_frames,
                env_stats_mean_rew,
                rew_change_mean_rew,
            ) = EnvAdv(
                env_name=self.args.task,
                render_mode="rgb_array",
                policy=self.policy,
                is_rdm=False,
                env_stats_episode=self.args.env_stats_episode,
                rew_change_episode=self.args.rew_change_episode,
            ).run()

            print("环境参数改变后,得到的rew值为:{:.1f}".format(env_stats_mean_rew))
            print("环境奖励结构改变后,得到的rew值为:{:.1f}".format(rew_change_mean_rew))
            # self.my_window.test_progress.append("正在保存结果......\n")
            # 环境异化之后的
            make_gif(
                env_stats_frames,
                is_rdm=False,
                is_atk=False,
                is_env_adv=True,
                env_stats_adv=True,
                env_rew_adv=False,
            )
            # self.ready.emit("stats_adv")
            # 没必要
            make_gif(
                rew_change_frames,
                is_rdm=False,
                is_atk=False,
                is_env_adv=True,
                env_stats_adv=False,
                env_rew_adv=True,
            )
            # if self.args.do_env_change:
            #     self.my_window.result_3.setText(str(env_stats_mean_rew))
            #     self.my_window.result_4.setText(str(rew_change_mean_rew))
            # self.my_window.test_progress.append("测试完毕！")
            print("test end...")
        return (
            asr_mean,
            robust_rate,
            env_stats_mean_rew,
            rew_change_mean_rew
        )


def make_gif(
        frames,
        is_rdm,
        is_atk,
        is_env_adv,
        env_stats_adv,
        env_rew_adv,
        gif_code=None,
):
    plt.figure()
    patch = plt.imshow(frames[0])
    # plt.figure()
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=30)

    if is_env_adv:
        if env_stats_adv:
            gif_path = "resource/env_adv/stats/stats_adv.gif"
        elif env_rew_adv:
            gif_path = "resource/env_adv/rew/rew_adv.gif"

    else:
        if is_rdm:
            gif_path = "resource/random/rdm_result.gif"
        elif is_atk is False:
            gif_path = "resource/origin/ori_result.gif"
        else:
            gif_path = "resource/attack/atk_result_{}.gif".format(gif_code)
    ensure_dir(gif_path)

    anim.save(gif_path, writer="pillow", fps=30)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
