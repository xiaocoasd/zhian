import json
from abc import ABC
import gymnasium as gym
import numpy as np

from tianshou.env import DummyVectorEnv

import os
import shutil
from copy import deepcopy

from adv_test.base_test.base_excute import BaseTestExe
from adv_test.algo_adv.algo_top import AlgoAtkAdv
from adv_test.net_adapter.net_top import NetAdapter
from adv_test.env_adv import EnvAdv
from adv_test.show.logits_plt import LogitsPlt
from adv_test.show.tree import TreeTop

# import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

def write_to_file(text):
    # 检查文件是否存在
    if not os.path.exists("../view/display.txt"):
        # 如果文件不存在，创建文件
        with open("../view/display.txt", 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        # 如果文件已存在，在末尾追加内容
        with open("../view/display.txt", 'a', encoding='utf-8') as f:
            f.write(text)
def write_result(text):
    # 检查文件是否存在
    if not os.path.exists("../view/result.txt"):
        # 如果文件不存在，创建文件
        with open("../view/result.txt", 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        # 如果文件已存在，在末尾追加内容
        with open("../view/result.txt", 'a', encoding='utf-8') as f:
            f.write(text)
class AdvTestTop(ABC):

    def __init__(
        self,
        args,
        policy,
        env: gym.Env,
    ) -> None:
        super().__init__()

        self.policy = policy
        self.args = args

        if self.args.is_based_on_gym:
            env_t = gym.make(self.args.task, render_mode="rgb_array")

            if self.args.task == "highway-v0":
                env_t.unwrapped.configure({"offscreen_rendering": True})

            self.env = DummyVectorEnv([lambda: env_t])

        else:
            self.env = deepcopy(env)

        # self.reset()

    def run(self):
        print("test begin...")
        write_to_file("开始测试！\n")
        # 清除已有的数据
        remove_previous_file("resource/random/")
        remove_previous_file("resource/origin/")
        remove_previous_file("resource/attack/")
        remove_previous_file("resource/env_adv/")
        remove_previous_file("resource/tree/")
        remove_previous_file("resource/act_pos/")
        remove_previous_file("resource/asr_3d")
        remove_previous_file("resource/reward_3d")

        asr_mean = 0.0
        rew_change_mean_rew = 0.0
        env_stats_mean_rew = 0.0
        abs_error = 0.0
        rsme_error = 0.0
        corr_matrix = [[0, 0], [0, 0]]

        atk_succ_rate = 0.0
        robust_rate = [0.0]

        net_t = NetAdapter(
            policy_type=self.args.policy_type,
            policy=deepcopy(self.policy),
            device=self.args.device,
        )
        write_to_file("正在进行冒烟测试......\n")
        print("冒烟测试开始")
        (
            rdm_frames_array,
            ori_frames_array,
            rdm_mean_rew,
            ori_mean_rew,
            ori_all_rew,
            ori_tree_input,
            ori_tree_output,
            ori_pos_array,
        ) = BaseTestExe(
            policy=self.policy,
            env=self.env,
            test_episode=self.args.atk_episode,
            net=net_t,
            atk_frequence=None,
            adv_obs_gene=None,
        ).run()

        ori_tree_input_value = [item for sublist in ori_tree_input for item in sublist]
        ori_tree_output_value = [
            item for sublist in ori_tree_output for item in sublist
        ]

        # print(len(ori_tree_output[0]))
        # print(ori_tree_output[0])
        # print(len(ori_tree_output_value))
        # print(len(ori_pos_array[0]))
        
        TreeTop(
            tree_input=ori_tree_input_value,
            tree_output=ori_tree_output_value,
            path="resource/tree/tree.png",
            env_name=self.args.task,
        )

        ori_pos_all = [item for sublist in ori_pos_array for item in sublist]
        ori_act_all = [item for sublist in ori_tree_output for item in sublist]

        ori_logits = LogitsPlt(
            data=ori_pos_all,
            act_data=ori_act_all,
        )

        ori_logits.plt(
            is_2d=True,
            is_3d=False,
            path="resource/act_pos/origin/result.png",
            title="action position for origin policy",
        )

        rdm_rdm_index = np.random.choice(len(rdm_frames_array), replace=False)
        ori_rdm_index = np.random.choice(len(ori_frames_array), replace=False)

        rdm_rdm_frame = rdm_frames_array[rdm_rdm_index]
        ori_rdm_frame = ori_frames_array[ori_rdm_index]
        write_to_file("正在打印随机模型行为......\n")
        make_gif(
            rdm_rdm_frame,
            is_rdm=True,
            is_atk=False,
            is_env_adv=False,
            env_rew_adv=False,
            env_stats_adv=False,
        )
        write_to_file("正在打印待测模型原始行为......\n")
        make_gif(
            ori_rdm_frame,
            is_rdm=False,
            is_atk=False,
            is_env_adv=False,
            env_rew_adv=False,
            env_stats_adv=False,
        )

        print("冒烟测试结束")
        write_to_file("冒烟测试结束！\n")
        if self.args.do_attack:
            print("对抗攻击开始")
            write_to_file("\n对抗攻击开始！\n")
            (
                atk_low_frames,
                atk_med_frames,
                atk_high_frames,
                atk_med_rew,
                atk_med_all_rew,
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
                args=self.args,
                policy=self.policy,
                net=net_t,
                env=self.env,
            ).run()

            print("对抗攻击结束")
            write_to_file("对抗攻击结束！\n")
            write_to_file("正在保存图片......\n")
            print("开始保存图片")
            write_to_file("正在保存对抗攻击下最差表现gif......\n")
            make_gif(
                atk_low_frames,
                is_rdm=False,
                is_atk=True,
                is_env_adv=False,
                env_stats_adv=False,
                env_rew_adv=False,
                gif_code=1,
            )
            write_to_file("正在保存对抗攻击下一般表现gif......\n")
            make_gif(
                atk_med_frames,
                is_rdm=False,
                is_atk=True,
                is_env_adv=False,
                env_stats_adv=False,
                env_rew_adv=False,
                gif_code=2,
            )
            write_to_file("正在保存对抗攻击下最优表现gif......\n")
            make_gif(
                atk_high_frames,
                is_rdm=False,
                is_atk=True,
                is_env_adv=False,
                env_stats_adv=False,
                env_rew_adv=False,
                gif_code=3,
            )
            # 处理数据

            succ_rate_value = np.concatenate(atk_succ_rate_array)
            rew_value = np.concatenate(atk_rew_array)

            sequence = np.arange(0.1, 1.1, 0.1)

            atk_frequence = np.tile(sequence, (100 // len(sequence) + 1))[:100]
            atk_eps = np.repeat(sequence, 10)
            rew_plt_value = [list(t) for t in zip(atk_eps, atk_frequence, rew_value)]
            asr_plt_value = [
                list(t) for t in zip(atk_eps, atk_frequence, succ_rate_value)
            ]

            atk_logits_high = LogitsPlt(data=atk_high_pos, act_data=atk_high_act)
            atk_logits_low = LogitsPlt(data=atk_low_pos, act_data=atk_low_act)
            atk_logits_med = LogitsPlt(data=atk_med_pos, act_data=atk_med_act)
            rew_plt_3d = LogitsPlt(data=rew_plt_value)
            asr_plt_3d = LogitsPlt(data=asr_plt_value)
            write_to_file("正在保存模型奖励三维图......\n")
            rew_plt_3d.plt(
                is_2d=False,
                is_3d=True,
                path="resource/reward_3d/reward_3d.gif",
                title="reward for attack_eps and attack_frequence",
                x_label="atk_eps",
                y_label="atk_frequence",
                z_label="reward",
            )
            write_to_file("正在保存攻击成功率三维图......\n")
            asr_plt_3d.plt(
                is_2d=False,
                is_3d=True,
                path="resource/asr_3d/asr_3d.gif",
                title="asr for attack_eps and attack_frequence",
                x_label="atk_eps",
                y_label="atk_frequence",
                z_label="asr",
            )
            write_to_file("正在保存模型行为散点图......\n")
            atk_logits_high.plt(
                is_2d=True,
                is_3d=False,
                path="resource/act_pos/attack/high_rew.png",
                title="action position after attack for high reward",
            )
            atk_logits_med.plt(
                is_2d=True,
                is_3d=False,
                path="resource/act_pos/attack/medium_rew.png",
                title="action position after attack for medium reward",
            )
            atk_logits_low.plt(
                is_2d=True,
                is_3d=False,
                path="resource/act_pos/attack/low_rew.png",
                title="action position after attack for low reward",
            )

            robust_rate = (atk_med_rew - rdm_mean_rew) / (ori_mean_rew - rdm_mean_rew)

            abs_error = np.mean(
                np.abs(
                    np.array(ori_all_rew).flatten()
                    - np.array(atk_med_all_rew).flatten()
                )
            )
            rsme_error = np.sqrt(
                np.mean(
                    np.square(
                        np.array(ori_all_rew).flatten()
                        - np.array(atk_med_all_rew).flatten()
                    )
                )
            )

            print("mae={}".format(abs_error))
            print("RSME:{0}".format(rsme_error))

            corr_matrix = np.corrcoef(
                np.array(ori_all_rew).flatten() + 1e-5,
                np.array(atk_med_all_rew).flatten(),
            )
            print("corr of pred and gt:{}".format(corr_matrix[0, 1]))

            print(
                "在{}攻击算法下，平均攻击成功率为：{:.2f}%".format(
                    self.args.atk_type, asr_mean
                )
            )

            print(
                "在{}攻击算法下，模型的相对鲁棒性为:{:.2f}".format(
                    self.args.atk_type, robust_rate[0]
                )
            )
            # print(robust_rate)

            print("对抗测试结束")
            write_to_file("对抗攻击测试结束！\n")
        if self.args.do_env_change:
            print("环境异化测试开始..")
            write_to_file("\n环境异化测试开始!\n")
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
                args=self.args,
            ).run()

            print("环境参数改变后,得到的rew值为:{:.1f}".format(env_stats_mean_rew))
            print("环境奖励结构改变后,得到的rew值为:{:.1f}".format(rew_change_mean_rew))

            make_gif(
                env_stats_frames,
                is_rdm=False,
                is_atk=False,
                is_env_adv=True,
                env_stats_adv=True,
                env_rew_adv=False,
            )
            # 没必要
            make_gif(
                rew_change_frames,
                is_rdm=False,
                is_atk=False,
                is_env_adv=True,
                env_stats_adv=False,
                env_rew_adv=True,
            )
            write_to_file("环境异化测试结束！\n")
            print("环境异化测试结束")

        print("test end...")
        if np.any(np.isnan(corr_matrix)) or np.all(np.array(corr_matrix) == [[0,0],[0,0]]):
            write_result(str(asr_mean) + " " + str(robust_rate[0]) + " " + str(env_stats_mean_rew) + " " + str(
                rew_change_mean_rew) + " " + str(abs_error) + " " + str(rsme_error) + " none")
        else:
            list_corr_matrix = [round(corr_matrix[0][0], 2), round(corr_matrix[0][1], 2), round(corr_matrix[1][0], 2), round(corr_matrix[1][1], 2)]
            corr_matrix_str = json.dumps(list_corr_matrix).replace("\n", "").replace(" ", "")
            write_result(str(asr_mean) + " " + str(robust_rate[0]) + " " + str(env_stats_mean_rew) + " " + str(
                rew_change_mean_rew) + " " + str(abs_error) + " " + str(rsme_error) + " " + corr_matrix_str)
        write_to_file("测试结束！\n")
        return (
            asr_mean,
            robust_rate,
            env_stats_mean_rew,
            rew_change_mean_rew,
            abs_error,
            rsme_error,
            corr_matrix,
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
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=60)

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

    anim.save(gif_path, writer="pillow", fps=15)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def remove_previous_file(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
