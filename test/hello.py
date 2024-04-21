import gymnasium as gym
import torch
import adv_test
import argparse

from typing import cast

from tianshou.env import DummyVectorEnv
from tianshou.utils.space_info import SpaceInfo


""" 
用户需要传递的参数：

已训练好的policy模型
相关参数

----任务名称

--对抗攻击部分：

----步数
----回合
----动作的形状
----使用的policy种类(网络类型)
----对于不同的攻击算法,会使用不同的接口，以下是所有可能需要的接口，需要根据选择的算法展示
------攻击算法的迭代次数
------最大最小像素值
------是否是有目标攻击
------动作形状（数据结构）

--环境异化部分
----改变环境属性后的回合数
----改变奖励结构的回合数
----如果是第三方环境需要传入具体的环境信息
----传入参数信息

"""


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--atk_type", type=str, default="fgsm")
    parser.add_argument("--act_shape", type=any, default=None)
    parser.add_argument("--atk_steps", type=int, default=None)
    parser.add_argument("--atk_episode", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--min_pixel", type=float, default=0.0)
    parser.add_argument("--max_pixel", type=float, default=255.0)
    parser.add_argument("--targeted", type=bool, default=False)
    parser.add_argument("--do_attack", type=bool,default=False)
    parser.add_argument("--do_env_change",type=bool,default=True)
    parser.add_argument("--env_stats_episode",type=int,default=10)
    parser.add_argument("--rew_change_episode",type=int,default=10)
    parser.add_argument("--is_based_on_gym", type=bool, default=True)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--policy-type", type=str, default="dqn")

    return parser.parse_args()


policy = torch.load("../test_model_pth/discrete/dqn.pth")
policy.eval()

args = set_args()

# 获取act_shape
env = gym.make(args.task)
env.action_space = cast(gym.spaces.Discrete, env.action_space)
space_info = SpaceInfo.from_env(env)

args.act_shape = space_info.action_info.action_shape

env = DummyVectorEnv([lambda: gym.make(args.task)])

test = adv_test.top.AdvTestTop(args, policy, env)

test.run()
