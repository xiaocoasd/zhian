import argparse
import base64
import os
import shutil
import gymnasium as gym
from typing import cast
import streamlit as st
from tianshou.env import DummyVectorEnv
from tianshou.utils.space_info import SpaceInfo
import torch
import sys

sys.path.append('./')
import adv_test
 

def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


def sidebar_bg(side_bg):
    side_bg_ext = 'png'

    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )


# 调用
sidebar_bg('../view/pic/sback.jpg')

# 调用
main_bg('../view/pic/background.jpg')


def my_test(policy, args):
    # policy.eval()
    env = gym.make(args.task)
    env.action_space = cast(gym.spaces.Discrete, env.action_space)
    space_info = SpaceInfo.from_env(env)
    args.act_shape = space_info.action_info.action_shape
    env = DummyVectorEnv([lambda: gym.make(args.task)])
    # st.write("###### 正在执行模型测试，去“测试结果”看看吧")
    test = adv_test.top.AdvTestTop(args, policy, env)
    atk_succ_rate, robust_rate, env_stats_mean_rew, rew_change_mean_rew, abs_error, rsme_error, corr_matrix, = test.run()
    return atk_succ_rate, robust_rate, env_stats_mean_rew, rew_change_mean_rew, abs_error, rsme_error, corr_matrix


if "slider1" in st.session_state:
    st.session_state["slider1"] = st.session_state["slider1"]
if "slider2" in st.session_state:
    st.session_state["slider2"] = st.session_state["slider2"]
if "slider3" in st.session_state:
    st.session_state["slider3"] = st.session_state["slider3"]
if "slider4" in st.session_state:
    st.session_state["slider4"] = st.session_state["slider4"]
if "slider5" in st.session_state:
    st.session_state["slider5"] = st.session_state["slider5"]
if "selectbox0" in st.session_state:
    st.session_state["selectbox0"] = st.session_state["selectbox0"]
if "selectbox1" in st.session_state:
    st.session_state["selectbox1"] = st.session_state["selectbox1"]
if "selectbox2" in st.session_state:
    st.session_state["selectbox2"] = st.session_state["selectbox2"]
if "selectbox3" in st.session_state:
    st.session_state["selectbox3"] = st.session_state["selectbox3"]
if "checkbox1" in st.session_state:
    st.session_state["checkbox1"] = st.session_state["checkbox1"]
if "checkbox2" in st.session_state:
    st.session_state["checkbox2"] = st.session_state["checkbox2"]
if "checkbox3" in st.session_state:
    st.session_state["checkbox3"] = st.session_state["checkbox3"]
if "checkbox4" in st.session_state:
    st.session_state["checkbox4"] = st.session_state["checkbox4"]
if "checkbox5" in st.session_state:
    st.session_state["checkbox5"] = st.session_state["checkbox5"]
if "checkbox6" in st.session_state:
    st.session_state["checkbox6"] = st.session_state["checkbox6"]
if "number_input1" in st.session_state:
    st.session_state["number_input1"] = st.session_state["number_input1"]
if "number_input2" in st.session_state:
    st.session_state["number_input2"] = st.session_state["number_input2"]
if "number_input3" in st.session_state:
    st.session_state["number_input3"] = st.session_state["number_input3"]
if "number_input4" in st.session_state:
    st.session_state["number_input4"] = st.session_state["number_input4"]

st.title("上传模型")

with st.sidebar:
    st.sidebar.warning("设置测试参数")
    st.title("设置参数")
    st.session_state["device"] = st.selectbox("您的设备", ["cpu", "cuda"], key="selectbox0")
    st.session_state["atk_episode"] = st.number_input("测试回合数", min_value=0, max_value=500, step=1,
                                                      key="number_input3")
    st.session_state["do_env_change"] = st.checkbox("是否执行环境异化", key="checkbox1")
    if st.session_state["do_env_change"]:

        st.session_state["env_stats_episode"] = st.number_input("改变环境属性后的回合数：", min_value=0,
                                                                max_value=500,
                                                                step=10, key="number_input1")
        st.session_state["rew_change_episode"] = st.number_input("改变奖励结构后的回合数：", min_value=0,
                                                                 max_value=500,
                                                                 step=10, key="number_input2")
    else:
        pass
    st.session_state["do_attack"] = st.checkbox("是否执行对抗攻击：", key="checkbox2")

    if st.session_state["do_attack"]:
        list3 = ["fgsm", "fgm", "GradientSignAttack", "cw",
                 "CarliniWagnerL2Attack",
                 "pgda",
                 "pgd", "PGDAttack", "LinfPGDAttack", "L2PGDAttack",
                 "SparseL1DescentAttack",
                 "MomentumIterativeAttack", "LinfMomentumIterativeAttack",
                 "ElasticNetL1Attack"]
        st.session_state["atk_type"] = st.selectbox("选择攻击方法", list3, key="selectbox1")
        st.session_state["n_iter"] = st.number_input("攻击迭代次数", min_value=0, max_value=500, step=10, key="slider1")
        st.session_state["targeted"] = st.checkbox("执行目标攻击", key="checkbox3")
        if st.session_state["atk_type"] == "cw" or st.session_state["atk_type"] == "CarliniWagnerL2Attack" or \
                st.session_state[
                    "atk_type"] == "ElasticNetL1Attack":
            st.session_state["learning_rate"] = st.slider("学习率", max_value=1.00, min_value=0.00, step=0.01,
                                                          key="slider2")
            st.session_state["confidence"] = st.slider("置信度", max_value=1.0, min_value=0.0, step=0.1, key="slider3")
            st.session_state["binary_search_steps"] = st.number_input("二分搜索步数", min_value=1, max_value=50,
                                                                      step=1, key="number_input4")
            st.session_state["abort_early"] = st.checkbox("提前终止", key="checkbox4")
            st.session_state["initial_const"] = st.slider("初始常数", min_value=0.000, max_value=1.000, step=0.001,
                                                          key="slider4")
        if st.session_state["atk_type"] == "pgd" or st.session_state["atk_type"] == "SparseL1DescentAttack":
            st.session_state["rand_init"] = st.checkbox("随机初始样本", key="checkbox5")
            st.session_state["l1_sparsity"] = st.slider("样本稀疏性", min_value=0.00, max_value=1.00, step=0.01,
                                                        key="slider5")

if "policy" not in st.session_state:
    st.session_state["policy"] = None
list1 = ["dqn", "drqn", "ddqn", "ppo", "a2c", "drpo", "ddpg", "td3", "sac", "pg"]
st.session_state["policy_type"] = st.selectbox("选择您的模型类别：", list1, key="selectbox2")
thild_fn = st.file_uploader("如果您在训练时使用了自定义函数，上传您包含该函数的.py文件", type="py")
if thild_fn is not None:
    file_contents = thild_fn.getvalue().decode("utf-8")
    # 执行上传的文件
    exec(file_contents)
if st.session_state["policy"] is None:
    uploaded_file = st.file_uploader("将模型文件拖放到此处", type="pth")
    if uploaded_file is not None:
        try:
            st.write("以下是你上传的文件：")
            policy = torch.load(uploaded_file)
            st.session_state['policy'] = policy
            st.write(st.session_state['policy'])
        except Exception as e:
            st.error("上传模型失败，可能是您使用了自定义函数！")
else:
    uploaded_file = st.file_uploader("将文件拖放到此处", type=None)
    try:
        if uploaded_file is not None:
            policy = torch.load(uploaded_file)
            st.session_state['policy'] = policy
        st.write("### 当前模型：")
        st.write(st.session_state['policy_type'])
        st.write(st.session_state['policy'])
    except Exception as e:
        st.error("上传模型失败，可能是您使用了自定义函数！")

list2 = ["CartPole-v1", "Pendulum-v1", "Pong-v0", "Breakout-v0",
         "SpaceInvaders-v0",
         "Humanoid-v3", "HalfCheetah-v3", "Ant-v3", "Taxi-v3", "highway-v0"]
st.session_state["task"] = st.selectbox("选择环境", list2, key="selectbox3")
if st.session_state["task"] == "CartPole-v1":
    st.image("../view/gif/cart_pole.gif", caption="当前环境")
elif st.session_state["task"] == "Pendulum-v1":
    st.image("../view/gif/pendulum.gif", caption="当前环境")
elif st.session_state["task"] == "Pong-v0":
    st.image("../view/gif/pong.gif", caption="当前环境")
elif st.session_state["task"] == "Breakout-v0":
    st.image("../view/gif/breakout.gif", caption="当前环境")
elif st.session_state["task"] == "SpaceInvaders-v0":
    st.image("../view/gif/space_invaders.gif", caption="当前环境")
elif st.session_state["task"] == "Humanoid-v3":
    st.image("../view/gif/humanoid.gif", caption="当前环境")
elif st.session_state["task"] == "HalfCheetah-v3":
    st.image("../view/gif/half_cheetah.gif", caption="当前环境")
elif st.session_state["task"] == "Ant-v3":
    st.image("../view/gif/ant.gif", caption="当前环境")
elif st.session_state["task"] == "Taxi-v3":
    st.image("../view/gif/taxi.gif", caption="当前环境")
elif st.session_state["task"] == "highway-v0":
    st.image("../view/gif/highway.gif", caption="当前环境")
st.session_state["is_not_based_on_gym"] = st.checkbox("使用其他环境", key="checkbox6")

test_begin = st.button("开始测试", type="primary")
if test_begin:
    if "robust_rate" in st.session_state:
        del st.session_state["robust_rate"]
    if "asr_mean" in st.session_state:
        del st.session_state["asr_mean"]
    if "env_stats_mean_rew" in st.session_state:
        del st.session_state["env_stats_mean_rew"]
    if "rew_change_mean_rew" in st.session_state:
        del st.session_state["rew_change_mean_rew"]

    if os.path.exists("../view/display.txt"):
        with open("../view/display.txt", 'w') as f:
            f.write('')
    if os.path.exists("../view/result.txt"):
        os.remove("../view/result.txt")
    if os.path.exists("resource/random/"):
        shutil.rmtree("resource/random/")
    if os.path.exists("resource/origin/"):
        shutil.rmtree("resource/origin/")
    if os.path.exists("resource/attack/"):
        shutil.rmtree("resource/attack/")
    if os.path.exists("resource/env_adv/"):
        shutil.rmtree("resource/env_adv/")
    if os.path.exists("resource/tree/"):
        shutil.rmtree("resource/tree/")
    if os.path.exists("resource/act_pos"):
        shutil.rmtree("resource/act_pos")
    if os.path.exists("resource/asr_3d"):
        shutil.rmtree("resource/asr_3d")
    if os.path.exists("resource/reward_3d"):
        shutil.rmtree("resource/reward_3d")
    args = argparse.Namespace()
    args.task = st.session_state["task"]
    args.atk_eps = 0.2
    if "rand_init" in st.session_state:
        args.rand_init = st.session_state["rand_init"]
    if "atk_type" in st.session_state:
        args.atk_type = st.session_state["atk_type"]
    args.policy_type = st.session_state["policy_type"]
    args.do_env_change = st.session_state["do_env_change"]
    args.do_attack = st.session_state["do_attack"]
    args.atk_frequence = 0.2
    args.action_shape = None
    args.atk_episode = st.session_state["atk_episode"]
    args.atk_steps = None
    if "n_iter" in st.session_state:
        args.n_iter = st.session_state["n_iter"]
    args.max_pixel = 255.0
    args.min_pixel = 0.0
    if "targeted" in st.session_state:
        args.targeted = st.session_state["targeted"]
    args.is_based_on_gym = not st.session_state["is_not_based_on_gym"]
    args.device = st.session_state["device"]
    if "env_stats_episode" in st.session_state:
        args.env_stats_episode = st.session_state["env_stats_episode"]
    if "rew_change_episode" in st.session_state:
        args.rew_change_episode = st.session_state["rew_change_episode"]
    if "confidence" in st.session_state:
        args.confidence = st.session_state["confidence"]
    if "learning_rate" in st.session_state:
        args.learning_rate = st.session_state["learning_rate"]
    if "initial_const" in st.session_state:
        args.initial_const = st.session_state["initial_const"]
    if "l1_sparsity" in st.session_state:
        args.l1_sparsity = st.session_state["l1_sparsity"]
    if "binary_search_steps" in st.session_state:
        args.binary_search_steps = st.session_state["binary_search_steps"]
    if "abort_early" in st.session_state:
        args.abort_early = st.session_state["abort_early"]
    if st.session_state['policy'] is not None:
        if args.atk_episode == 0:
            st.error("测试回合数不能为0！")
        else:
            policy = st.session_state['policy']
            with st.spinner("正在执行测试，请稍候..."):
                atk_succ_rate, robust_rate, env_stats_mean_rew, rew_change_mean_rew, abs_error, rsme_error, corr_matrix = my_test(
                    policy, args)
                st.success("测试完成！去测试结果中查看吧！")
    else:
        st.error("当前还没有上传模型")
