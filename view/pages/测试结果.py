import base64
import json
import os.path

import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
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
st.title("测试进度")
count = st_autorefresh(interval=4000, limit=100000, key="fizzbuzzcounter")
if "policy" not in st.session_state:
    st.session_state["policy"] = None
if st.session_state["policy"] is not None:
    with open("../view/display.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    st.write("##### 在这里查看您的测试进度")
    st.text(text)
    try:
        if os.path.exists("resource/tree/tree.png"):
            st.write("## 冒烟测试结果")
            st.write("### 在决策树中直观地了解您模型的决策")
            st.image("resource/tree/tree.png", caption='决策树')
        if os.path.exists("resource/origin/ori_result.gif"):
            st.write("### 原始模型行为")
            st.image("resource/origin/ori_result.gif", caption="待测模型动作")
        if os.path.exists("resource/random/rdm_result.gif"):
            st.write("### 随机模型行为")
            st.write("该模型为系统生成的随机模型，它只会执行随机动作，代表最差行为")
            st.image("resource/random/rdm_result.gif", caption="随机模型动作")
        if os.path.exists("resource/act_pos/origin/result.png"):
            st.write("### 行为预测图")
            st.write("系统将模型神经网络映射到一个二维空间，您可以观察您模型的决策能力")
            st.image("resource/act_pos/origin/result.png")
        if os.path.exists("resource/attack/atk_result_1.gif"):
            st.write("## 对抗测试结果")
            st.write("### 对抗攻击下模型的最差表现")
            st.image("resource/attack/atk_result_1.gif")
        if os.path.exists("resource/attack/atk_result_2.gif"):
            st.write("### 对抗攻击下模型的平均表现")
            st.image("resource/attack/atk_result_2.gif")
        if os.path.exists("resource/attack/atk_result_3.gif"):
            st.write("### 对抗攻击下模型的最好表现")
            st.image("resource/attack/atk_result_3.gif")
        if os.path.exists("resource/act_pos/attack/high_rew.png"):
            st.write("### 对抗攻击下模型动作散点图")
            st.write("##### 神经网络的输出在二维空间的映射")
            st.image("resource/act_pos/attack/high_rew.png")
        if os.path.exists("resource/act_pos/attack/medium_rew.png"):
            st.write("### 对抗攻击下模型一般表现")
            st.image("resource/act_pos/attack/medium_rew.png")
        if os.path.exists("resource/act_pos/attack/low_rew.png"):
            st.write("### 对抗攻击下模型最差表现")
            st.image("resource/act_pos/attack/low_rew.png")
        if os.path.exists("resource/asr_3d/asr_3d.gif") and os.path.exists("resource/reward_3d/reward_3d.gif"):
            st.write("### 攻击差异下的模型行为")
            st.write("系统测试了不同攻击强度和攻击频率下模型的表现，并以此绘制了三维曲面图。您可以观察模型在越来越高的攻击强度和攻击频率下模型的抵抗力。reward为您的模型取得的奖励。ASR（attack "
                     "success rate）为我们攻击的攻击成功率。对于目标攻击，模型做出目标动作记作攻击成功，对于无目标攻击，模型动作改变即视为攻击成功。")
            st.image("resource/asr_3d/asr_3d.gif", caption="攻击成功率三维图")
            st.image("resource/reward_3d/reward_3d.gif", caption="reward三维图")
        if os.path.exists("resource/env_adv/stats/stats_adv.gif"):
            st.write("## 环境异化结果")
            st.write("### 环境异化下模型的一般表现")
            st.image("resource/env_adv/stats/stats_adv.gif")
    except Exception as e:
        pass

    if os.path.exists("../view/result.txt"):
        with open("../view/result.txt", 'r', encoding='utf-8') as f:
            text = f.read()
        values = text.split()  # 按空格分割值
        # 解析每个值并存储到相应的变量中
        atk_succ_rate = round(100 - float(values[0]), 2)
        robust_rate = round(float(values[1]) * 100, 2)
        env_stats_mean_rew = round(float(values[2]), 2)
        rew_change_mean_rew = round(float(values[3]), 2)
        abs_error = round(float(values[4]), 2)
        rsme_error = round(float(values[5]), 2)
        st.write("### 您的模型得分")
        # 初始化一个布尔变量，用于记录条件语句是否都发生了
        all_conditions_met = True

        if atk_succ_rate != 100.00:
            st.write("#### 攻击抵抗力：" + str(atk_succ_rate))
            all_conditions_met = False

        if robust_rate != 0.00:
            st.write("#### 相对鲁棒性：" + str(robust_rate))
            all_conditions_met = False

        if env_stats_mean_rew != 0.00:
            st.write("#### 环境参数改变后的平均奖励：" + str(env_stats_mean_rew))
            all_conditions_met = False

        if rew_change_mean_rew != 0.00:
            st.write("#### 奖励结构改变后的平均奖励：" + str(rew_change_mean_rew))
            all_conditions_met = False

        if abs_error != 0.00:
            st.write("#### 平均绝对误差: " + str(abs_error))
            all_conditions_met = False

        if rsme_error != 0.00:
            st.write("#### 均方根误差: " + str(rsme_error))
            all_conditions_met = False

        if values[6] != "none":
            st.write("#### 相关性矩阵："+values[6])
            all_conditions_met = False

        # 检查所有条件是否都满足
        if all_conditions_met:
            st.write("##### 暂无结果")
        else:
            pass



else:
    st.markdown(
        """
        ### 去上传一个模型吧
    """
    )
