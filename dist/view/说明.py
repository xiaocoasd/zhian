import base64
import os
import streamlit as st
import pandas as pd

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

st.set_page_config(page_title="深度强化学习安全测试系统")


def main_bg(main_bg):
    main_bg_ext = "png"
    # current_dir = os.getcwd()
    # bg_path = os.path.join(current_dir, main_bg)
    bg_path = main_bg
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(bg_path, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


def sidebar_bg(side_bg):
    side_bg_ext = "png"
    current_dir = os.getcwd()
    bg_path = os.path.join(current_dir, side_bg)
    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(bg_path, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )


# st.sidebar.image("../view/pic/logo.png")
# # 使用os.getcwd()和os.path.join()构建相对路径
sidebar_bg("../view/pic/sback.jpg")
main_bg(os.path.join("../view/pic/background.jpg"))
st.write("# 欢迎使用深度强化学习安全测试系统")

st.sidebar.success("点击上传模型和参数开始使用。")

st.markdown(
    """
    #### 测试思路
    """
)

st.image("../view/pic/frame.png")

st.markdown(
    """

    - 本系统聚焦于强化学习模型测试，通过模型泛化性、对抗鲁棒性和模型的可解释性三个角度来评估强化学习模型的可信度和安全性，推动产业智能化的发展。
    - 针对模型的泛化性，在环境异化的思路下，通过负载管理和奖励结构的改变两个测试，观察模型在新的环境下的具体表现，最后综合测试强度和测试结果，最终得到模型的泛化能力。
    - 针对模型的对抗鲁棒性，我们通过不同攻击强度下的白盒黑盒攻击算法，对模型的输入数据加入扰动以近似模拟环境和现实环境之间的差距，观察模型在攻击下的具体表现，进而评估模型的鲁棒能力。
    - 对于模型的可解释性，将在原始环境下的输入数据和模型的动作作为训练数据，使用决策树算法，可以得到模型的具体的决策树。
      通过对决策树的分析，可以得到模型在何种情况下会做出怎样的决策，进而评估模型的可解释性。除此之外，我们还会得到模型的动作空间分布图，
      在图中可以看到模型对于不同动作的选择情况，借此衡量模型的可解释性和公平性。

    """
)

st.markdown(
    """
    #### 测试过程：

    - 冒烟测试：将强化学习模型置于和训练环境一样的环境下进行基础测试，并和采取随机动作的模型进行对比，观察强化学习模型是否是正收益，据此决定是否进行下一步测试。

    - 环境异化：通过对环境的相关属性的修改和奖励结构的改变，来测试和检验模型在相似环境下的具体行为表现，从而衡量模型的泛化能力。
      例如在"CartPole-v1"环境下，可将推杆的重量、长度等相关属性进行修改，在将模型至于此环境下进行测试。

    - 对抗测试：通过经典的攻击算法，对原始输入数据施加扰动，并将加入扰动的数据输入到模型，观察模型的表现，以此近似衡量模型的鲁棒性。
    ###### 我们实现的攻击算法：

    """
)

data = [
    {"攻击方法": "FGSM", "特点": "简单高效：一步操作完成，速度快。"},
    {"攻击方法": "CW", "特点": "扰动小且隐蔽：优化过程中重点在于扰动的最小化"},
    {"攻击方法": "PGDAttack", "特点": "迭代性强：通过多步迭代更细致地调整扰动"},
    {
        "攻击方法": "L2PGDAttack",
        "特点": "扰动特性：专注于L2范数，通常生成的扰动在整体能量上较为均衡",
    },
    {
        "攻击方法": "SparseL1DescentAttack",
        "特点": "扰动稀疏性：扰动集中在少数几个特征，可以造成局部但强烈的影响",
    },
    {"攻击方法": "MI", "特点": "稳定性和效果：使用动量帮助克服局部最小值和噪声的问题"},
    {
        "攻击方法": "ElasticNetL1Attack",
        "特点": "扰动特性：结合L1和L2范数的优化，生成的扰动既有稀疏性也有分散性",
    },
    {
        "攻击方法": "Zeroth Order Optimization",
        "特点": "黑盒攻击：不需要内部梯度信息，适合于没有模型内部访问权限的场景",
    },
]

df = pd.DataFrame(data)

st.dataframe(df)
st.markdown(
    """
    ### 使用说明
    - 首先选择您的的模型的训练算法，如dqn、ppo、ddpg等。之后请将您模型的pth文件传入文件接收器，此外如果您在训练的时候使用了外部的自定义函数（例如在ppo模型的训练过程中需要自定义的dist_fn函数）
      您需要将包含该函数的python文件传入。例如：
"""
)
code = """
import torch
from torch.distributions import Distribution, Independent, Normal
def dist(*logits: torch.Tensor) -> Distribution:
    return Independent(Normal(*logits), 1)
"""

st.code(code, language="python")
st.markdown(
    """
    - 之后选择您的模型所适配的环境，如gymnasium库中的CartPole-v1，Pendulum-v1等，或是基于gymnasium库接口的第三方环境如highway-env。
"""
)
st.markdown(
    """
    - 在页面右侧设置好基本参数如训练设备、测试的回合数等，再根据需求选择是否执行环境异化测试和对抗攻击测试，并依次设置好相关的参数。
      对于环境异化测试的参数设置，您需要设定环境属性改变后测试的回合数以及在奖励结构改变后的测试的回合数，本系统不支持同时改变环境属性和环境的奖励结构；
      此外，如果您需要对抗测试，则还需要选择具体攻击算法，如经典的白盒攻击算法fgsm，pgd等，或是黑盒攻击算法zoo等。还要依据不同算法的不同参数需要，设置相关的参数。
"""
)
st.markdown(
    """
    - 最后点击开始测试，进入测试结果页面后，可看到具体的测试进度，并实时显示测试结果。
"""
)
