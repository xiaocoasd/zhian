import streamlit as st
import pandas as pd
from st_aggrid import AgGrid,GridOptionsBuilder
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
st.set_page_config(
    page_title="智安一体化测试平台"
)
st.write("# 欢迎使用智安一体化测试平台")

st.sidebar.success("点击上传模型和参数开始使用。")

st.markdown(
    """
    智安是一个专为强化学习模型测试而构建的学习平台。\n
    ###### 我们提供的测试方法：
    - 环境异化：通过人为的修改环境条件来测试和提高模型在不同环境下的性能的方法。我们尝试更改环境状态，以及奖励机制，从而衡量模型的泛化能力。
    
    - 数据增强：在深度学习中指用于对输入数据施加扰动的策略。借助这一思想，我们对模型的自身属性进行调整，比如将“Cart Pole”的推杆长度进行调整，从而测试负载变化条件下的鲁棒性。
    ###### 我们实现的攻击算法：
    """)
# 创建表格数据
data = [
    {"攻击方法": "FGSM(Fast Gradient Sign Method)", "特点": "简单高效：一步操作完成，速度快。"},
    {"攻击方法": "CW2 (Carlini & Wagner L2 Attack)", "特点": "扰动小且隐蔽：优化过程中重点在于扰动的最小化"},
    {"攻击方法": "PGDAttack (Projected Gradient Descent Attack)", "特点": "迭代性强：通过多步迭代更细致地调整扰动"},
    {"攻击方法": "L2PGDAttack (Projected Gradient Descent Attack with L2 norm)", "特点": "扰动特性：专注于L2范数，通常生成的扰动在整体能量上较为均衡"},
    {"攻击方法": "SparseL1DescentAttack", "特点": "扰动稀疏性：扰动集中在少数几个特征，可以造成局部但强烈的影响"},
    {"攻击方法": "MI (Momentum Iterative Attack)", "特点": "稳定性和效果：使用动量帮助克服局部最小值和噪声的问题"},
    {"攻击方法": "ElasticNetL1Attack", "特点": "扰动特性：结合L1和L2范数的优化，生成的扰动既有稀疏性也有分散性"},
    {"攻击方法": "Zeroth Order Optimization", "特点": "黑盒攻击：不需要内部梯度信息，适合于没有模型内部访问权限的场景"}
]

# 将数据转换为DataFrame对象
df = pd.DataFrame(data)

# 创建 GridOptionsBuilder 对象
gob = GridOptionsBuilder.from_dataframe(df)

# 配置默认列选项
gob.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True, wrapText=True, autoHeight=True)

# 配置特定列选项
gob.configure_column("攻击方法", pinned="left", order=1)
gob.configure_column("特点", pinned="right", order=2)

# 生成 grid options
grid_options = gob.build()

# 绘制表格
AgGrid(df, grid_options, height=300)
st.markdown(
    """
    ### 使用说明
    - 将您模型的.pth文件传入文件接收器，如果您在训练的时候使用了自定义函数（例如ppo的训练需要dist_fn）
      您可以将包含该函数的python文件传入。例如：
"""
)
code = """
import torch
from torch.distributions import Distribution, Independent, Normal
def dist(*logits: torch.Tensor) -> Distribution:
    return Independent(Normal(*logits), 1)
"""

st.code(code, language='python')
st.markdown(
    """
    
    - 设置测试参数后点击开始测试即可在测试结果中查看
    
"""
)
st.markdown(
    """
    ### 想了解更多吗？
    - 查看我们的 [git仓库](https://github.com/xiaocoasd/zhian)
"""
)
