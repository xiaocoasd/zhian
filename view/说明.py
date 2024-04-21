import streamlit as st
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
    智安是一个专为强化学习模型测试而构建的学习平台。
    **👈 从侧边栏选择一个演示**，看看 Streamlit 能做什么吧！
    ### 想了解更多吗？
    - 查看 [streamlit.io](https://streamlit.io)
    - 阅读我们的 [文档](https://docs.streamlit.io)
    - 在我们的 [社区论坛](https://discuss.streamlit.io) 提问
    ### 查看更复杂的示例
    - 使用神经网络来 [分析 Udacity 自动驾驶汽车图像数据集](https://github.com/streamlit/demo-self-driving)
    - 探索一个 [纽约市乘车数据集](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
