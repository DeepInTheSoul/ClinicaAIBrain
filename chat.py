# UI框架
import os  # 导入操作系统模块
import streamlit as st  # 导入 Streamlit 并简写为 st
from rag import rag_chain, rag_chain_v2  # 从 rag 模块导入 rag_chain 函数
from dotenv import dotenv_values  # 从 dotenv 模块导入 dotenv_values 函数
from langchain_openai import ChatOpenAI  # 从 langchain_openai 模块导入 ChatOpenAI
from langchain_community.llms import QianfanLLMEndpoint  # 从 langchain_community.llms 模块导入 QianfanLLMEndpoint

# 加载 .env 文件中的环境变量
config = dotenv_values(".env")

# 设置 Qianfan 的 AK 和 SK 环境变量
os.environ["QIANFAN_AK"] = "rYndCW8UyNrh7ZIxAmxG0w1X"
os.environ["QIANFAN_SK"] = "KovKWoaJeKYeIQwLgOUxFof5KI1ggTRq"

# 初始化不同的大模型接口
ERNIE4_llm = QianfanLLMEndpoint(model="ERNIE-4.0-8K", streaming=False)
Yi_llm = QianfanLLMEndpoint(model="Yi-34B-Chat", streaming=False)
Llama_llm = QianfanLLMEndpoint(model="Meta-Llama-3-8B", streaming=False)
ChatGLM_llm = QianfanLLMEndpoint(model="ChatGLM2-6B-32K", streaming=False)

# 定义聊天页面函数
def chat_page():
    # UI界面的边栏，通过下拉列表切换LLM模型
    with st.sidebar:
        pattern_list = [
            "大模型问答",  # 对话模式选项
            "医疗知识库问答",
        ]

        model_list = [
            "文心一言4.0",  # 对话模型选项
            "ChatGLM-6B",
            "Llama3-8B",
            "Yi-34B",
        ]
        pattern = st.selectbox("对话模式选择：",  # 创建选择框供用户选择对话模式
                             pattern_list,
                             index=0
                             )
        st.write('You selected:', pattern)  # 显示用户选择的对话模式

        model = st.selectbox("对话模型选择：",  # 创建选择框供用户选择对话模型
                             model_list,
                             index=0
                             )
        st.write('You selected:', model)  # 显示用户选择的对话模型

    st.title("🤖 ClinicaAIBrain：智能医疗大脑")  # 设置页面标题
    # 聊天历史记录初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 聊天界面展示历史聊天
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
             st.markdown(message["content"])

    # 根据选择的对话模式和模型，初始化对应的 LLM 实例
    if pattern == "大模型问答":
        if model == "文心一言4.0":
            llm = ERNIE4_llm
        elif model == "ChatGLM-6B":
            llm = ChatGLM_llm
        elif model == "Llama3-8B":
            llm = Llama_llm
        elif model == "Yi-34B":
            llm = Yi_llm
    elif pattern == "医疗知识库问答":
        if model == "文心一言4.0":
            llm = rag_chain(ERNIE4_llm)
        elif model == "ChatGLM-6B":
            llm = rag_chain(ChatGLM_llm)
        elif model == "Llama3-8B":
            llm = rag_chain(Llama_llm)
        elif model == "Yi-34B":
            llm = rag_chain(Yi_llm)

    # 获取用户输入的聊天消息
    if chat_input := st.chat_input("What is up?"):
        st.chat_message("user").markdown(chat_input)  # 在聊天界面显示用户的消息
        # 将用户消息添加到聊天记录
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("assistant"):
            # 以流式输出的方式调用 LLM 的 API，并在 UI 界面显示
            response = st.write_stream(
                llm.stream(chat_input))
        # 将大模型的消息添加到聊天记录
        st.session_state.messages.append({"role": "assistant", "content": response})



# 定义聊天页面函数
def chat_page_v2():
    # UI界面的边栏，通过下拉列表切换LLM模型
    with st.sidebar:
        pattern_list = [
            "医疗知识库问答",
        ]

        model_list = [
            "文心一言4.0",  # 对话模型选项
            "ChatGLM-6B",
            "Llama3-8B",
            "Yi-34B",
        ]
        pattern = st.selectbox("对话模式选择：",  # 创建选择框供用户选择对话模式
                             pattern_list,
                             index=0
                             )
        st.write('You selected:', pattern)  # 显示用户选择的对话模式

        model = st.selectbox("对话模型选择：",  # 创建选择框供用户选择对话模型
                             model_list,
                             index=0
                             )
        st.write('You selected:', model)  # 显示用户选择的对话模型

    st.title("🤖 ClinicaAIBrain：智能医疗大脑")  # 设置页面标题
    # 聊天历史记录初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 聊天界面展示历史聊天
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
             st.markdown(message["content"])

    # 根据选择的对话模式和模型，初始化对应的 LLM 实例
    if pattern == "大模型问答":
        if model == "文心一言4.0":
            llm = ERNIE4_llm
        elif model == "ChatGLM-6B":
            llm = ChatGLM_llm
        elif model == "Llama3-8B":
            llm = Llama_llm
        elif model == "Yi-34B":
            llm = Yi_llm
    elif pattern == "医疗知识库问答":
        if model == "文心一言4.0":
            llm = ERNIE4_llm
        elif model == "ChatGLM-6B":
            llm = ChatGLM_llm
        elif model == "Llama3-8B":
            llm = Llama_llm
        elif model == "Yi-34B":
            llm = Yi_llm

    # 获取用户输入的聊天消息
    if chat_input := st.chat_input("What is up?"):
        st.chat_message("user").markdown(chat_input)  # 在聊天界面显示用户的消息
        # 将用户消息添加到聊天记录
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("assistant"):
            # 以流式输出的方式调用 LLM 的 API，并在 UI 界面显示
            response = st.write(
                rag_chain_v2(llm,chat_input))
        # 将大模型的消息添加到聊天记录
        st.session_state.messages.append({"role": "assistant", "content": response})